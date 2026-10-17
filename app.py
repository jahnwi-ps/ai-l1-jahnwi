import os
import sys
import shutil
import threading
from functools import lru_cache
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from ollama import Client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['DB_FOLDER'] = os.path.join(BASE_DIR, 'db_chroma')
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB upload limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Configuration (set in .env — see .env.example) ---
API_KEY = os.getenv("OLLAMA_API_KEY")
if not API_KEY:
    sys.exit("Error: OLLAMA_API_KEY missing. Copy .env.example to .env and set your key.")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-coder:480b-cloud")
ALLOWED_EXTENSIONS = {'.pdf', '.txt'}

ollama_client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + API_KEY}
)

# Global RAG components (guarded by rag_lock — Flask serves requests on multiple threads)
vectorstore = None
retriever = None
rag_lock = threading.Lock()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


@lru_cache(maxsize=1)
def get_embedding_function():
    """Returns the local HuggingFace embedding model (zero cost, no API needed). Loaded once."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_document(path):
    if path.lower().endswith('.pdf'):
        return PyPDFLoader(path).load()
    return TextLoader(path, encoding='utf-8').load()


def init_rag_system():
    """
    Loads data.txt plus any previously uploaded files, chunks them, embeds them,
    and stores them in ChromaDB. Called once at startup; can also be triggered lazily on first chat.
    """
    global vectorstore, retriever

    paths = [os.path.join(BASE_DIR, 'data.txt')]
    upload_dir = app.config['UPLOAD_FOLDER']
    paths += [os.path.join(upload_dir, f) for f in sorted(os.listdir(upload_dir))
              if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]

    docs = []
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            docs.extend(load_document(path))
        except Exception as e:
            print(f"Skipping {path}: {e}")

    # Step 1 — Chunking
    splits = text_splitter.split_documents(docs)
    if not splits:
        print("No data found. Upload a file or create data.txt.")
        return False

    # Step 2 — Embedding + ChromaDB storage
    print("Initializing ChromaDB Vector Store...")
    embedding_function = get_embedding_function()

    db_path = app.config['DB_FOLDER']
    if os.path.exists(db_path):
        shutil.rmtree(db_path)  # Rebuild fresh on each startup

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=db_path
    )
    retriever = vectorstore.as_retriever()
    print(f"RAG System Initialized with ChromaDB ({len(splits)} chunks).")
    return True


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/readme')
def readme():
    return send_from_directory(BASE_DIR, 'README.html')


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File is too large (max 20 MB)."}), 413


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    if not filename or os.path.splitext(filename)[1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only PDF and TXT files are supported."}), 400

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    global vectorstore, retriever

    try:
        splits = text_splitter.split_documents(load_document(save_path))
    except Exception as e:
        print(f"Failed to parse {filename}: {e}")
        os.remove(save_path)
        return jsonify({"error": "Could not read that file. Is it a valid PDF/UTF-8 text file?"}), 400

    if not splits:
        os.remove(save_path)
        return jsonify({"error": "No text could be extracted from that file."}), 400

    with rag_lock:
        if vectorstore is None:
            # First upload — create a new ChromaDB collection
            vectorstore = Chroma.from_documents(splits, get_embedding_function(),
                                                persist_directory=app.config['DB_FOLDER'])
        else:
            # Subsequent uploads — add to existing collection
            vectorstore.add_documents(splits)
        retriever = vectorstore.as_retriever()

    return jsonify({"message": "File processed and knowledge base updated!"})


@app.route('/chat', methods=['POST'])
def chat():
    global retriever
    # Lazy init if the server started without data
    with rag_lock:
        if retriever is None and not init_rag_system():
            return jsonify({"answer": "I have no knowledge yet. Please upload a document."})

    data = request.get_json(silent=True) or {}
    question = (data.get('message') or '').strip()

    if not question:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Step 3 — Retrieval: find relevant chunks from ChromaDB
        relevant_docs = retriever.invoke(question)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])

        # Step 4 — Prompt construction (RAG prompt)
        system_prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the following context. If the answer is not in the context, say you don't know.

Context:
{context_text}
"""

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ]

        # Step 5 — Generation via Ollama Cloud LLM (Qwen3-Coder 480b)
        response = ollama_client.chat(model=MODEL_NAME, messages=messages, stream=False)
        answer = response.message.content

        return jsonify({"answer": answer})

    except Exception as e:
        # Log details server-side; don't leak internals to the browser
        print(f"Ollama API Error: {e}")
        return jsonify({"error": "Error interacting with AI. Check the server logs."}), 500


if __name__ == '__main__':
    try:
        init_rag_system()
    except Exception as e:
        print(f"Initial load skipped: {e}")

    # Debug mode exposes an interactive code console — only enable it locally via FLASK_DEBUG=1
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, use_reloader=False, port=int(os.getenv("PORT", "5001")))
