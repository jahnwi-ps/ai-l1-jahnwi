@echo off
echo Starting AI Assistant...
if not exist .env (
    echo .env not found. Copy .env.example to .env and set OLLAMA_API_KEY first.
    pause
    exit /b 1
)
pip install -r requirements.txt
start http://127.0.0.1:5001
python app.py
pause
