from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
conversation_history = []

class ChatRequest(BaseModel):
    message: str

def validate_message(message: str):
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Empty message not allowed")
    if len(message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long")

def save_log(user_message: str, ai_reply: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("transcript.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] User : {user_message}\n")
        f.write(f"[{timestamp}] AI   : {ai_reply}\n")
        f.write("-" * 60 + "\n")

def call_groq(history: list) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history,
        max_tokens=1024,
    )
    return response.choices[0].message.content

@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message
    validate_message(message)
    conversation_history.append({"role": "user", "content": message})
    reply = call_groq(conversation_history)
    conversation_history.append({"role": "assistant", "content": reply})
    print(f"User : {message}")
    print(f"AI   : {reply}")
    print("-" * 50)
    save_log(message, reply)
    return {"reply": reply}

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "running", "version": "1.0.0"}