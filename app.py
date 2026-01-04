from google import genai

import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS fix (so browser preflight OPTIONS works)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev/hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()  # only needed if using .env

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set")

client = genai.Client(api_key=api_key)



@app.post("/api/analyze")
def analyze(body: dict):

    user_msg = body.get("message", "").strip()

    if not user_msg:
        return {"error": "Missing message"}
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents= user_msg,
    )
    return {"response": response.text}
