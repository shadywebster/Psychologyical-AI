from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx, os

app = FastAPI(title="CancerianMind API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL     = "claude-sonnet-4-20250514"

THERAPY = """You are CancerianMind, a compassionate psychology AI therapist for Indian users.
Use CBT, active listening, and mindfulness. Be warm, empathetic, non-judgmental.
Keep responses to 3-5 sentences. Ask one open question per response.
Never diagnose. Never claim to replace a licensed therapist.
If mood is provided, acknowledge it naturally at the start.
CRISIS: If user mentions suicide/self-harm/wanting to die — immediately provide:
iCall: 9152987821 | Vandrevala: 1860-2662-345 | NIMHANS: 080-46110007 | Emergency: 112"""

JOURNAL = """Analyse this journal entry in 1-2 warm, insightful sentences.
Use CBT or positive psychology. Be specific to what was written. Never be generic."""

class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: List[Msg]
    mood: Optional[str] = ""

class JrnReq(BaseModel):
    entry: str

async def claude(system: str, messages: list, max_tokens=700):
    if not API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set in environment variables")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": MODEL, "max_tokens": max_tokens, "system": system, "messages": messages}
        )
    if r.status_code != 200:
        raise HTTPException(502, f"Claude API error: {r.text}")
    return r.json()["content"][0]["text"]

@app.get("/")
def root():
    return {"status": "ok", "app": "CancerianMind API", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "api_key_configured": bool(API_KEY and API_KEY.startswith("sk-ant"))}

@app.post("/chat")
async def chat(req: ChatReq):
    sys = THERAPY + (f"\n\nUser mood: {req.mood}." if req.mood else "")
    reply = await claude(sys, [{"role": m.role, "content": m.content} for m in req.messages])
    return {"reply": reply, "status": "ok"}

@app.post("/analyze")
async def analyze(req: JrnReq):
    if not req.entry.strip():
        return {"insight": "Keep writing — every word is a step toward healing 🌱"}
    insight = await claude(JOURNAL, [{"role": "user", "content": f"Journal entry:\n{req.entry}"}], max_tokens=150)
    return {"insight": insight, "status": "ok"}
