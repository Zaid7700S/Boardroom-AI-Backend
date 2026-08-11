import json
import uuid
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import Response 
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from supabase_client import supabase
from agent_factory import stream_boardroom_debate
from workflow import build_workflow
import os

app = FastAPI()

# Get the frontend URL from environment variables (for Vercel)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["FRONTEND_URL"], # Change to your React port in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProblemRequest(BaseModel):
    problem: str
    groq_api_key: str

def verify_token(authorization: str):
    """Verifies Supabase JWT and returns user_id"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing auth token")
    token = authorization.split(" ")[1]
    try:
        # Verify token with Supabase
        res = supabase.auth.get_user(token)
        if not res.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return res.user.id, token
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# Add the Health Check endpoint (HEAD method)
@app.head("/health")
async def healthcheck():
    return Response(status_code=200)

@app.post("/api/stream")
async def stream_boardroom(req: ProblemRequest, authorization: str = Header(None)):
    """Streams the AutoGen debate, then runs LangGraph, saves to DB, and returns chart."""
    user_id, token = verify_token(authorization)
    
    async def event_generator():
        full_history = ""
        
        # 1. Stream the AutoGen Debate
        async for msg in stream_boardroom_debate(req.problem, req.groq_api_key):
            if msg["agent"] == "HISTORY":
                full_history = msg["content"]
            else:
                # Yield SSE event to frontend
                yield {
                    "event": "debate",
                    "data": json.dumps({"agent": msg["agent"], "content": msg["content"]})
                }

        # 2. Run LangGraph for Plan and Chart Generation
        yield {"event": "status", "data": json.dumps({"message": "Drafting Strategic Action Plan..."})}
        app_workflow = build_workflow()
        
        # Pass groq_api_key directly into the state dictionary
        result = app_workflow.invoke(
            {
                "problem": req.problem,
                "debate_history": full_history,
                "groq_api_key": req.groq_api_key
            }
        )
        
        # 3. Save to Supabase DB & Storage
        yield {"event": "status", "data": json.dumps({"message": "Saving to Database..."})}
        
        chart_bytes = result.get("chart_bytes", b"")
        chart_url = None
        
        if chart_bytes:
            file_name = f"{uuid.uuid4()}.png"
            # Upload to Supabase Storage
            supabase.storage.from_("charts").upload(file_name, chart_bytes, {"content-type": "image/png"})
            # Get Public URL
            chart_url = supabase.storage.from_("charts").get_public_url(file_name)
        
        # Insert Row into Database
        row_data = {
            "user_id": user_id,
            "problem": req.problem,
            "debate_history": full_history,
            "action_plan": result["action_plan"],
            "chart_url": chart_url
        }
        db_response = supabase.table("boardroom_sessions").insert(row_data).execute()
        session_id = db_response.data[0]["id"]
        
        # 4. Send final data to frontend
        yield {
            "event": "complete",
            "data": json.dumps({
                "session_id": session_id,
                "action_plan": result["action_plan"],
                "chart_url": chart_url
            })
        }

    return EventSourceResponse(event_generator())

@app.get("/api/history")
async def get_history(authorization: str = Header(None)):
    """Gets user's past sessions."""
    user_id, _ = verify_token(authorization)
    response = supabase.table("boardroom_sessions").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return response.data