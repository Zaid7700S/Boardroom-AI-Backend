import json
import uuid
import base64
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

# Remove the FRONTEND_URL variable, we don't need it anymore for CORS
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    # Use regex to allow ALL Vercel URLs and localhost for development
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProblemRequest(BaseModel):
    problem: str
    groq_api_key: str

def verify_token(authorization: str):
    """Verifies Supabase JWT and returns user_id, or None if guest"""
    if not authorization or not authorization.startswith("Bearer "):
        return None # Guest user
    
    token = authorization.split(" ")[1]
    try:
        # Verify token with Supabase
        res = supabase.auth.get_user(token)
        if not res.user:
            return None
        return res.user.id
    except Exception:
        return None

# Add the Health Check endpoint (HEAD method)
@app.head("/health")
async def healthcheck():
    return Response(status_code=200)

@app.post("/api/stream")
async def stream_boardroom(req: ProblemRequest, authorization: str = Header(None)):
    """Streams the AutoGen debate, then runs LangGraph, saves to DB (if logged in), and returns chart."""
    user_id = verify_token(authorization)
    
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
        
        chart_bytes = result.get("chart_bytes", b"")
        chart_url = None
        session_id = "guest_session"
        
        if user_id:
            # 3a. Logged-in user: Save to Supabase DB & Storage
            yield {"event": "status", "data": json.dumps({"message": "Saving to Database..."})}
            
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
        else:
            # 3b. Guest: Convert image to base64 string to send directly to frontend
            if chart_bytes:
                base64_str = base64.b64encode(chart_bytes).decode('utf-8')
                chart_url = f"data:image/png;base64,{base64_str}"
        
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
    """Gets user's past sessions. Returns empty list for guests."""
    user_id = verify_token(authorization)
    if not user_id:
        return []
    
    response = supabase.table("boardroom_sessions").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return response.data
