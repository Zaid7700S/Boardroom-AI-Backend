import json
import base64
import os
from fastapi import FastAPI, Header
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from supabase_client import supabase
from agent_factory import stream_boardroom_debate
from workflow import build_workflow

app = FastAPI()

# Exact production origin(s), comma-separated, e.g. "https://boardroom-ai.vercel.app,https://boardroomai.com"
_frontend_origins = os.getenv("FRONTEND_URL", "")
ALLOWED_ORIGINS = [o.strip() for o in _frontend_origins.split(",") if o.strip()]

# Optional: preview-deployment regex scoped to *your* Vercel project only, e.g.
# "https://boardroom-ai-.*\.vercel\.app" - NOT any *.vercel.app site. Also allows
# localhost for local development. Leave PREVIEW_ORIGIN_REGEX unset to disable
# preview-deployment access entirely and only trust ALLOWED_ORIGINS.
_preview_regex = os.getenv("PREVIEW_ORIGIN_REGEX", "")
_localhost_regex = r"http://localhost:\d+"
ALLOW_ORIGIN_REGEX = f"{_preview_regex}|{_localhost_regex}" if _preview_regex else _localhost_regex

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
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
        return None  # Guest user

    parts = authorization.split(" ")
    if len(parts) != 2:
        return None

    token = parts[1]
    try:
        res = supabase.auth.get_user(token)
        if not res.user:
            return None
        return res.user.id
    except Exception:
        return None


@app.head("/health")
async def healthcheck():
    return Response(status_code=200)


@app.post("/api/stream")
async def stream_boardroom(req: ProblemRequest, authorization: str = Header(None)):
    """Streams the AutoGen debate, then runs LangGraph, saves to DB (if logged in), and returns chart."""
    user_id = verify_token(authorization)

    async def event_generator():
        full_history = ""
        debate_messages = []  # [{agent, content}, ...] - kept as structured data, not just the joined string

        # 1. Stream the AutoGen Debate
        async for msg in stream_boardroom_debate(req.problem, req.groq_api_key):
            if msg["agent"] == "HISTORY":
                full_history = msg["content"]
            else:
                debate_messages.append({"agent": msg["agent"], "content": msg["content"]})
                yield {
                    "event": "debate",
                    "data": json.dumps({"agent": msg["agent"], "content": msg["content"]})
                }

        # 2. Run LangGraph for Plan and Chart Generation
        yield {"event": "status", "data": json.dumps({"message": "Drafting Strategic Action Plan..."})}

        try:
            app_workflow = build_workflow()

            result = app_workflow.invoke(
                {
                    "problem": req.problem,
                    "debate_history": full_history,
                    "groq_api_key": req.groq_api_key
                }
            )

            chart_bytes = result.get("chart_bytes", b"")
            chart_base64_str = base64.b64encode(chart_bytes).decode("utf-8") if chart_bytes else None
            # No storage bucket in this project - chart is stored/returned inline as base64 for both
            # guests and logged-in users, rather than uploaded to a bucket that doesn't exist.
            chart_url = f"data:image/png;base64,{chart_base64_str}" if chart_base64_str else None
            session_id = "guest_session"

            if user_id:
                # 3a. Logged-in user: save the plan + each debate turn to Supabase
                yield {"event": "status", "data": json.dumps({"message": "Saving to Database..."})}

                plan_row = {
                    "user_id": user_id,
                    "title": req.problem,
                    "markdown": result["action_plan"],
                    "chart_base64": chart_base64_str,
                }
                plan_response = supabase.table("plans").insert(plan_row).execute()
                plan_id = plan_response.data[0]["id"]
                session_id = plan_id

                if debate_messages:
                    chat_rows = [
                        {
                            "plan_id": plan_id,
                            "user_id": user_id,
                            "role": m["agent"],
                            "content": m["content"],
                        }
                        for m in debate_messages
                    ]
                    supabase.table("chats").insert(chat_rows).execute()

            # 4. Send final data to frontend
            yield {
                "event": "complete",
                "data": json.dumps({
                    "session_id": session_id,
                    "action_plan": result["action_plan"],
                    "chart_url": chart_url
                })
            }
        except Exception as e:
            # Without this, an exception here (bad model name, LLM/API failure, DB error) would
            # silently kill the generator - the frontend never gets a 'complete' event and just
            # hangs forever on whatever status was last shown. Surface it instead.
            yield {
                "event": "error",
                "data": json.dumps({"message": f"Failed to generate strategy: {str(e)[:300]}"})
            }

    return EventSourceResponse(event_generator())


@app.get("/api/history")
async def get_history(authorization: str = Header(None)):
    """Gets user's past sessions (plans + their debate turns). Returns empty list for guests."""
    user_id = verify_token(authorization)
    if not user_id:
        return []

    plans_response = (
        supabase.table("plans")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    plans = plans_response.data

    plan_ids = [p["id"] for p in plans]
    chats_by_plan = {}
    if plan_ids:
        chats_response = (
            supabase.table("chats")
            .select("*")
            .in_("plan_id", plan_ids)
            .order("created_at")
            .execute()
        )
        for c in chats_response.data:
            chats_by_plan.setdefault(c["plan_id"], []).append(
                {"agent": c["role"], "content": c["content"]}
            )

    sessions = []
    for p in plans:
        chart_b64 = p.get("chart_base64")
        sessions.append({
            "id": p["id"],
            "created_at": p["created_at"],
            "problem": p["title"],
            "action_plan": p["markdown"],
            "chart_url": f"data:image/png;base64,{chart_b64}" if chart_b64 else None,
            "messages": chats_by_plan.get(p["id"], []),
        })

    return sessions