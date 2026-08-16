import json
import io
import re
import matplotlib
matplotlib.use('Agg')  # Required for server environments
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict, List

MODEL_FAST = "openai/gpt-oss-20b"
MAX_STEPS = 4
DEFAULT_DURATION_DAYS = 14
MAX_DURATION_DAYS = 90


class StrategyState(TypedDict):
    problem: str
    debate_history: str
    action_plan: str
    chart_data: List[dict]
    chart_bytes: bytes
    groq_api_key: str


# ─── NODE 1: Compress Debate into an Action Plan ────────────────
def extract_plan(state: StrategyState) -> dict:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4, groq_api_key=state["groq_api_key"])

    prompt = f"""You are a strategic assistant. Based on the following executive debate, create a concise 4-step action plan.
Format it strictly as a numbered list. Keep each step to 1 sentence.

DEBATE:
{state['debate_history']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"action_plan": response.content}


# ─── NODE 2: Generate Chart DATA (not code) ─────────────────────
def generate_chart_data(state: StrategyState) -> dict:
    """
    Asks the LLM only for structured timeline data (step name + duration in days).
    The LLM's output is NEVER executed — it is parsed as JSON and validated below.
    This replaces the old pattern of asking the LLM for Python code and running it,
    which was a remote-code-execution risk.
    """
    llm = ChatGroq(model=MODEL_FAST, temperature=0.2, groq_api_key=state["groq_api_key"])

    prompt = f"""Based on the action plan below, output a JSON array with exactly {MAX_STEPS} items describing
a project timeline. Each item must have:
  - "step": a short (max 8 word) label for the step
  - "duration_days": an integer number of days (reasonable range: 3 to 30)

Respond with ONLY the raw JSON array. No markdown, no explanation, no code.

ACTION PLAN:
{state['action_plan']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    chart_data = _parse_chart_data(response.content)
    return {"chart_data": chart_data}


def _parse_chart_data(raw: str) -> List[dict]:
    """Strictly parse and validate the LLM's JSON output. Never eval/exec anything.
    Falls back to a safe default if the model returns anything malformed."""
    text = raw.strip()
    # Strip accidental markdown fences without ever executing the content
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    fallback = [
        {"step": "Assess the situation", "duration_days": DEFAULT_DURATION_DAYS},
        {"step": "Develop response plan", "duration_days": DEFAULT_DURATION_DAYS},
        {"step": "Execute plan", "duration_days": DEFAULT_DURATION_DAYS},
        {"step": "Review and communicate", "duration_days": DEFAULT_DURATION_DAYS},
    ]

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return fallback

    if not isinstance(parsed, list) or not parsed:
        return fallback

    cleaned = []
    for item in parsed[:MAX_STEPS]:
        if not isinstance(item, dict):
            continue
        step_name = str(item.get("step", "")).strip()[:80]  # cap length, force to str
        try:
            duration = int(item.get("duration_days", DEFAULT_DURATION_DAYS))
        except (TypeError, ValueError):
            duration = DEFAULT_DURATION_DAYS
        duration = max(1, min(duration, MAX_DURATION_DAYS))  # clamp to a sane range
        if step_name:
            cleaned.append({"step": step_name, "duration_days": duration})

    return cleaned if cleaned else fallback


# ─── NODE 3: Render the chart ourselves (fixed function, no exec) ───
def render_chart(state: StrategyState) -> dict:
    """Renders the Gantt chart with a function we control. The LLM never
    supplies code here, only the validated data from generate_chart_data."""
    steps = state.get("chart_data") or []
    if not steps:
        return {"chart_bytes": b""}

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        cursor = 0
        for i, item in enumerate(steps):
            duration = item["duration_days"]
            ax.barh(i, duration, left=cursor, height=0.5, color=plt.cm.tab20(i))
            cursor += duration

        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels([item["step"] for item in steps])
        ax.set_title("Recovery Strategy Timeline")
        ax.set_xlabel("Days")
        ax.set_ylabel("Step")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return {"chart_bytes": buf.getvalue()}
    except Exception:
        plt.close("all")
        return {"chart_bytes": b""}


# ─── Build the Graph ────────────────────────────────────────────
def build_workflow():
    workflow = StateGraph(StrategyState)
    workflow.add_node("extract_plan", extract_plan)
    workflow.add_node("generate_chart_data", generate_chart_data)
    workflow.add_node("render_chart", render_chart)

    workflow.set_entry_point("extract_plan")
    workflow.add_edge("extract_plan", "generate_chart_data")
    workflow.add_edge("generate_chart_data", "render_chart")
    workflow.add_edge("render_chart", END)

    return workflow.compile()