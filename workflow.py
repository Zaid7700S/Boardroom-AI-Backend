import json
import os
import subprocess
import matplotlib
matplotlib.use('Agg') # Required for server environments
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict

MODEL_FAST = "llama-3.1-8b-instant"

class StrategyState(TypedDict):
    problem: str
    debate_history: str
    action_plan: str
    chart_code: str
    chart_bytes: bytes
    groq_api_key: str  # Added this to the state!

# ─── NODE 1: Compress Debate into an Action Plan ────────────────
def extract_plan(state: StrategyState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, groq_api_key=state["groq_api_key"])
    
    prompt = f"""You are a strategic assistant. Based on the following executive debate, create a concise 4-step action plan.
Format it strictly as a numbered list. Keep each step to 1 sentence.

DEBATE:
{state['debate_history']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"action_plan": response.content}

# ─── NODE 2: Generate Python Code for a Gantt Chart ─────────────
def generate_chart_code(state: StrategyState) -> dict:
    llm = ChatGroq(model=MODEL_FAST, temperature=0.2, groq_api_key=state["groq_api_key"])
    
    prompt = f"""You are a Python expert. Given the action plan below, write a Python script using `matplotlib` to create a horizontal Gantt chart showing the timeline for these 4 steps.
Assume each step takes roughly 2 weeks. Use different colors for each step.
The script MUST save the file as `strategy_chart.png` in the current directory. 
Do NOT include any markdown formatting or explanations. ONLY output raw Python code.

ACTION PLAN:
{state['action_plan']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content.replace("```python", "").replace("```", "").strip()
    return {"chart_code": code}

# ─── NODE 3: Execute Code and get Image Bytes ───────────────────
def execute_chart_code(state: StrategyState) -> dict:
    script_path = "generate_chart.py"
    chart_path = os.path.abspath("strategy_chart.png")
    
    try:
        with open(script_path, "w") as f:
            f.write(state["chart_code"])
        
        subprocess.run(["python", script_path], check=True, capture_output=True, text=True)
        
        if os.path.exists(chart_path):
            with open(chart_path, "rb") as f:
                img_bytes = f.read()
            os.remove(chart_path) # cleanup file, we only need bytes
            return {"chart_bytes": img_bytes}
        else:
            return {"chart_bytes": b""}
            
    except Exception as e:
        return {"chart_bytes": b""}

# ─── Build the Graph ────────────────────────────────────────────
def build_workflow():
    workflow = StateGraph(StrategyState)
    workflow.add_node("extract_plan", extract_plan)
    workflow.add_node("generate_chart_code", generate_chart_code)
    workflow.add_node("execute_chart_code", execute_chart_code)
    
    workflow.set_entry_point("extract_plan")
    workflow.add_edge("extract_plan", "generate_chart_code")
    workflow.add_edge("generate_chart_code", "execute_chart_code")
    workflow.add_edge("execute_chart_code", END)
    
    return workflow.compile()