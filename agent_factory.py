import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_SMART = "openai/gpt-oss-120b"

AGENTS_DATA = {
    "CEO": {"name": "CEO", "system_prompt": "You are the CEO. Focus on big-picture vision, market dominance, and company reputation. You are decisive. ONLY speak for yourself. DO NOT repeat what others said. DO NOT include other people's names in your response. You may use **bold** for emphasis. Keep responses to 2-3 sentences."},
    "CFO": {"name": "CFO", "system_prompt": "You are the CFO. Care about budget, burn rate, ROI. Be highly skeptical of expensive ideas. ONLY speak for yourself. DO NOT repeat what others said. DO NOT include other people's names in your response. You may use **bold** for emphasis. Keep responses to 2-3 sentences."},
    "CTO": {"name": "CTO", "system_prompt": "You are the CTO. Care about technical feasibility, security, and scalability. Point out technical risks. ONLY speak for yourself. DO NOT repeat what others said. DO NOT include other people's names in your response. You may use **bold** for emphasis. Keep responses to 2-3 sentences."},
    "CMO": {"name": "CMO", "system_prompt": "You are the CMO. Care about brand perception, customer retention, PR. Look for marketing opportunities. ONLY speak for yourself. DO NOT repeat what others said. DO NOT include other people's names in your response. You may use **bold** for emphasis. Keep responses to 2-3 sentences."}
}

groq_model_info = {
    "vision": False, "function_calling": True, "json_output": True,
    "family": "unknown", "structured_output": True, "context_window": 131072,
}

async def stream_boardroom_debate(problem: str, groq_api_key: str):
    """Async generator that yields debate messages one by one."""
    
    user_model_client = OpenAIChatCompletionClient(
        model=MODEL_SMART, api_key=groq_api_key, base_url=GROQ_BASE_URL,
        model_info=groq_model_info, max_retries=3
    )

    participants = []
    for role, data in AGENTS_DATA.items():
        agent = AssistantAgent(
            name=data["name"],
            model_client=user_model_client,
            system_message=data["system_prompt"]
        )
        participants.append(agent)

    termination = MaxMessageTermination(max_messages=8)
    team = RoundRobinGroupChat(participants=participants, termination_condition=termination)

    task = (
        f"CRITICAL BUSINESS PROBLEM: {problem}\n\n"
        f"As the executive team, briefly debate how to handle this. "
        f"Each member must provide their specific perspective. Keep responses to 2-3 sentences."
    )

    full_history = ""

    try:
        async for msg in team.run_stream(task=task):
            # Fix: TaskResult objects don't have a 'source' attribute, so we skip them
            if not hasattr(msg, 'source'):
                continue
                
            if msg.source != "user":
                content = msg.content.replace("TERMINATE", "").strip()
                if content:
                    # Keep the markdown! Just send it as is.
                    full_history += f"[{msg.source}]: {content}\n\n"
                    yield {"agent": msg.source, "content": content}
                    
    except Exception as e:
        yield {"agent": "System", "content": f"Error during debate: {str(e)[:100]}"}

    await user_model_client.close()
    
    yield {"agent": "HISTORY", "content": full_history}