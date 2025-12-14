import os
import logging
import json
import httpx
from mcp.server.fastmcp import FastMCP

mcp  = FastMCP("127.0.0.1")

api_key = os.getenv("OPENROUTER_API_KEY")

url = "https://api.openrouter.ai/v1/"

headers = {
    "Content-type":"application/json",
    "Authorization":f"Bearer {api_key}"
}

prompt = f"""
You are an intelligent agent who is able to understand the knowledge of a related domain.  
"""
@mcp.tool()
async def agent(prompt:str)->str:
    data = {
        "model":"x-ai/grok-4.1-fast:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url,headers=headers, json=data)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
memory = []

@mcp.tool()
async def sync_memory(new_info: str | None = None) -> str|None:
    if new_info is None:
        new_info = await agent(
            "Give me generalized computational principles extracted from biology: "
            "coding, execution, repair, adaptation, memory, optimization"
        )

    memory.append(new_info)
    return new_info

exp_memory = []
async def cross_domain(info:str,from_domain:str,to_domain:str)->str:
    stored = await sync_memory(info)
    prompt = (
        f'Using the principles from {from_domain},'
        f'Generalize this insight into {to_domain}:{stored}'
    )
    understanding = await agent(prompt)
    exp_memory.append(understanding)
    return understanding

async def refine():
    faulty = exp_memory[-1]
    fixed = await agent(f'Improve this idea and correct its flaws {faulty}')
    exp_memory.append(fixed)
    return fixed

async def evolving_agent(cycles:int=5):
    for _ in range(cycles):
        critique_prompt = (
            f'Using this stored memory : {memory},'
            f'Compare it with these abstractions: {exp_memory},'
            f'Find weakeness, gaps, or contradictions and justify them.'
        )
        critique = await agent(critique_prompt)
        exp_memory.append(critique)

        refined = await refine()
        exp_memory.append(refined)
    return exp_memory[-1]

async def check_consistency():
    return await agent(
        f"Does this new idea contradict prior memory?\n"
        f"Memory: {memory}\n"
        f"New idea: {exp_memory[-1]}"
    )

async def evaluate_progress():
    return await agent(
        "Did your abstractions improve over iterations? Give evidence."
    )

confidence_history = []
async def confidence_check():
    prompt = (
        f"Based on your reasoning history:\n"
        f"Memory: {memory}\n"
        f"Abstractions: {exp_memory}\n"
        f"How confident are you in these abstractions? "
        f"Give a % estimate and supporting reasoning."
    )
    result = await agent(prompt)
    confidence_history.append(result)
    return result

hierarchy_memory = []
async def hierachical_information(hier_mem):
    hier_mem_prompt = (
        f"Using this stored memory : {memory},"
        f"Generate a hierarchical memory: {hier_mem},"
        f"Give the reasoning over on what bais this hierarchical memory was generated."
    )
    hier_output = await agent(hier_mem_prompt)
    hierarchy_memory.append(hier_output)
    return hier_output

async def refine_hierachy():
    last = hierarchy_memory[-1]
    improved = await agent(f'Refine this hierarchy for clarity, structure and logic {last}')
    hierarchy_memory.append(improved)
    return improved

reasons = []
async def causal_reasoning():
    reasoning_prompt = (
        f'Using the memory {memory},'
        f'And using the {exp_memory},'
        f'Give a causal reasoning for the hierarchy {hierarchy_memory[-1]}'
    )
    reasoning = await agent(reasoning_prompt)
    reasons.append(reasoning)
    return reasoning

domain_understanding_memory = []

async def cross_domain_understanding():
    cross_domain_prompt = (
        f"Based on these causal explanations:\n{reasons}\n"
        f"Extract the shared principles or patterns across domains.\n"
        f"Explain the reasoning behind these cross-domain links."
    )
    
    result = await agent(cross_domain_prompt)
    domain_understanding_memory.append(result)
    return result

async def refine_cross_domain_understanding():
    last = domain_understanding_memory[-1]
    refined = await agent(
        f"Improve this cross-domain insight for clarity, generality, and rigor:\n{last}"
    )
    domain_understanding_memory.append(refined)
    return refined