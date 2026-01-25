# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE AI AGENT
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations
import json
from typing import Any, Dict, List, TypedDict, Annotated
from operator import add
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from ddgs import DDGS

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Local configuration (NO getenv)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:8b"
TEMPERATURE_PLANNER = 0.0
TEMPERATURE_DRAFTERS = 0.3
TEMPERATURE_AGGREGATOR = 0.1
DDGS_MAX_RESULTS_PER_QUERY = 5
MAX_QUERIES = 6
MAX_SOURCES_IN_PROMPT = 12

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# State
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class AgentState(TypedDict):
    question: str
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    drafts: Annotated[List[str], add]   # fan-in reducer: concatenates lists
    final_answer: str

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Schemas
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class SearchPlan(BaseModel):
    queries: List[str] = Field(..., description="Short web-search queries (3 to 6).")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Helpers
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def build_llm(temperature: float) -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )

def _safe_json_extract(text: str) -> Dict[str, Any]:
    
    """
    Robust-ish JSON extraction:
    - Try direct parse
    - Else find first '{' and last '}' and parse that slice
    """
    
    text = text.strip()
    
    try:
        return json.loads(text)
    except Exception:
        pass

    l = text.find("{")
    r = text.rfind("}")
    
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(text[l : r + 1])
        except Exception:
            pass

    raise ValueError("Could not parse JSON from model output.")

def _summarize_sources(results: List[Dict[str, Any]], max_items: int) -> str:
    
    lines = []
    n = 0
    
    for item in results:
        if n >= max_items:
            break
        if item.get("error"):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("url") or item.get("href") or "").strip()
        body = (item.get("body") or item.get("snippet") or item.get("content") or "").strip()
        if not (title or url or body):
            continue
        body = body[:400]
        lines.append(f"- {title}\n  {url}\n  {body}")
        n += 1
    
    return "\n".join(lines) if lines else "(No useful sources returned.)"

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Nodes
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def node_plan_search(state: AgentState) -> Dict[str, Any]:
    llm = build_llm(TEMPERATURE_PLANNER)

    prompt = (
        "You are a web-search planner.\n"
        "Given the user question, produce 3 to 6 short, specific web search queries.\n"
        "Return ONLY valid JSON in the format:\n"
        '{"queries":["...","..."]}\n\n'
        f"User question:\n{state['question']}\n"
    )

    raw = llm.invoke(prompt).content
    data = _safe_json_extract(raw)

    try:
        plan = SearchPlan.model_validate(data)
        queries = [q.strip() for q in plan.queries if q.strip()][:MAX_QUERIES]
    except ValidationError:
       
        # fallback: minimal behavior
       
        queries = [state["question"][:120]]

    return {
        "search_queries": queries,
        "search_results": [],
        "drafts": [],
        "final_answer": "",
    }

def node_web_search(state: AgentState) -> Dict[str, Any]:

    out: List[Dict[str, Any]] = []
    
    # DDGS supports context manager in many versions
    
    try:
        with DDGS() as ddgs:
            for q in state["search_queries"]:
                try:
                    
                    # ddgs.text returns an iterator/list of dicts with keys like title/href/body
                   
                    for r in ddgs.text(q, max_results=DDGS_MAX_RESULTS_PER_QUERY):
                        if isinstance(r, dict):
                            out.append({"query": q, **r})
                        else:
                            out.append({"query": q, "raw": r})
                except Exception as e:
                    out.append({"query": q, "error": str(e)})
    except Exception:
        
        # fallback if DDGS doesn't support context manager
        
        ddgs = DDGS()
        
        for q in state["search_queries"]:
            try:
                for r in ddgs.text(q, max_results=DDGS_MAX_RESULTS_PER_QUERY):
                    if isinstance(r, dict):
                        out.append({"query": q, **r})
                    else:
                        out.append({"query": q, "raw": r})
            except Exception as e:
                out.append({"query": q, "error": str(e)})

    return {"search_results": out}

def make_responder_node(name: str, focus: str):
    def node(state: AgentState) -> Dict[str, Any]:
        
        llm = build_llm(TEMPERATURE_DRAFTERS)
        sources = _summarize_sources(state["search_results"], MAX_SOURCES_IN_PROMPT)

        prompt = (
            f"You are {name}.\n"
            f"Focus: {focus}\n\n"
            f"User question:\n{state['question']}\n\n"
            "Web sources (may include noise):\n"
            f"{sources}\n\n"
            "Write a concise, well-supported answer.\n"
            "If a claim is not supported by the sources, explicitly mark uncertainty.\n"
        )

        draft = llm.invoke(prompt).content.strip()
        return {"drafts": [f"[{name}]\n{draft}"]}
    
    return node

node_answer_1 = make_responder_node("Agent-1", "clear structured explanation")
node_answer_2 = make_responder_node("Agent-2", "limitations, caveats, counterpoints")
node_answer_3 = make_responder_node("Agent-3", "practical steps, recommendations, examples")

def node_aggregate(state: AgentState) -> Dict[str, Any]:
    
    llm = build_llm(TEMPERATURE_AGGREGATOR)

    sources = _summarize_sources(state["search_results"], MAX_SOURCES_IN_PROMPT)
    drafts = "\n\n".join(state["drafts"])

    prompt = (
        "You are an aggregator.\n"
        "Combine the three agent drafts into a single final answer.\n"
        "Rules:\n"
        "1) Remove redundancy.\n"
        "2) If agents disagree, explain the disagreement.\n"
        "3) Do NOT invent facts beyond the sources; mark uncertainty clearly.\n"
        "4) Keep it direct and complete.\n\n"
        f"User question:\n{state['question']}\n\n"
        f"Sources:\n{sources}\n\n"
        f"Agent drafts:\n{drafts}\n\n"
        "Final answer:\n"
    )

    final = llm.invoke(prompt).content.strip()
    
    return {"final_answer": final}

def build_graph():
    
    g = StateGraph(AgentState)

    g.add_node("plan_search", node_plan_search)
    g.add_node("web_search", node_web_search)
    g.add_node("answer_1", node_answer_1)
    g.add_node("answer_2", node_answer_2)
    g.add_node("answer_3", node_answer_3)
    g.add_node("aggregate", node_aggregate)

    g.set_entry_point("plan_search")
    g.add_edge("plan_search", "web_search")

    # fan-out

    g.add_edge("web_search", "answer_1")
    g.add_edge("web_search", "answer_2")
    g.add_edge("web_search", "answer_3")

    # fan-in

    g.add_edge("answer_1", "aggregate")
    g.add_edge("answer_2", "aggregate")
    g.add_edge("answer_3", "aggregate")

    g.add_edge("aggregate", END)

    return g.compile()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Main
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def main():
    question = input("Pergunta: ").strip()
    app = build_graph()

    init_state: AgentState = {
        "question": question,
        "search_queries": [],
        "search_results": [],
        "drafts": [],
        "final_answer": "",
    }

    out = app.invoke(init_state)
    print("\n=== RESPOSTA FINAL ===\n")
    print(out["final_answer"])

if __name__ == "__main__":
    main()
