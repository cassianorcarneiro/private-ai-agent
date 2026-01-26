# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE AI AGENT
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

import os
from dataclasses import dataclass
import json
from typing import Any, Dict, List, TypedDict, Annotated
from operator import add
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from ddgs import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import ollama

from config import Config

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Graph state
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class AgentState(TypedDict):
    
    history: List[Dict[str, str]]  # [{"role":"user|assistant","content":"..."}]

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
# Core class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

@dataclass
class MultiAgentWebAssistant:

    def __init__(self, config: Config):

        self.config = config
        self.console = Console()
        
        self.history: List[Dict[str, str]] = None  # type: ignore

        self._check_model()

        if self.history is None:
            self.history = []

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Check model
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def _check_model(self):

        try:
            models_response = ollama.list()
            
            model_names = []
            model_details = []
            
            if hasattr(models_response, 'models') and models_response.models:
                for model in models_response.models:
                    model_name = model.model
                    model_names.append(model_name)
                    model_details.append({
                        'name': model_name,
                        'size': model.size,
                        'modified': model.modified_at,
                        'parameters': getattr(model.details, 'parameter_size', 'N/A') if model.details else 'N/A'
                    })
            
            if not model_names:
                self.console.print("❌ [red]No models found in Ollama[/red]")
                raise Exception("No models available")
            
            # Find DeepSeek models

            selected_model = [
                model for model in model_details
                if self.config.ollama_model in model['name'].lower()
            ]
            
            if selected_model:

                # Use the first DeepSeek model found

                selected_model = selected_model[0]
                self.config.ollama_model = selected_model['name']
                
                self.console.print(Panel(
                    f"✅ [green]Selected model:[/green] {self.config.ollama_model}\n"
                    f"📊 [cyan]Size:[/cyan] {selected_model['size']/1024/1024/1024:.1f}GB\n"
                    f"⚙️  [yellow]Parameters:[/yellow] {selected_model['parameters']}\n"
                    f"📅 [magenta]Last modified date:[/magenta] "
                    f"{selected_model['modified'].strftime('%Y-%m-%d %H:%M')}",
                    title="🤖 Loaded Model",
                    border_style="green"
                ))

            else:

                # Use the first available model

                selected_model = model_details[0]
                self.config.ollama_model = selected_model['name']
                self.console.print(Panel(
                    f"⚠️ [yellow]Using available model:[/yellow] {self.config.ollama_model}\n"
                    f"📊 [cyan]Size:[/cyan] {selected_model['size']/1024/1024/1024:.1f}GB",
                    title="🤖 Alternative Model",
                    border_style="yellow"
                ))
                
        except Exception as e:
            self.console.print(f"❌ Error connecting to Ollama: {e}", style="bold red")
            self.console.print("\n🔧 [yellow]Possible solutions:[/yellow]")
            self.console.print("1. Check if Ollama is running: ollama serve")
            self.console.print("2. Install a model: ollama pull deepseek-coder")
            raise
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Create Ollama LLM session
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def _llm(self, temperature: float) -> ChatOllama:
        return ChatOllama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=temperature,
        )
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Utilities
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    @staticmethod
    def _safe_json_extract(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l : r + 1])

        raise ValueError("Could not parse JSON from model output.")

    @staticmethod
    def _summarize_sources(results: List[Dict[str, Any]], max_items: int) -> str:
        
        lines: List[str] = []
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

            lines.append(f"- {title}\n  {url}\n  {body[:400]}")
            n += 1

        return "\n".join(lines) if lines else "(No useful sources returned.)"

    def _history_block(self, max_turns: int = 6) -> str:

        recent = self.history[-2 * max_turns :]
        out = []

        for m in recent:
            role = m["role"]
            out.append(f"{role.upper()}: {m['content']}")
        
        return "\n".join(out) if out else "(no prior context)"
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Graph nodes (methods)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def node_plan_search(self, state: AgentState) -> Dict[str, Any]:
        llm = self._llm(self.config.temperature_planner)

        prompt = (
            "You are a web-search planner.\n"
            "Given the user question and recent chat context, output 3 to 6 short web queries.\n"
            'Return ONLY JSON: {"queries":["..."]}\n\n'
            f"Recent chat context:\n{self._history_block()}\n\n"
            f"User question:\n{state['question']}\n"
        )

        raw = llm.invoke(prompt).content
        try:
            data = self._safe_json_extract(raw)
            plan = SearchPlan.model_validate(data)
            queries = [q.strip() for q in plan.queries if q.strip()][: self.config.max_queries]
            if not queries:
                queries = [state["question"][:120]]
        except (ValueError, ValidationError):
            queries = [state["question"][:120]]

        return {"search_queries": queries, "search_results": [], "drafts": [], "final_answer": ""}

    def node_web_search(self, state: AgentState) -> Dict[str, Any]:
        
        out: List[Dict[str, Any]] = []

        try:
            with DDGS() as ddgs:
                for q in state["search_queries"]:
                    try:
                        for r in ddgs.text(q, max_results=self.config.ddgs_max_results_per_query):
                            if isinstance(r, dict):
                                out.append({"query": q, **r})
                            else:
                                out.append({"query": q, "raw": r})
                    except Exception as e:
                        out.append({"query": q, "error": str(e)})
        except Exception:
            ddgs = DDGS()
            for q in state["search_queries"]:
                try:
                    for r in ddgs.text(q, max_results=self.config.ddgs_max_results_per_query):
                        if isinstance(r, dict):
                            out.append({"query": q, **r})
                        else:
                            out.append({"query": q, "raw": r})
                except Exception as e:
                    out.append({"query": q, "error": str(e)})

        return {"search_results": out}

    def node_answer(self, name: str, focus: str):
        def _node(state: AgentState) -> Dict[str, Any]:
            llm = self._llm(self.config.temperature_drafters)
            sources = self._summarize_sources(state["search_results"], self.config.max_sources_in_prompt)

            prompt = (
                f"You are {name}.\nFocus: {focus}\n\n"
                f"Recent chat context:\n{self._history_block()}\n\n"
                f"User question:\n{state['question']}\n\n"
                f"Sources:\n{sources}\n\n"
                "Write a concise, well-supported answer. Mark uncertainty when needed.\n"
            )

            draft = llm.invoke(prompt).content.strip()
            return {"drafts": [f"[{name}]\n{draft}"]}
        return _node

    def node_aggregate(self, state: AgentState) -> Dict[str, Any]:
        llm = self._llm(self.config.temperature_aggregator)
        sources = self._summarize_sources(state["search_results"], self.config.max_sources_in_prompt)
        drafts = "\n\n".join(state["drafts"])

        prompt = (
            "You are an aggregator.\n"
            "Combine drafts into one final answer. No hallucinations; mark uncertainty.\n\n"
            f"Recent chat context:\n{self._history_block()}\n\n"
            f"User question:\n{state['question']}\n\n"
            f"Sources:\n{sources}\n\n"
            f"Drafts:\n{drafts}\n\n"
            "Final answer:\n"
        )

        final = llm.invoke(prompt).content.strip()
        return {"final_answer": final}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Graph build + run
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("plan_search", self.node_plan_search)
        g.add_node("web_search", self.node_web_search)
        g.add_node("answer_1", self.node_answer("Agent-1", "clear structured explanation"))
        g.add_node("answer_2", self.node_answer("Agent-2", "limitations, caveats, counterpoints"))
        g.add_node("answer_3", self.node_answer("Agent-3", "practical steps, recommendations, examples"))
        g.add_node("aggregate", self.node_aggregate)
        
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

    def chat_loop_ollama(model: str):
        messages = [
            {"role": "system", "content": "Responda em pt-BR. Seja objetivo."}
        ]

        while True:
            user_input = input("Você: ").strip()
            if user_input.lower() in {"sair", "exit", "quit"}:
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            res = ollama.chat(model=model, messages=messages)
            content = (res.message.content or "").strip() or (getattr(res.message, "thinking", "") or "").strip()

            messages.append({"role": "assistant", "content": content})
            print("\nAgente:", content, "\n")

    def ask(self, question: str) -> str:
        app = self.build_graph()

        init_state: AgentState = {
            "history": self.history,
            "question": question,
            "search_queries": [],
            "search_results": [],
            "drafts": [],
            "final_answer": "",
        }

        out = app.invoke(init_state)
        answer = out["final_answer"]

        # Atualiza memória da sessão (fora do grafo)
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return answer
        
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Graph build + run
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def main():

    config = Config()

    assistant = MultiAgentWebAssistant(config=config)

    assistant.console.print(Panel('Type "exit" to close.',
            title="",
            border_style="white"
        ))

    while True:
        
        print('\n')
        q = input("User question: ").strip()
        print('\n')

        if not q:
            continue
        if q.lower() in {"sair", "exit", "quit"}:
            break
        
        a = assistant.ask(q)
        
        assistant.console.print(Panel(a,
            title="🤖 Response",
            border_style="blue"
        ))

if __name__ == "__main__":
    os.system('cls')
    main()