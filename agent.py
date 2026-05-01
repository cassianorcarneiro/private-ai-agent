# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE AI AGENT
# REPOSITORY: https://github.com/cassianorcarneiro/private-multi-agent
# CASSIANO RIBEIRO CARNEIRO
#
# Pipeline:
#   plan_search ─> web_search ─┬─> explanation ─┐
#                               ├─> caveats ─────┼─> aggregate ─> END
#                               └─> examples ────┘
#
# Each drafter receives a SUBSET of filtered sources by intent compatible
# with its role, and produces Pydantic-validated JSON. The aggregator consumes
# the three structured outputs.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from ddgs import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import ollama

from config import Config
from schemas import SearchPlan, DrafterOutput, FinalAnswer
from io_utils import (
    rank_sources, filter_sources_by_intent, summarize_sources,
)
from llm_client import LLMClient, StructuredOutputError


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Graph state
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class AgentState(TypedDict, total=False):
    history: List[Dict[str, str]]
    question: str
    use_web_search: bool

    search_plan: SearchPlan
    search_results: List[Dict[str, Any]]

    # Reducer 'add' concatenates lists from parallel drafters
    drafter_outputs: Annotated[List[DrafterOutput], add]

    final: FinalAnswer


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Helpers
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def load_prompts(prompts_dir: str) -> Dict[str, str]:
    base = Path(prompts_dir)
    if not base.exists():
        alt = Path(__file__).parent / prompts_dir
        if alt.exists():
            base = alt
    needed = {
        "planner": "01_planner.txt",
        "explanation": "02_drafter_explanation.txt",
        "caveats": "03_drafter_caveats.txt",
        "examples": "04_drafter_examples.txt",
        "aggregator": "05_aggregator.txt",
    }
    out: Dict[str, str] = {}
    for name, fname in needed.items():
        p = base / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Prompt '{fname}' not found in {base.resolve()}."
            )
        out[name] = p.read_text(encoding="utf-8").strip()
    return out


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Core class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

@dataclass
class MultiAgentAssistant:
    config: Config

    def __post_init__(self):
        self.console = Console()
        self.history: List[Dict[str, str]] = []

        self.console.print("[dim]→ Loading prompts...[/dim]")
        self.prompts = load_prompts(self.config.prompts_dir)

        self.console.print("[dim]→ Checking models in Ollama...[/dim]")
        self._check_model()

        self.console.print("[dim]→ Initializing LLM client...[/dim]")
        self.llm = LLMClient(
            base_url=self.config.ollama_base_url,
            default_model=self.config.ollama_model,
            json_max_retries=self.config.json_max_retries,
        )

        self.console.print("[dim]→ Building agent graph...[/dim]")
        self.app = self.build_graph()
        self.console.print("[green]✓ Ready.[/green]\n")

    # ---- Model resolution ----------------------------------------------------------------------

    def _check_model(self):
        try:
            models_response = ollama.Client(host=self.config.ollama_base_url).list()
            model_details = []
            if hasattr(models_response, "models") and models_response.models:
                for model in models_response.models:
                    model_details.append({
                        "name": model.model,
                        "size": getattr(model, "size", 0),
                        "parameters": getattr(model.details, "parameter_size", "N/A") if model.details else "N/A",
                    })
            if not model_details:
                self.console.print("❌ [red]No models found in Ollama.[/red]")
                self.console.print("   Install one, e.g.: [cyan]ollama pull llama3.1:8b[/cyan]")
                raise RuntimeError("No models available")

            self.config.ollama_model = self._resolve_model(self.config.ollama_model, model_details, "default")

            # Resolve optional overrides
            for attr, label in [
                ("ollama_model_drafter_explanation", "drafter:explanation"),
                ("ollama_model_drafter_caveats", "drafter:caveats"),
                ("ollama_model_drafter_examples", "drafter:examples"),
                ("ollama_model_aggregator", "aggregator"),
            ]:
                requested = getattr(self.config, attr)
                if requested:
                    setattr(self.config, attr, self._resolve_model(requested, model_details, label))

        except Exception as e:
            # Re-raise as RuntimeError with a friendly message; main() handles it.
            raise RuntimeError(
                f"Could not connect to Ollama at {self.config.ollama_base_url}.\n"
                f"   Original error: {e}\n\n"
                "🔧 Possible solutions:\n"
                "   1. Check if Ollama is running:  ollama serve\n"
                "   2. Install a model:             ollama pull llama3.1:8b\n"
                "   3. Verify URL in config.ollama_base_url"
            ) from e

    def _resolve_model(self, requested: str, model_details: list, label: str) -> str:
        """Exact match → prefix → substring → fallback. Deterministic."""
        req_low = requested.lower()

        # 1. exact
        match = [m for m in model_details if m["name"].lower() == req_low]
        if match:
            self._log_model(label, match[0], exact_match=True)
            return match[0]["name"]
        # 2. prefix
        match = [m for m in model_details if m["name"].lower().startswith(req_low)]
        if match:
            self._log_model(label, match[0], exact_match=False)
            return match[0]["name"]
        # 3. substring
        match = [m for m in model_details if req_low in m["name"].lower()]
        if match:
            self._log_model(label, match[0], exact_match=False)
            return match[0]["name"]

        # fallback
        chosen = model_details[0]
        self.console.print(Panel(
            f"⚠️  [yellow]'{requested}' not found.[/yellow]\n"
            f"Using fallback: [bold]{chosen['name']}[/bold]",
            title=f"🤖 Model ({label}) — fallback",
            border_style="yellow",
        ))
        return chosen["name"]

    def _log_model(self, label: str, chosen: dict, exact_match: bool):
        size_gb = (chosen["size"] or 0) / 1024 / 1024 / 1024
        suffix = "" if exact_match else " [dim](inexact match)[/dim]"
        self.console.print(Panel(
            f"✅ [green]Model for {label}:[/green] {chosen['name']}{suffix}\n"
            f"📊 [cyan]Size:[/cyan] {size_gb:.1f} GB\n"
            f"⚙️  [yellow]Parameters:[/yellow] {chosen['parameters']}",
            title=f"🤖 Model ({label})",
            border_style="green",
        ))

    # ---- Utilities -----------------------------------------------------------------------------

    def _history_block(self) -> str:
        recent = self.history[-2 * self.config.history_max_turns :]
        out = [f"{m['role'].upper()}: {m['content']}" for m in recent]
        return "\n".join(out) if out else "(no previous context)"

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Nodes
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def node_plan_search(self, state: AgentState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [1/3] Planning queries...[/dim cyan]")

        if not state.get("use_web_search", True):
            return {"search_plan": SearchPlan(queries=[]), "search_results": []}

        prompt = (
            self.prompts["planner"]
            + f"\n\n[RECENT HISTORY]:\n{self._history_block()}"
            + f"\n\n[QUESTION]:\n{state['question']}"
        )
        try:
            plan = self.llm.chat_structured(
                prompt=prompt,
                schema=SearchPlan,
                temperature=self.config.temperature_planner,
            )
        except StructuredOutputError:
            plan = SearchPlan(queries=[
                {"query": state["question"][:120], "intent": "factual"}
            ])

        plan.queries = plan.queries[: self.config.max_queries]
        return {"search_plan": plan}

    def node_web_search(self, state: AgentState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> [2/3] Web search...[/dim cyan]")

        plan = state.get("search_plan") or SearchPlan()
        if not state.get("use_web_search", True) or not plan.queries:
            return {"search_results": []}

        out: List[Dict[str, Any]] = []
        for q_item in plan.queries:
            try:
                with DDGS() as ddgs:
                    results = ddgs.text(q_item.query, max_results=self.config.ddgs_max_results_per_query)
                    for r in results:
                        if isinstance(r, dict):
                            out.append({"query": q_item.query, "intent": q_item.intent, **r})
            except Exception as e:
                out.append({"query": q_item.query, "intent": q_item.intent, "error": str(e)})

        ranked = rank_sources(
            out,
            preferred_domains=self.config.preferred_sources,
            max_items=self.config.max_sources_in_prompt * 2,
        )
        return {"search_results": ranked}

    def _node_drafter(
        self,
        role: str,
        prompt_key: str,
        intents_allowed: List[str],
        temperature: float,
        model_attr: str,
    ):
        """Factory that creates a drafter node with intent-based source filtering."""
        def _node(state: AgentState) -> Dict[str, Any]:
            self.console.print(f"[dim cyan]>> [3/3] Drafter:{role}...[/dim cyan]")

            relevant_sources = filter_sources_by_intent(
                state.get("search_results", []),
                intents_allowed=intents_allowed,
            )
            sources_text = (
                summarize_sources(relevant_sources, self.config.max_sources_in_prompt)
                if state.get("use_web_search") and relevant_sources
                else "(web search disabled or no relevant sources for this role)"
            )

            prompt = (
                self.prompts[prompt_key]
                + f"\n\n[RECENT HISTORY]:\n{self._history_block()}"
                + f"\n\n[QUESTION]:\n{state['question']}"
                + f"\n\n[SOURCES (allowed intents: {', '.join(intents_allowed)})]:\n{sources_text}"
            )

            model = getattr(self.config, model_attr) or self.config.ollama_model
            try:
                output = self.llm.chat_structured(
                    prompt=prompt,
                    schema=DrafterOutput,
                    temperature=temperature,
                    model=model,
                )
            except StructuredOutputError as e:
                # Degraded fallback: create minimal output so aggregator doesn't break
                output = DrafterOutput(
                    role=role,
                    summary=f"(failed to generate structured output for drafter {role})",
                    key_points=[],
                    body_markdown=f"*Drafter {role} failed: {e}*",
                    confidence="low",
                )
            return {"drafter_outputs": [output]}
        return _node

    def node_aggregate(self, state: AgentState) -> Dict[str, Any]:
        self.console.print("[dim cyan]>> Aggregating...[/dim cyan]")

        drafts = state.get("drafter_outputs", [])
        drafts_json = "\n\n".join(
            f"[DRAFTER {d.role.upper()}]\n{d.model_dump_json(indent=2)}"
            for d in drafts
        )

        prompt = (
            self.prompts["aggregator"]
            + f"\n\n[RECENT HISTORY]:\n{self._history_block()}"
            + f"\n\n[QUESTION]:\n{state['question']}"
            + f"\n\n[STRUCTURED DRAFTS]:\n{drafts_json}"
        )

        model = self.config.ollama_model_aggregator or self.config.ollama_model
        try:
            final = self.llm.chat_structured(
                prompt=prompt,
                schema=FinalAnswer,
                temperature=self.config.temperature_aggregator,
                model=model,
            )
        except StructuredOutputError as e:
            # Fallback: assemble response from drafts in deterministic mode
            final = self._fallback_aggregate(drafts, error=str(e))
        return {"final": final}

    def _fallback_aggregate(self, drafts: List[DrafterOutput], error: str) -> FinalAnswer:
        parts: List[str] = []
        for d in drafts:
            parts.append(f"## {d.role.title()}\n\n{d.body_markdown}")
        parts.append(f"\n*Note: the automatic aggregator failed ({error[:120]}); "
                     "this response was assembled from individual drafts.*")
        confidences = [d.confidence for d in drafts]
        if all(c == "high" for c in confidences):
            level = "high"
        elif "low" in confidences:
            level = "low"
        else:
            level = "medium"
        return FinalAnswer(
            answer_markdown="\n\n".join(parts),
            confidence_level=level,
            open_questions=[],
        )

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Build & run
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    def build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("plan_search", self.node_plan_search)
        g.add_node("web_search", self.node_web_search)

        # Each drafter receives a different SUBSET of sources (filtered by intent).
        # This is the real difference compared to v1: it's not "three models looking
        # at the same soup" — each one focuses on sources relevant to its role.
        g.add_node("drafter_explanation", self._node_drafter(
            role="explanation",
            prompt_key="explanation",
            intents_allowed=["factual", "context"],
            temperature=self.config.temperature_drafter_explanation,
            model_attr="ollama_model_drafter_explanation",
        ))
        g.add_node("drafter_caveats", self._node_drafter(
            role="caveats",
            prompt_key="caveats",
            intents_allowed=["counterpoint", "context"],
            temperature=self.config.temperature_drafter_caveats,
            model_attr="ollama_model_drafter_caveats",
        ))
        g.add_node("drafter_examples", self._node_drafter(
            role="examples",
            prompt_key="examples",
            intents_allowed=["examples", "context"],
            temperature=self.config.temperature_drafter_examples,
            model_attr="ollama_model_drafter_examples",
        ))

        g.add_node("aggregate", self.node_aggregate)

        g.set_entry_point("plan_search")
        g.add_edge("plan_search", "web_search")

        # fan-out
        g.add_edge("web_search", "drafter_explanation")
        g.add_edge("web_search", "drafter_caveats")
        g.add_edge("web_search", "drafter_examples")

        # fan-in
        g.add_edge("drafter_explanation", "aggregate")
        g.add_edge("drafter_caveats", "aggregate")
        g.add_edge("drafter_examples", "aggregate")

        g.add_edge("aggregate", END)
        return g.compile()

    def ask(self, question: str, use_web_search: bool = True) -> FinalAnswer:
        init_state: AgentState = {
            "history": self.history,
            "question": question,
            "use_web_search": use_web_search,
        }

        try:
            out = self.app.invoke(init_state)
        except Exception as e:
            self.console.print(f"[bold red]Unexpected error in pipeline: {e}[/bold red]")
            return FinalAnswer(
                answer_markdown=(
                    f"## ⚠️ Technical Error\n\n"
                    f"An error occurred while processing your question: `{e}`\n\n"
                    "The history has been preserved. Try rephrasing or check if Ollama "
                    "is still running."
                ),
                confidence_level="low",
                open_questions=[],
            )

        final: FinalAnswer = out["final"]

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": final.answer_markdown})
        # Strict trim to prevent indefinite growth
        max_items = self.config.history_max_turns * 4
        if len(self.history) > max_items:
            self.history = self.history[-max_items:]

        return final


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# REPL (Read-Eval-Print Loop)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def main():
    console = Console()

    # Explicit start log — if you don't see this line, it's a terminal/encoding issue
    console.print("[bold]🤖 Starting Private Multi-Agent...[/bold]")

    # 1) Verify that prompts exist BEFORE trying to connect to Ollama
    try:
        config = Config()
    except Exception as e:
        console.print(f"[bold red]Error creating Config: {e}[/bold red]")
        return

    # 2) Attempt initialization — capture ANY failure and display it (no silent failure)
    try:
        assistant = MultiAgentAssistant(config=config)
    except FileNotFoundError as e:
        console.print(f"\n[bold red]❌ Prompt file missing:[/bold red]\n   {e}")
        console.print(
            "\n[yellow]Prompts should be in ./prompts/ next to agent.py.[/yellow]\n"
            "Ensure the folder was copied along with the .py files."
        )
        return
    except RuntimeError as e:
        console.print(f"\n[bold red]❌ {e}[/bold red]")
        return
    except Exception as e:
        # Ultimate safety net
        console.print(f"\n[bold red]❌ Unexpected initialization failure:[/bold red]")
        console.print_exception()  # prints full traceback via rich
        return

    assistant.console.print(Panel(
        'Available Commands:\n'
        '  [cyan]/search on|off[/cyan]   toggle web search\n'
        '  [cyan]/debug[/cyan]           show internal structure of last response\n'
        '  [cyan]exit[/cyan] / [cyan]quit[/cyan]      shutdown',
        title="🤖 Commands", border_style="white",
    ))

    use_search = True
    last_final: Optional[FinalAnswer] = None

    while True:
        assistant.console.print()
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            assistant.console.print("\n[yellow]Shutting down...[/yellow]")
            break
        assistant.console.print()

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        if q.lower().startswith("/search"):
            if "off" in q.lower():
                use_search = False
                assistant.console.print("🔴 [red]Web search disabled.[/red]")
            elif "on" in q.lower():
                use_search = True
                assistant.console.print("🟢 [green]Web search enabled.[/green]")
            continue

        if q.lower() == "/debug":
            if not last_final:
                assistant.console.print("[dim]No questions processed yet.[/dim]")
            else:
                assistant.console.print(Panel(
                    last_final.model_dump_json(indent=2),
                    title="[debug] FinalAnswer", border_style="magenta",
                ))
            continue

        final = assistant.ask(q, use_web_search=use_search)
        last_final = final

        assistant.console.print(Panel(
            Markdown(final.answer_markdown),
            title=f"🤖 Answer (confidence: {final.confidence_level})",
            border_style="blue",
        ))
        if final.open_questions:
            assistant.console.print(Panel(
                "\n".join(f"• {q}" for q in final.open_questions),
                title="🔎 Open Questions", border_style="dim cyan",
            ))


if __name__ == "__main__":
    main()