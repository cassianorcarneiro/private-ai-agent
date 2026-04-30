# 🔒 Private AI Agent

> Privacy-first, offline-capable AI assistant powered by local language models.

A multi-agent AI assistant that orchestrates specialized agents and optional web retrieval to deliver high-quality, well-grounded responses — without sharing your data with proprietary model providers.

<p align="center">
  <img alt="Stack" src="https://img.shields.io/badge/Stack-LangGraph%20%2B%20Ollama-blue?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
  <img alt="Status" src="https://img.shields.io/badge/Privacy-Local%20First-success?style=for-the-badge">
</p>

---

## 📦 Why this project

Most modern AI assistants ship your prompts, context, and conversation history to third-party APIs. This project takes the opposite approach:

- 🏠 **Local-first** — all reasoning happens on your machine via [Ollama](https://ollama.com)
- 🤖 **Multi-agent by design** — three specialist drafters work in parallel on the *same question* but with *different sources and roles*, and an aggregator merges their structured outputs into one balanced answer
- 🎯 **Intent-aware retrieval** — search queries carry intent labels (`factual` / `context` / `examples` / `counterpoint`) so each drafter sees only sources matching its role: explanation gets factual content, caveats gets counterpoints, examples gets tutorials and use cases
- 📊 **Source re-ranking** — results from trusted domains (Wikipedia, MDN, docs.python.org, ArXiv, Reuters, etc.) are bumped to the top before being summarized into prompts
- 🛡️ **Validated outputs end-to-end** — every agent returns Pydantic-validated JSON with `format=json` enforcement and automatic re-prompting on validation failure
- 🧩 **Versioned prompts** — every agent's instructions live in `./prompts/*.txt`, separate from code, so changes are diff-able
- 🔍 **Optional web retrieval** — DuckDuckGo search can be toggled on or off per question
- 🔬 **Inspectable state** — built on [LangGraph](https://github.com/langchain-ai/langgraph); the `/debug` command shows the full structured `FinalAnswer` of the last question

---

## 🏗️ How it works

```
                ┌─> Drafter: explanation  (sees factual + context sources)  ─┐
plan_search ──> web_search ─┼─> Drafter: caveats     (sees counterpoint + context sources) ─┼─> aggregate ──> END
                └─> Drafter: examples    (sees examples + context sources)   ─┘
```

| Stage | Role |
|-------|------|
| **Planner** | Generates 3–6 search queries, each tagged with an intent (`factual`, `context`, `examples`, `counterpoint`) |
| **Web Search** | Fetches results via DuckDuckGo (skipped when `/search off`); re-ranks by trusted-domain bonus |
| **Drafter: explanation** | Builds the core explanation. Sees only `factual` + `context` sources |
| **Drafter: caveats** | Surfaces limitations, debates, edge cases. Sees only `counterpoint` + `context` sources |
| **Drafter: examples** | Provides concrete examples, code, next steps. Sees only `examples` + `context` sources. Slightly higher temperature for controlled creativity |
| **Aggregator** | Merges the three structured drafts (Pydantic JSON, not text dumps) into a final markdown answer with calibrated confidence and a list of open questions |

Each drafter returns a `DrafterOutput` with `summary`, `key_points`, `body_markdown`, `confidence`, and `sources_used`. The aggregator consumes the JSON directly — it sees the structure, not just three text blobs.

The fan-out / fan-in pattern is implemented with LangGraph's `Annotated[List[DrafterOutput], add]` reducer, so the three drafters run independently and their structured outputs are concatenated automatically.

### Why this differs from a "three agents in parallel saying similar things" design

The earlier version of this project gave all three drafters the same sources and the same context. With small models, that produced near-identical drafts that the aggregator had to merge — a lot of compute for marginal gain. The current design enforces source-level division of labor: the explanation drafter literally cannot see the counterpoint sources, and vice-versa. The drafts that come back are genuinely complementary rather than redundant.

---

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama** running locally — get it at [ollama.com/download](https://ollama.com/download)
- **~5 GB free disk** for a typical model like `llama3.1:8b`
- **Internet** for the first run (pulling the model)

---

## 🚀 Quick start

### 1. Install Ollama and pull a model

```bash
ollama pull llama3.1:8b        # or qwen2.5:7b, mistral, deepseek-coder, etc.
ollama serve
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the agent

```bash
python agent.py
```

Then interact with the assistant:

```
Você: What are the trade-offs of using local LLMs vs cloud APIs?
```

### Commands

| Command | Effect |
|---------|--------|
| `/search on` | Enable DuckDuckGo web retrieval (default) |
| `/search off` | Disable web retrieval — model uses only internal knowledge |
| `/debug` | Print the structured `FinalAnswer` JSON of the last response |
| `exit` or `quit` | Leave the assistant |

### Session memory

The assistant maintains a rolling conversation history (last 6 turns by default), trimmed to a hard cap so RAM doesn't grow unbounded. History is held in memory only and discarded when the program exits — nothing is written to disk.

---

## ⚙️ Configuration

Edit `config.py` to tune behavior:

| Field | Purpose | Default |
|-------|---------|---------|
| `ollama_model` | Default model (exact match preferred, then prefix, then substring) | `llama3.1:8b` |
| `ollama_base_url` | Ollama server URL | `http://127.0.0.1:11434` |
| `ollama_model_drafter_*` | Per-drafter model overrides — empty means use default | `""` |
| `ollama_model_aggregator` | Override for the aggregator (e.g. larger model for the merge) | `""` |
| `temperature_planner` | Determinism of the search planner | `0.0` |
| `temperature_drafter_explanation` | Conservative for factual rigor | `0.2` |
| `temperature_drafter_caveats` | Low for careful critique | `0.2` |
| `temperature_drafter_examples` | Slightly higher for richer examples | `0.4` |
| `temperature_aggregator` | Determinism of the final merge | `0.1` |
| `json_max_retries` | Retries when an agent's JSON fails Pydantic validation | `2` |
| `max_queries` | Cap on web search queries per question | `6` |
| `ddgs_max_results_per_query` | Results fetched per query | `5` |
| `max_sources_in_prompt` | Sources passed to each drafter, post-filtering | `8` |
| `preferred_sources` | Domains the re-ranker boosts | Wikipedia, MDN, docs.python.org, GitHub, SO, ArXiv, Reuters, … |
| `prompts_dir` | Where versioned prompt files live | `./prompts` |

### Recommended models

For general-purpose Q&A:

- `llama3.1:8b` — strong all-rounder
- `qwen2.5:7b` — excellent reasoning, multilingual
- `mistral:7b` — fast and capable

For technical / coding questions:

- `deepseek-coder` — code-focused, very capable
- `qwen2.5-coder` — strong on code and explanation

Larger models (14B–70B) produce noticeably better aggregated answers if your hardware can run them.

**Hybrid setup:** keep a fast model as the default for the drafters, and set `ollama_model_aggregator` to a larger model — the aggregator does the most reasoning-heavy work.

### Editing prompts

Prompts live as plain text files in `./prompts/` (`01_planner.txt`, `02_drafter_explanation.txt`, etc.). Edit them directly; they're versioned by your VCS like any other source. The agent loads them at startup.

---

## 🔐 Privacy model

- ✅ Prompts and conversation history **never** leave your machine when web search is off
- ✅ When web search is on, only the **search queries** generated by the planner are sent to DuckDuckGo — never the full conversation or original question verbatim (unless the planner falls back to using the question as a query)
- ✅ No telemetry, no analytics, no API keys required
- ✅ The Ollama instance runs locally; you control which models are pulled and used
- ✅ Conversation is kept in RAM only, with a hard cap, and is wiped on exit

---

## 📁 Project structure

```
private-multi-agent/
├── agent.py            # MultiAgentAssistant class, graph nodes, REPL loop
├── config.py           # Config dataclass with models and behavior settings
├── schemas.py          # Pydantic schemas (SearchPlan, DrafterOutput, FinalAnswer)
├── io_utils.py         # Source ranking and intent-based filtering
├── llm_client.py       # Unified Ollama client with structured-JSON retry
├── prompts/
│   ├── 01_planner.txt
│   ├── 02_drafter_explanation.txt
│   ├── 03_drafter_caveats.txt
│   ├── 04_drafter_examples.txt
│   └── 05_aggregator.txt
├── requirements.txt
└── README.md
```

---

## 🛣️ Roadmap

- [ ] Persistent conversation history (opt-in)
- [ ] Configurable agent personas via YAML
- [ ] Pluggable search backends (SearXNG, Brave, etc.)
- [ ] Streaming responses
- [ ] Tool-use support (calculator, code execution, file reading)
- [ ] Per-node checkpoints so `/debug` can show every intermediate object

---

## 📜 License

MIT — see `LICENSE` file.

## 👤 Author

**Cassiano Ribeiro Carneiro** — [@cassianorcarneiro](https://github.com/cassianorcarneiro)

---

### 🤖 AI Assistance Disclosure

The codebase architecture, organizational structure, and stylistic formatting of this repository were refactored and optimized leveraging [Claude](https://www.anthropic.com/claude) by Anthropic. All core business logic and intellectual property remain the work of the repository authors and are governed by the project's license.

---

> *Built on the principle that useful AI shouldn't require giving up your data.*
