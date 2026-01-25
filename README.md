# aegis-mind
AegisMind is a privacy-first, offline-capable AI system built on local language models. It orchestrates multiple specialized agents and optional web retrieval to deliver high-quality responses without sharing user data with proprietary model providers.

Pipeline:
(1) Planner agent -> search queries
(2) Web search via DDGS
(3) 3 responder agents in parallel
(4) Aggregator agent -> final answer

Notes:
- Ollama must be running locally (default: http://localhost:11434).
- ddgs package provides DDGS class. Some environments still use duckduckgo_search.
This file supports both imports.