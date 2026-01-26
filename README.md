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

The algorithm considers using the "mixtral:8x7b" model (it is possible to change it in the config.py file).

(1) Install Ollama
(2) Open the terminal in Windows
(3) Enter the command: ollama pull deepseek-r1:8b
