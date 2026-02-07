## Overview

This project is a **privacy-first, offline-capable AI system** built on **local large language models**. Its core design philosophy emphasizes **explicit orchestration, transparency, and user data sovereignty**, avoiding any dependency on proprietary cloud-based model providers.

Unlike end-to-end monolithic chatbots that rely on implicit, internal task decomposition, this system adopts a **predefined multi-agent pipeline**. Each stage of reasoning is handled by a **specialized agent with a fixed role**, enabling predictable behavior, auditability, and fine-grained control over the reasoning process.

---

## Architectural Approach

The system follows a **static, role-based multi-agent architecture**, where the reasoning workflow is explicitly engineered rather than dynamically inferred at runtime. This design choice prioritizes:

- Deterministic execution
- Interpretability of each reasoning step
- Easier debugging and extension
- Suitability for private, regulated, or offline environments

While modern large chatbots often rely on *implicit latent planning*, this project deliberately externalizes the reasoning structure as an explicit computational graph.

---

## Pipeline

The processing flow is fixed and executed as follows:

1. **Planner Agent**  
   Analyzes the user prompt and formulates search queries when external information is required.

2. **Web Search (Optional)**  
   Queries DuckDuckGo using the DDGS interface for retrieval-augmented generation.

3. **Responder Agents (Parallel)**  
   Three independent responder agents generate candidate answers in parallel, increasing diversity and robustness.

4. **Aggregator Agent**  
   Synthesizes the responder outputs into a single coherent and high-quality final response.

---

## Key Characteristics

- Fully local inference (no data leaves the machine)
- Explicit agent roles and execution order
- Optional web retrieval, cleanly separated from reasoning
- Modular design, allowing agent replacement or extension
- Compatible with constrained or offline deployments

---

## Requirements & Notes

- **Ollama** must be running locally  
  Default endpoint: `http://localhost:11434`

- The project supports both:
  - `ddgs` (preferred, provides the `DDGS` class)
  - `duckduckgo_search` (legacy compatibility)

- The default model is:
  - `mixtral:8x7b`  
  This can be changed in the `config.py` file.

---

## Setup

1. Install **Ollama**
2. Open a terminal (Windows or compatible shell)
3. Pull a local model, for example:
   
   ```bash
   ollama pull deepseek-r1:8b

## Design Rationale

This project intentionally contrasts with dynamically orchestrated LLM systems by embracing explicit engineering over emergent behavior. The result is a system that trades some flexibility for control, safety, and clarity, making it well-suited for private assistants, research, and on-premises AI applications.
