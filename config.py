# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE AI AGENT
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
 
    # ----- Ollama settings -----
    ollama_model = "qwen2.5:1.5b" # mixtral:8x7b, deepseek-r1:8b, gemma3:27b, qwen2.5:1.5b, llama3.1
    ollama_base_url = "http://127.0.0.1:11434"
    
    # ----- LLM temperatures (lower = more deterministic, important for code) -----
    temperature_planner = 0.0
    temperature_drafters = 0.3
    temperature_aggregator = 0.1

    # ----- Web Search settings -----
    ddgs_max_results_per_query = 5
    max_queries = 6
    max_sources_in_prompt = 12