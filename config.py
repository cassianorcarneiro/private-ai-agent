# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE AI AGENT — V2
# CASSIANO RIBEIRO CARNEIRO
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:

    # ----- Ollama settings -----
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://127.0.0.1:11434"

    # Allows using different models per agent (advanced).
    # If empty, uses ollama_model. Useful for running a larger model only 
    # in the aggregator and smaller models in the drafters.
    ollama_model_drafter_explanation: str = ""
    ollama_model_drafter_caveats: str = ""
    ollama_model_drafter_examples: str = ""
    ollama_model_aggregator: str = ""

    # ----- Temperatures -----
    temperature_planner: float = 0.0
    # Drafters with spaced temperatures to diversify perspectives.
    # explanation = more conservative (factual)
    # caveats     = low (rigor)
    # examples    = slightly higher (controlled creativity in examples)
    temperature_drafter_explanation: float = 0.2
    temperature_drafter_caveats: float = 0.2
    temperature_drafter_examples: float = 0.4
    temperature_aggregator: float = 0.1

    # ----- Robustness -----
    json_max_retries: int = 2

    # ----- Web Search -----
    ddgs_max_results_per_query: int = 5
    max_queries: int = 6
    max_sources_in_prompt: int = 8

    # Trusted domains (generic, non-medical). These receive a boost in ranking.
    # The list is deliberately short — for an open-domain assistant,
    # diversity > purity. Adjust according to your typical usage.
    preferred_sources: List[str] = field(default_factory=lambda: [
        # Technical documentation and primary sources
        "wikipedia.org", "wikimedia.org",
        "github.com", "gitlab.com",
        "stackoverflow.com", "stackexchange.com",
        # Official standards and references
        "developer.mozilla.org", "docs.python.org", "rust-lang.org",
        "arxiv.org", "nature.com", "science.org",
        # News with editorial curation
        "reuters.com", "apnews.com", "bbc.com",
    ])

    # ----- History -----
    history_max_turns: int = 6

    # Directory for versioned prompts
    prompts_dir: str = "./prompts"