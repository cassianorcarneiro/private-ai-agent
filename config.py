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

    # Permite usar modelos diferentes por agente (avançado).
    # Se vazio, usa ollama_model. Útil para rodar um modelo maior só no
    # aggregator e modelos menores nos drafters.
    ollama_model_drafter_explanation: str = ""
    ollama_model_drafter_caveats: str = ""
    ollama_model_drafter_examples: str = ""
    ollama_model_aggregator: str = ""

    # ----- Temperaturas -----
    temperature_planner: float = 0.0
    # Drafters com temperaturas espaçadas para diversificar perspectivas.
    # explanation = mais conservador (factual)
    # caveats     = baixo (rigor)
    # examples    = um pouco mais alto (criatividade controlada nos exemplos)
    temperature_drafter_explanation: float = 0.2
    temperature_drafter_caveats: float = 0.2
    temperature_drafter_examples: float = 0.4
    temperature_aggregator: float = 0.1

    # ----- Robustez -----
    json_max_retries: int = 2

    # ----- Web Search -----
    ddgs_max_results_per_query: int = 5
    max_queries: int = 6
    max_sources_in_prompt: int = 8

    # Domínios confiáveis (genéricos, não médicos). Recebem boost no ranking.
    # A lista é deliberadamente curta — para um assistente de domínio aberto,
    # diversidade > pureza. Ajuste conforme seu uso típico.
    preferred_sources: List[str] = field(default_factory=lambda: [
        # Documentação técnica e fontes primárias
        "wikipedia.org", "wikimedia.org",
        "github.com", "gitlab.com",
        "stackoverflow.com", "stackexchange.com",
        # Padrões e referências oficiais
        "developer.mozilla.org", "docs.python.org", "rust-lang.org",
        "arxiv.org", "nature.com", "science.org",
        # Notícias com curadoria editorial
        "reuters.com", "apnews.com", "bbc.com",
    ])

    # ----- Histórico -----
    history_max_turns: int = 6

    # Diretório de prompts versionados
    prompts_dir: str = "./prompts"
