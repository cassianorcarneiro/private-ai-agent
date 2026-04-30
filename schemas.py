# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Schemas Pydantic para os outputs estruturados dos agentes.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------- Search planning ----------------------------------------------------------------------

class SearchQueryItem(BaseModel):
    query: str
    intent: Literal["factual", "context", "examples", "counterpoint"] = "factual"
    """
    factual:      busca de fatos/definições/dados (boa para o explanation drafter)
    context:      contexto, histórico, comparações
    examples:     exemplos práticos, casos de uso, tutoriais (drafter examples)
    counterpoint: críticas, limitações, debates abertos (drafter caveats)
    """


class SearchPlan(BaseModel):
    queries: List[SearchQueryItem] = Field(default_factory=list)


# ---------- Drafter output -----------------------------------------------------------------------

class DrafterOutput(BaseModel):
    """Estrutura comum a todos os drafters. Permite ao aggregator processar
    de forma uniforme em vez de receber 3 strings opacas."""
    role: Literal["explanation", "caveats", "examples"]
    summary: str = Field(..., description="1-2 frases resumindo o ponto central")
    key_points: List[str] = Field(default_factory=list, description="Bullets principais")
    body_markdown: str = Field(..., description="Conteúdo expandido em markdown")
    confidence: Literal["high", "medium", "low"] = "medium"
    sources_used: List[str] = Field(default_factory=list, description="URLs efetivamente úteis para esta resposta")


# ---------- Final answer -------------------------------------------------------------------------

class FinalAnswer(BaseModel):
    answer_markdown: str
    confidence_level: Literal["high", "medium", "low"] = "medium"
    open_questions: List[str] = Field(default_factory=list, description="Pontos onde dados disponíveis não bastam")
