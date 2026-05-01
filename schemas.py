# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Pydantic schemas for the agents' structured outputs.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------- Search planning ----------------------------------------------------------------------

class SearchQueryItem(BaseModel):
    query: str
    intent: Literal["factual", "context", "examples", "counterpoint"] = "factual"
    """
    factual:      search for facts/definitions/data (good for the explanation drafter)
    context:      context, history, comparisons
    examples:     practical examples, use cases, tutorials (examples drafter)
    counterpoint: criticisms, limitations, open debates (caveats drafter)
    """


class SearchPlan(BaseModel):
    queries: List[SearchQueryItem] = Field(default_factory=list)


# ---------- Drafter output -----------------------------------------------------------------------

class DrafterOutput(BaseModel):
    """Common structure for all drafters. Allows the aggregator to process 
    them uniformly instead of receiving 3 opaque strings."""
    role: Literal["explanation", "caveats", "examples"]
    summary: str = Field(..., description="1-2 sentences summarizing the central point")
    key_points: List[str] = Field(default_factory=list, description="Main bullet points")
    body_markdown: str = Field(..., description="Expanded content in markdown")
    confidence: Literal["high", "medium", "low"] = "medium"
    sources_used: List[str] = Field(default_factory=list, description="URLs that were actually useful for this response")


# ---------- Final answer -------------------------------------------------------------------------

class FinalAnswer(BaseModel):
    answer_markdown: str
    confidence_level: Literal["high", "medium", "low"] = "medium"
    open_questions: List[str] = Field(default_factory=list, description="Points where available data is insufficient")