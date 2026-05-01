# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# I/O Utilities: source ranking by trusted domain.
# Kept outside of agent.py to be testable in isolation.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

import re
from typing import Any, Dict, List


def domain_of(url: str) -> str:
    """Extracts the effective domain from a URL, without external dependencies."""
    if not url:
        return ""
    m = re.match(r"^https?://([^/]+)", url)
    if not m:
        return ""
    host = m.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def rank_sources(
    results: List[Dict[str, Any]],
    preferred_domains: List[str],
    max_items: int,
) -> List[Dict[str, Any]]:
    """
    Reorders results: trusted sources (substring match in domain) come
    first. Maintains relative order within each bucket.
    """
    preferred_lower = [d.lower() for d in preferred_domains]

    def score(item: Dict[str, Any]) -> int:
        if item.get("error"):
            return -1
        url = item.get("url") or item.get("href") or ""
        host = domain_of(url)
        if not host:
            return 0
        for d in preferred_lower:
            if d in host:
                return 2
        return 1

    annotated = [(score(r), i, r) for i, r in enumerate(results)]
    annotated.sort(key=lambda t: (-t[0], t[1]))
    ranked = [r for s, _, r in annotated if s >= 0]
    return ranked[:max_items]


def filter_sources_by_intent(
    sources: List[Dict[str, Any]],
    intents_allowed: List[str],
) -> List[Dict[str, Any]]:
    """Filters sources by the intent annotated when the query was run."""
    if not intents_allowed:
        return sources
    return [s for s in sources if s.get("intent") in intents_allowed]


def summarize_sources(sources: List[Dict[str, Any]], max_items: int) -> str:
    """Formats sources into compact text to be included in prompts."""
    lines: List[str] = []
    for item in sources[:max_items]:
        if item.get("error"):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("url") or item.get("href") or "").strip()
        body = (item.get("body") or item.get("snippet") or item.get("content") or "").strip()
        if not (title or url or body):
            continue
        lines.append(f"- {title}\n  {url}\n  {body[:400]}")
    return "\n".join(lines) if lines else "(No useful sources returned.)"