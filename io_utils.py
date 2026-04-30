# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Utilitários de I/O: ranking de fontes por domínio confiável.
# Mantido fora do agent.py para ser testável isoladamente.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

import re
from typing import Any, Dict, List


def domain_of(url: str) -> str:
    """Extrai o domínio efetivo de uma URL, sem dependências externas."""
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
    Reordena resultados: fontes confiáveis (substring match em domínio) vêm
    primeiro. Mantém ordem relativa dentro de cada bucket.
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
    """Filtra fontes pelo intent anotado quando a query foi rodada."""
    if not intents_allowed:
        return sources
    return [s for s in sources if s.get("intent") in intents_allowed]


def summarize_sources(sources: List[Dict[str, Any]], max_items: int) -> str:
    """Formata fontes em texto compacto para entrar nos prompts."""
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
    return "\n".join(lines) if lines else "(Nenhuma fonte útil retornada.)"
