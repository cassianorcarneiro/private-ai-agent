# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# AEGIS-MIND
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class Config:
    # Configurações do Ollama - será detectado automaticamente
    OLLAMA_HOST = "http://localhost:11434"
    MODEL_NAME = "deepseek-r1:8b"  # Será sobrescrito pelo detectado
    
    # Configurações de pesquisa
    MAX_SEARCH_RESULTS = 10
    SEARCH_TIMEOUT = 30
    
    # Configurações de monitoramento
    LOG_FILE = "agent_monitor.log"
    ENABLE_CONSOLE_MONITOR = True

    # Configurações de agentes colaborativos

    AGENT_ROLES = [
        {
            "name": "Pesquisador",
            "goal": "Encontrar fatos relevantes e pontos-chave para responder à pergunta.",
            "temperature": 0.3,
        },
        {
            "name": "Analista",
            "goal": "Organizar o raciocínio e explicar o tema de forma clara e estruturada.",
            "temperature": 0.5,
        },
        {
            "name": "Cético",
            "goal": "Apontar limitações, ambiguidades e riscos da resposta.",
            "temperature": 0.2,
        },
    ]

    timelimit = "y"          # últimos 12 meses
    per_domain_limit = 2     # diversidade
    min_score = 2            # corta ruído
    fetch_top_n = 3          # busca texto completo dos top 3
    fetch_chars = 1200       # limite de texto (evita prompt gigante)
    allow_domains = [
        "arxiv.org", "ieee.org", "acm.org", "springer.com",
        "nature.com", "sciencedirect.com", "openreview.net"
    ]
    block_domains = [
        "pinterest.com", "facebook.com", "instagram.com", "tiktok.com"
    ]