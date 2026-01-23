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
    MAX_SEARCH_RESULTS = 5
    SEARCH_TIMEOUT = 30
    
    # Configurações de monitoramento
    LOG_FILE = "agent_monitor.log"
    ENABLE_CONSOLE_MONITOR = True

    # Configurações de agentes colaborativos
    ENABLE_MULTI_AGENT = True
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
