# Configurações do agente
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