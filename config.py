# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE AI AGENT
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class Config:
 
    ollama_base_url = "http://localhost:11434"
    ollama_model = "mixtral:8x7b" # mixtral:8x7b, deepseek-r1:8b, gemma3:27b
    temperature_planner = 0.0
    temperature_drafters = 0.3
    temperature_aggregator = 0.1
    ddgs_max_results_per_query = 5
    max_queries = 6
    max_sources_in_prompt = 12