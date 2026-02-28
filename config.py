# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# PRIVATE AI AGENT
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class Config:
 
    ollama_base_url = "http://127.0.0.1:11434"
    ollama_model = "qwen2.5:1.5b" # mixtral:8x7b, deepseek-r1:8b, gemma3:27b, qwen2.5:1.5b, llama3.1
    temperature_planner = 0.0
    temperature_drafters = 0.3
    temperature_aggregator = 0.1
    ddgs_max_results_per_query = 5
    max_queries = 6
    max_sources_in_prompt = 12