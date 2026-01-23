from duckduckgo_search import DDGS
import time
from typing import List, Dict
from utils.monitoring import SearchMonitor

class WebSearcher:
    def __init__(self, monitor: SearchMonitor, max_results=5):
        self.ddgs = DDGS()
        self.monitor = monitor
        self.max_results = max_results
        self.search_history = []
    
    def search(self, query: str) -> List[Dict]:
        """
        Realiza uma pesquisa no DuckDuckGo e retorna os resultados
        """
        try:
            print(f"🔍 Pesquisando: {query}")
            
            # Realizar a pesquisa
            results = list(self.ddgs.text(
                keywords=query,
                max_results=self.max_results
            ))
            
            # Extrair fontes
            sources = [result.get('href', '') for result in results]
            
            # Registrar no monitor
            self.monitor.log_search(query, len(results), sources)
            
            # Adicionar ao histórico
            self.search_history.append({
                'timestamp': time.time(),
                'query': query,
                'results_count': len(results),
                'sources': sources
            })
            
            return results
            
        except Exception as e:
            print(f"❌ Erro na pesquisa: {e}")
            self.monitor.logger.error(f"Erro na pesquisa '{query}': {e}")
            return []
    
    def get_search_context(self, query: str) -> str:
        """
        Realiza pesquisa e formata os resultados como contexto para o modelo
        """
        results = self.search(query)
        
        if not results:
            return "Nenhum resultado encontrado na pesquisa."
        
        # Formatar resultados como contexto
        context = "Resultados da pesquisa:\n\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. {result.get('title', 'Sem título')}\n"
            context += f"   URL: {result.get('href', 'Sem URL')}\n"
            context += f"   Descrição: {result.get('body', 'Sem descrição')}\n\n"
        
        return context
    
    def get_search_history(self) -> List[Dict]:
        """Retorna o histórico de pesquisas"""
        return self.search_history.copy()