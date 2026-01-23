import logging
import threading
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json
from duckduckgo_search import DDGS
import time
from typing import List, Dict
from .monitoring import SearchMonitor

class SearchMonitor:
    def __init__(self, log_file="search_monitor.log"):
        self.log_file = log_file
        self.console = Console()
        self.lock = threading.Lock()
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_search(self, query, results_count, sources=None):
        """Registra uma pesquisa realizada"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "type": "SEARCH",
            "query": query,
            "results_count": results_count,
            "sources": sources or []
        }
        
        with self.lock:
            # Log para arquivo
            self.logger.info(f"Pesquisa: '{query}' - {results_count} resultados")
            
            # Exibição no console (opcional)
            if hasattr(self, 'console_monitor') and self.console_monitor:
                self._display_search_console(query, results_count, sources)
    
    def log_response(self, prompt, response, tokens_used=None):
        """Registra uma resposta do modelo"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "type": "RESPONSE",
            "prompt_length": len(prompt),
            "response_length": len(response),
            "tokens_used": tokens_used
        }
        
        with self.lock:
            self.logger.info(f"Resposta gerada - {len(response)} caracteres")
    
    def _display_search_console(self, query, results_count, sources):
        """Exibe informações da pesquisa no console de forma formatada"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Timestamp", style="dim")
        table.add_column("Query")
        table.add_column("Results")
        table.add_column("Sources")
        
        sources_str = ", ".join(sources[:3]) if sources else "N/A"
        if sources and len(sources) > 3:
            sources_str += f" ... (+{len(sources)-3})"
        
        table.add_row(
            datetime.now().strftime("%H:%M:%S"),
            query[:50] + "..." if len(query) > 50 else query,
            str(results_count),
            sources_str
        )
        
        self.console.print(Panel(table, title="🔍 Monitor de Pesquisas", border_style="green"))

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