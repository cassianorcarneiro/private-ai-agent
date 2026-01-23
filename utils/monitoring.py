import logging
import threading
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

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