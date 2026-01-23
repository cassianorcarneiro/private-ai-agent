import ollama
from config import Config
from utils.web_search import WebSearcher
from utils.monitoring import SearchMonitor
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import threading
import time
import json
from datetime import datetime

class DeepSeekAgent:
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.monitor = SearchMonitor(config.LOG_FILE)
        self.monitor.console_monitor = config.ENABLE_CONSOLE_MONITOR
        self.searcher = WebSearcher(self.monitor, config.MAX_SEARCH_RESULTS)
        
        # Verificar se o modelo está disponível
        self._check_model()
    
    def _check_model(self):
        """Verifica se o modelo DeepSeek está disponível no Ollama"""
        try:
            # Método corrigido para verificar modelos
            models_response = ollama.list()
            
            self.console.print(f"[dim]Tipo da resposta: {type(models_response)}[/dim]")
            self.console.print(f"[dim]Atributos disponíveis: {dir(models_response)}[/dim]")
            
            # Acessar corretamente a lista de modelos
            model_names = []
            model_details = []
            
            if hasattr(models_response, 'models') and models_response.models:
                for model in models_response.models:
                    model_name = model.model  # Acessar via atributo 'model'
                    model_names.append(model_name)
                    model_details.append({
                        'name': model_name,
                        'size': model.size,
                        'modified': model.modified_at,
                        'parameters': getattr(model.details, 'parameter_size', 'N/A') if model.details else 'N/A'
                    })
            
            self.console.print(f"[dim]Modelos encontrados: {model_names}[/dim]")
            
            if not model_names:
                self.console.print("❌ [red]Nenhum modelo encontrado no Ollama[/red]")
                raise Exception("Nenhum modelo disponível")
            
            # Encontrar modelos DeepSeek
            deepseek_models = [model for model in model_details if 'deepseek' in model['name'].lower()]
            
            if deepseek_models:
                # Usar o primeiro modelo DeepSeek encontrado
                selected_model = deepseek_models[0]
                self.config.MODEL_NAME = selected_model['name']
                
                self.console.print(Panel(
                    f"✅ [green]Modelo selecionado:[/green] {self.config.MODEL_NAME}\n"
                    f"📊 [cyan]Tamanho:[/cyan] {selected_model['size']/1024/1024/1024:.1f}GB\n"
                    f"⚙️ [yellow]Parâmetros:[/yellow] {selected_model['parameters']}\n"
                    f"📅 [magenta]Modificado:[/magenta] {selected_model['modified'].strftime('%Y-%m-%d %H:%M')}",
                    title="🤖 Modelo Carregado",
                    border_style="green"
                ))
            else:
                # Usar o primeiro modelo disponível
                selected_model = model_details[0]
                self.config.MODEL_NAME = selected_model['name']
                self.console.print(Panel(
                    f"⚠️ [yellow]Usando modelo disponível:[/yellow] {self.config.MODEL_NAME}\n"
                    f"📊 [cyan]Tamanho:[/cyan] {selected_model['size']/1024/1024/1024:.1f}GB",
                    title="🤖 Modelo Alternativo",
                    border_style="yellow"
                ))
                
        except Exception as e:
            self.console.print(f"❌ Erro ao conectar com Ollama: {e}", style="bold red")
            self.console.print("\n🔧 [yellow]Soluções possíveis:[/yellow]")
            self.console.print("1. Verifique se o Ollama está rodando: ollama serve")
            self.console.print("2. Instale um modelo: ollama pull deepseek-coder")
            raise
    
    def _should_search(self, user_input: str) -> bool:
        """
        Determina se uma pesquisa na internet é necessária
        """
        search_keywords = [
            'pesquisar', 'buscar', 'encontrar', 'procurar',
            'notícias', 'atual', 'recente', 'hoje', 'agora',
            'internet', 'web', 'online', 'duckduckgo', 'google',
            'qual é', 'quem é', 'o que é', 'quando', 'onde',
            'como', 'por que', 'atualização', 'últimas',
            'novidades', 'preço', 'cotação', 'clima', 'tempo'
        ]
        
        user_input_lower = user_input.lower()
        
        # Verificar palavras-chave
        for keyword in search_keywords:
            if keyword in user_input_lower:
                return True
        
        # Verificar se é uma pergunta factual
        question_prefixes = [
            'qual ', 'quem ', 'o que ', 'onde ', 'quando ',
            'como ', 'por que ', 'quantos ', 'quantas ', 'quanto '
        ]
        if any(user_input_lower.startswith(prefix) for prefix in question_prefixes):
            return True
        
        # Verificar se pergunta sobre eventos atuais
        current_events = ['eleição', 'presidente', 'governo', 'mercado', 'bolsa', 'bitcoin']
        if any(event in user_input_lower for event in current_events):
            return True
        
        return False
    
    def _extract_search_query(self, user_input: str) -> str:
        """
        Extrai a query de pesquisa do input do usuário
        """
        # Remove comandos explícitos de pesquisa
        remove_phrases = [
            'pesquise por', 'busque por', 'encontre', 'procure por',
            'pesquisar', 'buscar', 'encontrar', 'procurar',
            'quero saber sobre', 'preciso de informações sobre',
            'me mostre sobre', 'me fale sobre'
        ]
        
        query = user_input.lower()
        for phrase in remove_phrases:
            query = query.replace(phrase, '')
        
        return query.strip()
    
    def generate_response(self, user_input: str, search_results: str = "") -> str:
        """
        Gera uma resposta usando o modelo DeepSeek
        """
        try:
            # Construir o prompt
            if search_results and search_results != "Nenhum resultado encontrado na pesquisa.":
                prompt = f"""Com base nos resultados de pesquisa abaixo e no seu conhecimento, responda à pergunta do usuário de forma útil, precisa e bem estruturada.

RESULTADOS DA PESQUISA:
{search_results}

PERGUNTA DO USUÁRIO: {user_input}

INSTRUÇÕES:
- Responda em português claro e natural
- Seja informativo e direto
- Use os resultados da pesquisa quando relevantes
- Se os resultados não forem úteis, use seu conhecimento
- Formate a resposta de forma organizada"""
            else:
                prompt = f"""Responda à seguinte pergunta do usuário de forma útil, precisa e bem estruturada:

PERGUNTA: {user_input}

INSTRUÇÕES:
- Responda em português claro e natural
- Seja informativo e direto
- Formate a resposta de forma organizada"""
            
            # Gerar resposta
            self.console.print(f"[dim]🔄 Gerando resposta com {self.config.MODEL_NAME}...[/dim]")
            
            response = ollama.generate(
                model=self.config.MODEL_NAME,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 1000
                }
            )
            
            # Acessar a resposta corretamente
            response_text = response.response
            
            # Registrar no monitor
            self.monitor.log_response(prompt, response_text)
            
            return response_text
            
        except Exception as e:
            error_msg = f"❌ Erro ao gerar resposta: {e}"
            self.monitor.logger.error(error_msg)
            self.console.print(f"[dim]Detalhes do erro: {type(e).__name__}[/dim]")
            return error_msg
    
    def process_query(self, user_input: str) -> str:
        """
        Processa a query do usuário e retorna uma resposta
        """
        self.console.print(Panel(
            f"💭 [bold blue]Usuário:[/bold blue] {user_input}",
            border_style="blue"
        ))
        
        # Determinar se precisa pesquisar
        if self._should_search(user_input):
            search_query = self._extract_search_query(user_input)
            self.console.print(f"🔍 [yellow]Realizando pesquisa: '{search_query}'[/yellow]")
            search_context = self.searcher.get_search_context(search_query)
            
            # Gerar resposta com contexto da pesquisa
            response = self.generate_response(user_input, search_context)
        else:
            # Gerar resposta sem pesquisa
            response = self.generate_response(user_input)
        
        return response
    
    def chat_loop(self):
        """
        Loop principal de chat
        """
        self.console.print(Panel(
            f"🤖 [bold green]Agente DeepSeek Ativado[/bold green]\n"
            f"📚 Modelo: {self.config.MODEL_NAME}\n"
            f"🔍 Pesquisas automáticas ativadas\n"
            f"🌐 Conectado ao DuckDuckGo\n"
            f"📊 Monitoramento ativo\n"
            f"\n💬 [bold]Comandos:[/bold]\n"
            f"  • 'sair' - Encerrar\n"
            f"  • 'historico' - Pesquisas recentes\n"
            f"  • 'modelos' - Listar modelos\n"
            f"  • 'teste' - Testar modelo\n"
            f"  • 'status' - Status do sistema",
            border_style="green"
        ))
        
        while True:
            try:
                user_input = input("\n👤 Você: ").strip()
                
                if user_input.lower() == 'sair':
                    self.console.print("👋 Até logo!", style="bold yellow")
                    break
                elif user_input.lower() == 'historico':
                    self._show_search_history()
                    continue
                elif user_input.lower() == 'modelos':
                    self._show_available_models()
                    continue
                elif user_input.lower() == 'teste':
                    self._test_model()
                    continue
                elif user_input.lower() == 'status':
                    self._show_system_status()
                    continue
                elif not user_input:
                    continue
                
                # Processar a query
                start_time = time.time()
                response = self.process_query(user_input)
                response_time = time.time() - start_time
                
                # Exibir resposta
                self.console.print(Panel(
                    Markdown(response),
                    title=f"🤖 DeepSeek ({response_time:.1f}s)",
                    border_style="green"
                ))
                
            except KeyboardInterrupt:
                self.console.print("\n👋 Encerrado pelo usuário", style="bold yellow")
                break
            except Exception as e:
                self.console.print(f"❌ Erro: {e}", style="bold red")
    
    def _show_search_history(self):
        """Mostra o histórico de pesquisas"""
        history = self.searcher.get_search_history()
        
        if not history:
            self.console.print("📝 Nenhuma pesquisa realizada ainda.", style="yellow")
            return
        
        self.console.print("\n📊 [bold]Histórico de Pesquisas:[/bold]")
        for i, search in enumerate(history[-5:], 1):
            timestamp = time.strftime('%H:%M:%S', time.localtime(search['timestamp']))
            self.console.print(
                f"  {i}. [{timestamp}] '{search['query']}' - "
                f"{search['results_count']} resultados"
            )
    
    def _show_available_models(self):
        """Mostra modelos disponíveis"""
        try:
            models_response = ollama.list()
            self.console.print("\n📚 [bold]Modelos Disponíveis:[/bold]")
            
            if hasattr(models_response, 'models') and models_response.models:
                for model in models_response.models:
                    size_gb = model.size / 1024 / 1024 / 1024
                    params = getattr(model.details, 'parameter_size', 'N/A') if model.details else 'N/A'
                    
                    self.console.print(f"  ✅ {model.model}")
                    self.console.print(f"     📊 {size_gb:.1f}GB | ⚙️ {params} | 📅 {model.modified_at.strftime('%d/%m %H:%M')}")
            else:
                self.console.print("  ℹ️  Nenhum modelo encontrado")
                
        except Exception as e:
            self.console.print(f"  ❌ Erro ao listar modelos: {e}")
    
    def _test_model(self):
        """Testa o modelo com uma pergunta simples"""
        self.console.print("\n🧪 [bold]Testando o modelo...[/bold]")
        
        test_prompts = [
            "Explique o que é Python em uma frase.",
            "Qual é a capital do Brasil?",
            "Como fazer um bolo simples?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            self.console.print(f"\n📝 Teste {i}: {prompt}")
            
            try:
                start_time = time.time()
                response = ollama.generate(
                    model=self.config.MODEL_NAME,
                    prompt=prompt
                )
                response_time = time.time() - start_time
                
                if hasattr(response, 'response'):
                    self.console.print(Panel(
                        response.response,
                        title=f"✅ Resposta ({response_time:.1f}s)",
                        border_style="green"
                    ))
                else:
                    self.console.print("❌ [red]Resposta inesperada do modelo[/red]")
                    
            except Exception as e:
                self.console.print(f"❌ [red]Erro no teste: {e}[/red]")
    
    def _show_system_status(self):
        """Mostra status do sistema"""
        try:
            models_response = ollama.list()
            model_count = len(models_response.models) if hasattr(models_response, 'models') else 0
            
            search_history = self.searcher.get_search_history()
            search_count = len(search_history)
            
            self.console.print(Panel(
                f"🤖 [bold]Status do Sistema[/bold]\n\n"
                f"📚 Modelos carregados: {model_count}\n"
                f"🔍 Pesquisas realizadas: {search_count}\n"
                f"⚙️  Modelo atual: {self.config.MODEL_NAME}\n"
                f"🕒 Hora do sistema: {datetime.now().strftime('%H:%M:%S')}",
                border_style="blue"
            ))
            
        except Exception as e:
            self.console.print(f"❌ [red]Erro ao verificar status: {e}[/red]")

def main():
    # Configuração
    config = Config()
    
    try:
        # Inicializar agente
        agent = DeepSeekAgent(config)
        
        # Iniciar chat
        agent.chat_loop()
        
    except Exception as e:
        console = Console()
        console.print(f"❌ Erro ao iniciar agente: {e}", style="bold red")
        console.print("\n💡 Execute 'ollama serve' em outro terminal e tente novamente.")

if __name__ == "__main__":
    main()