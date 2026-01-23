import requests
import subprocess
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def check_ollama_connection():
    """Verifica a conexão com o Ollama de forma robusta"""
    console.print(Panel("🔍 [bold blue]Diagnóstico do Ollama[/bold blue]", border_style="blue"))
    
    # Verificar se o serviço está rodando
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        
        if response.status_code == 200:
            console.print("✅ [green]Ollama está respondendo na porta 11434[/green]")
            
            # Tentar parsear a resposta
            try:
                data = response.json()
                console.print(f"[dim]Estrutura da resposta: {list(data.keys())}[/dim]")
                
                if 'models' in data:
                    models = data['models']
                    console.print(f"📚 [green]Encontrados {len(models)} modelos:[/green]")
                    
                    table = Table(show_header=True, header_style="bold green")
                    table.add_column("Nome", style="cyan")
                    table.add_column("Tamanho", style="magenta")
                    table.add_column("Modificado", style="yellow")
                    
                    for model in models:
                        name = model.get('name', 'N/A')
                        size = model.get('size', 0)
                        size_gb = f"{size/1024/1024/1024:.1f}GB" if size > 0 else "N/A"
                        modified = model.get('modified_at', 'N/A')[:10] if model.get('modified_at') else 'N/A'
                        
                        table.add_row(name, size_gb, modified)
                    
                    console.print(table)
                else:
                    console.print("❌ [yellow]Resposta não contém 'models'[/yellow]")
                    console.print(f"[dim]Resposta completa: {data}[/dim]")
                    
            except json.JSONDecodeError as e:
                console.print(f"❌ [red]Erro ao decodificar JSON: {e}[/red]")
                
        else:
            console.print(f"❌ [red]Resposta HTTP {response.status_code}[/red]")
            
    except requests.exceptions.ConnectionError:
        console.print("❌ [red]Não foi possível conectar ao Ollama[/red]")
        console.print("\n🔧 [yellow]Soluções:[/yellow]")
        console.print("1. Inicie o Ollama: [green]ollama serve[/green]")
        console.print("2. Verifique se está instalado: [green]ollama --version[/green]")
        console.print("3. Verifique a porta: [green]netstat -tulpn | grep 11434[/green]")
        
    except requests.exceptions.Timeout:
        console.print("❌ [red]Timeout ao conectar com Ollama[/red]")
    except Exception as e:
        console.print(f"❌ [red]Erro inesperado: {e}[/red]")

def check_ollama_cli():
    """Verifica o CLI do Ollama"""
    console.print(Panel("💻 [bold blue]Verificando CLI do Ollama[/bold blue]", border_style="blue"))
    
    try:
        # Verificar versão
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"✅ [green]Ollama CLI: {result.stdout.strip()}[/green]")
        else:
            console.print("❌ [red]Ollama CLI não funciona[/red]")
            
        # Verificar modelos via CLI
        result = subprocess.run(['ollama', 'list', '--format', 'json'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                console.print("✅ [green]CLI list funciona corretamente[/green]")
                console.print(f"[dim]Resposta CLI: {list(data.keys())}[/dim]")
            except json.JSONDecodeError:
                console.print("❌ [red]Resposta do CLI não é JSON válido[/red]")
        else:
            console.print(f"❌ [red]CLI list falhou: {result.stderr}[/red]")
            
    except FileNotFoundError:
        console.print("❌ [red]Ollama não está instalado ou não está no PATH[/red]")
    except subprocess.TimeoutExpired:
        console.print("❌ [red]Timeout no comando Ollama[/red]")

def test_model_generation():
    """Testa a geração de texto com um modelo"""
    console.print(Panel("🧪 [bold blue]Teste de Geração de Texto[/bold blue]", border_style="blue"))
    
    try:
        import ollama
        
        # Listar modelos primeiro
        models = ollama.list()
        console.print(f"[dim]Estrutura ollama.list(): {list(models.keys())}[/dim]")
        
        if 'models' in models and models['models']:
            model_name = models['models'][0]['name']
            console.print(f"🔧 [yellow]Testando com modelo: {model_name}[/yellow]")
            
            # Testar geração
            response = ollama.generate(model=model_name, prompt="Olá! Responda brevemente: como você está?")
            
            console.print(f"[dim]Estrutura da resposta generate: {list(response.keys())}[/dim]")
            
            if 'response' in response:
                console.print("✅ [green]Geração funcionando![/green]")
                console.print(f"📝 Resposta: {response['response'][:100]}...")
            else:
                console.print(f"❌ [red]Resposta não contém 'response': {response}[/red]")
        else:
            console.print("❌ [yellow]Nenhum modelo disponível para teste[/yellow]")
            
    except Exception as e:
        console.print(f"❌ [red]Erro no teste de geração: {e}[/red]")

def main():
    """Função principal de diagnóstico"""
    console.print(Panel("🩺 [bold magenta]Diagnóstico Completo do Sistema Ollama[/bold magenta]", 
                      border_style="magenta"))
    
    check_ollama_connection()
    print()
    check_ollama_cli()
    print()
    test_model_generation()
    
    console.print(Panel("💡 [bold green]Próximos Passos[/bold green]", border_style="green"))
    console.print("1. Se houver erros, inicie o Ollama: [green]ollama serve[/green]")
    console.print("2. Instale um modelo: [green]ollama pull deepseek-coder[/green]")
    console.print("3. Execute o agente novamente: [green]python main.py[/green]")

if __name__ == "__main__":
    main()