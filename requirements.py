# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# AEGIS-MIND
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# Importando frameworks

import importlib
import subprocess
import sys
import os

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Intalar frameworks necessários
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def instalar_libraries():

    def corrigir_pandas_ta():
        python_exe = sys.executable
        venv_path = os.path.dirname(os.path.dirname(python_exe))
        site_packages = os.path.join(venv_path, "Lib", "site-packages")
        pandas_ta_path = os.path.join(site_packages, "pandas_ta")

        if not os.path.isdir(pandas_ta_path):
            print("Pasta pandas_ta não encontrada em:", pandas_ta_path)
            return

        corrigidos = 0
        for root, _, files in os.walk(pandas_ta_path):
            for f in files:
                if f.endswith(".py"):
                    arquivo = os.path.join(root, f)
                    with open(arquivo, "r", encoding="utf-8") as file:
                        conteudo = file.read()
                    if "from numpy import NaN" in conteudo:
                        novo_conteudo = conteudo.replace(
                            "from numpy import NaN",
                            "from numpy import nan"
                        )
                        with open(arquivo, "w", encoding="utf-8") as file:
                            file.write(novo_conteudo)
                        print(f"Corrigido em: {arquivo}")
                        corrigidos += 1

        if corrigidos == 0:
            print("Nenhuma ocorrência encontrada para corrigir.")
        else:
            print(f"Total de arquivos corrigidos: {corrigidos}")

            
    def instalar_se_necessario(pacote, nome_pip=None):
        
        try:
            importlib.import_module(pacote)
        except ImportError:
            print(f"Instalando o pacote '{nome_pip or pacote}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", nome_pip or pacote])

    # Lista de pacotes a instalar: (nome do import, nome no pip se for diferente)
    
    bibliotecas = [
        ('langgraph',),
        ('ddgs',),
        ('langchain',),
        ('langchain-ollama',),
        ('pydantic',),
    ]

    for args in bibliotecas:
        instalar_se_necessario(*args)

    corrigir_pandas_ta()

if __name__ == "__main__":
    instalar_libraries()