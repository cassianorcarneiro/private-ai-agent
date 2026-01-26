# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PRIVATE AI AGENT
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Importing frameworks

import importlib
import subprocess
import sys
import os

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Install required frameworks
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def install_libraries():

    def fix_pandas_ta():
        python_exe = sys.executable
        venv_path = os.path.dirname(os.path.dirname(python_exe))
        site_packages = os.path.join(venv_path, "Lib", "site-packages")
        pandas_ta_path = os.path.join(site_packages, "pandas_ta")

        if not os.path.isdir(pandas_ta_path):
            print("pandas_ta folder not found at:", pandas_ta_path)
            return

        fixed = 0
        for root, _, files in os.walk(pandas_ta_path):
            for f in files:
                if f.endswith(".py"):
                    file_path = os.path.join(root, f)
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                    if "from numpy import NaN" in content:
                        new_content = content.replace(
                            "from numpy import NaN",
                            "from numpy import nan"
                        )
                        with open(file_path, "w", encoding="utf-8") as file:
                            file.write(new_content)
                        print(f"Fixed in: {file_path}")
                        fixed += 1

        if fixed == 0:
            print("No occurrences found to fix.")
        else:
            print(f"Total files fixed: {fixed}")

            
    def install_if_needed(package, pip_name=None):
        
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"Installing package '{pip_name or package}'...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name or package]
            )

    # List of packages to install: (import name, pip name if different)
    
    libraries = [
        ('langgraph',),
        ('ddgs',),
        ('langchain',),
        ('langchain-ollama',),
        ('pydantic',),
    ]

    for args in libraries:
        install_if_needed(*args)

    fix_pandas_ta()

if __name__ == "__main__":
    install_libraries()
