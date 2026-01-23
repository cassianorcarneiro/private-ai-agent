# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# AEGIS-MIND
# CASSIANO RIBEIRO CARNEIRO
# V1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

# Import custom modules

from .web_search import WebSearcher
from .monitoring import SearchMonitor

# Define o que será importado com 'from utils import *'
__all__ = ['WebSearcher', 'SearchMonitor']

# Versão do pacote
__version__ = '1.0.0'

# Código de inicialização (opcional)
print(f"📦 Inicializando pacote utils versão {__version__}")