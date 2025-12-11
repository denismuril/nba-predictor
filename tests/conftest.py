# tests/conftest.py
# Configuração pytest para adicionar projeto ao path

import sys
import os

# Adicionar diretório raiz ao PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
