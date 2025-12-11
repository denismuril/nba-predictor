# Configurações do projeto

import os
from pathlib import Path

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Carregar variáveis de ambiente (ex.: chaves de API)
# Caso .env exista, ele será lido manualmente na aplicação principal

# Exemplo de constantes
DEFAULT_ITERATIONS = 300_000
HCA = 3.0  # Home Court Advantage
