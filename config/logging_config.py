# Configuração centralizada de logs

import logging
import sys
from pathlib import Path

# Diretório para arquivos de log (cria se não existir)
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "nba_predictor.log"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("nba_predictor")
