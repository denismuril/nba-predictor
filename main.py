#!/usr/bin/env python3
"""
NBA Predictor - Entry Point
Sistema profissional de análise quantitativa e previsão de jogos da NBA.
"""
import sys
import os
import logging
from pathlib import Path

# Adicionar diretório atual ao PYTHONPATH para garantir que imports funcionem
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Tentar importar logger config
try:
    from utils.logger_config import setup_logging
except ImportError:
    setup_logging = None

def load_env() -> bool:
    """
    Carrega variáveis do arquivo .env.
    Tenta usar python-dotenv, fallback para leitura manual.
    
    Returns:
        bool: True se o arquivo foi carregado com sucesso, False caso contrário.
    """
    env_path = Path(__file__).parent / ".env"
    
    # 1. Tentar python-dotenv
    try:
        from dotenv import load_dotenv
        if load_dotenv(dotenv_path=env_path, override=False):
            logging.info(f"✅ Arquivo .env carregado via python-dotenv")
            return True
    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"⚠️  Erro ao usar python-dotenv: {e}")

    # 2. Fallback manual
    if env_path.exists():
        try:
            logging.info(f"📂 Carregando .env manualmente de {env_path}")
            with open(env_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        try:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip().strip("'").strip('"')
                            if key and key not in os.environ: # Não sobrescrever se já existe
                                os.environ[key] = value
                        except ValueError:
                            pass
            return True
        except Exception as e:
            logging.error(f"❌ Erro ao ler .env manualmente: {e}")
            return False
    
    logging.warning(f"⚠️  Arquivo .env não encontrado em {env_path}")
    return False

if __name__ == "__main__":
    # 1. Configurar logging inicial (antes de tudo)
    if setup_logging:
        setup_logging(level="INFO")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )
    
    # 2. Carregar ambiente
    load_env()
    
    # 3. Importar e executar CLI
    try:
        from interfaces.cli import main
        main()
    except ImportError as e:
        logging.critical(f"❌ Erro fatal ao importar interfaces.cli: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"❌ Erro não tratado na execução: {e}", exc_info=True)
        sys.exit(1)
