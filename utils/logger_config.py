"""
Configuração centralizada de logging para o NBA Predictor.
Fornece logging estruturado e configurável.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configura logging para o aplicativo.
    
    Args:
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Caminho para arquivo de log (opcional).
        format_string: String de formatação customizada (opcional).
        
    Returns:
        Logger configurado.
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    
    # Converter string de nível para constante
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configurar handlers
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            handlers.append(file_handler)
        except PermissionError:
            # Fallback: Tentar criar arquivo com sufixo do usuário ou ignorar
            try:
                alt_log_file = log_file.parent / f"{log_file.stem}_user{log_file.suffix}"
                file_handler = logging.FileHandler(alt_log_file, encoding='utf-8')
                handlers.append(file_handler)
                print(f"Warning: Permissao negada em {log_file}. Usando {alt_log_file}")
            except Exception:
                print(f"Error: Falha ao configurar log em arquivo. Apenas console ativo.")
        except Exception as e:
            print(f"Error: Erro ao configurar log: {e}")
    
    # Configurar formato
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    for handler in handlers:
        handler.setFormatter(formatter)
    
    # Configurar root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True  # Sobrescrever configuração anterior
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configurado (nivel: {level})")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Obtém um logger com o nome especificado.
    
    Args:
        name: Nome do logger (geralmente __name__ do módulo).
        
    Returns:
        Logger configurado.
    """
    return logging.getLogger(name)

# Configuração padrão ao importar
_default_log_file = Path(__file__).parent.parent / "logs" / f"nba_predictor_{datetime.now().strftime('%Y%m%d')}.log"
setup_logging(level="INFO", log_file=_default_log_file)

