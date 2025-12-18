"""
Data Integrity Logger - Sistema de logging dedicado para validação de dados de odds.

Este módulo configura um logger separado especificamente para rastrear problemas
de integridade de dados, como:
- Odds fora do range válido (1.01-50.0)
- Scrapers que retornam dados vazios
- Tentativas de injeção de valores fixos (1.90)
- Falhas de normalização de nomes

v26.2: Logging dedicado para audit trail de integridade de dados.
"""

import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Configuração do logger de integridade de dados
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

INTEGRITY_LOG_FILE = LOG_DIR / "data_integrity.log"

# Formatter detalhado para audit trail
integrity_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Handler com rotação para evitar arquivos gigantes
integrity_handler = RotatingFileHandler(
    INTEGRITY_LOG_FILE,
    maxBytes=10_000_000,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
integrity_handler.setFormatter(integrity_formatter)
integrity_handler.setLevel(logging.WARNING)  # Apenas warnings e errors

# Logger dedicado
integrity_logger = logging.getLogger('nba_predictor.data_integrity')
integrity_logger.addHandler(integrity_handler)
integrity_logger.setLevel(logging.WARNING)
integrity_logger.propagate = False  # Não propaga para root logger


def log_invalid_odds(source: str, game: str, odds_value: float, reason: str):
    """
    Log odds inválidas para audit trail.
    
    Args:
        source: Nome do scraper/provider
        game: Identificação do jogo (ex: "Lakers vs Celtics")
        odds_value: Valor da odd que foi rejeitada
        reason: Razão da rejeição
    """
    integrity_logger.warning(
        f"INVALID_ODDS - Source: {source} | Game: {game} | "
        f"Value: {odds_value} | Reason: {reason}"
    )


def log_missing_data(source: str, game: str, url: str = None):
    """
    Log quando scraper não encontra dados reais.
    
    Args:
        source: Nome do scraper/provider
        game: Identificação do jogo
        url: URL que foi scrapeada (opcional)
    """
    url_info = f" | URL: {url}" if url else ""
    integrity_logger.warning(
        f"MISSING_DATA - Source: {source} | Game: {game}{url_info}"
    )


def log_fixed_value_attempt(source: str, value: float, location: str):
    """
    Log tentativas de injeção de valores fixos (1.90).
    
    Args:
        source: Nome do scraper/provider
        value: Valor fixo que foi detectado
        location: Localização no código (arquivo:linha)
    """
    integrity_logger.error(
        f"FIXED_VALUE_DETECTED - Source: {source} | Value: {value} | "
        f"Location: {location} | ACTION: Value rejected"
    )


def log_normalization_failure(source: str, raw_name: str, context: str = None):
    """
    Log falhas de normalização de nomes (times ou jogadores).
    
    Args:
        source: Nome do scraper/provider
        raw_name: Nome bruto que não foi normalizado
        context: Contexto adicional (opcional)
    """
    context_info = f" | Context: {context}" if context else ""
    integrity_logger.warning(
        f"NORMALIZATION_FAILED - Source: {source} | Raw: '{raw_name}'{context_info}"
    )


def validate_odds_range(odds_value: float, source: str, game: str = "unknown") -> bool:
    """
    Valida se odds está no range válido (1.01 - 50.0).
    
    Args:
        odds_value: Valor da odd a validar
        source: Provider/scraper que forneceu a odd
        game: Identificação do jogo
        
    Returns:
        True se válido, False se inválido (e loga)
    """
    MIN_ODDS = 1.01
    MAX_ODDS = 50.0
    
    if odds_value < MIN_ODDS:
        log_invalid_odds(source, game, odds_value, f"Below minimum ({MIN_ODDS})")
        return False
    
    if odds_value > MAX_ODDS:
        log_invalid_odds(source, game, odds_value, f"Above maximum ({MAX_ODDS})")
        return False
    
    # Detecta tentativa de usar valor fixo 1.90
    if abs(odds_value - 1.90) < 0.001:
        log_fixed_value_attempt(source, odds_value, f"Game: {game}")
        return False
    
    return True


# Exemplo de uso
if __name__ == "__main__":
    # Teste do logger
    log_invalid_odds("test_scraper", "Lakers vs Celtics", 0.50, "Below minimum")
    log_missing_data("test_scraper", "Heat vs Bulls", "https://example.com")
    log_fixed_value_attempt("test_scraper", 1.90, "test.py:123")
    log_normalization_failure("test_scraper", "LBJ", "Player name too short")
    
    print(f"Integrity log written to: {INTEGRITY_LOG_FILE}")
