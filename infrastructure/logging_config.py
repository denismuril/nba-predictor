"""
Logs Estruturados - JSON Logging para Monitoramento
====================================================
Logs em formato JSON estruturado compatível com ELK Stack,
Datadog, CloudWatch, e outras ferramentas de observabilidade.

Autor: NBA Predictor v22.0
"""

import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import os
import traceback


class StructuredJsonFormatter(logging.Formatter):
    """
    Formatador de logs em JSON estruturado.
    
    Campos padrão:
    - timestamp: ISO 8601 UTC
    - level: INFO, WARNING, ERROR, etc
    - logger: Nome do logger
    - message: Mensagem de log
    - module, function, line: Localização no código
    
    Campos extras contextuais são adicionados automaticamente.
    """
    
    def __init__(self, service_name: str = "nba-predictor", 
                 environment: str = None):
        super().__init__()
        self.service_name = service_name
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
    
    def format(self, record: logging.LogRecord) -> str:
        """Formata log record em JSON"""
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Adicionar campos extras do LogContext
        if hasattr(record, 'game_id'):
            log_entry['game_id'] = record.game_id
        if hasattr(record, 'team'):
            log_entry['team'] = record.team
        if hasattr(record, 'prediction_id'):
            log_entry['prediction_id'] = record.prediction_id
        if hasattr(record, 'latency_ms'):
            log_entry['latency_ms'] = record.latency_ms
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'bet_id'):
            log_entry['bet_id'] = record.bet_id
        if hasattr(record, 'flow_name'):
            log_entry['flow_name'] = record.flow_name
        if hasattr(record, 'task_name'):
            log_entry['task_name'] = record.task_name
        
        # Adicionar exception info se presente
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Adicionar campos extras genéricos
        if hasattr(record, 'extra_fields') and isinstance(record.extra_fields, dict):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Formatador colorido para console (desenvolvimento)"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Formato: [TIMESTAMP] [LEVEL] [LOGGER] MESSAGE
        timestamp = datetime.now().strftime('%H:%M:%S')
        prefix = f"{color}[{timestamp}] [{record.levelname:8}]{self.RESET}"
        
        return f"{prefix} [{record.name}] {record.getMessage()}"


def setup_structured_logging(
    level: str = "INFO",
    json_file: str = "logs/app.jsonl",
    console: bool = True,
    colored_console: bool = True
) -> logging.Logger:
    """
    Configura logging estruturado para a aplicação.
    
    Args:
        level: Nível de log (DEBUG, INFO, WARNING, ERROR)
        json_file: Caminho para arquivo de logs JSON
        console: Se True, também loga para console
        colored_console: Se True, usa cores no console
        
    Returns:
        Logger raiz configurado
    """
    from pathlib import Path
    
    # Criar diretório de logs
    Path(json_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remover handlers existentes
    root_logger.handlers.clear()
    
    # Handler JSON para arquivo
    json_handler = logging.FileHandler(json_file)
    json_handler.setFormatter(StructuredJsonFormatter())
    json_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(json_handler)
    
    # Handler para console
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        
        if colored_console and sys.stdout.isatty():
            console_handler.setFormatter(ConsoleFormatter())
        else:
            console_handler.setFormatter(StructuredJsonFormatter())
        
        console_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(console_handler)
    
    # Silenciar loggers muito verbosos
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    root_logger.info(f"Logging configurado: level={level}, json_file={json_file}")
    
    return root_logger


class LogContext:
    """
    Context manager para adicionar campos extras aos logs.
    
    Usage:
        with LogContext(game_id='LAL_BOS_2024-01-15', team='LAL'):
            logger.info("Processando jogo")  # Inclui game_id e team
    """
    
    _context: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        self.fields = kwargs
        self._old_factory = None
    
    def __enter__(self):
        # Salvar contexto anterior
        self._old_context = LogContext._context.copy()
        LogContext._context.update(self.fields)
        
        # Modificar record factory para adicionar campos
        self._old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            for key, value in LogContext._context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, *args):
        # Restaurar contexto e factory
        LogContext._context = self._old_context
        logging.setLogRecordFactory(self._old_factory)
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Obtém valor do contexto atual"""
        return cls._context.get(key, default)


def log_with_context(logger: logging.Logger, level: str, message: str, **extra):
    """
    Helper para logar com campos extras.
    
    Usage:
        log_with_context(logger, 'info', 'Aposta registrada', 
                         bet_id='123', stake=100.0)
    """
    record = logger.makeRecord(
        logger.name, 
        getattr(logging, level.upper()),
        '', 0, message, (), None
    )
    
    for key, value in extra.items():
        setattr(record, key, value)
    
    logger.handle(record)


# ============= HELPERS DE MÉTRICAS =============

class MetricsLogger:
    """
    Logger especializado para métricas de performance.
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger('metrics')
    
    def log_latency(self, operation: str, latency_ms: float, **extra):
        """Loga latência de uma operação"""
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO,
            '', 0, f"Latency: {operation}", (), None
        )
        record.latency_ms = latency_ms
        record.operation = operation
        for key, value in extra.items():
            setattr(record, key, value)
        self.logger.handle(record)
    
    def log_prediction(self, game_id: str, prob_home: float, 
                       confidence: str, latency_ms: float):
        """Loga uma previsão gerada"""
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO,
            '', 0, f"Prediction: {game_id}", (), None
        )
        record.game_id = game_id
        record.prob_home = prob_home
        record.confidence = confidence
        record.latency_ms = latency_ms
        self.logger.handle(record)
    
    def log_bet(self, bet_id: str, stake: float, odds: float, 
                ev_pct: float, kelly: float):
        """Loga uma aposta registrada"""
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO,
            '', 0, f"Bet: {bet_id}", (), None
        )
        record.bet_id = bet_id
        record.stake = stake
        record.odds = odds
        record.ev_pct = ev_pct
        record.kelly = kelly
        self.logger.handle(record)


# ============= CONFIGURAÇÃO AUTOMÁTICA =============

def get_logger(name: str) -> logging.Logger:
    """
    Obtém logger configurado.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Mensagem de log")
    """
    return logging.getLogger(name)


# Auto-configurar se executado diretamente
if __name__ == "__main__":
    setup_structured_logging(level="DEBUG")
    
    logger = get_logger("test")
    
    # Teste básico
    logger.info("Teste de log básico")
    logger.warning("Teste de warning")
    logger.error("Teste de erro")
    
    # Teste com contexto
    with LogContext(game_id="LAL_BOS_2024-01-15", team="LAL"):
        logger.info("Log com contexto de jogo")
    
    # Teste de métricas
    metrics = MetricsLogger()
    metrics.log_latency("prediction", 125.5, model="ensemble_v22")
    
    print("\n✅ Logs estruturados funcionando!")
