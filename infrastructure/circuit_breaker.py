"""
Circuit Breaker - Padrão de Proteção do Sistema
================================================
Implementa o padrão Circuit Breaker para proteger o sistema.
Se o modelo errar muito ou odds vierem zeradas, para de enviar tips.

Autor: NBA Predictor v22.0
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any, Optional, Dict
import asyncio
import logging
import os

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados possíveis do Circuit Breaker"""
    CLOSED = "closed"       # Normal, permitindo requests
    OPEN = "open"           # Falhou muito, bloqueado
    HALF_OPEN = "half_open" # Testando se voltou ao normal


class CircuitOpenError(Exception):
    """Exceção lançada quando o circuit está aberto"""
    
    def __init__(self, circuit_name: str, message: str = None):
        self.circuit_name = circuit_name
        self.message = message or f"Circuit '{circuit_name}' está OPEN"
        super().__init__(self.message)


class CircuitBreaker:
    """
    Implementação do padrão Circuit Breaker.
    
    Estados:
    - CLOSED: Funcionamento normal, requests são permitidos
    - OPEN: Muitas falhas, requests são bloqueados
    - HALF_OPEN: Testando recuperação com requests limitados
    
    Parâmetros:
    - failure_threshold: Número de falhas para abrir o circuit
    - recovery_timeout: Segundos para tentar recuperação
    - success_threshold: Sucessos necessários para fechar novamente
    """
    
    def __init__(
        self, 
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 3,
        notify_on_open: bool = True
    ):
        """
        Inicializa o Circuit Breaker.
        
        Args:
            name: Nome identificador do circuit
            failure_threshold: Falhas consecutivas para abrir
            recovery_timeout: Segundos até tentar reabrir
            success_threshold: Sucessos para fechar em HALF_OPEN
            notify_on_open: Se deve enviar alerta quando abrir
        """
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.notify_on_open = notify_on_open
        
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()
        
        # Estatísticas
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.times_opened = 0
    
    @property
    def is_closed(self) -> bool:
        """Verifica se circuit está fechado (normal)"""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Verifica se circuit está aberto (bloqueado)"""
        return self.state == CircuitState.OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executa função com proteção do circuit breaker.
        
        Args:
            func: Função a executar (pode ser async ou sync)
            *args, **kwargs: Argumentos para a função
            
        Returns:
            Resultado da função
            
        Raises:
            CircuitOpenError: Se circuit está aberto
        """
        self.total_calls += 1
        
        # Verificar estado
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitOpenError(self.name)
        
        try:
            # Executar função
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Verifica se deve tentar resetar o circuit"""
        if self.last_failure_time is None:
            return True
        
        elapsed = datetime.now() - self.last_failure_time
        return elapsed > timedelta(seconds=self.recovery_timeout)
    
    def _transition_to_half_open(self):
        """Transiciona para estado HALF_OPEN"""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.last_state_change = datetime.now()
        logger.info(f"🔄 Circuit '{self.name}': OPEN → HALF_OPEN")
    
    async def _on_success(self):
        """Registra sucesso na operação"""
        self.last_success_time = datetime.now()
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.last_state_change = datetime.now()
                logger.info(f"✅ Circuit '{self.name}': HALF_OPEN → CLOSED")
        else:
            # Em CLOSED, reseta contador de falhas
            self.failure_count = 0
    
    async def _on_failure(self, error: Exception):
        """Registra falha na operação"""
        self.last_failure_time = datetime.now()
        self.failure_count += 1
        self.success_count = 0
        self.total_failures += 1
        
        logger.warning(f"⚠️ Circuit '{self.name}': Falha {self.failure_count}/{self.failure_threshold}")
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.times_opened += 1
            self.last_state_change = datetime.now()
            logger.warning(f"🔴 Circuit '{self.name}': → OPEN após {self.failure_count} falhas")
            
            if self.notify_on_open:
                await self._send_alert(error)
    
    async def _send_alert(self, error: Exception):
        """Envia alerta quando circuit abre"""
        try:
            from telegram import Bot
            
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            admin_id = os.getenv('TELEGRAM_ADMIN_ID')
            
            if not token or not admin_id:
                logger.warning("Telegram não configurado para alertas")
                return
            
            bot = Bot(token)
            msg = (
                f"🚨 **CIRCUIT BREAKER ATIVADO**\n\n"
                f"📛 Componente: `{self.name}`\n"
                f"❌ Falhas: {self.failure_count}\n"
                f"📊 Estado: OPEN\n"
                f"⏱️ Timeout: {self.recovery_timeout}s\n"
                f"🔍 Erro: {str(error)[:100]}\n\n"
                f"⚠️ Tips automáticas **PAUSADAS**"
            )
            await bot.send_message(admin_id, msg, parse_mode='Markdown')
            logger.info(f"📱 Alerta enviado para admin")
            
        except Exception as e:
            logger.error(f"Erro enviando alerta: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do circuit"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_calls': self.total_calls,
            'total_failures': self.total_failures,
            'total_successes': self.total_successes,
            'times_opened': self.times_opened,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_success': self.last_success_time.isoformat() if self.last_success_time else None,
            'last_state_change': self.last_state_change.isoformat()
        }
    
    def reset(self):
        """Reseta o circuit para estado inicial"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = datetime.now()
        logger.info(f"🔄 Circuit '{self.name}' resetado")


# ============= CIRCUITS PRÉ-DEFINIDOS =============

class ModelCircuitBreaker(CircuitBreaker):
    """
    Circuit Breaker especializado para modelo de ML.
    
    Abre o circuit quando:
    - Odds vêm zeradas repetidamente
    - Probabilidades fora do range válido
    - Erros de inferência
    """
    
    def __init__(self):
        super().__init__(
            name="ml_model",
            failure_threshold=5,
            recovery_timeout=300,  # 5 minutos
            success_threshold=3
        )
    
    async def validate_prediction(self, prediction: Dict[str, Any]) -> bool:
        """
        Valida se uma previsão é saudável.
        
        Registra falha se:
        - Odds zeradas
        - Probabilidade fora do range [0.01, 0.99]
        - Campos obrigatórios ausentes
        
        Returns:
            True se previsão válida, False caso contrário
        """
        try:
            # Verificar odds zeradas
            odds_home = prediction.get('odds_home', prediction.get('Odd Casa', 0))
            if odds_home == 0:
                await self._on_failure(ValueError("Odds zeradas"))
                return False
            
            # Verificar probabilidade
            prob = prediction.get('prob_home', prediction.get('Prob Casa %', 0))
            if prob < 0.01 or prob > 0.99:
                await self._on_failure(ValueError(f"Probabilidade fora do range: {prob}"))
                return False
            
            # Verificar campos obrigatórios
            required = ['Casa', 'Visitante', 'Data']
            for field in required:
                if field not in prediction:
                    await self._on_failure(ValueError(f"Campo obrigatório ausente: {field}"))
                    return False
            
            await self._on_success()
            return True
            
        except Exception as e:
            await self._on_failure(e)
            return False


class OddsCircuitBreaker(CircuitBreaker):
    """
    Circuit Breaker para scrapers de odds.
    
    Abre quando fontes de odds falham repetidamente.
    """
    
    def __init__(self):
        super().__init__(
            name="odds_scraper",
            failure_threshold=3,
            recovery_timeout=180,  # 3 minutos
            success_threshold=2
        )


class DataCollectionCircuitBreaker(CircuitBreaker):
    """
    Circuit Breaker para coleta de dados.
    
    Protege contra falhas em APIs externas (nba_api, etc).
    """
    
    def __init__(self):
        super().__init__(
            name="data_collection",
            failure_threshold=5,
            recovery_timeout=600,  # 10 minutos
            success_threshold=3
        )


# ============= REGISTRY DE CIRCUITS =============

class CircuitBreakerRegistry:
    """
    Registro central de todos os circuit breakers.
    Permite monitoramento centralizado.
    """
    
    _circuits: Dict[str, CircuitBreaker] = {}
    
    @classmethod
    def register(cls, circuit: CircuitBreaker):
        """Registra um circuit breaker"""
        cls._circuits[circuit.name] = circuit
        logger.debug(f"Circuit '{circuit.name}' registrado")
    
    @classmethod
    def get(cls, name: str) -> Optional[CircuitBreaker]:
        """Obtém circuit pelo nome"""
        return cls._circuits.get(name)
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Retorna estatísticas de todos os circuits"""
        return {name: cb.get_stats() for name, cb in cls._circuits.items()}
    
    @classmethod
    def reset_all(cls):
        """Reseta todos os circuits"""
        for circuit in cls._circuits.values():
            circuit.reset()
    
    @classmethod
    def is_any_open(cls) -> bool:
        """Verifica se algum circuit está aberto"""
        return any(cb.is_open for cb in cls._circuits.values())


# Registrar circuits padrão
CircuitBreakerRegistry.register(ModelCircuitBreaker())
CircuitBreakerRegistry.register(OddsCircuitBreaker())
CircuitBreakerRegistry.register(DataCollectionCircuitBreaker())
