"""
Connection Pool para SQLite com gerenciamento robusto de recursos.

Resolve problemas de database locking através de:
1. Pool de conexões reutilizáveis (evita overhead de criar/destruir connections)
2. Timeouts configuráveis
3. Resource cleanup automático
4. Thread-safe operations
"""
import sqlite3
import threading
import logging
from queue import Queue, Empty, Full
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Pool de conexões SQLite para reduzir contenção e melhorar performance.
    
    Features:
    - Conexões pré-configuradas com WAL mode
    - Timeouts automáticos para evitar deadlocks
    - Cleanup automático de conexões idle
    - Thread-safe via Queue
    """
    
    def __init__(
        self,
        db_path: str,
        pool_size: int = 5,
        timeout: int = 30,
        busy_timeout: int = 30000  # 30 segundos
    ):
        """
        Inicializa connection pool.
        
        Args:
            db_path: Caminho para o arquivo SQLite
            pool_size: Número de conexões no pool (default: 5)
            timeout: Timeout para get_connection em segundos (default: 30)
            busy_timeout: PRAGMA busy_timeout em ms (default: 30000)
        """
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self.timeout = timeout
        self.busy_timeout = busy_timeout
        
        # Queue thread-safe para gerenciar conexões
        self.pool = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self._closed = False
        
        # Inicializar pool de conexões
        self._initialize_pool()
        
        logger.info(
            f"✅ Connection Pool inicializado: {pool_size} conexões, "
            f"timeout={timeout}s, busy_timeout={busy_timeout}ms"
        )
    
    def _initialize_pool(self):
        """Cria conexões iniciais para o pool."""
        for i in range(self.pool_size):
            try:
                conn = self._create_connection()
                self.pool.put(conn, block=False)
                logger.debug(f"  Conexão {i+1}/{self.pool_size} criada")
            except Full:
                logger.error(f"❌ Erro ao preencher pool: Queue cheia")
                break
            except Exception as e:
                logger.error(f"❌ Erro ao criar conexão {i+1}: {e}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """
        Cria e configura uma nova conexão SQLite.
        
        Returns:
            Conexão configurada com WAL mode e otimizações
        """
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,  # Permitir uso em múltiplas threads
            isolation_level=None  # Autocommit mode
        )
        
        # Configurar PRAGMAs para performance e concorrência
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')  # Balance entre segurança e performance
        conn.execute(f'PRAGMA busy_timeout={self.busy_timeout}')
        conn.execute('PRAGMA temp_store=MEMORY')  # Tabelas temp em memória
        conn.execute('PRAGMA cache_size=-64000')  # 64MB cache (negative = KB)
        
        return conn
    
    def get_connection(self, timeout: Optional[float] = None) -> sqlite3.Connection:
        """
        Obtém uma conexão do pool.
        
        Args:
            timeout: Timeout customizado em segundos (default: usa self.timeout)
            
        Returns:
            Conexão SQLite do pool
            
        Raises:
            Empty: Se pool está vazio e timeout expirou
            RuntimeError: Se pool foi fechado
        """
        if self._closed:
            raise RuntimeError("Connection pool foi fechado")
        
        timeout = timeout if timeout is not None else self.timeout
        
        try:
            conn = self.pool.get(timeout=timeout)
            
            # Verificar se conexão ainda está válida
            try:
                conn.execute('SELECT 1')
            except sqlite3.Error:
                # Conexão inválida, criar nova
                logger.warning("⚠️ Conexão inválida detectada, recriando...")
                conn = self._create_connection()
            
            return conn
            
        except Empty:
            raise RuntimeError(
                f"Connection pool exhausted! Todas as {self.pool_size} conexões estão em uso. "
                f"Timeout after {timeout}s."
            )
    
    def return_connection(self, conn: sqlite3.Connection):
        """
        Retorna uma conexão ao pool.
        
        Args:
            conn: Conexão a retornar
        """
        if self._closed:
            # Pool fechado, apenas close a conexão
            try:
                conn.close()
            except:
                pass
            return
        
        try:
            # Limpar transações pendentes
            if conn.in_transaction:
                conn.rollback()
            
            # Retornar ao pool
            self.pool.put(conn, block=False)
        except Full:
            # Pool cheio (não deveria acontecer), close conexão
            logger.warning("⚠️ Pool cheio ao retornar conexão, closing...")
            try:
                conn.close()
            except:
                pass
    
    @contextmanager
    def get_connection_context(self, timeout: Optional[float] = None):
        """
        Context manager para usar conexões com cleanup automático.
        
        Usage:
            with pool.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute(...)
                conn.commit()
        
        Args:
            timeout: Timeout customizado em segundos
            
        Yields:
            Conexão SQLite (retornada automaticamente ao pool no final)
        """
        conn = self.get_connection(timeout=timeout)
        try:
            yield conn
        finally:
            self.return_connection(conn)
    
    def close(self):
        """
        Fecha todas as conexões do pool.
        
        ATENÇÃO: Chamar apenas no shutdown da aplicação!
        """
        with self.lock:
            if self._closed:
                return
            
            self._closed = True
            closed_count = 0
            
            # Drenar queue e fechar todas conexões
            while not self.pool.empty():
                try:
                    conn = self.pool.get(block=False)
                    conn.close()
                    closed_count += 1
                except Empty:
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao fechar conexão: {e}")
            
            logger.info(f"✅ Connection Pool fechado. {closed_count} conexões encerradas.")
    
    def __del__(self):
        """Cleanup automático no garbage collection."""
        self.close()
    
    @property
    def available_connections(self) -> int:
        """Retorna número de conexões disponíveis no pool."""
        return self.pool.qsize()
    
    @property
    def in_use_connections(self) -> int:
        """Retorna número de conexões em uso."""
        return self.pool_size - self.pool.qsize()
    
    def get_stats(self) -> dict:
        """
        Retorna estatísticas do pool.
        
        Returns:
            Dict com métricas do pool
        """
        return {
            'pool_size': self.pool_size,
            'available': self.available_connections,
            'in_use': self.in_use_connections,
            'utilization_pct': (self.in_use_connections / self.pool_size) * 100,
            'closed': self._closed
        }


# Singleton global (opcional, mas recomendado)
_global_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_connection_pool(
    db_path: Optional[str] = None,
    pool_size: int = 5,
    recreate: bool = False
) -> ConnectionPool:
    """
    Obtém o pool de conexões global (Singleton pattern).
    
    Args:
        db_path: Caminho para o database (obrigatório na primeira chamada)
        pool_size: Tamanho do pool (default: 5)
        recreate: Se True, recria o pool mesmo se já existir
        
    Returns:
        Instância global de ConnectionPool
    """
    global _global_pool
    
    with _pool_lock:
        if _global_pool is None or recreate:
            if db_path is None:
                raise ValueError("db_path é obrigatório para criar o pool pela primeira vez")
            
            if _global_pool is not None:
                _global_pool.close()
            
            _global_pool = ConnectionPool(db_path, pool_size=pool_size)
        
        return _global_pool
