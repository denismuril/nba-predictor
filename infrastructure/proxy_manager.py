"""
Proxy Manager - Gerenciador de proxies rotativos para scrapers.

Este módulo fornece infraestrutura de resiliência para os scrapers:
- Rotação automática de proxies
- Detecção e marcação de proxies "burned" (403/429)
- Health check de proxies
- Integração com stealth_browser.py

Configuração via .env:
    PROXY_LIST=http://user:pass@proxy1:8080,http://user:pass@proxy2:8080
    
Ou via lista estática no código.

v27.0: Implementação inicial para arquitetura God Mode.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Set
from pathlib import Path

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProxyStatus(Enum):
    """Status possíveis de um proxy."""
    AVAILABLE = "available"      # Pronto para uso
    IN_USE = "in_use"           # Atualmente em uso
    COOLING_DOWN = "cooling_down"  # Em cooldown temporário
    BURNED = "burned"            # Banido permanentemente


@dataclass
class ProxyInfo:
    """
    Informações detalhadas de um proxy.
    
    Atributos:
        url: URL completa do proxy (http://user:pass@host:port)
        status: Status atual
        last_used: Timestamp do último uso
        fail_count: Contagem de falhas consecutivas
        success_count: Contagem de sucessos
        cooldown_until: Até quando está em cooldown
        burn_reason: Motivo do ban (se burned)
    """
    url: str
    status: ProxyStatus = ProxyStatus.AVAILABLE
    last_used: Optional[datetime] = None
    fail_count: int = 0
    success_count: int = 0
    cooldown_until: Optional[datetime] = None
    burn_reason: Optional[str] = None
    
    def is_available(self) -> bool:
        """Verifica se proxy está disponível para uso."""
        if self.status == ProxyStatus.BURNED:
            return False
        if self.status == ProxyStatus.COOLING_DOWN:
            if self.cooldown_until and datetime.now() > self.cooldown_until:
                self.status = ProxyStatus.AVAILABLE
                return True
            return False
        return self.status == ProxyStatus.AVAILABLE


@dataclass 
class ProxyManagerConfig:
    """
    Configuração do gerenciador de proxies.
    
    Atributos:
        max_fails_before_burn: Falhas consecutivas antes de marcar como burned
        cooldown_seconds: Segundos de cooldown após falha
        health_check_timeout: Timeout para health check em segundos
        min_success_rate: Taxa mínima de sucesso para manter proxy ativo
    """
    max_fails_before_burn: int = 3
    cooldown_seconds: int = 60
    health_check_timeout: float = 10.0
    min_success_rate: float = 0.3


class ProxyManager:
    """
    Gerenciador de proxies rotativos.
    
    Fornece proxies válidos para scrapers, com rotação automática
    e detecção de proxies bloqueados.
    
    Exemplo de uso:
        >>> pm = ProxyManager()
        >>> proxy = pm.get_proxy()
        >>> # Usar proxy...
        >>> if success:
        ...     pm.mark_success(proxy)
        ... else:
        ...     pm.mark_failed(proxy, "403 Forbidden")
    
    Integração com stealth_browser:
        >>> async with create_stealth_browser(use_proxy_manager=True) as (browser, ctx, page):
        ...     # Proxy é selecionado automaticamente
    """
    
    # Proxies padrão (gratuitos - baixa qualidade, apenas para fallback)
    DEFAULT_PROXIES: List[str] = []  # Deixar vazio - proxies gratuitos não funcionam bem
    
    def __init__(
        self, 
        proxies: Optional[List[str]] = None,
        config: Optional[ProxyManagerConfig] = None
    ):
        """
        Inicializa o gerenciador.
        
        Args:
            proxies: Lista de URLs de proxy. Se None, tenta carregar de .env
            config: Configuração customizada
        """
        self.config = config or ProxyManagerConfig()
        self._proxies: Dict[str, ProxyInfo] = {}
        self._current_index: int = 0
        self._lock = asyncio.Lock()
        
        # Carregar proxies
        proxy_list = proxies or self._load_from_env() or self.DEFAULT_PROXIES
        
        for url in proxy_list:
            if url and url.strip():
                self._proxies[url] = ProxyInfo(url=url.strip())
        
        logger.info(f"🔄 ProxyManager inicializado com {len(self._proxies)} proxies")
        
        if not self._proxies:
            logger.warning(
                "⚠️ Nenhum proxy configurado. Scrapers rodarão sem proxy.\n"
                "   Configure via PROXY_LIST no .env ou passe lista no construtor."
            )
    
    def _load_from_env(self) -> List[str]:
        """
        Carrega lista de proxies do .env.
        
        Formato esperado:
            PROXY_LIST=http://user:pass@host1:port,http://user:pass@host2:port
            
        Returns:
            Lista de URLs de proxy
        """
        env_value = os.getenv("PROXY_LIST", "")
        
        if not env_value:
            # Tentar carregar de arquivo .env manualmente
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                try:
                    with open(env_path, "r") as f:
                        for line in f:
                            if line.startswith("PROXY_LIST="):
                                env_value = line.split("=", 1)[1].strip().strip('"\'')
                                break
                except Exception as e:
                    logger.debug(f"Não foi possível ler .env: {e}")
        
        if not env_value:
            return []
        
        proxies = [p.strip() for p in env_value.split(",") if p.strip()]
        logger.info(f"✅ Carregados {len(proxies)} proxies do .env")
        return proxies
    
    def get_proxy(self) -> Optional[str]:
        """
        Obtém próximo proxy disponível (round-robin).
        
        Returns:
            URL do proxy ou None se nenhum disponível
        """
        if not self._proxies:
            return None
        
        available = [
            info for info in self._proxies.values() 
            if info.is_available()
        ]
        
        if not available:
            logger.warning("⚠️ Nenhum proxy disponível no momento")
            return None
        
        # Round-robin entre proxies disponíveis
        self._current_index = (self._current_index + 1) % len(available)
        selected = available[self._current_index]
        
        # Atualizar status
        selected.status = ProxyStatus.IN_USE
        selected.last_used = datetime.now()
        
        logger.debug(f"🔄 Usando proxy: {self._mask_proxy(selected.url)}")
        return selected.url
    
    def mark_success(self, proxy_url: str):
        """
        Marca proxy como bem-sucedido.
        
        Args:
            proxy_url: URL do proxy
        """
        if proxy_url not in self._proxies:
            return
        
        info = self._proxies[proxy_url]
        info.success_count += 1
        info.fail_count = 0  # Reset falhas consecutivas
        info.status = ProxyStatus.AVAILABLE
        
        logger.debug(
            f"✅ Proxy sucesso: {self._mask_proxy(proxy_url)} "
            f"(total: {info.success_count})"
        )
    
    def mark_failed(self, proxy_url: str, reason: str = "unknown"):
        """
        Marca proxy como falho.
        
        Args:
            proxy_url: URL do proxy
            reason: Motivo da falha
        """
        if proxy_url not in self._proxies:
            return
        
        info = self._proxies[proxy_url]
        info.fail_count += 1
        
        # Verificar se deve queimar
        if info.fail_count >= self.config.max_fails_before_burn:
            self.mark_burned(proxy_url, reason)
        else:
            # Cooldown temporário
            info.status = ProxyStatus.COOLING_DOWN
            info.cooldown_until = datetime.now() + timedelta(
                seconds=self.config.cooldown_seconds * info.fail_count
            )
            
            logger.warning(
                f"⚠️ Proxy falhou ({info.fail_count}x): {self._mask_proxy(proxy_url)} "
                f"- {reason}. Cooldown: {info.cooldown_until}"
            )
    
    def mark_burned(self, proxy_url: str, reason: str):
        """
        Marca proxy como permanentemente queimado.
        
        Args:
            proxy_url: URL do proxy
            reason: Motivo do ban
        """
        if proxy_url not in self._proxies:
            return
        
        info = self._proxies[proxy_url]
        info.status = ProxyStatus.BURNED
        info.burn_reason = reason
        
        logger.error(
            f"🔥 Proxy BURNED: {self._mask_proxy(proxy_url)} - {reason}"
        )
    
    def rotate(self) -> Optional[str]:
        """
        Força rotação para próximo proxy disponível.
        
        Returns:
            URL do novo proxy ou None
        """
        return self.get_proxy()
    
    async def health_check(self, proxy_url: str) -> bool:
        """
        Verifica se um proxy está funcional.
        
        Args:
            proxy_url: URL do proxy a verificar
            
        Returns:
            True se funcional, False caso contrário
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx não disponível para health check")
            return True  # Assume funcional
        
        test_url = "https://httpbin.org/ip"
        
        try:
            async with httpx.AsyncClient(
                proxy=proxy_url,
                timeout=self.config.health_check_timeout
            ) as client:
                response = await client.get(test_url)
                
                if response.status_code == 200:
                    logger.debug(f"✅ Health check OK: {self._mask_proxy(proxy_url)}")
                    return True
                
                logger.warning(
                    f"⚠️ Health check falhou ({response.status_code}): "
                    f"{self._mask_proxy(proxy_url)}"
                )
                return False
                
        except Exception as e:
            logger.warning(f"❌ Health check erro: {e}")
            return False
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Verifica saúde de todos os proxies.
        
        Returns:
            Dict mapeando proxy URL → status (True/False)
        """
        results = {}
        
        for url in self._proxies:
            results[url] = await self.health_check(url)
            await asyncio.sleep(0.5)  # Evitar rate limit
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Retorna estatísticas do pool de proxies.
        
        Returns:
            Dict com estatísticas
        """
        total = len(self._proxies)
        available = sum(1 for p in self._proxies.values() if p.is_available())
        burned = sum(1 for p in self._proxies.values() if p.status == ProxyStatus.BURNED)
        cooling = sum(1 for p in self._proxies.values() if p.status == ProxyStatus.COOLING_DOWN)
        
        return {
            "total": total,
            "available": available,
            "burned": burned,
            "cooling_down": cooling,
            "in_use": total - available - burned - cooling,
        }
    
    def reset_all(self):
        """
        Reseta status de todos os proxies.
        Útil para reiniciar após período de espera.
        """
        for info in self._proxies.values():
            info.status = ProxyStatus.AVAILABLE
            info.fail_count = 0
            info.cooldown_until = None
            info.burn_reason = None
        
        logger.info("🔄 Todos os proxies resetados")
    
    def _mask_proxy(self, url: str) -> str:
        """
        Mascara credenciais do proxy para logging seguro.
        
        Args:
            url: URL completa do proxy
            
        Returns:
            URL com credenciais mascaradas
        """
        if "@" in url:
            # http://user:pass@host:port → http://***:***@host:port
            protocol, rest = url.split("://", 1)
            if "@" in rest:
                creds, host = rest.rsplit("@", 1)
                return f"{protocol}://***:***@{host}"
        return url


# Singleton global para uso compartilhado
_proxy_manager_instance: Optional[ProxyManager] = None


def get_proxy_manager() -> ProxyManager:
    """
    Obtém instância singleton do ProxyManager.
    
    Returns:
        Instância compartilhada do ProxyManager
    """
    global _proxy_manager_instance
    
    if _proxy_manager_instance is None:
        _proxy_manager_instance = ProxyManager()
    
    return _proxy_manager_instance


def reset_proxy_manager():
    """Reseta instância singleton (útil para testes)."""
    global _proxy_manager_instance
    _proxy_manager_instance = None


# ============================================================================
# TESTE / DEMO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    async def demo():
        """Demo do gerenciador de proxies."""
        # Criar com proxies de teste
        pm = ProxyManager(proxies=[
            "http://test1:8080",
            "http://test2:8080",
            "http://test3:8080",
        ])
        
        print("\n📊 Estatísticas iniciais:")
        print(pm.get_stats())
        
        # Simular uso
        for i in range(5):
            proxy = pm.get_proxy()
            print(f"\n🔄 Tentativa {i+1}: {proxy}")
            
            if i % 2 == 0:
                pm.mark_success(proxy)
            else:
                pm.mark_failed(proxy, "403 Forbidden")
        
        print("\n📊 Estatísticas após uso:")
        print(pm.get_stats())
        
        # Testar reset
        pm.reset_all()
        print("\n📊 Após reset:")
        print(pm.get_stats())
    
    asyncio.run(demo())
