"""
Cache singleton para estatísticas de árbitros.
Evita recarregar dados repetidamente e problemas com variáveis globais.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from threading import Lock

logger = logging.getLogger(__name__)

class RefereeCache:
    """
    Singleton para cache de estatísticas de árbitros.
    Thread-safe e com lazy loading.
    """
    _instance: Optional['RefereeCache'] = None
    _lock = Lock()
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_loaded = False
        self._initialized = True
    
    def load_cache(self, csv_path: Optional[Path] = None) -> bool:
        """
        Carrega o cache de árbitros do arquivo CSV.
        
        Args:
            csv_path: Caminho para o arquivo CSV. Se None, usa o padrão.
            
        Returns:
            bool: True se carregado com sucesso, False caso contrário.
        """
        if self._cache_loaded:
            return True
        
        if csv_path is None:
            csv_path = Path(__file__).parent.parent / "data" / "referee_stats.csv"
        
        if not csv_path.exists():
            logger.warning(f"⚠️  Arquivo de estatísticas de árbitros não encontrado: {csv_path}")
            return False
        
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['referee_name', 'home_win_pct', 'foul_rate', 'games_officiated', 'years_experience']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"❌ Colunas faltando no CSV de árbitros: {missing_columns}")
                return False
            
            for _, row in df.iterrows():
                try:
                    ref_name = str(row['referee_name']).lower()
                    self._cache[ref_name] = {
                        'home_win_pct': float(row['home_win_pct']),
                        'foul_rate': float(row['foul_rate']),
                        'games': int(row['games_officiated']),
                        'experience': int(row['years_experience'])
                    }
                except (ValueError, KeyError) as e:
                    logger.warning(f"⚠️  Erro ao processar linha do árbitro: {e}")
                    continue
            
            self._cache_loaded = True
            logger.info(f"✅ Cache de árbitros carregado: {len(self._cache)} árbitros")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar cache de árbitros: {e}")
            return False
    
    def get_stats(self, ref_name: str) -> Dict[str, Any]:
        """
        Obtém estatísticas de um árbitro.
        
        Args:
            ref_name: Nome do árbitro.
            
        Returns:
            Dict com estatísticas do árbitro ou valores padrão se não encontrado.
        """
        if not self._cache_loaded:
            self.load_cache()
        
        ref_lower = ref_name.lower()
        
        # Busca exata
        if ref_lower in self._cache:
            return self._cache[ref_lower]
        
        # Busca parcial (fuzzy matching)
        for cached_name, stats in self._cache.items():
            if ref_lower in cached_name or cached_name in ref_lower:
                return stats
        
        # Retornar valores padrão se não encontrado
        logger.debug(f"⚠️  Árbitro '{ref_name}' não encontrado no cache. Usando valores padrão.")
        from config.constants import REFEREE_ADJUSTMENTS
        return {
            "home_win_pct": REFEREE_ADJUSTMENTS.get('default_home_win_pct', 0.55),
            "foul_rate": 42.0,
            "games": 0,
            "experience": 0
        }
    
    def clear_cache(self):
        """Limpa o cache (útil para testes)."""
        self._cache.clear()
        self._cache_loaded = False
    
    def get_cache_size(self) -> int:
        """Retorna o número de árbitros no cache."""
        return len(self._cache)

# Instância global (singleton)
_referee_cache_instance = RefereeCache()

def get_referee_stats(ref_name: str) -> Dict[str, Any]:
    """
    Função helper para obter estatísticas de árbitro.
    Usa o singleton RefereeCache.
    
    Args:
        ref_name: Nome do árbitro.
        
    Returns:
        Dict com estatísticas do árbitro.
    """
    return _referee_cache_instance.get_stats(ref_name)

