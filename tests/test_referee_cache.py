"""
Testes unitários para o cache de árbitros.
"""
import pytest
import pandas as pd
from pathlib import Path
import tempfile
from core.referee_cache import RefereeCache, get_referee_stats


class TestRefereeCache:
    """Testes para a classe RefereeCache."""
    
    def test_singleton_pattern(self):
        """Testa que RefereeCache é singleton."""
        cache1 = RefereeCache()
        cache2 = RefereeCache()
        assert cache1 is cache2
    
    def test_get_stats_default(self):
        """Testa obtenção de stats com valores padrão."""
        cache = RefereeCache()
        cache.clear_cache()
        stats = cache.get_stats("Unknown Referee")
        
        assert stats['home_win_pct'] == 0.58
        assert stats['foul_rate'] == 42.0
        assert stats['games'] == 0
        assert stats['experience'] == 0
    
    def test_load_cache_from_csv(self):
        """Testa carregamento de cache a partir de CSV."""
        cache = RefereeCache()
        cache.clear_cache()
        
        # Criar CSV temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("referee_name,home_win_pct,foul_rate,games_officiated,years_experience\n")
            f.write("John Doe,0.58,40.5,500,15\n")
            f.write("Jane Smith,0.52,45.0,300,10\n")
            temp_path = Path(f.name)
        
        try:
            result = cache.load_cache(temp_path)
            assert result is True
            assert cache.get_cache_size() == 2
            
            stats = cache.get_stats("John Doe")
            assert stats['home_win_pct'] == 0.58
            assert stats['foul_rate'] == 40.5
            assert stats['games'] == 500
            assert stats['experience'] == 15
        finally:
            temp_path.unlink()
    
    def test_load_cache_missing_file(self):
        """Testa carregamento com arquivo faltando."""
        cache = RefereeCache()
        cache.clear_cache()
        
        result = cache.load_cache(Path("nonexistent_file.csv"))
        assert result is False
        assert cache.get_cache_size() == 0
    
    def test_load_cache_invalid_csv(self):
        """Testa carregamento com CSV inválido."""
        cache = RefereeCache()
        cache.clear_cache()
        
        # Criar CSV com colunas faltando
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("referee_name,home_win_pct\n")
            f.write("John Doe,0.58\n")
            temp_path = Path(f.name)
        
        try:
            result = cache.load_cache(temp_path)
            assert result is False
        finally:
            temp_path.unlink()
    
    def test_fuzzy_matching(self):
        """Testa matching parcial de nomes."""
        cache = RefereeCache()
        cache.clear_cache()
        
        # Criar CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("referee_name,home_win_pct,foul_rate,games_officiated,years_experience\n")
            f.write("John Michael Doe,0.58,40.5,500,15\n")
            temp_path = Path(f.name)
        
        try:
            cache.load_cache(temp_path)
            
            # Deve encontrar com nome parcial
            stats1 = cache.get_stats("John Doe")
            assert stats1['home_win_pct'] == 0.58
            
            stats2 = cache.get_stats("John Michael")
            assert stats2['home_win_pct'] == 0.58
        finally:
            temp_path.unlink()
    
    def test_clear_cache(self):
        """Testa limpeza do cache."""
        cache = RefereeCache()
        cache.clear_cache()
        
        # Adicionar dados manualmente (simular)
        cache._cache['test'] = {'home_win_pct': 0.5}
        assert cache.get_cache_size() == 1
        
        cache.clear_cache()
        assert cache.get_cache_size() == 0
        assert not cache._cache_loaded


class TestGetRefereeStats:
    """Testes para função helper get_referee_stats."""
    
    def test_get_stats_function(self):
        """Testa função helper."""
        stats = get_referee_stats("Unknown")
        assert isinstance(stats, dict)
        assert 'home_win_pct' in stats
        assert 'foul_rate' in stats
        assert 'games' in stats
        assert 'experience' in stats

