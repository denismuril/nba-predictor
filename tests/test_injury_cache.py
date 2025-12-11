"""
Testes do Sistema de Cache de Lesões

Testa:
- CacheManager: TTL, serialização, thread-safety
- InjuryManager: Orquestração de scrapers
- InjuryReport: Estrutura de dados
"""
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.scrapers.injury_scraper_v2 import (
    InjuryReport,
    CacheManager,
    DataCleaner,
    InjuryManager,
    CACHE_FILE,
    CACHE_TTL_MINUTES,
)


class TestInjuryReport:
    """Testes para a estrutura InjuryReport."""
    
    def test_injury_report_creation(self):
        """InjuryReport deve ser criável com todos os campos."""
        report = InjuryReport(
            player_name="LeBron James",
            team="LAL",
            status="OUT",
            description="Ankle sprain",
            source="Rotowire",
            updated_at="2024-01-01T12:00:00"
        )
        
        assert report.player_name == "LeBron James"
        assert report.team == "LAL"
        assert report.status == "OUT"
    
    def test_is_critical_out(self):
        """Status OUT deve ser considerado crítico."""
        report = InjuryReport(
            player_name="Test",
            team="TST",
            status="OUT",
            description="Test",
            source="Test",
            updated_at="2024-01-01T12:00:00"
        )
        assert report.is_critical() is True
    
    def test_is_critical_doubtful(self):
        """Status DOUBTFUL deve ser considerado crítico."""
        report = InjuryReport(
            player_name="Test",
            team="TST",
            status="DOUBTFUL",
            description="Test",
            source="Test",
            updated_at="2024-01-01T12:00:00"
        )
        assert report.is_critical() is True
    
    def test_is_critical_questionable(self):
        """Status QUESTIONABLE NÃO deve ser considerado crítico."""
        report = InjuryReport(
            player_name="Test",
            team="TST",
            status="QUESTIONABLE",
            description="Test",
            source="Test",
            updated_at="2024-01-01T12:00:00"
        )
        assert report.is_critical() is False


class TestDataCleaner:
    """Testes para normalização de dados."""
    
    def test_normalize_name_with_jr(self):
        """Deve remover Jr. do nome."""
        assert DataCleaner.normalize_name("Marcus Morris Jr.") == "Marcus Morris"
        assert DataCleaner.normalize_name("Gary Trent Jr") == "Gary Trent"
    
    def test_normalize_name_with_suffix(self):
        """Deve remover sufixos II, III, IV."""
        assert DataCleaner.normalize_name("Robert Williams III") == "Robert Williams"
        assert DataCleaner.normalize_name("Kelly Oubre Jr.") == "Kelly Oubre"
    
    def test_normalize_status_out(self):
        """Deve normalizar status OUT."""
        assert DataCleaner.normalize_status("out") == "OUT"
        assert DataCleaner.normalize_status("Out (ankle)") == "OUT"
        assert DataCleaner.normalize_status("OUT - injury") == "OUT"
    
    def test_normalize_status_questionable(self):
        """Deve normalizar status QUESTIONABLE."""
        assert DataCleaner.normalize_status("questionable") == "QUESTIONABLE"
        assert DataCleaner.normalize_status("Questionable (knee)") == "QUESTIONABLE"
    
    def test_normalize_status_gtd(self):
        """Deve normalizar status GTD (Game Time Decision)."""
        assert DataCleaner.normalize_status("GTD") == "GTD"
        assert DataCleaner.normalize_status("Game Time Decision") == "GTD"


class TestCacheManager:
    """Testes para o gerenciador de cache."""
    
    def test_cache_save_and_load(self, tmp_path):
        """Cache deve salvar e carregar corretamente."""
        # Criar report de teste
        report = InjuryReport(
            player_name="Test Player",
            team="TST",
            status="OUT",
            description="Test injury",
            source="Test",
            updated_at=datetime.now().isoformat()
        )
        
        # Mockar o arquivo de cache
        with patch('data.scrapers.injury_scraper_v2.CACHE_FILE', tmp_path / "test_cache.json"):
            with patch('data.scrapers.injury_scraper_v2.CACHE_DIR', tmp_path):
                # Salvar
                result = CacheManager.save_cache([report])
                assert result is True
                
                # Carregar
                loaded = CacheManager.load_cache()
                assert loaded is not None
                assert len(loaded) == 1
                assert loaded[0].player_name == "Test Player"
    
    def test_cache_expired(self, tmp_path):
        """Cache expirado deve retornar None."""
        cache_file = tmp_path / "expired_cache.json"
        
        # Criar cache velho (2 horas atrás)
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        cache_data = {
            'timestamp': old_time,
            'count': 1,
            'data': [{
                'player_name': 'Old Player',
                'team': 'OLD',
                'status': 'OUT',
                'description': 'Old',
                'source': 'Old',
                'updated_at': old_time
            }]
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        with patch('data.scrapers.injury_scraper_v2.CACHE_FILE', cache_file):
            # Com TTL de 30 min, cache de 2h deve ser inválido
            loaded = CacheManager.load_cache()
            assert loaded is None
    
    def test_cache_corrupted(self, tmp_path):
        """Cache corrompido deve retornar None."""
        cache_file = tmp_path / "corrupted_cache.json"
        
        with open(cache_file, 'w') as f:
            f.write("{ invalid json }")
        
        with patch('data.scrapers.injury_scraper_v2.CACHE_FILE', cache_file):
            loaded = CacheManager.load_cache()
            assert loaded is None
    
    def test_cache_missing_file(self, tmp_path):
        """Cache inexistente deve retornar None."""
        non_existent = tmp_path / "non_existent.json"
        
        with patch('data.scrapers.injury_scraper_v2.CACHE_FILE', non_existent):
            loaded = CacheManager.load_cache()
            assert loaded is None


class TestInjuryManager:
    """Testes para o orquestrador principal."""
    
    def test_manager_uses_cache_when_valid(self, tmp_path):
        """Manager deve usar cache quando válido."""
        cache_file = tmp_path / "valid_cache.json"
        
        # Criar cache recente (5 minutos atrás)
        recent_time = (datetime.now() - timedelta(minutes=5)).isoformat()
        cache_data = {
            'timestamp': recent_time,
            'count': 1,
            'ttl_minutes': 30,
            'data': [{
                'player_name': 'Cached Player',
                'team': 'CAC',
                'status': 'OUT',
                'description': 'From cache',
                'source': 'Cache',
                'updated_at': recent_time
            }]
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        with patch('data.scrapers.injury_scraper_v2.CACHE_FILE', cache_file):
            manager = InjuryManager()
            injuries = manager.get_latest_injuries()
            
            assert len(injuries) == 1
            assert injuries[0].player_name == "Cached Player"
            assert injuries[0].source == "Cache"
    
    def test_manager_calculates_impact(self):
        """Manager deve calcular impacto corretamente."""
        manager = InjuryManager()
        
        # Simular lesões
        test_injuries = [
            InjuryReport("LeBron James", "LAL", "OUT", "Test", "Test", "2024-01-01"),
            InjuryReport("Unknown Player", "LAL", "QUESTIONABLE", "Test", "Test", "2024-01-01"),
        ]
        
        impact = manager.calculate_team_injury_impact("LAL", test_injuries)
        
        # Impact deve ser negativo (lesões prejudicam)
        assert impact < 0
        # LeBron OUT tem alto impacto
        assert impact < -0.1


class TestIntegration:
    """Testes de integração."""
    
    def test_full_flow_with_mocked_scrapers(self, tmp_path):
        """Fluxo completo com scrapers mockados."""
        cache_file = tmp_path / "integration_cache.json"
        
        # Mock que simula scraper retornando dados
        mock_injuries = [
            InjuryReport("Stephen Curry", "GSW", "OUT", "Ankle", "Mock", "2024-01-01"),
            InjuryReport("Kevin Durant", "PHX", "QUESTIONABLE", "Knee", "Mock", "2024-01-01"),
        ]
        
        with patch('data.scrapers.injury_scraper_v2.CACHE_FILE', cache_file):
            with patch('data.scrapers.injury_scraper_v2.CACHE_DIR', tmp_path):
                manager = InjuryManager()
                
                # Mockar todos os scrapers para retornarem os dados mockados
                for scraper in manager.scrapers:
                    scraper.scrape = MagicMock(return_value=mock_injuries)
                
                injuries = manager.get_latest_injuries(force_refresh=True)
                
                assert len(injuries) == 2
                assert injuries[0].player_name == "Stephen Curry"
                
                # Verificar que cache foi criado
                assert cache_file.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
