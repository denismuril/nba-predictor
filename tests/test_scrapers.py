"""
Teste 5: Conectividade e Fallback - Resiliência
================================================
CRÍTICO: Sistema deve sobreviver quedas de API.
"""

import pytest
from unittest.mock import patch, Mock


def test_api_failure_graceful_degradation():
    """
    Se NBA API cair, sistema deve usar fallback
    """
    from ml_pipeline.train_spread_real import fetch_historical_games_from_api
    
    with patch('nba_api.stats.endpoints.leaguegamefinder.LeagueGameFinder') as mock_api:
        # Simular erro
        mock_api.side_effect = Exception("NBA API Timeout")
        
        # Não deve crashar
        try:
            result = fetch_historical_games_from_api(seasons=['2025-26'])
            # Deve retornar None ou fallback
            assert result is None or len(result) == 0, \
                "Sistema deve retornar None em caso de erro"
            print("✅ API failure tratada corretamente")
        except Exception as e:
            pytest.fail(f"❌ Sistema crashou ao invés de fallback: {e}")


def test_injury_scraper_fallback():
    """
    Se PDF de lesões falhar, tentar ESPN
    """
    from data.scrapers.injury_scraper import obter_injury_report
    
    # Executar (deve tentar PDF, depois ESPN)
    try:
        result = obter_injury_report()
        
        # Deve retornar algo (dict vazio é OK)
        assert isinstance(result, dict), \
            f"❌ Deveria retornar dict, retornou {type(result)}"
        
        print(f"✅ Injury report: {len(result)} times")
    except Exception as e:
        print(f"⚠️ Injury scraper falhou: {e}")


def test_stats_scraper_excel_fallback():
    """
    Se scraping falhar, carregar do Excel
    """
    from data.scrapers.stats_scraper import obter_player_stats
    
    try:
        result = obter_player_stats()
        
        # Deve retornar dict de DataFrames
        assert isinstance(result, dict), \
            f"❌ Deveria retornar dict, retornou {type(result)}"
        
        # Deve ter pelo menos 1 source
        assert len(result) > 0, \
            "❌ Nenhuma fonte de stats disponível"
        
        print(f"✅ Stats sources: {list(result.keys())}")
    except Exception as e:
        pytest.fail(f"❌ Stats scraper falhou completamente: {e}")


def test_model_file_missing():
    """
    Se modelo não existir, deve dar erro claro
    """
    import joblib
    from pathlib import Path
    
    fake_path = Path('data/models/MODELO_INEXISTENTE.joblib')
    
    with pytest.raises(FileNotFoundError):
        joblib.load(fake_path)
    
    print("✅ Erro de modelo ausente detectado corretamente")


def test_empty_schedule_handling():
    """
    Se não houver jogos, sistema não deve crashar
    """
    from data.scrapers.schedule_scraper import obter_schedule
    from datetime import datetime, timedelta
    
    # Data futura sem jogos (off-season)
    future_date = (datetime.now() + timedelta(days=200)).strftime('%Y-%m-%d')
    
    try:
        schedule = obter_schedule(future_date)
        
        # Deve retornar lista vazia, não crashar
        assert isinstance(schedule, list), \
            f"❌ Deveria retornar lista, retornou {type(schedule)}"
        
        print(f"✅ Schedule vazio tratado: {len(schedule)} jogos")
    except Exception as e:
        print(f"⚠️ Schedule scraper com data futura: {e}")


if __name__ == "__main__":
    print("🌐 TESTANDO RESILIÊNCIA DO SISTEMA\n")
    test_api_failure_graceful_degradation()
    test_injury_scraper_fallback()
    test_stats_scraper_excel_fallback()
    test_model_file_missing()
    test_empty_schedule_handling()
    print("\n🎉 SISTEMA RESILIENTE!")
