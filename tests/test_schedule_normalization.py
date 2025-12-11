"""
Testes de Normalização de Schedule

Valida que o schedule_scraper retorna nomes normalizados (IDs de 3 letras)
e que há consistência entre schedule e database.
"""
import pytest
import pandas as pd
from data.scrapers.schedule_scraper import obter_schedule
from utils.team_normalization import normalize_team


def test_schedule_returns_normalized_names():
    """Verifica que schedule retorna nomes normalizados (3 letras uppercase)"""
    # Tentar obter schedule de hoje
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    schedule = obter_schedule(today)
    
    if not schedule:
        pytest.skip("Nenhum jogo encontrado para hoje - teste inconclusivo")
    
    for game in schedule:
        home = game['home']
        away = game['away']
        
        # Todos os nomes devem ser IDs de 3 letras UPPERCASE
        assert len(home) == 3, f"Home team não normalizado: '{home}' (len={len(home)})"
        assert len(away) == 3, f"Away team não normalizado: '{away}' (len={len(away)})"
        assert home.isupper(), f"Home team não está uppercase: '{home}'"
        assert away.isupper(), f"Away team não está uppercase: '{away}'"
        assert home.isalpha(), f"Home team contém caracteres inválidos: '{home}'"
        assert away.isalpha(), f"Away team contém caracteres inválidos: '{away}'"


def test_normalization_consistency():
    """Verifica que schedule e database usam a mesma normalização"""
    from data.repositories.db_manager import get_db_manager
    
    db = get_db_manager()
    df_history = db.get_history()
    
    if df_history.empty:
        pytest.skip("Database vazio - teste inconclusivo")
    
    # Verificar formato do histórico (sample de 10 times únicos)
    sample_home_teams = df_history['home_team'].dropna().head(20).unique()
    
    assert len(sample_home_teams) > 0, "Nenhum time encontrado no histórico"
    
    for team in sample_home_teams:
        assert len(team) == 3, f"Database tem nome não-normalizado: '{team}' (len={len(team)})"
        assert team.isupper(), f"Database tem nome não-uppercase: '{team}'"
        assert team.isalpha(), f"Database tem caracteres inválidos: '{team}'"


def test_normalize_team_function():
    """Testa a função normalize_team com casos comuns"""
    # Nomes completos
    assert normalize_team("Los Angeles Lakers") == "LAL"
    assert normalize_team("Golden State Warriors") == "GSW"
    assert normalize_team("Boston Celtics") == "BOS"
    
    # Apelidos
    assert normalize_team("Lakers") == "LAL"
    assert normalize_team("Warriors") == "GSW"
    assert normalize_team("Celtics") == "BOS"
    
    # IDs já normalizados
    assert normalize_team("LAL") == "LAL"
    assert normalize_team("GSW") == "GSW"
    assert normalize_team("BOS") == "BOS"
    
    # Case insensitive
    assert normalize_team("lakers") == "LAL"
    assert normalize_team("LAKERS") == "LAL"
    assert normalize_team("LaKeRs") == "LAL"
    
    # Inválidos
    assert normalize_team("Invalid Team") is None
    assert normalize_team("") is None
    assert normalize_team(None) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
