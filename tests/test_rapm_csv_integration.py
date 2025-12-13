"""
Teste de integração RAPM CSV - Requer dados reais de RAPM.
Marcado como skip quando dados não estão disponíveis.
"""
import pytest
import asyncio
from pathlib import Path
from data.scrapers.stats_scraper import StatsScraper


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "data/nba_rapm.csv").exists(),
    reason="Requer data/nba_rapm.csv para teste de integração"
)
@pytest.mark.xfail(reason="Requer dados RAPM reais e configuração asyncio")
@pytest.mark.asyncio
async def test_rapm_integration():
    """Testa carregamento de RAPM do CSV."""
    scraper = StatsScraper()
    rapm_data = await scraper.get_rapm()

    assert not rapm_data.empty, "RAPM não deve estar vazio"
    assert 'Player' in rapm_data.columns or 'PLAYER_NAME' in rapm_data.columns
