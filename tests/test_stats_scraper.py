"""
Testes unitários para StatsScraper - Fallback de Métricas RAPM
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from data.scrapers.stats_scraper import StatsScraper


class TestRAPMFallback:
    """Testes para o fallback hierárquico de métricas RAPM"""
    
    @pytest.mark.asyncio
    async def test_rapm_external_success(self):
        """Deve usar RAPM externo quando disponível (Prioridade 1)"""
        scraper = StatsScraper()
        
        # Mock de resposta bem-sucedida do nbarapm.com
        mock_data = [
            {'player_name': 'LeBron James', 'team': 'LAL', 
             'rapm_timedecay': 5.2, 'orapm_timedecay': 3.1, 'drapm_timedecay': 2.1},
            {'player_name': 'Stephen Curry', 'team': 'GSW',
             'rapm_timedecay': 4.8, 'orapm_timedecay': 3.5, 'drapm_timedecay': 1.3}
        ]
        
        with patch.object(scraper, 'fetch_json', return_value=mock_data):
            result = await scraper.get_rapm()
            
            # Verificações
            assert not result.empty
            assert len(result) == 2
            assert 'RAPM_SOURCE' in result.columns
            assert all(result['RAPM_SOURCE'] == 'EXTERNAL')
            assert 'Player' in result.columns
            assert 'RAPM' in result.columns
            assert result.loc[0, 'Player'] == 'LeBron James'
            assert pytest.approx(result.loc[0, 'RAPM'], 0.1) == 5.2
    
    @pytest.mark.asyncio
    async def test_rapm_fallback_to_netrting(self):
        """Deve usar NetRtg quando RAPM externo falhar (Prioridade 2)"""
        scraper = StatsScraper()
        
        # Mock falha do RAPM externo (retorna None)
        # Mock sucesso da NBA API com NetRtg
        mock_nba_data = pd.DataFrame({
            'PLAYER_NAME': ['Giannis Antetokounmpo', 'Nikola Jokic'],
            'TEAM_ABBREVIATION': ['MIL', 'DEN'],
            'NET_RATING': [12.5, 15.2]
        })
        
        with patch.object(scraper, 'fetch_json', return_value=None):
            with patch('data.scrapers.stats_scraper.leaguedashplayerstats.LeagueDashPlayerStats') as mock_api:
                mock_api.return_value.get_data_frames.return_value = [mock_nba_data]
                
                result = await scraper.get_rapm()
                
                # Verificações
                assert not result.empty
                assert 'RAPM_SOURCE' in result.columns
                assert all(result['RAPM_SOURCE'] == 'NET_RTG_NBA')
                # NetRtg deve ser clipped para [-8, +8]
                assert result['RAPM'].max() <= 8
                assert result['RAPM'].min() >= -8
    
    @pytest.mark.asyncio
    async def test_rapm_fallback_to_game_score(self):
        """Deve calcular Game Score quando todos os fallbacks falharem (Prioridade 3)"""
        scraper = StatsScraper()
        
        # Mock dados básicos para cálculo de Game Score
        mock_basic_stats = pd.DataFrame({
            'PLAYER_NAME': ['Kevin Durant', 'Luka Doncic'],
            'TEAM_ABBREVIATION': ['PHO', 'DAL'],
            'PTS': [28.5, 32.1],
            'FGM': [10.2, 11.5],
            'FGA': [18.5, 22.3],
            'FTM': [7.1, 8.2],
            'FTA': [8.0, 9.5],
            'REB': [7.2, 8.9],
            'AST': [5.1, 8.7],
            'STL': [0.9, 1.2],
            'BLK': [1.1, 0.5],
            'TOV': [3.2, 4.1],
            'PF': [2.1, 2.5]
        })
        
        # Mock todas as fontes anteriores falhando
        with patch.object(scraper, 'fetch_json', return_value=None):
            with patch('data.scrapers.stats_scraper.leaguedashplayerstats.LeagueDashPlayerStats') as mock_api:
                # Primeira chamada (NetRtg) falha
                # Segunda chamada (basic stats) retorna dados
                mock_api.return_value.get_data_frames.side_effect = [
                    [pd.DataFrame()],  # NetRtg vazio
                    [mock_basic_stats]  # Basic stats OK
                ]
                
                result = await scraper.get_rapm()
                
                # Verificações
                assert not result.empty
                assert 'RAPM_SOURCE' in result.columns
                assert all(result['RAPM_SOURCE'] == 'GAME_SCORE')
                assert 'Player' in result.columns
                assert 'RAPM' in result.columns
                # Game Score deve estar normalizado para range [-8, +8]
                assert result['RAPM'].max() <= 8
                assert result['RAPM'].min() >= -8
    
    def test_calculate_local_metrics_formula(self):
        """Deve calcular Game Score corretamente usando fórmula de Hollinger"""
        scraper = StatsScraper()
        
        # Dados de teste para um jogador hipotético
        test_data = pd.DataFrame({
            'PLAYER_NAME': ['Test Player'],
            'TEAM_ABBREVIATION': ['LAL'],
            'PTS': [20.0],
            'FGM': [8.0],
            'FGA': [15.0],
            'FTM': [3.0],
            'FTA': [4.0],
            'REB': [10.0],  # será dividido em OREB/DREB
            'AST': [5.0],
            'STL': [1.0],
            'BLK': [1.0],
            'TOV': [2.0],
            'PF': [2.0]
        })
        
        result = scraper._calculate_local_metrics(test_data)
        
        # Verificações básicas
        assert not result.empty
        assert len(result) == 1
        assert result.loc[0, 'Player'] == 'Test Player'
        assert result.loc[0, 'RAPM_SOURCE'] == 'GAME_SCORE'
        
        # Verificar que RAPM está dentro do range esperado
        assert -8 <= result.loc[0, 'RAPM'] <= 8
        
        # Verificar divisão O/DRAPM
        rapm_total = result.loc[0, 'RAPM']
        orapm = result.loc[0, 'ORAPM']
        drapm = result.loc[0, 'DRAPM']
        
        # ORAPM deve ser ~60% do total, DRAPM ~40%
        assert pytest.approx(orapm, abs=0.1) == rapm_total * 0.6
        assert pytest.approx(drapm, abs=0.1) == rapm_total * 0.4
    
    def test_calculate_local_metrics_missing_columns(self):
        """Deve retornar DataFrame vazio se colunas necessárias faltarem"""
        scraper = StatsScraper()
        
        # Dados incompletos (faltando PTS, FGM, etc.)
        incomplete_data = pd.DataFrame({
            'PLAYER_NAME': ['Test Player'],
            'TEAM_ABBREVIATION': ['LAL']
        })
        
        result = scraper._calculate_local_metrics(incomplete_data)
        
        # Deve retornar vazio
        assert result.empty
    
    def test_calculate_local_metrics_normalization(self):
        """Deve normalizar Game Score usando Z-Score corretamente"""
        scraper = StatsScraper()
        
        # Criar dataset com múltiplos jogadores para testar normalização
        test_data = pd.DataFrame({
            'PLAYER_NAME': ['Player A', 'Player B', 'Player C', 'Player D'],
            'TEAM_ABBREVIATION': ['LAL', 'GSW', 'BOS', 'MIA'],
            'PTS': [30.0, 20.0, 15.0, 10.0],
            'FGM': [12.0, 8.0, 6.0, 4.0],
            'FGA': [20.0, 15.0, 12.0, 10.0],
            'FTM': [5.0, 3.0, 2.0, 1.0],
            'FTA': [6.0, 4.0, 3.0, 2.0],
            'REB': [10.0, 8.0, 6.0, 4.0],
            'AST': [8.0, 6.0, 4.0, 2.0],
            'STL': [2.0, 1.5, 1.0, 0.5],
            'BLK': [1.5, 1.0, 0.5, 0.2],
            'TOV': [3.0, 2.5, 2.0, 1.5],
            'PF': [2.0, 2.0, 2.0, 2.0]
        })
        
        result = scraper._calculate_local_metrics(test_data)
        
        # Verificações
        assert len(result) == 4
        
        # Valores devem estar normalizados (clipped para [-8, +8])
        assert result['RAPM'].max() <= 8
        assert result['RAPM'].min() >= -8
        
        # Melhor jogador (Player A) deve ter RAPM mais alto
        assert result.loc[result['Player'] == 'Player A', 'RAPM'].values[0] > \
               result.loc[result['Player'] == 'Player D', 'RAPM'].values[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
