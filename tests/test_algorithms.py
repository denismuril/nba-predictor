"""
Testes unitários para core/algorithms.py

Testa funções críticas:
- normalizar_metrica
- calcular_net_rating_v11
- calcular_power_rating_v11

NOTA: Atualizado para usar as assinaturas corretas das funções (v11).
"""

import pytest
import pandas as pd
import numpy as np

from core.algorithms import (
    normalizar_metrica,
    calcular_net_rating_v11,
    calcular_power_rating_v11,
)


class TestNormalizarMetrica:
    """Testes para normalizar_metrica()."""

    def test_normalizar_metrica_valid_range(self):
        """Testa normalização com valores válidos."""
        # Valor no meio do range (5 no range 0-10) = 50
        result = normalizar_metrica(5.0, 0.0, 10.0)
        assert result == 50.0

    def test_normalizar_metrica_extremos(self):
        """Testa nos extremos do range."""
        # Valor mínimo = 0
        assert normalizar_metrica(0.0, 0.0, 10.0) == 0.0
        # Valor máximo = 100
        assert normalizar_metrica(10.0, 0.0, 10.0) == 100.0

    def test_normalizar_metrica_division_by_zero(self):
        """Testa quando min == max (divisão por zero)."""
        result = normalizar_metrica(5.0, 5.0, 5.0)
        # Deve retornar 50 (fallback) quando range é zero
        assert result == 50.0

    def test_normalizar_metrica_clipping(self):
        """Testa que valores são clipped entre 0 e 100."""
        # Valor acima do máximo deve ser clipped para 100
        result = normalizar_metrica(15.0, 0.0, 10.0)
        assert result == 100.0

        # Valor abaixo do mínimo deve ser clipped para 0
        result = normalizar_metrica(-5.0, 0.0, 10.0)
        assert result == 0.0

    def test_normalizar_metrica_negative_range(self):
        """Testa com range negativo."""
        # -5 no range -10 a -2
        # ((-5 - (-10)) / (-2 - (-10))) * 100 = 62.5
        result = normalizar_metrica(-5.0, -10.0, -2.0)
        assert abs(result - 62.5) < 0.01


class TestCalcularNetRatingV11:
    """Testes para calcular_net_rating_v11().
    
    NOTA: Esta função espera Dict[str, pd.DataFrame], não Dict[str, float].
    O segundo parâmetro 'dfs' deve conter DataFrames com colunas apropriadas.
    """
    
    @pytest.fixture
    def sample_rapm_df(self):
        """Cria DataFrame RAPM de exemplo."""
        return pd.DataFrame({
            'Team': ['LAL', 'GSW', 'BOS'],
            'Time Decay ORAPM': [3.5, 4.2, 2.8],
            'Time Decay DRAPM': [2.1, 1.9, 3.0]
        })
    
    @pytest.fixture
    def sample_lebron_df(self):
        """Cria DataFrame LEBRON de exemplo."""
        return pd.DataFrame({
            'Team': ['LAL', 'GSW', 'BOS'],
            'O-LEBRON': [2.5, 3.1, 2.0],
            'D-LEBRON': [1.8, 2.2, 2.5]
        })

    def test_net_rating_complete_data(self, sample_rapm_df, sample_lebron_df):
        """Testa com dados completos em DataFrames."""
        dfs = {
            'RAPM': sample_rapm_df,
            'lebron': sample_lebron_df
        }
        
        result = calcular_net_rating_v11('LAL', dfs)
        
        # Deve retornar um float
        assert isinstance(result, (int, float))
        
        # Net rating deve estar em range razoável
        assert -100 <= result <= 100

    def test_net_rating_empty_dfs(self):
        """Testa com DataFrames vazios."""
        dfs = {
            'RAPM': pd.DataFrame(),
            'lebron': pd.DataFrame()
        }
        
        result = calcular_net_rating_v11('Lakers', dfs)
        
        # Deve retornar 0 ou valor default
        assert result == 0.0

    def test_net_rating_missing_team(self, sample_rapm_df):
        """Testa com time que não existe nos dados."""
        dfs = {'RAPM': sample_rapm_df}
        
        result = calcular_net_rating_v11('NONEXISTENT_TEAM', dfs)
        
        # Deve retornar 0 quando time não é encontrado
        assert result == 0.0

    def test_net_rating_consistency(self, sample_rapm_df, sample_lebron_df):
        """Testa que mesmos inputs geram mesmo output."""
        dfs = {
            'RAPM': sample_rapm_df.copy(),
            'lebron': sample_lebron_df.copy()
        }
        
        result1 = calcular_net_rating_v11('LAL', dfs)
        result2 = calcular_net_rating_v11('LAL', dfs)
        
        assert result1 == result2


class TestCalcularPowerRatingV11:
    """Testes para calcular_power_rating_v11().
    
    Assinatura: calcular_power_rating_v11(
        home_team, away_team, injuries, standings, dfs,
        referees=None, shot_quality_data=None, recent_games_df=None
    )
    """
    
    @pytest.fixture
    def sample_dfs(self):
        """DataFrames de estatísticas de exemplo."""
        return {
            'RAPM': pd.DataFrame({
                'Team': ['LAL', 'BOS'],
                'Time Decay ORAPM': [3.5, 2.8],
                'Time Decay DRAPM': [2.1, 3.0]
            }),
            'lebron': pd.DataFrame({
                'Team': ['LAL', 'BOS'],
                'O-LEBRON': [2.5, 2.0],
                'D-LEBRON': [1.8, 2.5]
            })
        }
    
    @pytest.fixture
    def empty_injuries(self):
        """Dicionário de lesões vazio."""
        return {}
    
    @pytest.fixture
    def empty_standings(self):
        """Classificação vazia."""
        return {}

    def test_power_rating_basic(self, sample_dfs, empty_injuries, empty_standings):
        """Testa cálculo básico de Power Rating."""
        result = calcular_power_rating_v11(
            home_team='LAL',
            away_team='BOS',
            injuries=empty_injuries,
            standings=empty_standings,
            dfs=sample_dfs
        )
        
        # Deve retornar dict com keys esperadas
        assert isinstance(result, dict)
        assert 'pr_casa' in result
        assert 'pr_visitante' in result
        assert 'prob_casa' in result
        assert 'prob_visitante' in result
        
        # Power rating deve ser numérico positivo
        assert isinstance(result['pr_casa'], (int, float))
        assert result['pr_casa'] > 0

    def test_power_rating_probabilities_sum_to_100(self, sample_dfs, empty_injuries, empty_standings):
        """Testa que probabilidades somam aproximadamente 100."""
        result = calcular_power_rating_v11(
            home_team='LAL',
            away_team='BOS',
            injuries=empty_injuries,
            standings=empty_standings,
            dfs=sample_dfs
        )
        
        total_prob = result['prob_casa'] + result['prob_visitante']
        assert abs(total_prob - 100.0) < 0.01

    def test_power_rating_home_advantage(self, sample_dfs, empty_injuries, empty_standings):
        """Testa que home advantage é aplicado."""
        # Mesmo time casa e fora com estatísticas iguais
        # O resultado deve mostrar favor para a casa
        dfs = {
            'RAPM': pd.DataFrame({
                'Team': ['LAL'],
                'Time Decay ORAPM': [0.0],
                'Time Decay DRAPM': [0.0]
            })
        }
        
        result = calcular_power_rating_v11(
            home_team='LAL',
            away_team='LAL',  # Mesmo time
            injuries=empty_injuries,
            standings=empty_standings,
            dfs=dfs
        )
        
        # Com estatísticas iguais, home team deve ter vantagem
        assert result['pr_casa'] >= result['pr_visitante']

    def test_power_rating_with_injuries(self, sample_dfs, empty_standings):
        """Testa impacto de lesões no rating."""
        injuries = {
            'LAL': {
                'LeBron James': 'Out',
                'Anthony Davis': 'Questionable'
            }
        }
        
        result = calcular_power_rating_v11(
            home_team='LAL',
            away_team='BOS',
            injuries=injuries,
            standings=empty_standings,
            dfs=sample_dfs
        )
        
        # Deve retornar resultado válido com lesões
        assert isinstance(result, dict)
        assert 'fator_lesao_casa' in result

    def test_power_rating_with_referees(self, sample_dfs, empty_injuries, empty_standings):
        """Testa com lista de árbitros."""
        result = calcular_power_rating_v11(
            home_team='LAL',
            away_team='BOS',
            injuries=empty_injuries,
            standings=empty_standings,
            dfs=sample_dfs,
            referees=['Scott Foster', 'Tony Brothers']
        )
        
        # Deve retornar resultado válido
        assert isinstance(result, dict)
        assert 'ajuste_referee' in result


class TestEdgeCases:
    """Testes para casos extremos."""

    def test_nan_handling(self):
        """Testa tratamento de NaN."""
        dfs = {
            'RAPM': pd.DataFrame({
                'Team': ['LAL'],
                'Time Decay ORAPM': [np.nan],
                'Time Decay DRAPM': [np.nan]
            })
        }
        
        result = calcular_net_rating_v11('Lakers', dfs)
        
        # Não deve levantar exceção
        assert isinstance(result, (int, float))
        # Deve tratar NaN corretamente (retornar 0)
        assert result == 0.0 or not np.isnan(result)

    def test_empty_team_name(self):
        """Testa com nome de time vazio."""
        dfs = {'RAPM': pd.DataFrame()}
        
        result = calcular_net_rating_v11('', dfs)
        
        # Deve retornar 0 para nome vazio
        assert result == 0.0


def test_full_rating_pipeline():
    """Teste de integração do pipeline completo de ratings."""
    # Criar dados mock
    rapm_df = pd.DataFrame({
        'Team': ['LAL', 'BOS', 'GSW'],
        'Time Decay ORAPM': [3.0, 2.5, 4.0],
        'Time Decay DRAPM': [1.5, 2.0, 2.5]
    })
    
    dfs = {'RAPM': rapm_df}
    
    # Calcular net rating para um time
    nr = calcular_net_rating_v11('LAL', dfs)
    
    # Net rating deve ser calculado
    assert isinstance(nr, (int, float))
    
    # Calcular power rating para confronto
    result = calcular_power_rating_v11(
        home_team='LAL',
        away_team='BOS',
        injuries={},
        standings={},
        dfs=dfs
    )
    
    # Deve retornar resultado completo
    assert 'pr_casa' in result
    assert 'prob_casa' in result
    assert result['prob_casa'] + result['prob_visitante'] == pytest.approx(100.0, abs=0.01)
