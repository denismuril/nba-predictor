"""
Testes unitários para core/algorithms.py

Testa funções críticas:
- normalizar_metrica
- calcular_net_rating_v11
- calcular_power_rating_v11
- Expected eFG%
- Shot Quality
"""

import pytest
import pandas as pd
import numpy as np
from core.algorithms import (
    normalizar_metrica,
    calcular_net_rating_v11,
    calcular_power_rating_v11,
    calcular_expected_efg,
    calcular_shot_quality_adjustment
)


class TestNormalizarMetrica:
    """Testes para normalizar_metrica()."""
    
    def test_normalizar_metrica_valid_range(self):
        """Testa normalização com valores válidos."""
        result = normalizar_metrica(5.0, min_val=-10, max_val=10)
        
        # Deve estar entre 0 e 100
        assert 0 <= result <= 100
        
        # 5 está a 75% do caminho entre -10 e 10
        # ((5 - (-10)) / (10 - (-10))) * 100 = 75
        assert result == 75.0
    
    def test_normalizar_metrica_extremos(self):
        """Testa valores nos extremos."""
        # Valor mínimo
        assert normalizar_metrica(-10, -10, 10) == 0.0
        
        # Valor máximo
        assert normalizar_metrica(10, -10, 10) == 100.0
        
        # Valor médio
        assert normalizar_metrica(0, -10, 10) == 50.0
    
    def test_normalizar_metrica_division_by_zero(self):
        """Testa proteção contra divisão por zero."""
        # min_val == max_val deve retornar 50 (meio)
        result = normalizar_metrica(5, 5, 5)
        assert result == 50.0
    
    def test_normalizar_metrica_clipping(self):
        """Testa que valores fora do range são clippados."""
        # Valor acima do max
        result = normalizar_metrica(15, -10, 10)
        assert result == 100.0
        
        # Valor abaixo do min
        result = normalizar_metrica(-15, -10, 10)
        assert result == 0.0
    
    def test_normalizar_metrica_negative_range(self):
        """Testa com range completamente negativo."""
        result = normalizar_metrica(-5, -10, -2)
        
        # -5 está a 62.5% do caminho entre -10 e -2
        # ((-5 - (-10)) / (-2 - (-10))) * 100 = 62.5
        assert abs(result - 62.5) < 0.01


class TestCalcularNetRatingV11:
    """Testes para calcular_net_rating_v11()."""
    
    def test_net_rating_complete_data(self):
        """Testa com dados completos."""
        stats = {
            'RAPM': 3.5,
            'BPM': 4.2,
            'PIE': 0.58,
            'LEBRON': 2.1
        }
        
        result = calcular_net_rating_v11('Lakers', stats)
        
        # Deve retornar um float
        assert isinstance(result, (int, float))
        
        # Deve estar em um range razoável (-15 a 15)
        assert -15 <= result <= 15
    
    def test_net_rating_partial_data(self):
        """Testa com dados parciais (algumas métricas faltando)."""
        stats = {
            'RAPM': 3.5,
            'BPM': None,
            'PIE': 0.58,
            'LEBRON': None
        }
        
        result = calcular_net_rating_v11('Lakers', stats)
        
        # Deve funcionar mesmo sem todas as métricas
        assert isinstance(result, (int, float))
        assert -15 <= result <= 15
    
    def test_net_rating_no_data(self):
        """Testa com todos os dados faltando."""
        stats = {
            'RAPM': None,
            'BPM': None,
            'PIE': None,
            'LEBRON': None
        }
        
        result = calcular_net_rating_v11('Lakers', stats)
        
        # Deve retornar 0 ou valor default
        assert result == 0.0
    
    def test_net_rating_extreme_values(self):
        """Testa com valores extremos."""
        stats = {
            'RAPM': 10.0,  # Muito alto
            'BPM': 12.0,   # Muito alto
            'PIE': 0.95,   # Muito alto
            'LEBRON': 8.0  # Muito alto
        }
        
        result = calcular_net_rating_v11('Lakers', stats)
        
        # Mesmo com valores extremos, deve estar no range
        assert -15 <= result <= 15
    
    def test_net_rating_consistency(self):
        """Testa que mesmos inputs geram mesmo output."""
        stats = {
            'RAPM': 3.5,
            'BPM': 4.2,
            'PIE': 0.58,
            'LEBRON': 2.1
        }
        
        result1 = calcular_net_rating_v11('Lakers', stats)
        result2 = calcular_net_rating_v11('Lakers', stats)
        
        assert result1 == result2


class TestCalcularPowerRatingV11:
    """Testes para calcular_power_rating_v11()."""
    
    def test_power_rating_basic(self):
        """Testa cálculo básico de Power Rating."""
        team_stats = {
            'Lakers': {
                'net_rating': 5.2,
                'injury_factor': 0.95,
                'rest_days': 2,
                'is_home': True
            }
        }
        
        result = calcular_power_rating_v11('Lakers', team_stats)
        
        # Deve retornar dict com keys esperadas
        assert isinstance(result, dict)
        assert 'power_rating' in result
        assert 'net_rating_adjusted' in result
        
        # Power rating deve ser numérico
        assert isinstance(result['power_rating'], (int, float))
    
    def test_power_rating_home_advantage(self):
        """Testa que home advantage é aplicado."""
        team_stats = {
            'Lakers': {
                'net_rating': 5.0,
                'injury_factor': 1.0,
                'rest_days': 1,
                'is_home': True
            }
        }
        
        home_result = calcular_power_rating_v11('Lakers', team_stats)
        
        # Mudar para away
        team_stats['Lakers']['is_home'] = False
        away_result = calcular_power_rating_v11('Lakers', team_stats)
        
        # Home deve ter power rating maior
        assert home_result['power_rating'] > away_result['power_rating']
    
    def test_power_rating_injury_impact(self):
        """Testa impacto de lesões."""
        team_stats_healthy = {
            'Lakers': {
                'net_rating': 5.0,
                'injury_factor': 1.0,  # Totalmente saudável
                'rest_days': 1,
                'is_home': True
            }
        }
        
        team_stats_injured = {
            'Lakers': {
                'net_rating': 5.0,
                'injury_factor': 0.7,  # 30% de força perdida
                'rest_days': 1,
                'is_home': True
            }
        }
        
        healthy_result = calcular_power_rating_v11('Lakers', team_stats_healthy)
        injured_result = calcular_power_rating_v11('Lakers', team_stats_injured)
        
        # Lesões devem diminuir power rating
        assert healthy_result['power_rating'] > injured_result['power_rating']
    
    def test_power_rating_rest_days(self):
        """Testa impacto de dias de descanso."""
        team_stats_rested = {
            'Lakers': {
                'net_rating': 5.0,
                'injury_factor': 1.0,
                'rest_days': 3,  # Bem descansado
                'is_home': True
            }
        }
        
        team_stats_tired = {
            'Lakers': {
                'net_rating': 5.0,
                'injury_factor': 1.0,
                'rest_days': 0,  # Back-to-back
                'is_home': True
            }
        }
        
        rested_result = calcular_power_rating_v11('Lakers', team_stats_rested)
        tired_result = calcular_power_rating_v11('Lakers', team_stats_tired)
        
        # Mais descanso = melhor rating
        assert rested_result['power_rating'] >= tired_result['power_rating']


class TestCalcularExpectedEFG:
    """Testes para calcular_expected_efg()."""
    
    def test_expected_efg_valid(self):
        """Testa cálculo de Expected eFG%."""
        team_stats = {
            'eFG%': 0.54,
            'opponent_deFG%': 0.52,
            'shot_quality': 1.02
        }
        
        result = calcular_expected_efg(team_stats)
        
        # Deve retornar percentual válido
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_expected_efg_no_data(self):
        """Testa com dados faltando."""
        result = calcular_expected_efg({})
        
        # Deve retornar valor default (0.5 ou league average)
        assert result is not None
        assert 0.4 <= result <= 0.6


class TestCalcularShotQuality:
    """Testes para calcular_shot_quality_adjustment()."""
    
    def test_shot_quality_basic(self):
        """Testa cálculo básico de Shot Quality."""
        team_data = {
            '3PA': 35.0,
            'FTA': 22.0,
            'assisted_2PM': 15.0,
            'total_2PM': 20.0
        }
        
        result = calcular_shot_quality_adjustment(team_data)
        
        # Deve retornar multiplicador próximo de 1.0
        assert isinstance(result, float)
        assert 0.8 <= result <= 1.2
    
    def test_shot_quality_high_quality(self):
        """Testa com shot selection de alta qualidade."""
        high_quality = {
            '3PA': 40.0,  # Muitos 3s
            'FTA': 25.0,  # Muitos free throws
            'assisted_2PM': 18.0,
            'total_2PM': 20.0  # Alta % de assistências
        }
        
        result = calcular_shot_quality_adjustment(high_quality)
        
        # Alta qualidade = multiplicador > 1.0
        assert result >= 1.0


class TestEdgeCases:
    """Testes para casos extremos."""
    
    def test_nan_handling(self):
        """Testa que NaN é tratado corretamente."""
        stats_with_nan = {
            'RAPM': np.nan,
            'BPM': 4.2,
            'PIE': 0.58,
            'LEBRON': np.nan
        }
        
        result = calcular_net_rating_v11('Lakers', stats_with_nan)
        
        # Não deve retornar NaN
        assert not np.isnan(result)
        assert isinstance(result, (int, float))
    
    def test_zero_values(self):
        """Testa com todos os valores zero."""
        stats_zero = {
            'RAPM': 0.0,
            'BPM': 0.0,
            'PIE': 0.0,
            'LEBRON': 0.0
        }
        
        result = calcular_net_rating_v11('Lakers', stats_zero)
        
        # Deve retornar 0 ou próximo
        assert abs(result) < 1.0
    
    def test_negative_values(self):
        """Testa com valores negativos."""
        stats_negative = {
            'RAPM': -3.5,
            'BPM': -2.1,
            'PIE': 0.35,  # PIE não pode ser negativo
            'LEBRON': -1.8
        }
        
        result = calcular_net_rating_v11('Lakers', stats_negative)
        
        # Deve aceitar valores negativos
        assert isinstance(result, (int, float))
        assert result < 0  # Resultado deve ser negativo


# Integration test
def test_full_rating_pipeline():
    """Teste end-to-end do pipeline de rating."""
    # Dados simulados
    team_stats = {
        'Lakers': {
            'RAPM': 3.5,
            'BPM': 4.2,
            'PIE': 0.58,
            'LEBRON': 2.1
        }
    }
    
    # 1. Calcular Net Rating
    net_rating = calcular_net_rating_v11('Lakers', team_stats['Lakers'])
    assert isinstance(net_rating, (int, float))
    
    # 2. Preparar dados para Power Rating
    power_stats = {
        'Lakers': {
            'net_rating': net_rating,
            'injury_factor': 0.95,
            'rest_days': 2,
            'is_home': True
        }
    }
    
    # 3. Calcular Power Rating
    power_rating = calcular_power_rating_v11('Lakers', power_stats)
    
    assert isinstance(power_rating, dict)
    assert 'power_rating' in power_rating
    assert isinstance(power_rating['power_rating'], (int, float))
    
    # Power rating deve incorporar net rating
    assert power_rating['power_rating'] != 0
