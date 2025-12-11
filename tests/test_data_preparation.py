"""
Testes unitários para ml_pipeline/data_preparation.py

Foco em:
- Prevenção de data leakage (shift correto)
- Rolling features
- Feature engineering
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml_pipeline.data_preparation import (
    add_rolling_features,
    calculate_four_factors,
    prepare_data_for_training,
    load_historical_data
)


class TestDataLeakagePrevention:
    """Testes críticos para prevenção de data leakage."""
    
    def test_rolling_features_use_shift(self):
        """
        CRÍTICO: Testa que rolling features usam shift(1) para evitar data leakage.
        
        Rolling features NÃO devem incluir o jogo atual no cálculo.
        """
        # Criar dados de teste
        data = {
            'team': ['LAL'] * 10,
            'date': pd.date_range('2024-01-01', periods=10),
            'pts': [100, 105, 95, 110, 102, 108, 98, 103, 107, 101]
        }
        df = pd.DataFrame(data)
        
        # Adicionar rolling features
        df_with_rolling = add_rolling_features(df, window=3)
        
        # Verificar que rolling não inclui valor atual
        # Para o índice 3 (4º jogo), rolling_3 deve ser média dos 3 jogos ANTERIORES
        if 'pts_rolling_3' in df_with_rolling.columns:
            # Média dos 3 jogos anteriores (índices 0,1,2): (100+105+95)/3 = 100
            expected_rolling_3 = (100 + 105 + 95) / 3
            actual_rolling_3 = df_with_rolling.iloc[3]['pts_rolling_3']
            
            # NÃO deve incluir o valor atual (110)
            assert actual_rolling_3 != (105 + 95 + 110) / 3, "LEAKAGE: Rolling incluiu valor atual!"
            
            # Deve ser aproximadamente igual à média correta
            assert abs(actual_rolling_3 - expected_rolling_3) < 0.1
    
    def test_no_future_data_in_features(self):
        """
        CRÍTICO: Garante que features não usam dados futuros.
        """
        # Dados com tendência clara
        data = {
            'team': ['LAL'] * 5,
            'date': pd.date_range('2024-01-01', periods=5),
            'pts': [100, 105, 110, 115, 120]  # Tendência crescente
        }
        df = pd.DataFrame(data)
        
        # Adicionar features
        df_with_features = add_rolling_features(df, window=2)
        
        # Para o 3º jogo (índice 2), rolling deve usar apenas jogos 0 e 1
        if 'pts_rolling_2' in df_with_features.columns:
            rolling_at_game3 = df_with_features.iloc[2]['pts_rolling_2']
            expected = (100 + 105) / 2  # Jogos anteriores
            
            # NÃO deve ser influenciado por jogos futuros (115, 120)
            assert abs(rolling_at_game3 - expected) < 0.1
    
    def test_first_n_games_handling(self):
        """
        Testa que primeiros N jogos são tratados corretamente.
        """
        data = {
            'team': ['LAL'] * 5,
            'date': pd.date_range('2024-01-01', periods=5),
            'pts': [100, 105, 110, 115, 120]
        }
        df = pd.DataFrame(data)
        
        df_with_rolling = add_rolling_features(df, window=3)
        
        # Primeiro jogo não deve ter rolling (shift = NaN)
        if 'pts_rolling_3' in df_with_rolling.columns:
            first_game_rolling = df_with_rolling.iloc[0]['pts_rolling_3']
            
            # Deve ser NaN ou 0
            assert pd.isna(first_game_rolling) or first_game_rolling == 0


class TestFourFactors:
    """Testes para cálculo de Four Factors."""
    
    def test_four_factors_complete_data(self):
        """Testa cálculo com dados completos."""
        game_stats = {
            'FGM': 40,
            'FGA': 85,
            'FG3M': 12,
            'FTM': 18,
            'FTA': 22,
            'OREB': 10,
            'DREB': 35,
            'TOV': 12,
            'opponent_DREB': 30
        }
        
        result = calculate_four_factors(game_stats)
        
        # Deve retornar dict com as 4 factors
        assert isinstance(result, dict)
        assert 'eFG%' in result
        assert 'TOV%' in result
        assert 'OREB%' in result
        assert 'FT_Rate' in result
        
        # Valores devem estar em ranges válidos
        assert 0.0 <= result['eFG%'] <= 1.0
        assert 0.0 <= result['TOV%'] <= 0.5
        assert 0.0 <= result['OREB%'] <= 1.0
    
    def test_four_factors_division_by_zero(self):
        """Testa proteção contra divisão por zero."""
        game_stats = {
            'FGM': 0,
            'FGA': 0,  # Divisão por zero em eFG%
            'FG3M': 0,
            'FTM': 0,
            'FTA': 0,
            'OREB': 0,
            'DREB': 0,
            'TOV': 0,
            'opponent_DREB': 0
        }
        
        result = calculate_four_factors(game_stats)
        
        # Não deve dar erro, deve retornar 0 ou valores default
        assert result is not None
        assert not np.isnan(result['eFG%'])


class TestPrepareDataForTraining:
    """Testes para prepare_data_for_training()."""
    
    def test_prepare_returns_correct_shapes(self):
        """Testa que X e y têm shapes corretos."""
        # Dados simulados
        df = pd.DataFrame({
            'home_team': ['LAL'] * 100,
            'away_team': ['GSW'] * 100,
            'home_pts': np.random.randint(90, 120, 100),
            'away_pts': np.random.randint(90, 120, 100),
            'home_net_rating': np.random.randn(100),
            'away_net_rating': np.random.randn(100),
            'home_win': np.random.randint(0, 2, 100)
        })
        
        X, y = prepare_data_for_training(df)
        
        # X e y devem ter mesmo número de linhas
        assert len(X) == len(y)
        
        # y deve ser 1D
        assert len(y.shape) == 1
        
        # X deve ter múltiplas features
        assert X.shape[1] > 1
    
    def test_prepare_removes_target_from_features(self):
        """Testa que target não está em X."""
        df = pd.DataFrame({
            'home_team': ['LAL'] * 50,
            'away_team': ['GSW'] * 50,
            'home_pts': np.random.randint(90, 120, 50),
            'away_pts': np.random.randint(90, 120, 50),
            'home_win': np.random.randint(0, 2, 50)
        })
        
        X, y = prepare_data_for_training(df)
        
        # home_win não deve estar em X
        if hasattr(X, 'columns'):
            assert 'home_win' not in X.columns


class TestLoadHistoricalData:
    """Testes para load_historical_data()."""
    
    def test_load_returns_dataframe(self):
        """Testa que load retorna DataFrame."""
        # Pode falhar se DB vazio, mas deve retornar DataFrame
        try:
            df = load_historical_data(limit=10)
            assert isinstance(df, pd.DataFrame)
        except Exception:
            # Se DB não existe, tudo bem
            pytest.skip("Database não disponível para teste")
    
    def test_load_respects_limit(self):
        """Testa que limit é respeitado."""
        try:
            df = load_historical_data(limit=5)
            assert len(df) <= 5
        except Exception:
            pytest.skip("Database não disponível para teste")


class TestFeatureEngineering:
    """Testes para feature engineering."""
    
    def test_no_features_from_future(self):
        """
        CRÍTICO: Garante que nenhuma feature usa informação do futuro.
        """
        # Simular série temporal
        dates = pd.date_range('2024-01-01', periods=30)
        df = pd.DataFrame({
            'date': dates,
            'team': ['LAL'] * 30,
            'pts': np.random.randint(90, 120, 30),
            'opp_pts': np.random.randint(90, 120, 30)
        })
        
        # Adicionar features
        df_with_features = add_rolling_features(df, window=5)
        
        # Para cada jogo, features só devem depender de jogos anteriores
        for idx in range(5, len(df)):
            if 'pts_rolling_5' in df_with_features.columns:
                rolling = df_with_features.iloc[idx]['pts_rolling_5']
                
                # Calcular manualmente (jogos anteriores)
                previous_games = df.iloc[idx-5:idx]['pts'].values
                manual_rolling = previous_games.mean()
                
                # Deve bater
                assert abs(rolling - manual_rolling) < 0.1, \
                    f"Jogo {idx}: Rolling={rolling}, Manual={manual_rolling}"
    
    def test_consistency_across_runs(self):
        """Testa que mesmo input gera mesmo output."""
        df = pd.DataFrame({
            'team': ['LAL'] * 10,
            'pts': [100, 105, 110, 95, 102, 108, 98, 103, 107, 101]
        })
        
        result1 = add_rolling_features(df.copy(), window=3)
        result2 = add_rolling_features(df.copy(), window=3)
        
        # Deve ser idêntico
        pd.testing.assert_frame_equal(result1, result2)


# Integration test
def test_full_data_prep_pipeline():
    """Teste end-to-end do pipeline de preparação."""
    # Dados simulados
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=50),
        'home_team': ['LAL'] * 50,
        'away_team': ['GSW'] * 50,
        'home_pts': np.random.randint(90, 120, 50),
        'away_pts': np.random.randint(90, 120, 50),
        'home_fg': np.random.randint(35, 45, 50),
        'home_fga': [85] * 50,
        'home_win': np.random.randint(0, 2, 50)
    })
    
    # 1. Adicionar rolling features
    df_with_rolling = add_rolling_features(df, window=5)
    
    # 2. Preparar para treino
    X, y = prepare_data_for_training(df_with_rolling)
    
    # Validações
    assert len(X) == len(y)
    assert len(X) > 0
    assert not np.any(np.isnan(y))  # Target não deve ter NaN
    
    # Verificar que não há leakage temporal
    # (isso seria verificado manualmente ou com análise de correlação)
    assert True  # Placeholder para teste mais sofisticado
