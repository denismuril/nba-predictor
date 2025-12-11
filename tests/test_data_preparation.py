"""
Testes unitários para ml_pipeline/data_preparation.py

Foco em:
- Prevenção de data leakage (shift correto)
- Rolling features
- Feature engineering

ATUALIZADO v21.5: Assinaturas atualizadas para corresponder ao código de produção
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml_pipeline.data_preparation import (
    add_rolling_features,
    calculate_four_factors,
    prepare_data_for_training
)


class TestDataLeakagePrevention:
    """Testes críticos para prevenção de data leakage."""
    
    def test_rolling_features_use_shift(self):
        """
        CRÍTICO: Testa que rolling features usam shift(1) para evitar data leakage.
        
        Rolling features NÃO devem incluir o jogo atual no cálculo.
        """
        # Criar dados de teste com estrutura correta para add_rolling_features
        dates = pd.date_range('2024-01-01', periods=20)
        data = {
            'date': dates,
            'home_team': ['LAL'] * 20,
            'away_team': ['GSW'] * 20,
            'home_score': [100 + i for i in range(20)],
            'away_score': [95 + i for i in range(20)],
            'home_efg': [0.5] * 20,
            'away_efg': [0.5] * 20,
            'home_tov_pct': [0.12] * 20,
            'away_tov_pct': [0.12] * 20,
            'home_orb_pct': [0.25] * 20,
            'away_orb_pct': [0.25] * 20,
            'home_ftr': [0.25] * 20,
            'away_ftr': [0.25] * 20,
        }
        df = pd.DataFrame(data)
        
        # Adicionar rolling features (usa windows=[5, 10] internamente)
        df_with_rolling = add_rolling_features(df, windows=[5])
        
        # Verificar que rolling existe
        rolling_cols = [c for c in df_with_rolling.columns if 'rolling' in c]
        assert len(rolling_cols) > 0, "Rolling features não foram criadas"
        
        # Verificar que primeiro valor é NaN (shift aplicado)
        sample_col = [c for c in rolling_cols if 'home' in c][0]
        assert pd.isna(df_with_rolling.iloc[0][sample_col]), \
            "LEAKAGE: Primeiro valor deveria ser NaN (shift não aplicado)"
    
    def test_no_future_data_in_features(self):
        """
        CRÍTICO: Garante que features não usam dados futuros.
        """
        # Dados com tendência clara
        dates = pd.date_range('2024-01-01', periods=15)
        data = {
            'date': dates,
            'home_team': ['LAL'] * 15,
            'away_team': ['GSW'] * 15,
            'home_score': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170],
            'away_score': [90] * 15,
            'home_efg': [0.5] * 15,
            'away_efg': [0.5] * 15,
            'home_tov_pct': [0.12] * 15,
            'away_tov_pct': [0.12] * 15,
            'home_orb_pct': [0.25] * 15,
            'away_orb_pct': [0.25] * 15,
            'home_ftr': [0.25] * 15,
            'away_ftr': [0.25] * 15,
        }
        df = pd.DataFrame(data)
        
        # Adicionar features
        df_with_features = add_rolling_features(df, windows=[5])
        
        # A rolling no jogo 6 (índice 5) deve usar apenas jogos 0-4 (100-120)
        # e NÃO os jogos futuros (125+)
        if 'home_rolling_5_points' in df_with_features.columns:
            rolling_at_game6 = df_with_features.iloc[5]['home_rolling_5_points']
            # Média de 100,105,110,115,120 com EWMA (não é média simples)
            # Mas definitivamente não deve incluir 125, 130, etc.
            assert rolling_at_game6 < 122, f"Rolling={rolling_at_game6} parece incluir dados futuros!"


class TestFourFactors:
    """Testes para cálculo de Four Factors."""
    
    def test_four_factors_complete_data(self):
        """Testa cálculo com DataFrame de dados completos."""
        # calculate_four_factors espera um DataFrame, não dict
        df = pd.DataFrame({
            'fgm': [40],
            'fga': [85],
            'fg3m': [12],
            'fta': [22],
            'tov': [12],
            'oreb': [10],
            'dreb': [35],
            'opp_fgm': [38],
            'opp_fga': [82],
            'opp_fg3m': [10],
            'opp_fta': [20],
            'opp_tov': [14],
            'opp_oreb': [8],
            'opp_dreb': [32],
            'home_score': [110],
            'away_score': [105],
        })
        
        result = calculate_four_factors(df)
        
        # Deve retornar DataFrame com as colunas de Four Factors
        assert isinstance(result, pd.DataFrame)
        assert 'home_efg' in result.columns
        assert 'home_tov_pct' in result.columns
        assert 'home_orb_pct' in result.columns
        assert 'home_ftr' in result.columns
        
        # Valores devem estar em ranges válidos
        assert 0.0 <= result['home_efg'].iloc[0] <= 1.0
        assert 0.0 <= result['home_tov_pct'].iloc[0] <= 0.5
        assert 0.0 <= result['home_orb_pct'].iloc[0] <= 1.0
    
    def test_four_factors_fallback_without_detailed_data(self):
        """Testa fallback quando dados detalhados não existem."""
        # Apenas scores, sem box score detalhado
        df = pd.DataFrame({
            'home_score': [110],
            'away_score': [105],
        })
        
        result = calculate_four_factors(df)
        
        # Não deve dar erro, deve usar valores default
        assert result is not None
        assert 'home_efg' in result.columns
        # Valores default são 0.5 para efg
        assert result['home_efg'].iloc[0] == 0.5


class TestPrepareDataForTraining:
    """Testes para prepare_data_for_training()."""
    
    def test_prepare_returns_correct_shapes(self):
        """Testa que X e y têm shapes corretos."""
        # Dados simulados com features seguras (rolling, elo)
        df = pd.DataFrame({
            'home_team': ['LAL'] * 100,
            'away_team': ['GSW'] * 100,
            'home_rolling_10_points': np.random.randn(100) + 110,
            'away_rolling_10_points': np.random.randn(100) + 105,
            'home_elo': np.random.randn(100) + 1500,
            'away_elo': np.random.randn(100) + 1500,
            'elo_diff': np.random.randn(100) * 50,
            'winner': np.random.choice(['HOME', 'AWAY'], 100)
        })
        
        X, y = prepare_data_for_training(df, target='winner')
        
        # X deve ter features numéricas
        assert len(X.columns) > 0, "X não tem nenhuma feature"
        
        # y deve existir
        assert y is not None, "y não foi retornado"
        
        # X e y devem ter mesmo número de linhas
        assert len(X) == len(y)
    
    def test_prepare_removes_target_from_features(self):
        """Testa que target não está em X."""
        df = pd.DataFrame({
            'home_team': ['LAL'] * 50,
            'away_team': ['GSW'] * 50,
            'home_rolling_10_points': np.random.randn(50) + 110,
            'away_rolling_10_points': np.random.randn(50) + 105,
            'home_elo': np.random.randn(50) + 1500,
            'away_elo': np.random.randn(50) + 1500,
            'winner': np.random.choice(['HOME', 'AWAY'], 50)
        })
        
        X, y = prepare_data_for_training(df, target='winner')
        
        # winner não deve estar em X
        if hasattr(X, 'columns'):
            assert 'winner' not in X.columns


class TestFeatureEngineering:
    """Testes para feature engineering."""
    
    def test_rolling_creates_expected_columns(self):
        """Testa que rolling cria as colunas esperadas."""
        dates = pd.date_range('2024-01-01', periods=20)
        data = {
            'date': dates,
            'home_team': ['LAL'] * 20,
            'away_team': ['GSW'] * 20,
            'home_score': [100 + i for i in range(20)],
            'away_score': [95 + i for i in range(20)],
            'home_efg': [0.5] * 20,
            'away_efg': [0.5] * 20,
            'home_tov_pct': [0.12] * 20,
            'away_tov_pct': [0.12] * 20,
            'home_orb_pct': [0.25] * 20,
            'away_orb_pct': [0.25] * 20,
            'home_ftr': [0.25] * 20,
            'away_ftr': [0.25] * 20,
        }
        df = pd.DataFrame(data)
        
        # Adicionar rolling com janela de 5
        df_with_features = add_rolling_features(df, windows=[5])
        
        # Deve criar colunas home_rolling_5_* e away_rolling_5_*
        expected_patterns = ['home_rolling_5', 'away_rolling_5']
        for pattern in expected_patterns:
            cols_with_pattern = [c for c in df_with_features.columns if pattern in c]
            assert len(cols_with_pattern) > 0, f"Colunas com '{pattern}' não encontradas"
    
    def test_consistency_across_runs(self):
        """Testa que mesmo input gera mesmo output."""
        dates = pd.date_range('2024-01-01', periods=15)
        data = {
            'date': dates,
            'home_team': ['LAL'] * 15,
            'away_team': ['GSW'] * 15,
            'home_score': [100 + i for i in range(15)],
            'away_score': [95 + i for i in range(15)],
            'home_efg': [0.5] * 15,
            'away_efg': [0.5] * 15,
            'home_tov_pct': [0.12] * 15,
            'away_tov_pct': [0.12] * 15,
            'home_orb_pct': [0.25] * 15,
            'away_orb_pct': [0.25] * 15,
            'home_ftr': [0.25] * 15,
            'away_ftr': [0.25] * 15,
        }
        df = pd.DataFrame(data)
        
        result1 = add_rolling_features(df.copy(), windows=[5])
        result2 = add_rolling_features(df.copy(), windows=[5])
        
        # Deve ser idêntico
        pd.testing.assert_frame_equal(result1, result2)


# Integration test
def test_full_data_prep_pipeline():
    """Teste end-to-end do pipeline de preparação."""
    dates = pd.date_range('2024-01-01', periods=50)
    df = pd.DataFrame({
        'date': dates,
        'home_team': ['LAL'] * 50,
        'away_team': ['GSW'] * 50,
        'home_score': np.random.randint(90, 120, 50),
        'away_score': np.random.randint(90, 120, 50),
        'home_efg': [0.5] * 50,
        'away_efg': [0.5] * 50,
        'home_tov_pct': [0.12] * 50,
        'away_tov_pct': [0.12] * 50,
        'home_orb_pct': [0.25] * 50,
        'away_orb_pct': [0.25] * 50,
        'home_ftr': [0.25] * 50,
        'away_ftr': [0.25] * 50,
        'winner': np.random.choice(['HOME', 'AWAY'], 50)
    })
    
    # 1. Adicionar rolling features
    df_with_rolling = add_rolling_features(df, windows=[5, 10])
    
    # Verificar que rolling features foram criadas
    rolling_cols = [c for c in df_with_rolling.columns if 'rolling' in c]
    assert len(rolling_cols) > 0, "Rolling features não foram criadas"
    
    # 2. Preparar para treino
    X, y = prepare_data_for_training(df_with_rolling, target='winner')
    
    # Validações
    assert X is not None, "X não foi retornado"
    assert len(X.columns) > 0, "X não tem features"
