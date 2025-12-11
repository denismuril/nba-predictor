"""
Testes para validar player features e V21 FIX.
"""
import pytest
import sys
import pandas as pd

sys.path.insert(0, '.')


class TestPlayerAggregation:
    """Testes para o módulo player_aggregation."""

    def test_import_player_aggregation(self):
        """Teste 1: Import do módulo player_aggregation."""
        from ml_pipeline.player_aggregation import aggregate_player_stats_by_team
        assert aggregate_player_stats_by_team is not None

    def test_aggregation_with_mock_data(self):
        """Teste 2: Agregação com dados mock."""
        from ml_pipeline.player_aggregation import aggregate_player_stats_by_team

        df_mock = pd.DataFrame({
            'Player': ['LeBron James', 'Anthony Davis', 'Austin Reaves'],
            'Team': ['LAL', 'LAL', 'LAL'],
            'RAPM': [5.2, 3.8, 1.2],
            'ORAPM': [3.1, 2.5, 0.8],
            'DRAPM': [2.1, 1.3, 0.4],
            'BPM': [8.5, 6.2, 2.1],
            'MP': [35.0, 33.5, 28.0]
        })

        result = aggregate_player_stats_by_team(df_mock, top_n=3)

        assert not result.empty
        assert result['Team'].values[0] == 'LAL'
        assert 'rapm_avg' in result.columns
        assert 'rapm_top' in result.columns
        assert 'depth_score' in result.columns
        assert result['rapm_avg'].values[0] == pytest.approx(3.4, rel=0.1)

    def test_load_historical_data(self):
        """Teste 3: Load historical data."""
        from ml_pipeline.data_preparation import load_historical_data

        df_hist = load_historical_data(
            seasons=['2024-25'],
            apply_weights=False,
            enable_player_features=False
        )

        assert df_hist is not None
        assert not df_hist.empty
        assert len(df_hist) > 100  # Deve ter pelo menos 100 jogos


class TestV21Fixes:
    """Testes para validar as correções V21."""

    def test_expanding_window_in_opponent_stats(self):
        """Teste V21: Expanding window no opponent_adjusted_stats."""
        from ml_pipeline.opponent_adjusted_stats import calcular_stats_ajustados_oponente

        # Criar dados mock
        df_mock = pd.DataFrame({
            'date': pd.date_range('2024-10-22', periods=20),
            'home_team': ['LAL', 'BOS'] * 10,
            'away_team': ['DEN', 'MIA'] * 10,
            'home_off_rating': [115.0] * 20,
            'away_off_rating': [110.0] * 20,
            'home_def_rating': [108.0] * 20,
            'away_def_rating': [112.0] * 20,
        })

        result = calcular_stats_ajustados_oponente(df_mock)

        assert 'liga_ortg_avg' in result.columns
        assert 'liga_drtg_avg' in result.columns
        assert 'home_ortg_adj' in result.columns
        assert 'away_ortg_adj' in result.columns

        # V21 FIX: Primeiro jogo deve ter fallback 112.0
        assert result['liga_ortg_avg'].iloc[0] == pytest.approx(112.0, rel=0.01)

    def test_odds_shopping_ttl(self):
        """Teste V21: TTL parameter no compare_lines."""
        from market.odds_shopping import compare_lines

        # Verificar que a função aceita max_age_minutes
        import inspect
        sig = inspect.signature(compare_lines)
        assert 'max_age_minutes' in sig.parameters

    def test_predict_fail_fast(self):
        """Teste V21: FAIL FAST no predict.py."""
        from ml_pipeline.predict import predict_next_games

        # Função deve existir e ser importável
        assert predict_next_games is not None
