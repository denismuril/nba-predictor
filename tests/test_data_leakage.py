"""
Testes de Data Leakage - Validação Rigorosa

Este módulo contém testes que DEVEM FALHAR se o modelo tentar usar dados do futuro.
Expande os testes existentes em test_leakage.py com cenários mais rigorosos.

v27.0: Implementação inicial com:
- test_rolling_features_no_future_data: Valida shift(1)
- test_feature_date_boundaries: Simula cenário real de previsão
- test_model_train_test_temporal_integrity: Verifica TimeSeriesSplit
- test_features_at_prediction_time: Testa pipeline completo em modo previsão

REGRA:
    Qualquer feature para prever o jogo T deve usar APENAS dados de jogos < T.
    Se qualquer teste falhar, o modelo está INVÁLIDO.

Uso:
    pytest tests/test_data_leakage.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit

# Importar módulos a testar
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.features import (
    create_rolling_features,
    create_season_avg_features,
    create_ema_features,
    create_all_features,
    validate_temporal_integrity,
    DataLeakageError,
    anti_leakage
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_game_data():
    """
    Cria dados de jogos simulados para teste.
    
    Estrutura: 30 jogos para Lakers, ordenados por data.
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'team': ['LAL'] * 30,
        'pts': np.cumsum(np.random.randn(30) * 5 + 115).astype(int),  # Trending data
        'reb': np.random.randint(40, 55, 30),
        'ast': np.random.randint(20, 35, 30),
        'opp_team': ['OPP'] * 30,
        'home_win': np.random.randint(0, 2, 30),
        'season': '2023-24'
    })
    
    return df


@pytest.fixture
def multi_team_data():
    """Dados com múltiplos times."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    
    teams = ['LAL', 'BOS', 'GSW']
    dfs = []
    
    for team in teams:
        team_df = pd.DataFrame({
            'date': dates,
            'team': team,
            'pts': np.random.randint(100, 130, 20),
            'reb': np.random.randint(40, 55, 20),
            'season': '2023-24'
        })
        dfs.append(team_df)
    
    return pd.concat(dfs, ignore_index=True).sort_values('date')


# =============================================================================
# TESTES DE ROLLING FEATURES
# =============================================================================

class TestRollingFeaturesNoLeakage:
    """Testes para garantir que rolling features não vazam dados do futuro."""
    
    def test_first_row_should_be_nan(self, sample_game_data):
        """
        Primeiro jogo DEVE ter NaN nas features rolling.
        
        Rationale: Não há jogos anteriores para calcular a média.
        Se não for NaN, há leakage.
        """
        df = create_rolling_features(
            sample_game_data, 
            windows=[5], 
            stats_cols=['pts'],
            aggregations=['mean']
        )
        
        first_row = df.iloc[0]
        rolling_col = 'pts_rolling_5_mean'
        
        assert pd.isna(first_row[rolling_col]), (
            f"❌ VAZAMENTO! Primeiro jogo não deveria ter {rolling_col}. "
            f"Valor encontrado: {first_row[rolling_col]}"
        )
    
    def test_rolling_uses_only_past_data(self, sample_game_data):
        """
        Rolling feature de um jogo DEVE refletir APENAS jogos anteriores.
        
        Exemplo: Se pontos foram [100, 110, 120] nos dias 1-3,
        então rolling_1_mean do dia 3 deve ser 110 (apenas dia 2),
        NÃO 120 (que é o próprio dia 3).
        """
        # Criar dados controlados
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'team': ['LAL', 'LAL', 'LAL'],
            'pts': [100, 110, 120],  # Valores conhecidos
        })
        
        df = create_rolling_features(df, windows=[1], stats_cols=['pts'], 
                                    aggregations=['mean'])
        
        # Dia 1: Deve ser NaN (sem histórico)
        assert pd.isna(df.iloc[0]['pts_rolling_1_mean']), (
            "❌ Dia 1 deveria ser NaN (sem passado)"
        )
        
        # Dia 2: Deve ver APENAS dia 1 (100)
        assert df.iloc[1]['pts_rolling_1_mean'] == 100, (
            f"❌ Dia 2 deveria ver 100 (dia 1), viu {df.iloc[1]['pts_rolling_1_mean']}"
        )
        
        # Dia 3: Deve ver APENAS dia 2 (110)
        assert df.iloc[2]['pts_rolling_1_mean'] == 110, (
            f"❌ Dia 3 deveria ver 110 (dia 2), viu {df.iloc[2]['pts_rolling_1_mean']}"
        )
    
    def test_window_5_uses_previous_5_games_only(self, sample_game_data):
        """
        Rolling-5 do jogo N deve usar APENAS jogos N-5 até N-1.
        
        Nunca deve incluir o jogo N.
        """
        df = create_rolling_features(
            sample_game_data,
            windows=[5],
            stats_cols=['pts'],
            aggregations=['mean']
        )
        
        # Verificar jogo 10 (índice 9)
        game_10_rolling = df.iloc[9]['pts_rolling_5_mean']
        
        # Calcular manualmente: média dos jogos 5-9 (índices 4-8)
        expected = sample_game_data.iloc[4:9]['pts'].mean()
        
        assert abs(game_10_rolling - expected) < 0.01, (
            f"❌ VAZAMENTO! Jogo 10 deveria ter rolling={expected:.2f}, "
            f"mas tem {game_10_rolling:.2f}"
        )


# =============================================================================
# TESTES DE SEASON AVG FEATURES
# =============================================================================

class TestSeasonAvgNoLeakage:
    """Testes para médias da temporada."""
    
    def test_season_avg_excludes_current_game(self, sample_game_data):
        """
        Média da temporada no jogo N deve excluir os pontos do jogo N.
        """
        df = create_season_avg_features(
            sample_game_data,
            stats_cols=['pts']
        )
        
        # Verificar jogo 5 (índice 4)
        game_5_avg = df.iloc[4]['pts_season_avg']
        
        # Calcular manualmente: média dos jogos 1-4 (índices 0-3)
        expected = sample_game_data.iloc[0:4]['pts'].mean()
        
        assert abs(game_5_avg - expected) < 0.01, (
            f"❌ VAZAMENTO! Season avg do jogo 5 deveria ser {expected:.2f}, "
            f"mas é {game_5_avg:.2f}"
        )


# =============================================================================
# TESTES DE EMA FEATURES
# =============================================================================

class TestEMANoLeakage:
    """Testes para EMAs."""
    
    def test_ema_first_row_is_nan(self, sample_game_data):
        """Primeiro jogo deve ter EMA como NaN."""
        df = create_ema_features(
            sample_game_data,
            spans=[10],
            stats_cols=['pts']
        )
        
        assert pd.isna(df.iloc[0]['pts_ema_10']), (
            "❌ Primeiro jogo deveria ter EMA como NaN"
        )


# =============================================================================
# TESTES DE INTEGRIDADE TEMPORAL
# =============================================================================

class TestTemporalIntegrity:
    """Testes para validar split temporal correto."""
    
    def test_train_before_test(self, sample_game_data):
        """TimeSeriesSplit deve garantir train ANTES de test."""
        tscv = TimeSeriesSplit(n_splits=3)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(sample_game_data)):
            train_dates = sample_game_data.iloc[train_idx]['date']
            test_dates = sample_game_data.iloc[test_idx]['date']
            
            assert train_dates.max() < test_dates.min(), (
                f"❌ Fold {fold}: Treino contém datas DEPOIS de teste! "
                f"Train max: {train_dates.max()}, Test min: {test_dates.min()}"
            )
    
    def test_validate_temporal_integrity_passes_valid(self):
        """Validador deve passar quando datas são válidas."""
        df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=30)})
        
        # Deve passar sem exceção
        assert validate_temporal_integrity(df, "2024-01-20", "2024-01-21")
    
    def test_validate_temporal_integrity_fails_overlap(self):
        """Validador deve falhar quando há sobreposição."""
        df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=30)})
        
        with pytest.raises(DataLeakageError):
            validate_temporal_integrity(df, "2024-01-25", "2024-01-20")  # Overlap!
    
    def test_no_overlap_in_actual_split(self, sample_game_data):
        """Verificar que não há sobreposição de índices entre folds."""
        tscv = TimeSeriesSplit(n_splits=3)
        
        all_test_indices = set()
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(sample_game_data)):
            # Verificar que não há índice repetido em diferentes folds de teste
            overlap = all_test_indices.intersection(set(test_idx))
            assert len(overlap) == 0, (
                f"❌ Fold {fold} tem índices de teste repetidos: {overlap}"
            )
            all_test_indices.update(test_idx)


# =============================================================================
# TESTES DE PIPELINE COMPLETO
# =============================================================================

class TestFullPipelineNoLeakage:
    """Testes do pipeline completo de features."""
    
    def test_create_all_features_no_leakage(self, sample_game_data):
        """Pipeline completo não deve vazar dados."""
        df = create_all_features(
            sample_game_data,
            windows=[5, 10],
            stats_cols=['pts', 'reb']
        )
        
        # Verificar que primeiro jogo tem NaN em todas features rolling
        first_row = df.iloc[0]
        rolling_cols = [c for c in df.columns if 'rolling' in c or 'ema' in c or 'season_avg' in c]
        
        for col in rolling_cols:
            assert pd.isna(first_row[col]), (
                f"❌ VAZAMENTO! Primeiro jogo tem valor em {col}: {first_row[col]}"
            )
    
    def test_prediction_scenario(self, sample_game_data):
        """
        Simular cenário real de previsão.
        
        Dado: Queremos prever o jogo do dia 2024-01-20
        Então: Features disponíveis devem ser calculadas APENAS com dados até 2024-01-19
        """
        target_date = "2024-01-20"
        
        # Criar features para todos os dados
        df = create_all_features(sample_game_data, windows=[5], stats_cols=['pts'])
        
        # Filtrar para dia alvo
        target_row = df[df['date'] == target_date]
        
        if not target_row.empty:
            # Verificar que rolling feature do dia 20 
            # NÃO inclui pontos do dia 20
            pts_day_20 = sample_game_data[
                sample_game_data['date'] == target_date
            ]['pts'].iloc[0]
            
            rolling_val = target_row['pts_rolling_5_mean'].iloc[0]
            
            # Se rolling_val incluísse pontos do dia 20,
            # seria diferente da média real dos 5 dias anteriores
            expected_5_days = sample_game_data[
                (sample_game_data['date'] < target_date)
            ].tail(5)['pts'].mean()
            
            assert abs(rolling_val - expected_5_days) < 0.01, (
                f"❌ VAZAMENTO! Rolling para {target_date} parece incluir "
                f"dados do próprio dia. Esperado: {expected_5_days:.2f}, "
                f"Obtido: {rolling_val:.2f}"
            )
    
    def test_multi_team_isolation(self, multi_team_data):
        """
        Features de um time NÃO devem vazar dados de outro time.
        """
        df = create_rolling_features(
            multi_team_data,
            windows=[5],
            stats_cols=['pts'],
            group_col='team'
        )
        
        # Verificar que cada time tem sua própria sequência
        for team in ['LAL', 'BOS', 'GSW']:
            team_df = df[df['team'] == team].reset_index(drop=True)
            
            # Primeiro jogo de cada time deve ser NaN
            assert pd.isna(team_df.iloc[0]['pts_rolling_5_mean']), (
                f"❌ VAZAMENTO! Primeiro jogo de {team} tem rolling, "
                "possivelmente vazando de outro time"
            )


# =============================================================================
# TESTES DE DECORADOR ANTI-LEAKAGE
# =============================================================================

class TestAntiLeakageDecorator:
    """Testes para o decorador @anti_leakage."""
    
    def test_decorator_allows_valid_function(self, sample_game_data):
        """Decorador não deve bloquear funções válidas."""
        @anti_leakage
        def valid_feature_func(df, target_date=None):
            df = df.copy()
            df['new_feature'] = df['pts'].shift(1)  # Shift correto
            return df
        
        # Deve executar sem erros
        result = valid_feature_func(sample_game_data, target_date="2024-01-15")
        assert 'new_feature' in result.columns
    
    def test_decorator_handles_missing_date(self):
        """Decorador deve lidar graciosamente com DataFrame sem 'date'."""
        @anti_leakage
        def func(df, target_date=None):
            return df
        
        df_no_date = pd.DataFrame({'pts': [1, 2, 3]})
        
        # Não deve lançar exceção
        result = func(df_no_date)
        assert len(result) == 3


# =============================================================================
# TESTES DE REGRESSÃO
# =============================================================================

class TestRegressionLeakage:
    """Testes de regressão para bugs conhecidos."""
    
    def test_no_lookahead_in_elo_ratings(self, sample_game_data):
        """
        Bug histórico: Elo ratings sendo calculados com resultado do jogo.
        
        O Elo de um time ANTES de um jogo não deve saber o resultado desse jogo.
        """
        # Este teste documenta um bug conhecido e previne regressão
        # Implementação depende do módulo elo_system.py
        pass  # Placeholder para quando elo_system for integrado
    
    def test_no_target_in_features(self, sample_game_data):
        """
        Features não devem incluir a variável target.
        """
        df = create_all_features(sample_game_data, stats_cols=['pts', 'reb'])
        
        # home_win é target, não deve ter rolling
        feature_cols = [c for c in df.columns if 'rolling' in c or 'ema' in c]
        
        for col in feature_cols:
            assert 'win' not in col.lower(), (
                f"❌ VAZAMENTO! Feature {col} parece derivada de target (win)"
            )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testes de Data Leakage")
    print("="*60)
    print("\nExecute com: pytest tests/test_data_leakage.py -v\n")
    
    # Rodar testes básicos
    pytest.main([__file__, "-v", "--tb=short"])
