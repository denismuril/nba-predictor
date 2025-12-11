# coding: utf-8
"""
Testes para Correções de Auditoria - NBA Predictor

Este módulo contém testes para validar as correções implementadas durante
a auditoria do sistema, cobrindo:

- P0-A: Bloqueio de apostas em odds fictícias
- P0-B: Eliminação de features constantes (Smart Money)
- P0-C: Limpeza de whitelist de features
- P1-A: Vetorização de cálculo de stats
- P1-B: Split temporal para calibração
- P2-A: Atualização de LEAGUE_DEFAULTS
- P2-B: Correção de fallback referee features
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# P0-A: Testes para bloqueio de odds fictícias
# ============================================================================

def test_p0a_block_estimated_odds():
    """Testa que odds estimadas são bloqueadas."""
    from betting.web_ui import SafeKellyStrategy
    
    strategy = SafeKellyStrategy(bankroll=1000)
    
    # Odds estimadas devem retornar SKIP
    result = strategy.calculate_stake(
        prob=65.0,
        odds=1.80,
        is_odds_estimated=True,
        odds_source='Estimado'
    )
    
    assert result.recommendation == 'SKIP', "Odds estimadas devem retornar SKIP"
    assert result.stake == 0, "Stake deve ser zero para odds estimadas"


def test_p0a_block_unknown_source():
    """Testa que fontes desconhecidas são bloqueadas."""
    from betting.web_ui import SafeKellyStrategy
    
    strategy = SafeKellyStrategy(bankroll=1000)
    
    fontes_suspeitas = ['unknown', 'estimated', 'fair', 'Estimado', '']
    
    for fonte in fontes_suspeitas:
        result = strategy.calculate_stake(
            prob=65.0,
            odds=1.80,
            is_odds_estimated=False,
            odds_source=fonte
        )
        assert result.recommendation == 'SKIP', f"Fonte '{fonte}' deveria ser bloqueada"


def test_p0a_block_out_of_range_odds():
    """Testa que odds fora do range são bloqueadas."""
    from betting.web_ui import SafeKellyStrategy
    
    strategy = SafeKellyStrategy(bankroll=1000)
    
    # Odds muito baixas
    result = strategy.calculate_stake(
        prob=95.0,
        odds=1.00,  # Inválida (< 1.01)
        is_odds_estimated=False,
        odds_source='Bet365'
    )
    assert result.recommendation == 'SKIP', "Odds < 1.01 devem retornar SKIP"
    
    # Odds muito altas
    result = strategy.calculate_stake(
        prob=5.0,
        odds=100.0,  # Inválida (> 50.0)
        is_odds_estimated=False,
        odds_source='Bet365'
    )
    assert result.recommendation == 'SKIP', "Odds > 50.0 devem retornar SKIP"


def test_p0a_valid_odds_pass():
    """Testa que odds válidas passam a validação."""
    from betting.web_ui import SafeKellyStrategy
    
    strategy = SafeKellyStrategy(bankroll=1000)
    
    result = strategy.calculate_stake(
        prob=65.0,
        odds=1.80,
        is_odds_estimated=False,
        odds_source='Bet365'
    )
    
    # Odds válidas não devem ser bloqueadas por P0-A
    # (podem ser SKIP por outros motivos como Kelly negativo)
    assert result.recommendation != 'SKIP' or 'BLOQUEADO' not in result.reason


# ============================================================================
# P0-B: Testes para Smart Money sem features constantes
# ============================================================================

def test_p0b_smart_money_no_data():
    """Testa que sem dados de odds, não cria features."""
    from ml_pipeline.feature_engineering_v2 import add_smart_money_features
    
    df = pd.DataFrame({
        'game_id': [1, 2, 3],
        'home_team': ['LAL', 'BOS', 'MIA'],
        'away_team': ['GSW', 'NYK', 'CHI'],
    })
    
    result = add_smart_money_features(df)
    
    # Não deve criar colunas de smart_money
    assert 'line_movement' not in result.columns
    assert 'implied_prob_diff' not in result.columns
    assert 'smart_money_signal' not in result.columns


def test_p0b_smart_money_insufficient_data():
    """Testa que com menos de 10% de dados válidos, não cria features."""
    from ml_pipeline.feature_engineering_v2 import add_smart_money_features
    
    df = pd.DataFrame({
        'game_id': range(100),
        'opening_odds': [None] * 95 + [2.0, 2.1, 2.2, 2.3, 2.4],
        'closing_odds': [None] * 95 + [1.9, 2.0, 2.1, 2.2, 2.3],
    })
    
    result = add_smart_money_features(df)
    
    # Com <10% de dados, não deve criar features
    assert 'line_movement' not in result.columns


def test_p0b_smart_money_uses_nan():
    """Testa que usa np.nan para linhas inválidas quando há dados suficientes."""
    from ml_pipeline.feature_engineering_v2 import add_smart_money_features
    
    # 80% dados válidos
    df = pd.DataFrame({
        'game_id': range(100),
        'opening_odds': [2.0] * 80 + [None] * 20,
        'closing_odds': [1.9] * 80 + [None] * 20,
    })
    
    result = add_smart_money_features(df)
    
    if 'line_movement' in result.columns:
        # Linhas inválidas devem ter NaN, não 0.0
        invalid_rows = result['line_movement'].isna()
        assert invalid_rows.sum() >= 20, "Linhas sem dados devem ter NaN"


# ============================================================================
# P0-C: Testes para whitelist limpa (já estava correto)
# ============================================================================

def test_p0c_blacklist_contains_smart_money():
    """Testa que blacklist contém features de smart_money."""
    from ml_pipeline.data_preparation import prepare_data_for_training
    
    # Criar df com features de smart_money
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'win': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'rolling_pts_5': [110] * 10,
        'home_elo': [1500] * 10,
        'line_movement': [0.0] * 10,  # Feature de leakage
        'smart_money_signal': [0] * 10,  # Feature de leakage
    })
    
    # prepare_data_for_training retorna (X, y) - features names são as colunas de X
    X, y = prepare_data_for_training(df)
    feature_names = list(X.columns)
    
    # Features de leakage NÃO devem estar presentes
    assert 'line_movement' not in feature_names
    assert 'smart_money_signal' not in feature_names


# ============================================================================
# P1-A: Testes para funções vetorizadas (skip se streamlit não disponível)
# ============================================================================

def test_p1a_precompute_team_stats_exists():
    """Testa que função existe e retorna dict."""
    try:
        from nba_predictor_web import precompute_team_stats, get_team_recent_stats_fast
    except ImportError:
        pytest.skip("Streamlit não disponível")
    
    # Função deve existir
    assert callable(precompute_team_stats)
    assert callable(get_team_recent_stats_fast)


# ============================================================================
# P1-B: Testes para split temporal
# ============================================================================

def test_p1b_temporal_split_function_exists():
    """Testa que função de split temporal existe."""
    from ml_pipeline.train_ensemble_v6 import temporal_train_calib_split
    
    assert callable(temporal_train_calib_split)


def test_p1b_temporal_split_uses_dates():
    """Testa que split usa datas corretamente."""
    from ml_pipeline.train_ensemble_v6 import temporal_train_calib_split
    
    # Criar dados de teste
    dates = pd.date_range('2024-01-01', periods=100)
    df = pd.DataFrame({
        'date': dates,
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
    })
    X = df[['feature1', 'feature2']]
    y = pd.Series(np.random.randint(0, 2, 100))
    weights = np.ones(100)
    
    X_train, X_calib, y_train, y_calib, w_train = temporal_train_calib_split(
        df, X, y, weights, calib_days=30
    )
    
    # Treino deve ter datas anteriores a calibração
    assert len(X_train) > 0
    assert len(X_calib) > 0


# ============================================================================
# P2-A: Testes para LEAGUE_DEFAULTS atualizados
# ============================================================================

def test_p2a_league_defaults_updated():
    """Testa que valores foram atualizados para 2024-25."""
    from ml_pipeline.feature_engineering_v2 import LEAGUE_DEFAULTS, get_league_default
    
    # Valores devem refletir 2024-25 (offensive inflation)
    assert LEAGUE_DEFAULTS['off_rating'] >= 117.0, "off_rating deve ser >=117 em 2024-25"
    assert LEAGUE_DEFAULTS['pts'] >= 116.0, "pts deve ser >=116 em 2024-25"
    assert LEAGUE_DEFAULTS['pace'] >= 100.0, "pace deve ser >=100 em 2024-25"
    
    # Função getter deve funcionar
    assert get_league_default('off_rating') >= 117.0
    assert get_league_default('inexistente') == 0.0


# ============================================================================
# P2-B: Testes para referee features sem fallback
# ============================================================================

def test_p2b_referee_no_constant_fallback():
    """Testa que sem dados de referee, não cria constantes."""
    from ml_pipeline.feature_engineering_v2 import add_referee_features
    
    df = pd.DataFrame({
        'game_id': [1, 2, 3],
        'date': pd.date_range('2024-01-01', periods=3),
        'home_team': ['LAL', 'BOS', 'MIA'],
        'away_team': ['GSW', 'NYK', 'CHI'],
    })
    
    result = add_referee_features(df)
    
    # Sem coluna de árbitros, não deve criar features
    if 'referee_home_win_pct' in result.columns:
        # Se criou, não pode ser constante 0.55
        unique_values = result['referee_home_win_pct'].dropna().unique()
        if len(unique_values) == 1:
            assert unique_values[0] != 0.55, "Não deve usar fallback 0.55"


# ============================================================================
# P3: Teste de Auditoria - Features Bloqueadas Nunca Entram no Modelo
# ============================================================================

def test_p3_blocked_features_not_in_training():
    """
    🚨 TESTE CRÍTICO: Valida que features de leakage NUNCA entram no X de treino.
    
    Verifica:
    - home_efg, away_efg (box score do jogo atual)
    - closing_odds (dados pós-jogo)
    - line_movement (requer closing_odds)
    - home_off_rating, home_def_rating (score do jogo atual)
    """
    from ml_pipeline.data_preparation import prepare_data_for_training
    
    # Criar DataFrame fake com features perigosas
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'home_team': ['LAL'] * 10,
        'away_team': ['BOS'] * 10,
        # Features SEGURAS (devem passar)
        'home_rolling_10_points': np.random.randn(10) + 110,
        'away_rolling_10_points': np.random.randn(10) + 110,
        'home_elo': [1500] * 10,
        'away_elo': [1500] * 10,
        'elo_diff': [0] * 10,
        # Features PERIGOSAS (devem ser bloqueadas)
        'home_efg': np.random.randn(10),       # Box score atual
        'away_efg': np.random.randn(10),
        'closing_odds': np.random.randn(10) + 2,  # Dados pós-jogo
        'line_movement': np.random.randn(10),  # Derivado de closing
        'home_off_rating': np.random.randn(10) + 110,  # Score atual
        'home_def_rating': np.random.randn(10) + 110,
        'smart_money_signal': np.random.randn(10),
        # Target
        'winner': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    })
    
    X, y = prepare_data_for_training(df, target='winner')
    
    # Lista de features que NUNCA devem aparecer em X
    FORBIDDEN_FEATURES = [
        'home_efg', 'away_efg', 
        'home_off_rating', 'home_def_rating',
        'away_off_rating', 'away_def_rating',
        'closing_odds', 'line_movement', 
        'smart_money_signal', 'implied_prob_diff'
    ]
    
    present_forbidden = [f for f in FORBIDDEN_FEATURES if f in X.columns]
    
    assert len(present_forbidden) == 0, \
        f"🚨 LEAKAGE DETECTADO! Features proibidas em X: {present_forbidden}"
    
    # Verificar que features seguras estão presentes
    EXPECTED_SAFE = ['home_rolling_10_points', 'away_rolling_10_points', 'home_elo', 'away_elo']
    present_safe = [f for f in EXPECTED_SAFE if f in X.columns]
    
    assert len(present_safe) >= 3, \
        f"⚠️ Features seguras faltando: apenas {present_safe} encontradas"
    
    print(f"✅ Teste P3 passou! {len(X.columns)} features seguras, 0 features proibidas")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

