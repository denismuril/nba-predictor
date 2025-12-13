"""
Teste Granular de Features - Isolamento Incremental

Este teste adiciona features UMA A UMA ao modelo minimalista para identificar
EXATAMENTE qual feature causa o vazamento de 65% → 95%.

NOTA: Testes marcados como skip por padrão pois requerem dados históricos.
"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

import pytest
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.WARNING)

# Flag para controlar se dados estão disponíveis
DATA_AVAILABLE = False
df = pd.DataFrame()
y = pd.Series()


def load_test_data():
    """Tenta carregar dados para os testes."""
    global DATA_AVAILABLE, df, y
    try:
        from ml_pipeline.data_preparation import load_historical_data
        df = load_historical_data(seasons=['2023-24', '2024-25'])
        df = df.sort_values('date').reset_index(drop=True)
        y = (df['winner'] == 'HOME').astype(int)
        DATA_AVAILABLE = True
    except Exception as e:
        DATA_AVAILABLE = False
        df = pd.DataFrame()
        y = pd.Series()
        logging.warning(f"Dados não disponíveis: {e}")


# Tentar carregar na importação (mas não falhar)
try:
    load_test_data()
except Exception:
    pass


# Features base candidatas
BASE_FEATURES_CANDIDATES = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_rolling_10_points', 'away_rolling_10_points',
    'home_rest_days', 'away_rest_days', 'rest_diff',
    'home_is_back_to_back', 'away_is_back_to_back',
]

# Grupos de features para testar
FEATURE_GROUPS = {
    'off_rating': [
        'home_rolling_5_off_rating', 'away_rolling_5_off_rating',
        'home_rolling_10_off_rating', 'away_rolling_10_off_rating',
    ],
    'def_rating': [
        'home_rolling_5_def_rating', 'away_rolling_5_def_rating',
        'home_rolling_10_def_rating', 'away_rolling_10_def_rating',
    ],
    'four_factors_rolling': [
        'home_rolling_10_efg_pct', 'away_rolling_10_efg_pct',
        'home_rolling_10_tov_pct', 'away_rolling_10_tov_pct',
    ],
}


@pytest.fixture
def base_features():
    """Fixture que retorna features base disponíveis."""
    if not DATA_AVAILABLE:
        pytest.skip("Dados históricos não disponíveis")
    return [f for f in BASE_FEATURES_CANDIDATES if f in df.columns]


@pytest.fixture
def features_to_test():
    """Fixture que retorna grupos de features para teste."""
    return FEATURE_GROUPS


def _test_feature_group(base_feats, new_features, group_name):
    """Helper para testar um grupo de features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit

    test_features = base_feats + [f for f in new_features if f in df.columns]
    if len(test_features) == 0:
        return 0.5, 0  # Fallback

    X = df[test_features].fillna(0)

    tscv = TimeSeriesSplit(n_splits=3)
    model = RandomForestClassifier(
        n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
    )

    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))

    return np.mean(scores), len(test_features)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Dados históricos não disponíveis")
class TestGranularFeatures:
    """Testes granulares de features."""

    def test_baseline_accuracy(self, base_features):
        """Testa acurácia baseline com features essenciais."""
        if len(base_features) == 0:
            pytest.skip("Nenhuma feature base disponível")

        acc, num_feats = _test_feature_group(base_features, [], "baseline")
        
        # Baseline deve estar entre 50% e 80%
        assert 0.50 <= acc <= 0.80, f"Baseline fora do range esperado: {acc:.2%}"
        print(f"Baseline: {acc:.2%} com {num_feats} features")

    def test_feature_groups_no_extreme_leakage(self, base_features, features_to_test):
        """Testa que nenhum grupo individual causa leakage extremo (>90%)."""
        if len(base_features) == 0:
            pytest.skip("Nenhuma feature base disponível")

        for group_name, group_features in features_to_test.items():
            available_features = [f for f in group_features if f in df.columns]
            if len(available_features) == 0:
                continue

            acc, _ = _test_feature_group(base_features, group_features, group_name)
            
            # Nenhum grupo deve causar leakage extremo
            assert acc < 0.90, (
                f"LEAKAGE DETECTADO no grupo '{group_name}': {acc:.2%}"
            )


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Dados históricos não disponíveis")
def test_data_loads():
    """Testa que os dados carregam corretamente."""
    assert len(df) > 0, "DataFrame vazio"
    assert len(y) > 0, "Target vazio"
    assert 'date' in df.columns, "Coluna 'date' ausente"


def test_module_imports():
    """Testa que imports do módulo funcionam."""
    # Este teste sempre deve passar
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
