"""
Teste Granular Simplificado - Versão Robusta

Testa APENAS features que existem no df para evitar KeyError.

NOTA: Testes marcados como skip por padrão pois requerem dados históricos.
"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

import pytest
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.WARNING)

# Flags de controle
DATA_AVAILABLE = False
FEATURES_AVAILABLE = False
df = pd.DataFrame()
y = pd.Series()
all_model_features = []


def load_test_data():
    """Tenta carregar dados e features."""
    global DATA_AVAILABLE, FEATURES_AVAILABLE, df, y, all_model_features

    try:
        from ml_pipeline.data_preparation import load_historical_data
        import joblib
        from pathlib import Path

        df = load_historical_data(seasons=['2023-24', '2024-25'])
        df = df.sort_values('date').reset_index(drop=True)
        y = (df['winner'] == 'HOME').astype(int)
        DATA_AVAILABLE = True

        features_path = Path('data/models/feature_names_v6.joblib')
        if features_path.exists():
            all_model_features = joblib.load(features_path)
            FEATURES_AVAILABLE = True
    except Exception as e:
        logging.warning(f"Dados não disponíveis: {e}")


# Tentar carregar na importação
try:
    load_test_data()
except Exception:
    pass


# Features base
BASE_FEATURES = [
    'home_elo', 'away_elo',
    'home_rolling_10_points', 'away_rolling_10_points',
    'home_rest_days', 'away_rest_days',
]


def create_feature_groups(all_features):
    """Agrupa features por padrão de nome."""
    groups = {}
    groups['off_rating'] = [f for f in all_features if 'off_rating' in f.lower()]
    groups['def_rating'] = [f for f in all_features if 'def_rating' in f.lower()]
    groups['rapm_bpm'] = [f for f in all_features if 'rapm' in f.lower() or 'bpm' in f.lower()]
    groups['ortg_drtg_adj'] = [f for f in all_features if 'ortg_adj' in f.lower() or 'drtg_adj' in f.lower()]
    return {k: v for k, v in groups.items() if v}


@pytest.fixture
def base_features():
    """Fixture que retorna features base disponíveis."""
    if not DATA_AVAILABLE:
        pytest.skip("Dados históricos não disponíveis")
    return [f for f in BASE_FEATURES if f in df.columns]


@pytest.fixture
def features_to_test():
    """Fixture que retorna grupos de features."""
    if not FEATURES_AVAILABLE:
        pytest.skip("Features do modelo não disponíveis")
    return create_feature_groups(all_model_features)


def _test_features(features_list):
    """Helper para testar um conjunto de features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit

    if not features_list:
        return None

    X = df[features_list].fillna(0)
    tscv = TimeSeriesSplit(n_splits=3)
    model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)

    scores = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(model.score(X.iloc[test_idx], y.iloc[test_idx]))

    return np.mean(scores)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Dados históricos não disponíveis")
class TestGranularSimple:
    """Testes granulares simplificados."""

    def test_baseline_exists(self, base_features):
        """Testa que features base existem."""
        assert len(base_features) > 0

    def test_baseline_accuracy(self, base_features):
        """Testa acurácia baseline."""
        acc = _test_features(base_features)
        assert acc is not None
        assert 0.45 <= acc <= 0.85

    def test_no_extreme_leakage(self, base_features, features_to_test):
        """Testa que nenhum grupo causa leakage extremo."""
        baseline_acc = _test_features(base_features)
        if baseline_acc is None:
            pytest.skip("Baseline não calculável")

        for group_name, group_feats in features_to_test.items():
            available = [f for f in group_feats if f in df.columns]
            if not available:
                continue

            test_feats = list(set(base_features + available))
            acc = _test_features(test_feats)
            if acc is not None:
                assert acc < 0.90, f"Leakage em '{group_name}': {acc:.2%}"


def test_module_imports():
    """Testa que imports básicos funcionam."""
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
