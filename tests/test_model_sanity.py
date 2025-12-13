"""
Teste 3: Sanidade do Modelo - Smoke Test
=========================================
CRÍTICO: Verifica que o modelo não "enlouqueceu".

NOTA: Os testes usam as features reais do modelo carregando-as do arquivo.
"""

import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


MODEL_PATH = Path('data/models/spread_model.joblib')
FEATURES_PATH = Path('data/models/spread_feature_names.joblib')


def test_model_exists():
    """Modelo deve existir."""
    if not MODEL_PATH.exists():
        pytest.skip("Modelo não encontrado em data/models/spread_model.joblib")
    assert MODEL_PATH.exists()


def test_features_file_exists():
    """Arquivo de features deve existir."""
    if not FEATURES_PATH.exists():
        pytest.skip("Features não encontradas em data/models/spread_feature_names.joblib")
    assert FEATURES_PATH.exists()


@pytest.fixture
def model_and_features():
    """Carrega modelo e features."""
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        pytest.skip("Modelo ou features não encontrados")

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


def test_model_loads(model_and_features):
    """Modelo deve carregar sem erros."""
    model, features = model_and_features

    assert model is not None
    assert features is not None
    assert len(features) > 0
    print(f"✅ Modelo carregado: {len(features)} features")


def test_predictions_in_reasonable_range(model_and_features):
    """Previsões devem estar em range NBA realista."""
    model, features = model_and_features

    # Criar input com todas as features reais do modelo, valores médios
    test_input = pd.DataFrame({feat: [0.0] for feat in features})

    # Preencher com valores aproximados se reconhecermos a feature
    for col in test_input.columns:
        if 'efg' in col.lower():
            test_input[col] = 0.50
        elif 'elo' in col.lower():
            test_input[col] = 1500.0
        elif 'rolling' in col.lower() and 'points' in col.lower():
            test_input[col] = 105.0
        elif 'rest' in col.lower():
            test_input[col] = 1.0

    prediction = model.predict(test_input)[0]

    # Spread na NBA: -30 a +30 (raramente mais extremo)
    assert -30 < prediction < 30, (
        f"❌ Previsão absurda: {prediction:.1f} (esperado: -30 a +30)"
    )
    print(f"✅ Previsão razoável: {prediction:.1f} pts")


def test_model_not_constant(model_and_features):
    """Modelo não deve prever sempre o mesmo valor."""
    model, features = model_and_features

    predictions = []
    for multiplier in [0.8, 1.0, 1.2]:
        test_input = pd.DataFrame({feat: [0.0 * multiplier] for feat in features})

        # Variar algumas features conhecidas
        for col in test_input.columns:
            if 'efg' in col.lower():
                test_input[col] = 0.50 * multiplier
            elif 'elo' in col.lower():
                test_input[col] = 1500.0 * multiplier
            elif 'rolling' in col.lower():
                test_input[col] = 100.0 * multiplier

        pred = model.predict(test_input)[0]
        predictions.append(pred)

    # Deve haver variação
    assert len(set([round(p, 1) for p in predictions])) > 1, (
        f"❌ Modelo prevendo sempre o mesmo: {predictions}"
    )
    print(f"✅ Modelo varia: {predictions}")


def test_feature_count_reasonable(model_and_features):
    """Modelo deve ter número razoável de features."""
    _, features = model_and_features

    # Deve ter entre 5 e 500 features
    assert 5 <= len(features) <= 500, (
        f"Número de features suspeito: {len(features)}"
    )


if __name__ == "__main__":
    print("💨 SMOKE TESTS DO MODELO\n")
    pytest.main([__file__, "-v"])
