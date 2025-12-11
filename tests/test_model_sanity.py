"""
Teste 3: Sanidade do Modelo - Smoke Test
=========================================
CRÍTICO: Verifica que o modelo não "enlouqueceu".
"""

import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def test_model_exists():
    """
    Modelo deve existir
    """
    model_path = Path('data/models/spread_model.joblib')
    assert model_path.exists(), "❌ Modelo não encontrado!"
    print("✅ Modelo existe")


def test_model_loads():
    """
    Modelo deve carregar sem erros
    """
    try:
        model = joblib.load('data/models/spread_model.joblib')
        features = joblib.load('data/models/spread_feature_names.joblib')
        
        assert model is not None
        assert features is not None
        assert len(features) > 0
        
        print(f"✅ Modelo carregado: {len(features)} features")
    except Exception as e:
        pytest.fail(f"❌ Erro ao carregar modelo: {e}")


def test_predictions_in_reasonable_range():
    """
    Previsões devem estar em range NBA realista
    """
    model = joblib.load('data/models/spread_model.joblib')
    features = joblib.load('data/models/spread_feature_names.joblib')
    
    # Input fake mas realista
    # Features: home_efg, away_efg, home_rolling_5, away_rolling_5
    test_input = pd.DataFrame({
        'home_efg': [0.52],  # 52% eFG
        'away_efg': [0.48],  # 48% eFG
        'home_team_rolling_5': [110],  # 110 pts/jogo
        'away_team_rolling_5': [105]   # 105 pts/jogo
    })
    
    prediction = model.predict(test_input)[0]
    
    # Spread na NBA: -30 a +30 (raramente mais extremo)
    assert -30 < prediction < 30, \
        f"❌ Previsão absurda: {prediction:.1f} (esperado: -30 a +30)"
    
    print(f"✅ Previsão razoável: {prediction:.1f} pts")


def test_strong_vs_weak_team():
    """
    Time forte em casa deve ter spread positivo
    """
    model = joblib.load('data/models/spread_model.joblib')
    
    # Cenário: Celtics (forte) vs Pistons (fraco)
    strong_home = pd.DataFrame({
        'home_efg': [0.56],  # Elite
        'away_efg': [0.44],  # Ruim
        'home_team_rolling_5': [120],
        'away_team_rolling_5': [95]
    })
    
    prediction = model.predict(strong_home)[0]
    
    # Time forte deve vencer por margin razoável
    assert prediction > 3.0, \
        f"❌ Forte vs Fraco deveria dar +3+, deu {prediction:.1f}"
    
    print(f"✅ Forte vs Fraco: +{prediction:.1f} (correto)")


def test_model_not_constant():
    """
    Modelo não deve prever sempre o mesmo valor
    """
    model = joblib.load('data/models/spread_model.joblib')
    
    predictions = []
    
    for efg_home in [0.45, 0.50, 0.55]:
        test = pd.DataFrame({
            'home_efg': [efg_home],
            'away_efg': [0.50],
            'home_team_rolling_5': [110],
            'away_team_rolling_5': [110]
        })
        pred = model.predict(test)[0]
        predictions.append(pred)
    
    # Deve haver variação
    assert len(set([round(p, 1) for p in predictions])) > 1, \
        f"❌ Modelo prevendo sempre o mesmo: {predictions}"
    
    print(f"✅ Modelo varia: {predictions}")


if __name__ == "__main__":
    print("💨 SMOKE TESTS DO MODELO\n")
    test_model_exists()
    test_model_loads()
    test_predictions_in_reasonable_range()
    test_strong_vs_weak_team()
    test_model_not_constant()
    print("\n🎉 MODELO PASSOU EM TODOS OS TESTES!")
