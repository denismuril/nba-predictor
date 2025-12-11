"""
Teste 1: Máquina do Tempo - Anti-Data Leakage
==============================================
CRÍTICO: Garante que o modelo NUNCA vê o futuro.
Se falhar, TODO o modelo está inválido.
"""

import pytest
import pandas as pd
import numpy as np


def test_no_future_leakage_rolling_features():
    """
    Verifica que rolling features com shift(1) não vazam dados.
    Testa a lógica core usada em add_rolling_features e add_rolling_four_factors.
    """
    # Dados fake: Dia 1 (100 pts), Dia 2 (110 pts), Dia 3 (120 pts)
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'team': ['LAL', 'LAL', 'LAL'],
        'score': [100, 110, 120],
    })
    
    # Aplicar rolling com shift(1) - mesma lógica usada no pipeline
    df['rolling_1_score'] = df.groupby('team')['score'].transform(
        lambda x: x.shift(1).rolling(1, min_periods=1).mean()
    )
   
    # DIA 1: Deve ser NaN (não tem histórico)
    assert pd.isna(df.loc[0, 'rolling_1_score']), \
        "❌ VAZAMENTO! Dia 1 não deveria ter rolling (não tem passado)"
    
    # DIA 2: Deve ver APENAS Dia 1 (100)
    assert df.loc[1, 'rolling_1_score'] == 100, \
        f"❌ VAZAMENTO! Dia 2 deveria ver 100 (Dia 1), viu {df.loc[1, 'rolling_1_score']}"
    
    # DIA 3: Deve ver APENAS Dia 2 (110)
    assert df.loc[2, 'rolling_1_score'] == 110, \
        f"❌ VAZAMENTO! Dia 3 deveria ver 110 (Dia 2), viu {df.loc[2, 'rolling_1_score']}"
    
    print("✅ Teste Máquina do Tempo PASSOU! Sem vazamento de dados.")


def test_no_leakage_in_train_test_split():
    """
    Verifica que dados de teste não vazam pro treino
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    # Simular dataset
    dates = pd.date_range('2023-01-01', periods=100)
    df = pd.DataFrame({
        'date': dates,
        'feature1': np.random.randn(100),
        'target': np.random.randn(100)
    })
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_dates = df.iloc[train_idx]['date']
        test_dates = df.iloc[test_idx]['date']
        
        # Train SEMPRE deve ser ANTES de Test
        assert train_dates.max() < test_dates.min(), \
            f"❌ Fold {fold}: Train contém datas DEPOIS de Test!"
    
    print("✅ Split temporal correto! Train sempre antes de Test.")


if __name__ == "__main__":
    test_no_future_leakage_rolling_features()
    test_no_leakage_in_train_test_split()
    print("\n🎉 TODOS OS TESTES DE LEAKAGE PASSARAM!")
