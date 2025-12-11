import joblib
import json
from pathlib import Path

try:
    # Tentar ler metadados se existirem (o script V4 deveria ter salvo, mas talvez não salvou json específico, só o joblib)
    # O script V4 salvou o modelo em data/models/ensemble_model_v4.joblib
    # Vamos carregar o modelo e ver se tem metadados acoplados ou se precisamos re-avaliar
    
    # Mas espere, o script V4 imprimiu a acurácia no final.
    # Vamos tentar carregar o modelo e fazer uma avaliação rápida se não tiver metadata.
    
    # Melhor: O script V3 salvava metadata.json. O V4 eu não coloquei para salvar metadata.json explicitamente no código que escrevi (erro meu).
    # Mas ele imprimiu no log.
    
    # Vamos fazer um load e score rápido com dados de teste
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from ml_pipeline.data_preparation import load_historical_data
    import pandas as pd
    
    print("Carregando dados...")
    df, _ = load_historical_data(seasons=['2023-24', '2024-25', '2025-26'], apply_weights=True)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Recriar X com as features do V4
    feature_names = joblib.load('data/models/feature_names_v4.joblib')
    
    # Pré-processamento básico igual ao treino
    X = pd.get_dummies(df, columns=['home_team', 'away_team'], drop_first=False)
    
    # Alinhar
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    y = (df['winner'] == 'HOME').astype(int)
    
    # Split 80/20
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print("Carregando modelo V4...")
    model = joblib.load('data/models/ensemble_model_v4.joblib')
    
    print("Avaliando...")
    acc = model.score(X_test, y_test)
    print(f"ACCURACY_V4: {acc:.4f}")
    
except Exception as e:
    print(f"Erro: {e}")
