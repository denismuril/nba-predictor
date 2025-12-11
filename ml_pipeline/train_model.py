import joblib
from sklearn.ensemble import RandomForestClassifier
from .data_preparation import load_historical_data

def train_and_save_model():
    """Treina o modelo de Machine Learning e salva em disco.
    Usa validação temporal (Time Series Split) para evitar look-ahead bias.
    """
    df = load_historical_data()
    
    # CRITICAL: Sort by date to ensure temporal integrity
    df = df.sort_values('date').reset_index(drop=True)
    
    # Definir features e target
    drop_cols = ['winner', 'correct', 'date', 'prediction', 'home_score', 'away_score', 'pt_diff']
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['winner'].apply(lambda x: 1 if x == 'HOME' else 0)
    
    # Time Series Split: 80% treino, 20% teste (dados mais recentes)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Hiperparâmetros otimizados via GridSearch
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    # Salvar modelo
    joblib.dump(model, 'data/models/ml_model.joblib')
    return model.score(X_test, y_test)
