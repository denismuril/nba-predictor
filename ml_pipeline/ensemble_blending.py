import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from ml_pipeline.data_preparation import load_historical_data, add_rolling_features, add_advanced_features
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_ensemble_blending():
    logger.info("🧪 Iniciando Treinamento de Ensemble Blending (Meta-Learner)...")
    
    # 1. Carregar Dados
    df = load_historical_data(seasons=['2023-24', '2024-25', '2025-26'], apply_weights=False)
    df = add_rolling_features(df)
    df = add_advanced_features(df)
    
    # Features (usar as selecionadas)
    try:
        features = joblib.load('data/models/feature_names_final.joblib')
    except:
        logger.warning("⚠️ Features selecionadas não encontradas. Usando todas numéricas.")
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f not in ['home_score', 'away_score', 'total_points', 'winner', 'spread', 'actual_spread']]

    X = df[features].fillna(0)
    y = df['winner']
    
    # 2. Definir Modelos Base
    # Idealmente carregaríamos os melhores hiperparâmetros salvos
    # Aqui vamos usar configurações robustas padrão para demonstração
    models = {
        'rf': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'xgb': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42),
        'lgb': lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    }
    
    # 3. Gerar Previsões Out-of-Fold (OOF)
    logger.info("🔄 Gerando previsões Out-of-Fold...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = pd.DataFrame(index=X.index)
    
    for name, model in models.items():
        logger.info(f"   Treinando {name}...")
        oof_col = f'pred_{name}'
        oof_preds[oof_col] = 0.0
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            oof_preds.loc[val_idx, oof_col] = model.predict_proba(X_val)[:, 1]
            
    # 4. Treinar Meta-Learner (Logistic Regression)
    logger.info("🧠 Treinando Meta-Learner (Logistic Regression)...")
    meta_X = oof_preds
    meta_y = y
    
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_X, meta_y)
    
    # Avaliar Meta-Learner (simples, no mesmo dataset OOF - ideal seria holdout separado)
    meta_preds = meta_model.predict_proba(meta_X)[:, 1]
    acc = accuracy_score(meta_y, (meta_preds >= 0.5).astype(int))
    ll = log_loss(meta_y, meta_preds)
    
    logger.info(f"🏆 Meta-Learner Accuracy (CV estimate): {acc:.2%}")
    logger.info(f"📉 Meta-Learner Log Loss: {ll:.4f}")
    
    # Pesos aprendidos
    weights = dict(zip(models.keys(), meta_model.coef_[0]))
    logger.info(f"⚖️ Pesos aprendidos: {weights}")
    
    # 5. Salvar
    # Precisamos salvar os modelos base treinados em TODO o dataset e o meta-modelo
    logger.info("💾 Salvando modelos finais...")
    
    final_models = {}
    for name, model in models.items():
        model.fit(X, y)
        final_models[name] = model
        
    joblib.dump(final_models, 'data/models/blending_base_models.joblib')
    joblib.dump(meta_model, 'data/models/blending_meta_model.joblib')
    
    logger.info("✅ Ensemble Blending concluído e salvo!")

if __name__ == "__main__":
    train_ensemble_blending()
