"""
SHAP Feature Importance Analysis

Descobre quais features realmente importam para o modelo.
"""
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_pipeline.data_preparation import load_historical_data
import logging

logger = logging.getLogger(__name__)

def analyze_feature_importance():
    """
    Usa SHAP para analisar importância de features.
    """
    try:
        import shap
    except ImportError:
        logger.error("❌ SHAP não instalado. Rode: pip install shap")
        return
    
    logger.info("🔍 Analisando Feature Importance com SHAP...")
    
    # Load model and data
    model = joblib.load('data/models/ml_model.joblib')
    df = load_historical_data()
    
    # Prepare features
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff']
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # Sample (SHAP é lento em datasets grandes)
    X_sample = X.sample(min(500, len(X)), random_state=42)
    
    logger.info(f"   Calculando SHAP values para {len(X_sample)} jogos...")
    
    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Se classificador binário, pegar classe positiva
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Feature importance (média absoluta)
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    # Print top 20
    print("\n" + "="*60)
    print("📊 TOP 20 FEATURES MAIS IMPORTANTES")
    print("="*60)
    for idx, row in feature_importance.head(20).iterrows():
        print(f"{row['feature']:.<50} {row['importance']:.4f}")
    print("="*60 + "\n")
    
    # Salvar ranking completo
    feature_importance.to_csv('feature_importance_ranking.csv', index=False)
    logger.info("💾 Ranking salvo: feature_importance_ranking.csv")
    
    # Plot summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    logger.info("📊 Gráfico salvo: shap_summary.png")
    
    # Identificar features fracas (< 0.01)
    weak_features = feature_importance[feature_importance['importance'] < 0.01]
    logger.info(f"\n⚠️  {len(weak_features)} features com baixa importância (<0.01):")
    for feat in weak_features['feature'].head(10):
        logger.info(f"   - {feat}")
    
    return feature_importance

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyze_feature_importance()
