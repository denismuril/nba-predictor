"""
Feature Analysis V3 - Importance & Selection

Analisa a importância das features dos modelos V3 (Totals e Moneyline)
para identificar variáveis irrelevantes ou prejudiciais.
"""

import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_feature_importance():
    """Carrega modelos e features, e gera relatório de importância."""
    
    logger.info("=" * 70)
    logger.info("🔍 FEATURE ANALYSIS V3")
    logger.info("=" * 70)
    
    models_dir = Path('data/models')
    
    # 1. Analisar Totals (V17)
    try:
        totals_model = joblib.load(models_dir / 'totals_model_v17.joblib')
        totals_features = joblib.load(models_dir / 'totals_feature_names_v17.joblib')
        
        logger.info(f"\n🏀 Totals Model V17")
        logger.info(f"   Features: {len(totals_features)}")
        
        if hasattr(totals_model, 'feature_importances_'):
            importances = totals_model.feature_importances_
            
            df_imp = pd.DataFrame({
                'feature': totals_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Salvar CSV
            df_imp.to_csv('data/totals_feature_importance.csv', index=False)
            logger.info("   💾 Saved: data/totals_feature_importance.csv")
            
            # Top 20
            print("\n   🏆 Top 20 Features (Totals):")
            print(df_imp.head(20).to_string(index=False))
            
            # Bottom 20 (Zero importance?)
            zero_imp = df_imp[df_imp['importance'] == 0]
            logger.info(f"\n   ⚠️ Features com importância ZERO: {len(zero_imp)}")
            if not zero_imp.empty:
                print(zero_imp['feature'].head(10).tolist())
                
    except Exception as e:
        logger.error(f"❌ Erro ao analisar Totals: {e}")

    # 2. Analisar Moneyline (V7)
    try:
        ensemble_model = joblib.load(models_dir / 'ensemble_model_v7.joblib')
        ensemble_features = joblib.load(models_dir / 'ensemble_feature_names_v7.joblib')
        
        logger.info(f"\n🏆 Moneyline Model V7")
        logger.info(f"   Features: {len(ensemble_features)}")
        
        if hasattr(ensemble_model, 'feature_importances_'):
            importances = ensemble_model.feature_importances_
            
            df_imp = pd.DataFrame({
                'feature': ensemble_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Salvar CSV
            df_imp.to_csv('data/moneyline_feature_importance.csv', index=False)
            logger.info("   💾 Saved: data/moneyline_feature_importance.csv")
            
            # Top 20
            print("\n   🏆 Top 20 Features (Moneyline):")
            print(df_imp.head(20).to_string(index=False))
            
            # Bottom 20
            zero_imp = df_imp[df_imp['importance'] == 0]
            logger.info(f"\n   ⚠️ Features com importância ZERO: {len(zero_imp)}")
            if not zero_imp.empty:
                print(zero_imp['feature'].head(10).tolist())
                
        elif hasattr(ensemble_model, 'coef_'):
            # Logistic Regression fallback (se não for XGBoost)
            importances = np.abs(ensemble_model.coef_[0])
            df_imp = pd.DataFrame({
                'feature': ensemble_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n   🏆 Top 20 Features (Logistic Regression Coefs):")
            print(df_imp.head(20).to_string(index=False))

    except Exception as e:
        logger.error(f"❌ Erro ao analisar Moneyline: {e}")

if __name__ == "__main__":
    analyze_feature_importance()
