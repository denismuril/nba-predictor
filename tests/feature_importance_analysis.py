"""
Feature Importance Analysis - Final P2.2

Analisa importância das 15 features P2.2 no modelo final.

Usage:
    python tests/feature_importance_analysis.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_feature_importance():
    """Analisa importância das features no modelo final."""
    
    logger.info("📊 Feature Importance Analysis\n")
    logger.info("="*60)
    
    # Load model
    model_path = Path('models/ml_model.joblib')
    
    if not model_path.exists():
        logger.error("❌ Modelo não encontrado em models/ml_model.joblib")
        return
    
    try:
        model_data = joblib.load(model_path)
        
        # Handle different model formats
        if isinstance(model_data, dict):
            model = model_data.get('model') or model_data.get('clf')
            feature_names = model_data.get('feature_names', [])
        else:
            # Model directly
            model = model_data
            feature_names = []
        
        if model is None:
            logger.error("❌ Não foi possível extrair modelo do arquivo")
            return
            
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return
    
    if not hasattr(model, 'feature_importances_'):
        logger.error("❌ Modelo não suporta feature importance")
        return
    
    # Get importances
    importances = model.feature_importances_
    
    # Create DataFrame
    df_imp = pd.DataFrame({
        'feature': feature_names if feature_names else [f'feature_{i}' for i in range(len(importances))],
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Identify P2.2 features
    p22_keywords = [
        'pace', 'def_rating', 'def_matchup', 'reb', 'three_pt', 'tov', 'turnover',
        'clutch', 'playoff', 'ts_pct', 'ast_tov', 'injury', 'schedule', 'travel',
        'fastbreak', 'paint', 'second_chance'
    ]
    
    df_imp['is_p22'] = df_imp['feature'].apply(
        lambda x: any(kw in x.lower() for kw in p22_keywords)
    )
    
    # Summary
    logger.info(f"\n📈 Top 20 Features:")
    logger.info(f"{'Feature':<40} {'Importance':<12} {'P2.2'}")
    logger.info("-"*60)
    
    for idx, row in df_imp.head(20).iterrows():
        p22_mark = "✅" if row['is_p22'] else "  "
        logger.info(f"{row['feature']:<40} {row['importance']:<12.4f} {p22_mark}")
    
    # P2.2 Stats
    p22_features = df_imp[df_imp['is_p22']]
    total_p22_importance = p22_features['importance'].sum()
    
    logger.info(f"\n🎯 P2.2 Features Summary:")
    logger.info(f"   Total P2.2 features: {len(p22_features)}")
    logger.info(f"   Total importance: {total_p22_importance:.4f} ({total_p22_importance*100:.1f}%)")
    logger.info(f"   Average importance: {p22_features['importance'].mean():.4f}")
    
    # Top P2.2
    logger.info(f"\n⭐ Top 10 P2.2 Features:")
    for idx, row in p22_features.head(10).iterrows():
        logger.info(f"   {row['feature']:<35} {row['importance']:.4f}")
    
    # Save
    report_path = Path('reports/feature_importance.csv')
    report_path.parent.mkdir(exist_ok=True, parents=True)
    df_imp.to_csv(report_path, index=False)
    
    logger.info(f"\n💾 Relatório salvo: {report_path}")
    logger.info("="*60 + "\n")
    
    return df_imp


if __name__ == '__main__':
    logger.info("🏀 Feature Importance Analysis - P2.2\n")
    
    df_imp = analyze_feature_importance()
    
    if df_imp is not None:
        print(f"\n✅ Análise completa!")
        print(f"   Features analisadas: {len(df_imp)}")
        print(f"   P2.2 features: {df_imp['is_p22'].sum()}")
