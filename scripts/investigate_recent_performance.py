#!/usr/bin/env python3
"""
Investigação de Performance Recente (30 dias)

Analisa por que a acurácia caiu nos últimos jogos.
Foca em:
1. Drift de features (mudança no padrão dos dados)
2. Times específicos com alta taxa de erro
3. Lesões (se dados disponíveis)
4. Comparação com baseline
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, log_loss

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_recent_performance():
    logger.info("="*80)
    logger.info("🕵️  INVESTIGAÇÃO DE PERFORMANCE RECENTE (30 DIAS)")
    logger.info("="*80)
    
    # 1. Carregar dados
    from ml_pipeline.data_preparation import load_historical_data
    
    df = load_historical_data(seasons=['2024-25', '2025-26'])
    
    if df is None or df.empty:
        logger.error("❌ Nenhum dado encontrado.")
        return
        
    # Filtrar últimos 30 dias
    max_date = df['date'].max()
    cutoff_date = max_date - timedelta(days=30)
    df_recent = df[df['date'] >= cutoff_date].copy()
    
    logger.info(f"📅 Período: {cutoff_date.date()} a {max_date.date()}")
    logger.info(f"📊 Total de jogos: {len(df_recent)}")
    
    if len(df_recent) < 10:
        logger.warning("⚠️  Poucos jogos para análise estatística robusta.")
    
    # 2. Carregar modelo e features
    model_path = Path('data/models/ensemble_model_calibrated_isotonic.joblib')
    features_path = Path('data/models/feature_names_final.joblib')
    
    if not model_path.exists():
        logger.error("❌ Modelo não encontrado.")
        return
        
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    
    # 3. Preparar X e y
    X = df_recent.copy()
    # Garantir colunas
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]
    y = (df_recent['winner'] == 'HOME').astype(int)
    
    # 4. Gerar previsões
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    # 5. Métricas Gerais
    acc = accuracy_score(y, preds)
    ll = log_loss(y, probs)
    
    logger.info("\n📊 PERFORMANCE GERAL:")
    logger.info(f"   Accuracy: {acc*100:.2f}%")
    logger.info(f"   Log Loss: {ll:.4f}")
    
    # 6. Análise de Erros
    df_recent['prob_home'] = probs
    df_recent['prediction'] = np.where(probs > 0.5, 'HOME', 'AWAY')
    df_recent['correct'] = (df_recent['prediction'] == df_recent['winner'])
    
    # 6.1 Por Time (Home/Away)
    logger.info("\n🏆 PIORES TIMES (HOME):")
    home_acc = df_recent.groupby('home_team')['correct'].agg(['mean', 'count']).sort_values('mean')
    logger.info(home_acc.head(5))
    
    logger.info("\n🏆 PIORES TIMES (AWAY):")
    away_acc = df_recent.groupby('away_team')['correct'].agg(['mean', 'count']).sort_values('mean')
    logger.info(away_acc.head(5))
    
    # 6.2 Por Confiança
    df_recent['confidence'] = np.abs(df_recent['prob_home'] - 0.5) * 2
    
    logger.info("\n🎯 POR CONFIANÇA:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df_recent['conf_bin'] = pd.cut(df_recent['confidence'], bins=bins, labels=labels)
    
    conf_acc = df_recent.groupby('conf_bin')['correct'].agg(['mean', 'count'])
    logger.info(conf_acc)
    
    # 6.3 Drift Check (Comparar médias das features chave com treino)
    # Carregar stats de treino (se salvos) ou calcular rápido
    # Simplificado: verificar se SOS mudou muito
    avg_sos = df_recent[['home_sos_10', 'away_sos_10']].mean()
    logger.info("\n🌊 FEATURE DRIFT (SOS):")
    logger.info(f"   Média SOS Recente: {avg_sos.mean():.4f}")
    # (Idealmente compararia com média histórica salva)
    
    # 7. Salvar Relatório
    report = {
        'period': f"{cutoff_date.date()} to {max_date.date()}",
        'accuracy': acc,
        'log_loss': ll,
        'worst_home_teams': home_acc.head(5).to_dict(),
        'worst_away_teams': away_acc.head(5).to_dict(),
        'confidence_breakdown': conf_acc.to_dict()
    }
    
    report_file = Path('data/monitoring/recent_performance_investigation.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
        
    logger.info(f"\n💾 Relatório salvo: {report_file}")
    
    return report

if __name__ == "__main__":
    investigate_recent_performance()
