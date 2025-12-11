#!/usr/bin/env python3
"""
Weighted Voting Optimization - Fase 2/3

Este script implementa a lógica de "Dynamic Weighted Voting":
1. Carrega o ensemble atual (Stacking)
2. Extrai os modelos base (RF, XGB, LGBM, Extra)
3. Avalia performance recente (últimos 30 dias)
4. Calcula pesos dinâmicos baseados na performance
5. Compara Stacking vs Weighted Voting
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.preprocessing import MinMaxScaler

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_last_30_days():
    """Carrega dados dos últimos 30 dias para validação."""
    from ml_pipeline.data_preparation import load_historical_data
    
    # Carregar tudo
    df = load_historical_data(seasons=['2024-25', '2025-26']) # Garantir dados recentes
    
    if df is None or df.empty:
        return None, None, None
        
    # Filtrar últimos 30 dias
    max_date = df['date'].max()
    cutoff_date = max_date - timedelta(days=30)
    
    df_recent = df[df['date'] >= cutoff_date].copy()
    
    logger.info(f"📅 Dados recentes: {len(df_recent)} jogos ({cutoff_date.date()} a {max_date.date()})")
    
    # Preparar X e y
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X = df_recent.drop(columns=drop_cols, errors='ignore')
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    
    # Carregar features selecionadas para garantir alinhamento
    features_path = Path('data/models/feature_names_final.joblib')
    if features_path.exists():
        selected_features = joblib.load(features_path)
        # Adicionar colunas faltantes com 0
        for col in selected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[selected_features]
    
    y = (df_recent['winner'] == 'HOME').astype(int)
    
    return X, y, df_recent

def extract_base_estimators(model):
    """Extrai estimadores base de um CalibratedClassifierCV(StackingClassifier)."""
    
    # 1. Desembrulhar CalibratedClassifierCV
    if hasattr(model, 'estimator'):
        ensemble = model.estimator
    elif hasattr(model, 'base_estimator'):
        ensemble = model.base_estimator
    else:
        # Pode ser o próprio StackingClassifier se não foi calibrado
        ensemble = model
        
    logger.info(f"📦 Ensemble type: {type(ensemble).__name__}")
    
    # 2. Extrair estimadores do StackingClassifier
    if hasattr(ensemble, 'estimators_'):
        return ensemble.estimators_
    elif hasattr(ensemble, 'estimators'):
        # Lista de tuplas (name, estimator)
        return [est for name, est in ensemble.estimators]
    else:
        logger.error("❌ Não foi possível encontrar estimators_ no ensemble")
        return []

def calculate_dynamic_weights(estimators, X, y):
    """Calcula pesos baseados em Log Loss recente."""
    weights = {}
    metrics = {}
    
    logger.info("\n📊 Avaliando performance individual (últimos 30 dias):")
    
    scores = []
    
    for i, est in enumerate(estimators):
        name = type(est).__name__
        try:
            # Predição
            probs = est.predict_proba(X)[:, 1]
            
            # Métricas
            ll = log_loss(y, probs)
            acc = accuracy_score(y, (probs > 0.5).astype(int))
            
            logger.info(f"   🔹 {name}: LogLoss={ll:.4f}, Acc={acc:.2%}")
            
            # Score para peso (inverso do log loss)
            # Quanto menor o log loss, maior o score
            score = 1 / (ll + 1e-6) # Evitar divisão por zero
            scores.append(score)
            
            metrics[name] = {'log_loss': ll, 'accuracy': acc}
            
        except Exception as e:
            logger.warning(f"   ⚠️  Erro ao avaliar {name}: {e}")
            scores.append(0)
    
    # Calcular pesos (Softmax ou normalização simples)
    # Vamos usar normalização simples dos scores inversos
    total_score = sum(scores)
    normalized_weights = [s / total_score for s in scores]
    
    logger.info("\n⚖️  Pesos Calculados:")
    for i, est in enumerate(estimators):
        name = type(est).__name__
        logger.info(f"   🔹 {name}: {normalized_weights[i]:.4f}")
        
    return normalized_weights, metrics

def evaluate_weighted_voting(estimators, weights, X, y):
    """Avalia o ensemble com pesos manuais."""
    final_probs = np.zeros(len(X))
    
    for i, est in enumerate(estimators):
        probs = est.predict_proba(X)[:, 1]
        final_probs += probs * weights[i]
        
    ll = log_loss(y, final_probs)
    acc = accuracy_score(y, (final_probs > 0.5).astype(int))
    
    return ll, acc, final_probs

def main():
    # 1. Carregar modelo
    model_path = Path('data/models/ensemble_model_calibrated_isotonic.joblib')
    if not model_path.exists():
        logger.error("Modelo não encontrado!")
        return
        
    model = joblib.load(model_path)
    
    # 2. Carregar dados recentes
    X, y, df = load_data_last_30_days()
    if X is None:
        return
        
    # 3. Extrair estimadores
    estimators = extract_base_estimators(model)
    if not estimators:
        return
        
    # 4. Calcular pesos dinâmicos
    weights, metrics = calculate_dynamic_weights(estimators, X, y)
    
    # 5. Avaliar Weighted Voting
    wv_ll, wv_acc, _ = evaluate_weighted_voting(estimators, weights, X, y)
    
    # 6. Avaliar Stacking Original (Baseline)
    stacking_probs = model.predict_proba(X)[:, 1]
    st_ll = log_loss(y, stacking_probs)
    st_acc = accuracy_score(y, (stacking_probs > 0.5).astype(int))
    
    logger.info("\n" + "="*60)
    logger.info("🏆 COMPARAÇÃO FINAL (Últimos 30 dias)")
    logger.info("="*60)
    logger.info(f"1️⃣ Stacking (Atual):      LogLoss={st_ll:.4f}, Acc={st_acc:.2%}")
    logger.info(f"2️⃣ Weighted Voting (Novo): LogLoss={wv_ll:.4f}, Acc={wv_acc:.2%}")
    
    diff_acc = wv_acc - st_acc
    if diff_acc > 0:
        logger.info(f"\n✅ Weighted Voting venceu por +{diff_acc*100:.2f}%!")
        
        # Salvar pesos
        weights_file = Path('data/models/dynamic_weights.json')
        import json
        with open(weights_file, 'w') as f:
            json.dump({
                'weights': weights,
                'estimators': [type(e).__name__ for e in estimators],
                'timestamp': datetime.now().isoformat(),
                'metrics': {'log_loss': wv_ll, 'accuracy': wv_acc}
            }, f, indent=2)
        logger.info(f"💾 Pesos salvos em: {weights_file}")
        
    else:
        logger.info(f"\n⚠️  Stacking ainda é melhor (Meta-learner > Linear weights).")

if __name__ == "__main__":
    main()
