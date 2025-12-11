#!/usr/bin/env python3
"""
Walk-Forward Validation Expandido - Fase 2

Implementa validação temporal rigorosa com múltiplos folds mensais.
Simula deployment real onde sempre prevemos o futuro baseado no passado.

Estratégia:
- Fold 1: Train[Oct-Mar] → Test[Apr]
- Fold 2: Train[Oct-Apr] → Test[May]
- Fold 3: Train[Oct-May] → Test[Jun]
... e assim por diante

Detecta:
- Drift sazonal (início vs fim de temporada)
- Degradação de performance ao longo do tempo
- Necessidade de retreinamento

Usage:
    python scripts/walk_forward_validation.py [--model ensemble|spread|totals]
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime, timedelta

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_monthly_folds(df, min_train_months=6):
    """
    Cria folds mensais para walk-forward validation.
    
    Args:
        df: DataFrame com coluna 'date'
        min_train_months: Mínimo de meses para treino inicial
    
    Returns:
        Lista de tuplas (train_idx, test_idx, test_month)
    """
    df = df.sort_values('date').reset_index(drop=True)
    df['year_month'] = df['date'].dt.to_period('M')
    
    unique_months = df['year_month'].unique()
    
    if len(unique_months) < min_train_months + 1:
        logger.warning(f"⚠️  Dados insuficientes para {min_train_months} meses de treino")
        min_train_months = len(unique_months) - 1
    
    folds = []
    
    for i in range(min_train_months, len(unique_months)):
        # Treino: todos os meses até i (exclusive)
        train_months = unique_months[:i]
        # Teste: mês i
        test_month = unique_months[i]
        
        train_idx = df[df['year_month'].isin(train_months)].index.tolist()
        test_idx = df[df['year_month'] == test_month].index.tolist()
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx, str(test_month)))
    
    logger.info(f"📊 Criados {len(folds)} folds mensais")
    logger.info(f"   Treino inicial: {min_train_months} meses")
    logger.info(f"   Folds de teste: {len(folds)} meses")
    
    return folds

def walk_forward_ensemble(df, weights, folds):
    """Walk-forward validation para ensemble moneyline."""
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    import joblib
    
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        use_full = True
    except:
        use_full = False
    
    # Preparar features
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X = df.drop(columns=drop_cols, errors='ignore')
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    y = (df['winner'] == 'HOME').astype(int)
    
    # Carregar features selecionadas se disponível
    selected_file = Path('data/models/selected_features.joblib')
    if selected_file.exists():
        selected_features = joblib.load(selected_file)
        X = X[selected_features]
        logger.info(f"✅ Usando {len(selected_features)} features selecionadas")
    
    results = []
    
    for fold_num, (train_idx, test_idx, test_month) in enumerate(folds, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📅 Fold {fold_num}/{len(folds)}: Testando {test_month}")
        logger.info(f"   Treino: {len(train_idx)} jogos")
        logger.info(f"   Teste: {len(test_idx)} jogos")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = weights[train_idx]
        
        # Treinar ensemble
        rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train, sample_weight=w_train)
        
        if use_full:
            base = [
                ('rf', rf),
                ('xgb', XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)),
                ('lgbm', LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)),
                ('extra', ExtraTreesClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1))
            ]
        else:
            base = [('rf', rf), ('extra', ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))]
        
        meta = LogisticRegression(max_iter=1000, random_state=42)
        model = StackingClassifier(estimators=base, final_estimator=meta, cv=3, n_jobs=-1)  # Reduzido de 5 para 3
        model.fit(X_train, y_train, sample_weight=w_train)
        
        # Avaliar
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_proba)
        
        logger.info(f"   ✅ Accuracy: {acc*100:.2f}%")
        logger.info(f"   📊 Log Loss: {logloss:.4f}")
        
        results.append({
            'fold': fold_num,
            'test_month': test_month,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'accuracy': float(acc),
            'log_loss': float(logloss)
        })
    
    return results

def analyze_results(results):
    """Analisa resultados do walk-forward."""
    df = pd.DataFrame(results)
    
    logger.info("\n" + "="*80)
    logger.info("📊 ANÁLISE WALK-FORWARD")
    logger.info("="*80)
    
    mean_acc = df['accuracy'].mean()
    std_acc = df['accuracy'].std()
    min_acc = df['accuracy'].min()
    max_acc = df['accuracy'].max()
    
    logger.info(f"\n📈 Accuracy:")
    logger.info(f"   Média: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)")
    logger.info(f"   Mínima: {min_acc*100:.2f}% ({df.loc[df['accuracy'].idxmin(), 'test_month']})")
    logger.info(f"   Máxima: {max_acc*100:.2f}% ({df.loc[df['accuracy'].idxmax(), 'test_month']})")
    
    # Tendência ao longo do tempo
    if len(results) >= 3:
        recent_3 = df.tail(3)['accuracy'].mean()
        first_3 = df.head(3)['accuracy'].mean()
        trend = recent_3 - first_3
        
        logger.info(f"\n📉 Tendência temporal:")
        logger.info(f"   Primeiros 3 folds: {first_3*100:.2f}%")
        logger.info(f"   Últimos 3 folds: {recent_3*100:.2f}%")
        logger.info(f"   Drift: {trend*100:+.2f}%")
        
        if abs(trend) > 0.03:  # 3% de drift
            logger.warning(f"   ⚠️  DRIFT DETECTADO! ({trend*100:+.2f}%)")
            logger.warning(f"   💡 Recomendação: Retreinamento periódico necessário")
    
    # Performance por trimestre
    df['quarter'] = pd.to_datetime(df['test_month'].str[:7] + '-01').dt.quarter
    if df['quarter'].nunique() > 1:
        logger.info(f"\n📅 Performance por trimestre:")
        for q in sorted(df['quarter'].unique()):
            q_df = df[df['quarter'] == q]
            q_acc = q_df['accuracy'].mean()
            logger.info(f"   Q{q}: {q_acc*100:.2f}% ({len(q_df)} folds)")
    
    logger.info("="*80)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Validation')
    parser.add_argument('--model', choices=['ensemble', 'spread', 'totals'], default='ensemble',
                       help='Modelo para validar')
    parser.add_argument('--min-train-months', type=int, default=6,
                       help='Mínimo de meses para treino inicial')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info(f"🚀 WALK-FORWARD VALIDATION - {args.model.upper()}")
    logger.info("="*80)
    
    # Carregar dados
    from ml_pipeline.data_preparation import load_historical_data
    from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG
    
    df, weights = load_historical_data(
        seasons=ML_SEASONS,
        apply_weights=True,
        weight_config=ML_SAMPLE_WEIGHT_CONFIG
    )
    
    # Criar folds
    folds = create_monthly_folds(df, min_train_months=args.min_train_months)
    
    # Executar walk-forward
    if args.model == 'ensemble':
        results = walk_forward_ensemble(df, weights, folds)
    else:
        logger.error(f"❌ Modelo {args.model} ainda não implementado")
        return 1
    
    # Analisar
    results_df = analyze_results(results)
    
    # Salvar
    output_file = Path(f'data/models/walk_forward_{args.model}_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Resultados salvos: {output_file}")
    
    # Salvar CSV também
    csv_file = Path(f'data/models/walk_forward_{args.model}_results.csv')
    results_df.to_csv(csv_file, index=False)
    logger.info(f"💾 CSV salvo: {csv_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
