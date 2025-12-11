#!/usr/bin/env python3
"""
Feature Selection Module

Implementa múltiplas estratégias de seleção de features:
1. Correlation Analysis - Remove features altamente correlacionadas (r > threshold)
2. Feature Importance - Remove features com baixa importância
3. Variance Threshold - Remove features com variância muito baixa

Usage:
    from ml_pipeline.feature_selection import select_features
    X_selected, selected_cols = select_features(X, y, method='combined')
"""
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

def remove_correlated_features(X, threshold=0.95, exclude_cols=None):
    """
    Remove features altamente correlacionadas.
    
    Args:
        X: DataFrame com features
        threshold: Threshold de correlação (default: 0.95)
        exclude_cols: Lista de colunas para NÃO remover (ex: odds, team dummies)
    
    Returns:
        DataFrame com features não-correlacionadas
    """
    if exclude_cols is None:
        exclude_cols = []
    
    logger.info(f"🔍 Analisando correlação entre {X.shape[1]} features...")
    
    # Calcular matriz de correlação
    corr_matrix = X.corr().abs()
    
    # Criar máscara triangular superior (evitar duplicatas)
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Encontrar features para remover
    to_drop = []
    for column in upper_tri.columns:
        # Pular colunas protegidas
        if column in exclude_cols:
            continue
            
        # Se essa coluna tem correlação alta com alguma outra
        correlated_features = upper_tri.index[upper_tri[column] > threshold].tolist()
        
        if correlated_features:
            # Remover apenas se a outra feature não está na lista de exclusão
            for feat in correlated_features:
                if feat not in exclude_cols and column not in to_drop:
                    to_drop.append(column)
                    logger.debug(f"   Removendo {column} (r={upper_tri[column][feat]:.3f} com {feat})")
                    break
    
    if to_drop:
        logger.info(f"✂️  Removendo {len(to_drop)} features correlacionadas (r > {threshold})")
        X_reduced = X.drop(columns=to_drop)
    else:
        logger.info(f"✅ Nenhuma feature altamente correlacionada encontrada")
        X_reduced = X.copy()
    
    return X_reduced, to_drop

def select_by_importance(X, y, threshold=0.001, n_estimators=100, sample_weight=None):
    """
    Seleciona features baseado em importância do Random Forest.
    
    Args:
        X: DataFrame com features
        y: Target
        threshold: Importância mínima para manter feature
        n_estimators: Número de árvores no RF
        sample_weight: Pesos das amostras (opcional)
    
    Returns:
        DataFrame com features importantes, lista de features removidas
    """
    logger.info(f"🎯 Calculando feature importance (RF com {n_estimators} árvores)...")
    
    # Treinar RF rápido
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y, sample_weight=sample_weight)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Salvar rankings
    importance_file = Path('data/models/feature_importance_ranking.csv')
    importance_file.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_file, index=False)
    logger.info(f"💾 Feature importance salvo em: {importance_file}")
    
    # Log top features
    logger.info(f"\n🏆 Top 10 Features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Selecionar features acima do threshold
    selected_features = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()
    removed_features = importance_df[importance_df['importance'] < threshold]['feature'].tolist()
    
    logger.info(f"\n✂️  Removendo {len(removed_features)} features com importance < {threshold}")
    logger.info(f"✅ Mantendo {len(selected_features)} features importantes")
    
    if removed_features:
        logger.debug(f"\n📋 Features removidas por baixa importância:")
        for feat in removed_features[:20]:  # Mostrar apenas primeiras 20
            imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
            logger.debug(f"   {feat}: {imp:.6f}")
    
    X_selected = X[selected_features]
    
    return X_selected, removed_features

def remove_low_variance_features(X, threshold=0.01):
    """
    Remove features com variância muito baixa (quase constantes).
    
    Args:
        X: DataFrame com features
        threshold: Variância mínima
    
    Returns:
        DataFrame sem features de baixa variância
    """
    logger.info(f"📊 Analisando variância das features...")
    
    # Calcular variância
    variances = X.var()
    low_var_features = variances[variances < threshold].index.tolist()
    
    if low_var_features:
        logger.info(f"✂️  Removendo {len(low_var_features)} features com variância < {threshold}")
        X_reduced = X.drop(columns=low_var_features)
    else:
        logger.info(f"✅ Todas features têm variância adequada")
        X_reduced = X.copy()
    
    return X_reduced, low_var_features

def select_features(X, y, method='combined', 
                   corr_threshold=0.95, 
                   importance_threshold=0.001,
                   variance_threshold=0.01,
                   sample_weight=None,
                   exclude_from_corr=None):
    """
    Pipeline completo de seleção de features.
    
    Args:
        X: DataFrame com features
        y: Target
        method: 'correlation', 'importance', 'variance', ou 'combined'
        corr_threshold: Threshold para correlação
        importance_threshold: Threshold para importância
        variance_threshold: Threshold para variância
        sample_weight: Pesos das amostras
        exclude_from_corr: Features para não remover na análise de correlação
    
    Returns:
        X_selected, report_dict
    """
    logger.info("="*80)
    logger.info("🎯 FEATURE SELECTION PIPELINE")
    logger.info("="*80)
    logger.info(f"📊 Features iniciais: {X.shape[1]}")
    logger.info(f"🎯 Método: {method}")
    
    report = {
        'initial_features': X.shape[1],
        'removed_by_correlation': [],
        'removed_by_importance': [],
        'removed_by_variance': [],
        'final_features': 0,
        'selected_columns': []
    }
    
    X_working = X.copy()
    
    # 1. Remover baixa variância
    if method in ['variance', 'combined']:
        X_working, low_var = remove_low_variance_features(X_working, variance_threshold)
        report['removed_by_variance'] = low_var
    
    # 2. Remover correlacionadas
    if method in ['correlation', 'combined']:
        X_working, corr_removed = remove_correlated_features(
            X_working, 
            corr_threshold,
            exclude_cols=exclude_from_corr
        )
        report['removed_by_correlation'] = corr_removed
    
    # 3. Selecionar por importância
    if method in ['importance', 'combined']:
        X_working, imp_removed = select_by_importance(
            X_working, 
            y, 
            importance_threshold,
            sample_weight=sample_weight
        )
        report['removed_by_importance'] = imp_removed
    
    report['final_features'] = X_working.shape[1]
    report['selected_columns'] = X_working.columns.tolist()
    
    # Resumo
    logger.info("\n" + "="*80)
    logger.info("📋 RESUMO DA SELEÇÃO")
    logger.info("="*80)
    logger.info(f"📊 Features iniciais: {report['initial_features']}")
    logger.info(f"❌ Removidas por baixa variância: {len(report['removed_by_variance'])}")
    logger.info(f"❌ Removidas por correlação: {len(report['removed_by_correlation'])}")
    logger.info(f"❌ Removidas por baixa importância: {len(report['removed_by_importance'])}")
    logger.info(f"✅ Features finais: {report['final_features']}")
    logger.info(f"📉 Redução: {100 * (1 - report['final_features']/report['initial_features']):.1f}%")
    logger.info("="*80)
    
    # Salvar report
    import json
    report_file = Path('data/models/feature_selection_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"\n💾 Report salvo em: {report_file}")
    
    # Salvar lista de features selecionadas
    selected_file = Path('data/models/selected_features.joblib')
    joblib.dump(report['selected_columns'], selected_file)
    logger.info(f"💾 Features selecionadas salvas em: {selected_file}")
    
    return X_working, report

if __name__ == "__main__":
    # Teste básico
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    from ml_pipeline.data_preparation import load_historical_data
    from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG
    
    logger.info("🧪 Testando Feature Selection...")
    
    # Carregar dados
    df, weights = load_historical_data(
        seasons=ML_SEASONS,
        apply_weights=True,
        weight_config=ML_SAMPLE_WEIGHT_CONFIG
    )
    
    # Preparar X e y
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
    
    # Proteger team dummies e odds da remoção por correlação
    team_dummies = [c for c in X.columns if 'team_' in c]
    odds_features = [c for c in X.columns if 'odds' in c]
    protected = team_dummies + odds_features
    
    # Executar seleção
    X_selected, report = select_features(
        X, y, 
        method='combined',
        corr_threshold=0.95,
        importance_threshold=0.0005,  # Mais agressivo
        sample_weight=weights,
        exclude_from_corr=protected
    )
    
    logger.info(f"\n✅ Feature Selection concluída!")
    logger.info(f"   {X.shape[1]} → {X_selected.shape[1]} features")
