"""
Ensemble Model V3 - Multi-Season Training com Sample Weighting

Melhorias implementadas:
- Suporte a multi-season data (combina 2023-24, 2024-25, 2025-26)
- Sample weighting (jogos recentes têm peso maior)
- Validação de rolling features antes do treino
- Logging detalhado de métricas por temporada
- Comparação de performance com/sem pesos
- Salvamento de metadados do treinamento

Usa stacking de:
- Random Forest (otimizado)
- XGBoost  
- LightGBM
- Extra Trees

Meta-model: Logistic Regression
"""
import joblib
import pandas as pd
import numpy as np
from ml_pipeline.data_preparation import load_historical_data
import logging
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Constantes de configuração
# Temporada atual: 2025-26 (começou em outubro de 2025)
ML_SEASONS = ['2023-24', '2024-25', '2025-26']
ML_SAMPLE_WEIGHT_CONFIG = {
    'enabled': True,
    'recent_30_days': 3.0,
    'recent_60_days': 2.0,
    'recent_90_days': 1.5,
    'default': 1.0
}

def train_ensemble_model_v3(use_sample_weights=True, seasons=None):
    """
    Treina ensemble V3 com multi-season data e sample weighting.
    
    Args:
        use_sample_weights: Se True, aplica pesos baseados em recência
        seasons: Lista de temporadas (default: ML_SEASONS)
    
    Returns:
        Tuple: (model, accuracy, metadata_dict)
    """
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        use_full_ensemble = True
    except ImportError:
        logger.warning("⚠️  XGBoost/LightGBM não disponíveis. Usando RF+ExtraTrees apenas.")
        use_full_ensemble = False
    
    logger.info("="*80)
    logger.info("🎯 TREINANDO ENSEMBLE MODEL V3 (Multi-Season + Sample Weighting)")
    logger.info("="*80)
    
    # Usar seasons padrão se não especificado
    if seasons is None:
        seasons = ML_SEASONS
    
    # Carregar dados
    logger.info(f"📦 Carregando dados de temporadas: {seasons}")
    
    if use_sample_weights:
        df, sample_weights = load_historical_data(
            seasons=seasons, 
            apply_weights=True,
            weight_config=ML_SAMPLE_WEIGHT_CONFIG,
            enable_player_features=True  # NOVO: Player impact metrics
        )
    else:
        df = load_historical_data(
            seasons=seasons, 
            apply_weights=False,
            enable_player_features=True  # NOVO: Player impact metrics
        )
        sample_weights = None
    
    if df is None or df.empty:
        logger.error("❌ Nenhum dado carregado. Abortando treinamento.")
        return None, 0, {}
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # Features e target
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 # REMOVER LEAKAGE (Stats do próprio jogo)
                 'pts', 'opp_pts', 'game_id', 'id',
                 'home_off_rating', 'home_def_rating', 'home_efg_pct', 'home_ts_pct', 'home_pace', 'home_pie',
                 'away_off_rating', 'away_def_rating', 'away_efg_pct', 'away_ts_pct', 'away_pace', 'away_pie',
                 'ast', 'opp_ast', 'reb', 'opp_reb', 'tov', 'opp_tov', 'stl', 'opp_stl', 'blk', 'opp_blk',
                 'pf', 'opp_pf', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_fg3a', 'opp_ftm', 'opp_fta',
                 # REMOVER BOX SCORES BRUTOS (já estavam comentados, mas reforçando)
                 'oreb', 'dreb', 'opp_oreb', 'opp_dreb',
                 # REMOVER FOUR FACTORS BRUTOS (calculados a partir de box scores)
                 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 # REMOVER prob_home e prob_away (são do modelo antigo)
                 'prob_home', 'prob_away']
    
    # Remover colunas irrelevantes
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # One-Hot Encoding para times
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    
    # Salvar nomes das features para garantir alinhamento na predição
    feature_names = X.columns.tolist()
    rolling_features = [f for f in feature_names if 'rolling' in f]
    roster_features = [f for f in feature_names if 'roster' in f]
    team_dummies = [f for f in feature_names if 'team_' in f]
    odds_features = [f for f in feature_names if 'odds' in f]
    
    logger.info(f"📋 Features selecionadas: {len(feature_names)}")
    logger.info(f"   Rolling features (5 e 10 jogos): {len(rolling_features)}")
    logger.info(f"   Roster features: {len(roster_features)}")
    logger.info(f"   Odds features: {len(odds_features)}")
    logger.info(f"   Team dummies: {len(team_dummies)}")
    
    # Validação de rolling features críticas
    required_rolling = ['home_rolling_5_points', 'home_rolling_10_points',
                       'away_rolling_5_points', 'away_rolling_10_points']
    missing = [f for f in required_rolling if f not in feature_names]
    if missing:
        logger.error(f"❌ Features críticas faltando: {missing}")
        return None, 0, {}
    else:
        logger.info(f"✅ Rolling features críticas validadas")
    
    y = (df['winner'] == 'HOME').astype(int)
    
    # Time Series Split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if use_sample_weights:
        weights_train = sample_weights[:split_idx]
        weights_test = sample_weights[split_idx:]
        logger.info(f"✅ Sample weights preparados (treino: {len(weights_train)}, teste: {len(weights_test)})")
    else:
        weights_train = None
        weights_test = None
    
    # Treinar Random Forest primeiro com hiperparâmetros otimizados
    logger.info("🔧 Treinando Random Forest base...")
    best_rf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=8, 
        random_state=42, 
        n_jobs=-1
    )
    best_rf.fit(X_train, y_train, sample_weight=weights_train)
    logger.info(f"✅ Random Forest treinado")
    
    # Base models
    if use_full_ensemble:
        base_estimators = [
            ('rf', best_rf),
            ('xgb', XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)),
            ('lgbm', LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)),
            ('extra', ExtraTreesClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1))
        ]
    else:
        base_estimators = [
            ('rf', best_rf),
            ('extra', ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
        ]
    
    # Meta model
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    
    # Stacking
    ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )
    
    logger.info(f"🔄 Treinando {len(base_estimators)} modelos base + meta-model...")
    logger.info(f"   Com sample weighting: {'Sim' if use_sample_weights else 'Não'}")
    
    ensemble.fit(X_train, y_train, sample_weight=weights_train)
    
    # Avaliar
    accuracy = ensemble.score(X_test, y_test, sample_weight=weights_test)
    
    logger.info(f"✅ Ensemble V3 treinado")
    logger.info(f"   Acurácia no teste: {accuracy*100:.2f}%")
    
    # Feature Importance
    logger.info("📊 Calculando feature importance...")
    rf_importance = best_rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importance
    }).sort_values('importance', ascending=False)
    
    # Salvar feature importance
    feature_importance_df.to_csv('data/models/feature_importance.csv', index=False)
    
    # Log top 10 features
    logger.info("🏆 Top 10 features mais importantes:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Metadados do treinamento
    training_metadata = {
        'version': 'V3',
        'timestamp': datetime.now().isoformat(),
        'seasons': seasons,
        'sample_weighting_enabled': use_sample_weights,
        'weight_config': ML_SAMPLE_WEIGHT_CONFIG if use_sample_weights else None,
        'total_games': len(df),
        'train_games': len(X_train),
        'test_games': len(X_test),
        'accuracy': float(accuracy),
        'num_features': len(feature_names),
        'rolling_features': len(rolling_features),
        'base_models': len(base_estimators),
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        }
    }
    
    # Salvar metadados
    metadata_path = Path('data/models/training_metadata.json')
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    logger.info(f"💾 Metadados salvos: {metadata_path}")
    
    # Salvar modelos
    joblib.dump(ensemble, 'data/models/ensemble_model.joblib')
    joblib.dump(ensemble, 'data/models/ml_model.joblib')
    logger.info("💾 Modelo salvo: data/models/ensemble_model.joblib")
    logger.info("💾 Modelo salvo: data/models/ml_model.joblib")
    
    logger.info("="*80)
    logger.info(f"✅ TREINAMENTO V3 CONCLUÍDO COM SUCESSO!")
    logger.info(f"   Acurácia: {accuracy*100:.2f}%")
    logger.info(f"   Jogos: {len(df)} ({len(X_train)} treino / {len(X_test)} teste)")
    logger.info(f"   Temporadas: {', '.join(seasons)}")
    logger.info("="*80)
    
    return ensemble, accuracy, training_metadata

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    model, acc, metadata = train_ensemble_model_v3(
        use_sample_weights=True,
        seasons=ML_SEASONS
    )
    
    if model:
        print(f"\n✅ Ensemble V3 pronto! Acurácia: {acc*100:.1f}%\n")
    else:
        print("\n❌ Falha no treinamento.\n")
