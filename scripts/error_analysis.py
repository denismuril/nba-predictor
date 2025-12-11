#!/usr/bin/env python3
"""
Error Analysis Detalhado - Fase 2

Analisa onde o modelo erra mais para identificar padrões e oportunidades de melhoria.

Segmentações:
1. Por status (Favorito vs Underdog)
2. Por magnitude do spread (-3, -6, -10, etc)
3. Por total esperado (low-scoring vs high-scoring)
4. Por time específico
5. Por back-to-back vs descansado
6. Por strength of schedule

Usage:
    python scripts/error_analysis.py [--save-report]
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib

# Tentar importar matplotlib/seaborn (opcional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sem GUI
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️  Matplotlib não disponível - plots desabilitados")

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_data():
    """Carrega modelo final e dados de teste."""
    from ml_pipeline.data_preparation import load_historical_data
    from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG
    
    # Carregar dados
    df, weights = load_historical_data(
        seasons=ML_SEASONS,
        apply_weights=True,
        weight_config=ML_SAMPLE_WEIGHT_CONFIG
    )
    
    # Carregar modelo final
    model_file = Path('data/models/ensemble_model_final.joblib')
    features_file = Path('data/models/feature_names_final.joblib')
    
    if not model_file.exists():
        logger.warning("⚠️  Modelo final não encontrado, usando ensemble_model.joblib")
        model_file = Path('data/models/ensemble_model.joblib')
        features_file = Path('data/models/feature_names.joblib')
    
    model = joblib.load(model_file)
    selected_features = joblib.load(features_file)
    
    logger.info(f"✅ Modelo carregado: {model_file}")
    logger.info(f"✅ Features: {len(selected_features)}")
    
    return model, df, weights, selected_features

def prepare_predictions(model, df, selected_features):
    """Prepara previsões para análise."""
    # Preparar X
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X = df.drop(columns=drop_cols, errors='ignore')
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    X = X[selected_features]
    
    # Fazer previsões
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # Probabilidade de home win
    
    # Adicionar ao dataframe
    df_analysis = df.copy()
    df_analysis['predicted_winner'] = ['HOME' if p == 1 else 'AWAY' for p in y_pred]
    df_analysis['home_win_prob'] = y_proba
    df_analysis['correct_prediction'] = (df_analysis['predicted_winner'] == df_analysis['winner'])
    
    # Calcular point differential esperado (simplificado)
    df_analysis['expected_spread'] = df_analysis['home_score'] - df_analysis['away_score']
    
    return df_analysis

def analyze_by_favorite_status(df):
    """Analisa performance em favoritos vs underdogs."""
    logger.info("\n" + "="*80)
    logger.info("📊 ANÁLISE: FAVORITO VS UNDERDOG")
    logger.info("="*80)
    
    # Determinar favorito baseado em odds (se disponível) ou win probability
    if 'odds_home' in df.columns and df['odds_home'].notna().sum() > 0:
        df['favorite'] = np.where(df['odds_home'] < df['odds_away'], 'HOME', 'AWAY')
    else:
        df['favorite'] = np.where(df['home_win_prob'] > 0.5, 'HOME', 'AWAY')
    
    df['picked_favorite'] = df['predicted_winner'] == df['favorite']
    
    # Análise
    fav_picks = df[df['picked_favorite'] == True]
    dog_picks = df[df['picked_favorite'] == False]
    
    fav_acc = fav_picks['correct_prediction'].mean() if len(fav_picks) > 0 else 0
    dog_acc = dog_picks['correct_prediction'].mean() if len(dog_picks) > 0 else 0
    
    logger.info(f"\n🏆 Apostas em FAVORITOS:")
    logger.info(f"   Total: {len(fav_picks)} jogos")
    logger.info(f"   Accuracy: {fav_acc*100:.2f}%")
    
    logger.info(f"\n🐶 Apostas em UNDERDOGS:")
    logger.info(f"   Total: {len(dog_picks)} jogos")
    logger.info(f"   Accuracy: {dog_acc*100:.2f}%")
    
    if len(fav_picks) > 0 and len(dog_picks) > 0:
        diff = fav_acc - dog_acc
        logger.info(f"\n📊 Diferença: {diff*100:+.2f}%")
        if abs(diff) > 0.05:
            logger.warning(f"   ⚠️  Viés significativo detectado!")
    
    return {'favorite_acc': fav_acc, 'underdog_acc': dog_acc}

def analyze_by_spread_range(df):
    """Analisa performance por range de spread."""
    logger.info("\n" + "="*80)
    logger.info("📊 ANÁLISE: POR RANGE DE SPREAD")
    logger.info("="*80)
    
    # Calcular spread implícito da probabilidade
    # spread ≈ -13 * log(prob_away / prob_home) (fórmula simplificada)
    df['implied_spread'] = -13 * np.log((1 - df['home_win_prob']) / df['home_win_prob'])
    df['spread_magnitude'] = df['implied_spread'].abs()
    
    # Criar bins
    bins = [0, 3, 6, 10, 100]
    labels = ['0-3', '3-6', '6-10', '10+']
    df['spread_range'] = pd.cut(df['spread_magnitude'], bins=bins, labels=labels)
    
    logger.info(f"\n📈 Performance por range de spread:")
    for range_label in labels:
        range_df = df[df['spread_range'] == range_label]
        if len(range_df) > 0:
            acc = range_df['correct_prediction'].mean()
            logger.info(f"   {range_label} pts: {acc*100:.2f}% ({len(range_df)} jogos)")
    
    return df.groupby('spread_range')['correct_prediction'].mean().to_dict()

def analyze_by_total_range(df):
    """Analisa performance por range de total esperado."""
    logger.info("\n" + "="*80)
    logger.info("📊 ANÁLISE: POR RANGE DE TOTAL")
    logger.info("="*80)
    
    # Criar bins de total
    bins = [0, 215, 230, 1000]
    labels = ['Low (<215)', 'Medium (215-230)', 'High (>230)']
    df['total_range'] = pd.cut(df['total_points'], bins=bins, labels=labels)
    
    logger.info(f"\n🎯 Performance por range de total:")
    for range_label in labels:
        range_df = df[df['total_range'] == range_label]
        if len(range_df) > 0:
            acc = range_df['correct_prediction'].mean()
            logger.info(f"   {range_label}: {acc*100:.2f}% ({len(range_df)} jogos)")
    
    return df.groupby('total_range')['correct_prediction'].mean().to_dict()

def analyze_by_back_to_back(df):
    """Analisa performance em jogos back-to-back vs descansados."""
    logger.info("\n" + "="*80)
    logger.info("📊 ANÁLISE: BACK-TO-BACK vs DESCANSADO")
    logger.info("="*80)
    
    if 'home_is_back_to_back' not in df.columns or 'away_is_back_to_back' not in df.columns:
        logger.warning("⚠️  Features de back-to-back não encontradas")
        return {}
    
    # Jogos onde um time está b2b e outro não
    df['b2b_advantage'] = 'Neither'
    df.loc[(df['home_is_back_to_back'] == 0) & (df['away_is_back_to_back'] == 1), 'b2b_advantage'] = 'Home rested'
    df.loc[(df['home_is_back_to_back'] == 1) & (df['away_is_back_to_back'] == 0), 'b2b_advantage'] = 'Away rested'
    df.loc[(df['home_is_back_to_back'] == 1) & (df['away_is_back_to_back'] == 1), 'b2b_advantage'] = 'Both B2B'
    
    logger.info(f"\n💤 Performance por situação de descanso:")
    for situation in ['Home rested', 'Away rested', 'Both B2B', 'Neither']:
        sit_df = df[df['b2b_advantage'] == situation]
        if len(sit_df) > 0:
            acc = sit_df['correct_prediction'].mean()
            logger.info(f"   {situation}: {acc*100:.2f}% ({len(sit_df)} jogos)")
    
    return df.groupby('b2b_advantage')['correct_prediction'].mean().to_dict()

def analyze_worst_teams(df, n=5):
    """Identifica times onde o modelo erra mais."""
    logger.info("\n" + "="*80)
    logger.info(f"📊 ANÁLISE: {n} PIORES TIMES (Mais erros)")
    logger.info("="*80)
    
    # Análise por home team
    home_errors = df.groupby('home_team').agg({
        'correct_prediction': ['mean', 'count']
    }).round(4)
    home_errors.columns = ['accuracy', 'games']
    home_errors = home_errors[home_errors['games'] >= 20]  # Mínimo 20 jogos
    
    worst_home = home_errors.nsmallest(n, 'accuracy')
    
    logger.info(f"\n🏠 Piores times (como mandante):")
    for team, row in worst_home.iterrows():
        logger.info(f"   {team}: {row['accuracy']*100:.2f}% ({int(row['games'])} jogos)")
    
    # Away teams
    away_errors = df.groupby('away_team').agg({
        'correct_prediction': ['mean', 'count']
    }).round(4)
    away_errors.columns = ['accuracy', 'games']
    away_errors = away_errors[away_errors['games'] >= 20]
    
    worst_away = away_errors.nsmallest(n, 'accuracy')
    
    logger.info(f"\n✈️  Piores times (como visitante):")
    for team, row in worst_away.iterrows():
        logger.info(f"   {team}: {row['accuracy']*100:.2f}% ({int(row['games'])} jogos)")
    
    return {'worst_home': worst_home.to_dict(), 'worst_away': worst_away.to_dict()}

def create_error_distribution_plot(df, output_dir='data/models/plots'):
    """Cria visualização da distribuição de erros."""
    if not HAS_PLOTTING:
        logger.warning("⚠️  Matplotlib não disponível - pulando plots")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Accuracy por confidence level
    df['confidence'] = df['home_win_prob'].apply(lambda x: max(x, 1-x))
    df['conf_bin'] = pd.cut(df['confidence'], bins=[0.5, 0.6, 0.7, 0.8, 1.0], 
                            labels=['50-60%', '60-70%', '70-80%', '80-100%'])
    
    plt.figure(figsize=(10, 6))
    conf_acc = df.groupby('conf_bin')['correct_prediction'].agg(['mean', 'count'])
    
    ax = conf_acc['mean'].plot(kind='bar', color='steelblue')
    plt.title('Accuracy por Nível de Confiança', fontsize=14, fontweight='bold')
    plt.xlabel('Confiança da Previsão')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    
    # Adicionar contagens
    for i, (idx, row) in enumerate(conf_acc.iterrows()):
        plt.text(i, row['mean'] + 0.02, f"n={int(row['count'])}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_by_confidence.png', dpi=150, bbox_inches='tight')
    logger.info(f"💾 Plot salvo: {output_dir}/accuracy_by_confidence.png")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Error Analysis')
    parser.add_argument('--save-report', action='store_true', help='Salvar report JSON')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🔍 ERROR ANALYSIS - FASE 2")
    logger.info("="*80)
    
    # Carregar modelo e dados
    model, df, weights, features = load_model_and_data()
    
    # Usar apenas set de teste (últimos 20%)
    df_sorted = df.sort_values('date').reset_index(drop=True)
    test_start = int(len(df_sorted) * 0.8)
    df_test = df_sorted.iloc[test_start:].reset_index(drop=True)
    
    logger.info(f"📊 Analisando {len(df_test)} jogos do set de teste")
    
    # Preparar previsões
    df_analysis = prepare_predictions(model, df_test, features)
    
    # Executar análises
    report = {}
    report['overall_accuracy'] = float(df_analysis['correct_prediction'].mean())
    logger.info(f"\n✅ Accuracy geral: {report['overall_accuracy']*100:.2f}%")
    
    report['by_favorite'] = analyze_by_favorite_status(df_analysis)
    report['by_spread'] = analyze_by_spread_range(df_analysis)
    report['by_total'] = analyze_by_total_range(df_analysis)
    report['by_rest'] = analyze_by_back_to_back(df_analysis)
    report['worst_teams'] = analyze_worst_teams(df_analysis, n=5)
    
    # Criar plots
    create_error_distribution_plot(df_analysis)
    
    # Salvar report
    if args.save_report:
        report_file = Path('data/models/error_analysis_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\n💾 Report salvo: {report_file}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ ERROR ANALYSIS CONCLUÍDA")
    logger.info("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
