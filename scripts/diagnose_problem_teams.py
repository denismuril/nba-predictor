#!/usr/bin/env python3
"""
Diagnóstico de Times Problemáticos

Investiga POR QUÊ Denver Nuggets e Cleveland Cavaliers estão com baixa accuracy.
Analisa:
1. Distribuição temporal dos dados (recentes vs antigos)
2. Qualidade das features para esses times
3. Padrões específicos de erro
4. Dados faltantes ou anômalos

Usage:
    python scripts/diagnose_problem_teams.py --teams "Denver Nuggets,Cleveland Cavaliers"
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
import argparse

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_and_model():
    """Carrega dados e modelo final."""
    from ml_pipeline.data_preparation import load_historical_data
    from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG
    
    df, weights = load_historical_data(
        seasons=ML_SEASONS,
        apply_weights=True,
        weight_config=ML_SAMPLE_WEIGHT_CONFIG
    )
    
    model_file = Path('models/ensemble_v7.joblib')
    features_file = Path('models/feature_names_v7.joblib')
    
    model = joblib.load(model_file)
    selected_features = joblib.load(features_file)
    
    return df, weights, model, selected_features

def analyze_team_temporal_distribution(df, team_name):
    """Analisa distribuição temporal dos dados do time."""
    logger.info(f"\n{'='*80}")
    logger.info(f"📅 ANÁLISE TEMPORAL: {team_name}")
    logger.info(f"{'='*80}")
    
    # Filtrar jogos do time (home ou away)
    team_games = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].copy()
    team_games = team_games.sort_values('date')
    
    logger.info(f"\n📊 Total de jogos: {len(team_games)}")
    logger.info(f"   Data início: {team_games['date'].min().date()}")
    logger.info(f"   Data fim: {team_games['date'].max().date()}")
    
    # Distribuição por temporada
    team_games['season'] = team_games['date'].dt.year.astype(str) + '-' + (team_games['date'].dt.year + 1).astype(str).str[-2:]
    
    logger.info(f"\n📈 Jogos por temporada:")
    for season in sorted(team_games['season'].unique()):
        season_games = team_games[team_games['season'] == season]
        logger.info(f"   {season}: {len(season_games)} jogos")
    
    # Últimos 30/60/90 dias
    max_date = team_games['date'].max()
    days_30 = len(team_games[team_games['date'] >= (max_date - pd.Timedelta(days=30))])
    days_60 = len(team_games[team_games['date'] >= (max_date - pd.Timedelta(days=60))])
    days_90 = len(team_games[team_games['date'] >= (max_date - pd.Timedelta(days=90))])
    
    logger.info(f"\n⏰ Distribuição recente:")
    logger.info(f"   Últimos 30 dias: {days_30} jogos")
    logger.info(f"   Últimos 60 dias: {days_60} jogos")
    logger.info(f"   Últimos 90 dias: {days_90} jogos")
    
    # Verificar gaps temporais
    team_games['days_since_prev'] = team_games['date'].diff().dt.days
    avg_gap = team_games['days_since_prev'].mean()
    max_gap = team_games['days_since_prev'].max()
    
    logger.info(f"\n📊 Gaps entre jogos:")
    logger.info(f"   Média: {avg_gap:.1f} dias")
    logger.info(f"   Máximo: {max_gap:.0f} dias")
    
    if max_gap > 30:
        logger.warning(f"   ⚠️  Gap grande detectado! Pode indicar dados faltantes")
    
    return team_games

def analyze_team_features(df, team_name, selected_features):
    """Analisa qualidade das features para o time."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 ANÁLISE DE FEATURES: {team_name}")
    logger.info(f"{'='*80}")
    
    team_games = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].copy()
    
    # Preparar features para análise
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away', 'home_team', 'away_team']
    
    X_team = team_games.drop(columns=drop_cols, errors='ignore')
    
    # Features com valores faltantes
    missing_features = []
    for col in selected_features:
        if col in X_team.columns:
            missing_pct = X_team[col].isna().sum() / len(X_team) * 100
            if missing_pct > 10:
                missing_features.append((col, missing_pct))
    
    if missing_features:
        logger.warning(f"\n⚠️  Features com >10% dados faltantes:")
        for feat, pct in sorted(missing_features, key=lambda x: x[1], reverse=True)[:10]:
            logger.warning(f"   {feat}: {pct:.1f}% missing")
    else:
        logger.info(f"\n✅ Todas features com <10% missing")
    
    # Advanced features específicas
    adv_features = ['home_sos_10', 'away_sos_10', 'home_win_streak', 'away_win_streak',
                   'home_rest_days', 'away_rest_days', 'home_is_back_to_back', 'away_is_back_to_back']
    
    logger.info(f"\n🎯 Advanced Features (quando {team_name} joga em casa):")
    home_games = team_games[team_games['home_team'] == team_name]
    if len(home_games) > 0:
        for feat in ['home_sos_10', 'home_win_streak', 'home_rest_days']:
            if feat in home_games.columns:
                mean_val = home_games[feat].mean()
                logger.info(f"   {feat}: média = {mean_val:.2f}")
    
    logger.info(f"\n🎯 Advanced Features (quando {team_name} joga fora):")
    away_games = team_games[team_games['away_team'] == team_name]
    if len(away_games) > 0:
        for feat in ['away_sos_10', 'away_win_streak', 'away_rest_days']:
            if feat in away_games.columns:
                mean_val = away_games[feat].mean()
                logger.info(f"   {feat}: média = {mean_val:.2f}")
    
    return team_games

def analyze_team_predictions(df, team_name, model, selected_features):
    """Analisa previsões do modelo para o time."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🤖 ANÁLISE DE PREVISÕES: {team_name}")
    logger.info(f"{'='*80}")
    
    team_games = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].copy()
    
    # Preparar X
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X_team = team_games.drop(columns=drop_cols, errors='ignore')
    X_team = pd.get_dummies(X_team, columns=['home_team', 'away_team'], drop_first=False)
    
    # Adicionar colunas faltantes (preenchidas com 0)
    missing_cols = set(selected_features) - set(X_team.columns)
    for col in missing_cols:
        X_team[col] = 0
    
    # Reordenar para match com modelo
    X_team = X_team[selected_features]
    
    # Fazer previsões
    y_pred = model.predict(X_team)
    y_proba = model.predict_proba(X_team)[:, 1]
    
    team_games['predicted_winner'] = ['HOME' if p == 1 else 'AWAY' for p in y_pred]
    team_games['home_win_prob'] = y_proba
    team_games['correct'] = (team_games['predicted_winner'] == team_games['winner'])
    
    # Accuracy geral
    overall_acc = team_games['correct'].mean()
    logger.info(f"\n📊 Accuracy geral para {team_name}: {overall_acc*100:.2f}%")
    
    # Quando favorito vs underdog
    team_games['is_favorite'] = np.where(
        ((team_games['home_team'] == team_name) & (team_games['home_win_prob'] > 0.5)) |
        ((team_games['away_team'] == team_name) & (team_games['home_win_prob'] < 0.5)),
        True, False
    )
    
    fav_acc = team_games[team_games['is_favorite']]['correct'].mean()
    dog_acc = team_games[~team_games['is_favorite']]['correct'].mean()
    
    logger.info(f"\n📈 Como FAVORITO: {fav_acc*100:.2f}% ({team_games['is_favorite'].sum()} jogos)")
    logger.info(f"📉 Como UNDERDOG: {dog_acc*100:.2f}% ({(~team_games['is_favorite']).sum()} jogos)")
    
    # Padrões de erro
    errors = team_games[~team_games['correct']]
    logger.info(f"\n❌ Erros: {len(errors)} jogos")
    
    if len(errors) > 0:
        # Média de confiança nos erros
        avg_conf_errors = errors['home_win_prob'].apply(lambda x: max(x, 1-x)).mean()
        logger.info(f"   Confiança média nos erros: {avg_conf_errors*100:.1f}%")
        
        # Erros recentes vs antigos
        max_date = team_games['date'].max()
        recent_errors = errors[errors['date'] >= (max_date - pd.Timedelta(days=60))]
        old_errors = errors[errors['date'] < (max_date - pd.Timedelta(days=60))]
        
        logger.info(f"\n   Erros recentes (60 dias): {len(recent_errors)}")
        logger.info(f"   Erros antigos (>60 dias): {len(old_errors)}")
        
        if len(recent_errors) > len(old_errors) * 0.5:
            logger.warning(f"   ⚠️  Muitos erros RECENTES! Pode ser mudança no time/roster")
    
    return team_games

def main():
    parser = argparse.ArgumentParser(description='Diagnóstico de Times Problemáticos')
    parser.add_argument('--teams', type=str, default='Denver Nuggets,Cleveland Cavaliers',
                       help='Times para diagnosticar (separados por vírgula)')
    args = parser.parse_args()
    
    teams = [t.strip() for t in args.teams.split(',')]
    
    logger.info("="*80)
    logger.info("🔍 DIAGNÓSTICO DE TIMES PROBLEMÁTICOS")
    logger.info("="*80)
    logger.info(f"Times a analisar: {', '.join(teams)}")
    
    # Carregar dados e modelo
    df, weights, model, features = load_data_and_model()
    
    logger.info(f"Times encontrados no dataset: {sorted(df['home_team'].unique())}")
    
    # Usar apenas set de teste
    df_sorted = df.sort_values('date').reset_index(drop=True)
    test_start = int(len(df_sorted) * 0.8)
    df_test = df_sorted.iloc[test_start:].reset_index(drop=True)
    
    logger.info(f"\n📊 Analisando set de teste: {len(df_test)} jogos")
    
    # Analisar cada time
    report = {}
    
    for team in teams:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# {team.upper()}")
        logger.info(f"{'#'*80}")
        
        # 1. Distribuição temporal
        temporal_data = analyze_team_temporal_distribution(df, team)
        
        # 2. Qualidade de features
        feature_data = analyze_team_features(df, team, features)
        
        # 3. Análise de previsões (apenas no set de teste)
        prediction_data = analyze_team_predictions(df_test, team, model, features)
        
        report[team] = {
            'total_games': len(temporal_data),
            'test_games': len(prediction_data),
            'test_accuracy': float(prediction_data['correct'].mean()),
            'errors_recent': int((~prediction_data['correct'] & 
                                 (prediction_data['date'] >= prediction_data['date'].max() - pd.Timedelta(days=60))).sum()),
            'errors_old': int((~prediction_data['correct'] & 
                              (prediction_data['date'] < prediction_data['date'].max() - pd.Timedelta(days=60))).sum())
        }
    
    # Salvar report
    report_file = Path('data/models/team_diagnosis_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n\n{'='*80}")
    logger.info(f"💾 Report salvo: {report_file}")
    logger.info(f"{'='*80}")
    
    # Conclusões
    logger.info(f"\n📋 CONCLUSÕES:")
    for team, data in report.items():
        logger.info(f"\n{team}:")
        logger.info(f"   Accuracy no teste: {data['test_accuracy']*100:.2f}%")
        logger.info(f"   Erros recentes: {data['errors_recent']}")
        logger.info(f"   Erros antigos: {data['errors_old']}")
        
        if data['errors_recent'] > data['errors_old']:
            logger.warning(f"   ⚠️  PROBLEMA RECENTE! Investigar mudanças no roster/forma")
        elif data['errors_old'] > data['errors_recent']:
            logger.info(f"   ✅ Melhorando com o tempo")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
