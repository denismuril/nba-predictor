#!/usr/bin/env python3
"""
Investigação Detalhada - Cleveland Cavaliers
Foco: Anomalia nos rest_days (7.61 dias médio)
"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

import pandas as pd
import logging
from ml_pipeline.data_preparation import load_historical_data
from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carregar dados
df, weights = load_historical_data(
    seasons=ML_SEASONS,
    apply_weights=True,
    weight_config=ML_SAMPLE_WEIGHT_CONFIG
)

# Filtrar jogos do Cleveland
cle_games = df[(df['home_team'] == 'CLE') | (df['away_team'] == 'CLE')].copy()
cle_games = cle_games.sort_values('date')

logger.info(f"\n{'='*80}")
logger.info(f"INVESTIGAÇÃO CLEVELAND - REST DAYS ANOMALY")
logger.info(f"{'='*80}\n")

# 1. Verificar rest_days quando CLE joga fora
cle_away = cle_games[cle_games['away_team'] == 'CLE'].copy()
logger.info(f"Total de jogos fora: {len(cle_away)}")

if 'away_rest_days' in cle_away.columns:
    logger.info(f"\n📊 Estatísticas de away_rest_days:")
    logger.info(f"   Média: {cle_away['away_rest_days'].mean():.2f} dias")
    logger.info(f"   Mediana: {cle_away['away_rest_days'].median():.2f} dias")
    logger.info(f"   Mínimo: {cle_away['away_rest_days'].min():.2f} dias")
    logger.info(f"   Máximo: {cle_away['away_rest_days'].max():.2f} dias")
    logger.info(f"   Std Dev: {cle_away['away_rest_days'].std():.2f} dias")
    
    # Percentis
    logger.info(f"\n   Percentis:")
    for p in [25, 50, 75, 90, 95]:
        val = cle_away['away_rest_days'].quantile(p/100)
        logger.info(f"      {p}%: {val:.2f} dias")
    
    # Distribuição
    logger.info(f"\n📈 Distribuição de valores:")
    logger.info(f"   0-2 dias: {(cle_away['away_rest_days'] <= 2).sum()} jogos")
    logger.info(f"   2-5 dias: {((cle_away['away_rest_days'] > 2) & (cle_away['away_rest_days'] <= 5)).sum()} jogos")
    logger.info(f"   5-10 dias: {((cle_away['away_rest_days'] > 5) & (cle_away['away_rest_days'] <= 10)).sum()} jogos")
    logger.info(f"   >10 dias: {(cle_away['away_rest_days'] > 10).sum()} jogos")
    logger.info(f"   >30 dias: {(cle_away['away_rest_days'] > 30).sum()} jogos")
    logger.info(f"   >100 dias: {(cle_away['away_rest_days'] > 100).sum()} jogos")
    
    # Casos extremos (>30 dias)
    extreme_cases = cle_away[cle_away['away_rest_days'] > 30].copy()
    if len(extreme_cases) > 0:
        logger.warning(f"\n⚠️ {len(extreme_cases)} jogos com >30 dias de descanso:")
        for idx, row in extreme_cases.head(10).iterrows():
            logger.warning(f"   {row['date'].date()}: {row['away_team']}@{row['home_team']} - {row['away_rest_days']:.0f} dias")

# 2. Comparar com outros times
logger.info(f"\n{'='*80}")
logger.info(f"COMPARAÇÃO COM OUTROS TIMES")
logger.info(f"{'='*80}\n")

all_teams = sorted(df['home_team'].unique())
team_rest_stats = []

for team in all_teams:
    team_away = df[df['away_team'] == team]
    if 'away_rest_days' in team_away.columns and len(team_away) > 0:
        mean_rest = team_away['away_rest_days'].mean()
        median_rest = team_away['away_rest_days'].median()
        max_rest = team_away['away_rest_days'].max()
        team_rest_stats.append({
            'team': team,
            'mean': mean_rest,
            'median': median_rest,
            'max': max_rest
        })

rest_df = pd.DataFrame(team_rest_stats).sort_values('mean', ascending=False)

logger.info("Top 10 times com MAIOR média de rest_days (fora):")
for idx, row in rest_df.head(10).iterrows():
    flag = "🔴" if row['team'] == 'CLE' else "  "
    logger.info(f"   {flag} {row['team']}: média={row['mean']:.2f}, mediana={row['median']:.2f}, max={row['max']:.0f}")

logger.info("\nTop 10 times com MENOR média de rest_days (fora):")
for idx, row in rest_df.tail(10).iterrows():
    logger.info(f"      {row['team']}: média={row['mean']:.2f}, mediana={row['median']:.2f}, max={row['max']:.0f}")

# 3. Verificar sequência de datas
logger.info(f"\n{'='*80}")
logger.info(f"VERIFICAÇÃO DE DATAS - CLEVELAND")
logger.info(f"{'='*80}\n")

cle_games['days_since_prev'] = cle_games['date'].diff().dt.days
logger.info("Últimos 20 jogos (ordem cronológica):")
for idx, row in cle_games.tail(20).iterrows():
    home_away = "CASA" if row['home_team'] == 'CLE' else "FORA"
    days_gap = row['days_since_prev'] if pd.notna(row['days_since_prev']) else 0
    rest_val = row['home_rest_days'] if row['home_team'] == 'CLE' else row.get('away_rest_days', 0)
    logger.info(f"   {row['date'].date()} | {home_away:4} | Gap real: {days_gap:3.0f}d | rest_days: {rest_val:.1f}d")

logger.info(f"\n✅ Investigação concluída.")
