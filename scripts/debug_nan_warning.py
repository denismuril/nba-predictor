
import sys
import pandas as pd
import logging
from pathlib import Path

# Setup paths
sys.path.append(str(Path.cwd()))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ml_pipeline.data_preparation import load_historical_data
from utils.team_normalization import normalize_team

def debug_nans():
    print("🚀 Iniciando debug de NaNs...")
    
    # Carregar dados
    df = load_historical_data(seasons=['2023-24', '2024-25'], raw=False)
    
    if df is None:
        print("❌ Nenhum dado carregado.")
        return

    # Verificar se as colunas rolling existem
    rolling_cols = [c for c in df.columns if 'rolling' in c]
    if not rolling_cols:
        print("❌ Nenhuma coluna rolling encontrada.")
        return
        
    sample_col = 'home_rolling_5_points'
    if sample_col not in df.columns:
        print(f"❌ Coluna {sample_col} não encontrada.")
        return

    # Filtrar NaNs
    nans = df[df[sample_col].isna()]
    nan_pct = len(nans) / len(df) * 100
    
    print(f"\n📊 Total de jogos: {len(df)}")
    print(f"⚠️ Jogos com NaN em {sample_col}: {len(nans)} ({nan_pct:.1f}%)")
    
    if len(nans) > 0:
        print("\n🔍 Primeiros 5 jogos com NaN:")
        print(nans[['date', 'home_team', 'away_team', sample_col]].head(5))
        
        # Verificar o histórico desses times
        team = nans.iloc[0]['home_team']
        date = nans.iloc[0]['date']
        print(f"\n🔍 Histórico do time {team} antes de {date}:")
        
        team_games = df[((df['home_team'] == team) | (df['away_team'] == team)) & (df['date'] <= date)].sort_values('date')
        print(team_games[['date', 'home_team', 'away_team', 'home_score', 'away_score']].tail(5))
        
        # Verificar se é o primeiro jogo
        first_game = team_games.iloc[0]['date']
        print(f"\n📅 Primeiro jogo de {team} no dataset: {first_game}")
        if first_game == date:
            print("✅ É o primeiro jogo do time no dataset. NaN é esperado.")
        else:
            print("❌ NÃO é o primeiro jogo. Problema de normalização ou gap?")
            
    else:
        print("✅ Zero NaNs encontrados!")

if __name__ == "__main__":
    debug_nans()
