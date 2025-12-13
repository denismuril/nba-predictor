
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

def debug_focused():
    print("🚀 Iniciando debug focado em LAL vs GSW...")
    
    # Carregar dados
    df = load_historical_data(seasons=['2023-24', '2024-25'], raw=False)
    
    if df is None: return

    # Procurar o jogo específico
    target_date = '2024-01-01'
    # Ajuste de fuso horário pode ser necessário, vamos buscar por string aproximada
    mask = (df['date'].astype(str).str.contains('2024-01-27')) & (df['home_team'] == 'GSW') & (df['away_team'] == 'LAL')
    # Nota: O jogo LAL @ GSW foi em 27/01/2024 foi duplo OT.
    # O user reportou 2024-01-01 LAL GSW... pode ser erro de data no report ou outro jogo.
    
    # Vamos filtrar qualquer LAL vs GSW
    gs_lal = df[((df['home_team'] == 'LAL') & (df['away_team'] == 'GSW')) | 
                ((df['home_team'] == 'GSW') & (df['away_team'] == 'LAL'))]
    
    print(f"\nJogos LAL vs GSW encontrados: {len(gs_lal)}")
    print(gs_lal[['date', 'home_team', 'away_team', 'home_rolling_5_points']].head(10))

    # Verificar se algum tem NaN
    nans = gs_lal[gs_lal['home_rolling_5_points'].isna()]
    if not nans.empty:
        print("\n⚠️ NaNs encontrados nestes jogos:")
        print(nans[['date', 'home_team', 'away_team']])
        
        # Verificar quantos jogos anteriores existiam para o time da casa
        for idx, row in nans.iterrows():
            team = row['home_team']
            date = row['date']
            prev_games = df[(df['date'] < date) & ((df['home_team'] == team) | (df['away_team'] == team))]
            print(f"Jogos anteriores de {team} antes de {date}: {len(prev_games)}")
    else:
        print("\n✅ Nenhum NaN encontrado nos jogos LAL vs GSW.")

if __name__ == "__main__":
    debug_focused()
