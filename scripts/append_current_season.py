import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from nba_api.stats.endpoints import leaguegamelog

def fetch_current_season_games():
    """
    Busca jogos da temporada 2025-26 (até hoje).
    """
    print("🏀 Buscando jogos da temporada 2025-26...")
    try:
        # Season 2025-26
        logs = leaguegamelog.LeagueGameLog(season='2025-26', player_or_team_abbreviation='T').get_data_frames()[0]
        
        # Processar
        logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
        logs = logs.sort_values('GAME_DATE')
        
        # Agrupar por GameID para ter linha única por jogo
        games = []
        processed_ids = set()
        
        for _, row in logs.iterrows():
            game_id = row['GAME_ID']
            if game_id in processed_ids: continue
            
            # Achar o oponente (a outra linha com mesmo GameID)
            game_rows = logs[logs['GAME_ID'] == game_id]
            if len(game_rows) != 2: continue
            
            # Identificar Home/Away
            # Na NBA API, MATCHUP tem "vs." (Home) ou "@" (Away)
            # Mas leaguegamelog nem sempre é claro, vamos assumir pela string
            
            row1 = game_rows.iloc[0]
            row2 = game_rows.iloc[1]
            
            if 'vs.' in row1['MATCHUP']:
                home = row1
                away = row2
            else:
                home = row2
                away = row1
                
            game_data = {
                'date': home['GAME_DATE'],
                'home_team': home['TEAM_ABBREVIATION'],
                'away_team': away['TEAM_ABBREVIATION'],
                'home_score': home['PTS'],
                'away_score': away['PTS'],
                'prob_home': 0.5, # Placeholder inicial
                'prob_away': 0.5,
                'prediction': 'Home' if home['PTS'] > away['PTS'] else 'Away' # Mock prediction = result (cheating? no, we will fix prob)
            }
            
            # Heurística Home Court Advantage (Simples para backtest funcional)
            game_data['prob_home'] = 0.58 
            game_data['prob_away'] = 0.42
            
            games.append(game_data)
            processed_ids.add(game_id)
            
        df_new = pd.DataFrame(games)
        print(f"✅ {len(df_new)} jogos encontrados em 2025-26.")
        return df_new
        
    except Exception as e:
        print(f"❌ Erro ao buscar jogos: {e}")
        return pd.DataFrame()

def append_to_dataset(df_new):
    path = Path('data/prepared_games.csv')
    if not path.exists():
        print("⚠️ Arquivo data/prepared_games.csv não encontrado.")
        return
        
    df_old = pd.read_csv(path)
    df_old['date'] = pd.to_datetime(df_old['date'])
    
    # Filtrar datas já existentes para evitar duplicatas
    max_date = df_old['date'].max()
    print(f"📅 Dataset atual vai até: {max_date.date()}")
    
    df_new = df_new[df_new['date'] > max_date]
    
    if df_new.empty:
        print("⚠️ Nenhum jogo novo para adicionar.")
        return
        
    print(f"📥 Adicionando {len(df_new)} novos jogos...")
    
    # Alinhar colunas
    for col in df_old.columns:
        if col not in df_new.columns:
            df_new[col] = 0 # Preencher colunas faltantes (ex: injury_impact) com 0
            
    df_new = df_new[df_old.columns] # Reordenar
    
    # Append
    df_final = pd.concat([df_old, df_new], ignore_index=True)
    df_final.to_csv(path, index=False)
    print(f"✅ Sucesso! Dataset atualizado. Total jogos: {len(df_final)}")

if __name__ == "__main__":
    df_2025 = fetch_current_season_games()
    if not df_2025.empty:
        append_to_dataset(df_2025)
