
import sys
import os
import pandas as pd
import sqlite3
from datetime import timedelta

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.repositories.db_manager import get_db_manager


# Import normalization util
from utils.team_normalization import normalize_team

def align_dates():
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    print("🚀 Iniciando alinhamento de datas das previsões...")
    
    # 1. Fetch orphaned predictions
    print("🔍 Buscando previsões órfãs...")
    
    # Selecting ALL predictions to verify alignment
    df_preds = pd.read_sql_query("SELECT game_id, date, home_team, away_team FROM predictions", conn)
    df_games = pd.read_sql_query("SELECT game_id, date, home_team, away_team FROM games", conn)
    
    # Prepare games lookup: (HomeAbbr, AwayAbbr) -> [(Date, GameID)]
    games_lookup = {}
    for _, row in df_games.iterrows():
        # Games table uses Abbreviations (e.g. MIL, PHI)
        h_abbr = row['home_team']
        a_abbr = row['away_team']
        
        key = (h_abbr, a_abbr)
        if key not in games_lookup:
            games_lookup[key] = []
        games_lookup[key].append( (pd.to_datetime(row['date']), row['game_id']) )
    
    updates_count = 0
    ph = "%s" if db_manager.db_type == 'postgres' else "?"
    
    for _, pred in df_preds.iterrows():
        # Predictions table uses Full Names (e.g. Milwaukee Bucks)
        # We must normalize to match the lookup keys
        p_home_full = pred['home_team']
        p_away_full = pred['away_team']
        
        p_home_abbr = normalize_team(p_home_full)
        p_away_abbr = normalize_team(p_away_full)
        
        p_date = pd.to_datetime(pred['date'])
        p_id = pred['game_id']
        
        # Look for match using ABBREVIATIONS
        candidates = games_lookup.get((p_home_abbr, p_away_abbr), [])
        
        best_match = None
        
        for g_date, g_id in candidates:
            # Check date diff (allow +/- 1 day)
            diff = abs((g_date - p_date).days)
            if diff <= 1:
                # Found a match!
                best_match = (g_id, g_date)
                break
        
        if best_match:
            new_id, new_date_ts = best_match
            new_date_str = new_date_ts.strftime('%Y-%m-%d')
            
            # If current ID doesn't match the Found Game ID, update it
            if new_id != p_id:
                print(f"🔄 Corrigindo: {p_id} ({p_date.date()}) -> {new_id} ({new_date_str}) [Matches: {p_home_abbr} vs {p_away_abbr}]")
                
                try:
                    # Update Prediction
                    # Handle duplicate key constraint: if target ID exists, delete this orphan row.
                    # First, try update.
                    query = f"UPDATE predictions SET game_id = {ph}, date = {ph} WHERE game_id = {ph}"
                    cursor.execute(query, (new_id, new_date_str, p_id))
                    updates_count += 1
                except Exception as e:
                    # Likely Duplicate Key Error (Unique Constraint)
                    # If so, this orphan is a duplicate. DELETE it.
                    conn.rollback()
                    try:
                        print(f"⚠️ Update falhou (possível duplicata). Tentando DELETAR órfão {p_id}...")
                        query_del = f"DELETE FROM predictions WHERE game_id = {ph}"
                        cursor.execute(query_del, (p_id,))
                        print(f"🗑️ Deletado com sucesso: {p_id}")
                        updates_count += 1
                    except Exception as e2:
                        conn.rollback()
                        print(f"❌ Falha fatal ao deletar {p_id}: {e2}")

    conn.commit()
    conn.close()
    
    print(f"\n✅ Alinhamento concluído! {updates_count} registros corrigidos (atualizados ou deletados).")

if __name__ == "__main__":
    align_dates()
