
import sys
import os
import pandas as pd
import sqlite3

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.repositories.db_manager import get_db_manager

def inspect():
    db = get_db_manager()
    conn = db.get_connection()
    
    print("--- Predictions for 2025-11-21 (Targeting MIL vs PHI) ---")
    query_preds = "SELECT date, home_team, away_team FROM predictions WHERE date LIKE '2025-11-21%'"
    try:
        df_preds = pd.read_sql_query(query_preds, conn)
        print(df_preds)
    except Exception as e:
        print(f"Error preds: {e}")

    print("\n--- Games Table for MIL or PHI (Any Date) ---")
    # Check if the game exists at all
    query_games = "SELECT game_id, date, home_team, away_team, home_score FROM games WHERE (home_team = 'MIL' AND away_team = 'PHI') OR (home_team = 'PHI' AND away_team = 'MIL')"
    try:
        df_games = pd.read_sql_query(query_games, conn)
        print(df_games)
    except Exception as e:
        print(f"Error games: {e}")
        
    print("\n--- Games Table around 2025-11-21 ---")
    query_games_date = "SELECT game_id, date, home_team, away_team FROM games WHERE date BETWEEN '2025-11-20' AND '2025-11-22'"
    try:
        df_games_date = pd.read_sql_query(query_games_date, conn)
        print(df_games_date)
    except Exception as e:
        print(f"Error games date: {e}")

    db.return_connection(conn)

if __name__ == "__main__":
    inspect()
