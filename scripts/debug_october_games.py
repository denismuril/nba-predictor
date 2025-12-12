
import sys
import os
import pandas as pd
import sqlite3

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.repositories.db_manager import get_db_manager

def debug_oct():
    db = get_db_manager()
    conn = db.get_connection()
    
    print("--- Games found in DB for October 2025 ---")
    query_games = "SELECT game_id, date, home_team, away_team, home_score FROM games WHERE date BETWEEN '2025-10-01' AND '2025-10-31' ORDER BY date LIMIT 20"
    try:
        df_games = pd.read_sql_query(query_games, conn)
        print(df_games)
    except Exception as e:
        print(f"Error games: {e}")

    print("\n--- Predictions in DB for October 2025 ---")
    query_preds = "SELECT game_id, date, home_team, away_team FROM predictions WHERE date BETWEEN '2025-10-01' AND '2025-10-31' ORDER BY date LIMIT 20"
    try:
        df_preds = pd.read_sql_query(query_preds, conn)
        print(df_preds)
    except Exception as e:
        print(f"Error preds: {e}")

    db.return_connection(conn)

if __name__ == "__main__":
    debug_oct()
