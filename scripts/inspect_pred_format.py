
import sys
import os
import pandas as pd
import sqlite3

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.repositories.db_manager import get_db_manager

def inspect_format():
    db = get_db_manager()
    conn = db.get_connection()
    
    print("--- Inspecting a sample of Predictions ---")
    query = "SELECT * FROM predictions LIMIT 5"
    try:
        df = pd.read_sql_query(query, conn)
        print(df[['date', 'home_team', 'away_team', 'game_id']])
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n--- Checking for 'Full Name' predictions ---")
    query_full = "SELECT count(*) as count FROM predictions WHERE home_team LIKE 'Milwaukee%'"
    try:
        df_full = pd.read_sql_query(query_full, conn)
        print(f"Contagem Milwaukee: {df_full.iloc[0]['count']}")
    except Exception as e:
        print(f"Error full: {e}")

    db.return_connection(conn)

if __name__ == "__main__":
    inspect_format()
