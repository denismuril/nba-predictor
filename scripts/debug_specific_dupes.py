
import sys
import os
import pandas as pd
import sqlite3

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.repositories.db_manager import get_db_manager

def check_dupes():
    db = get_db_manager()
    conn = db.get_connection()
    
    print("--- Checking Predictions for MIL vs PHI (Any Date) ---")
    query = """
    SELECT * FROM predictions 
    WHERE (home_team LIKE 'Milwaukee%' AND away_team LIKE 'Phil%') 
       OR (home_team LIKE 'Phil%' AND away_team LIKE 'Milwaukee%')
    ORDER BY date
    """
    try:
        df = pd.read_sql_query(query, conn)
        print(df[['game_id', 'date', 'home_team', 'away_team', 'home_score']])
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Checking Predictions for SAS vs ATL (Any Date) ---")
    query_sas = """
    SELECT * FROM predictions 
    WHERE (home_team = 'SAS' AND away_team = 'ATL') 
       OR (home_team = 'ATL' AND away_team = 'SAS')
    ORDER BY date
    """
    try:
        df_sas = pd.read_sql_query(query_sas, conn)
        print(df_sas[['game_id', 'date', 'home_team', 'away_team', 'home_score']])
    except Exception as e:
        print(f"Error SAS: {e}")

    db.return_connection(conn)

if __name__ == "__main__":
    check_dupes()
