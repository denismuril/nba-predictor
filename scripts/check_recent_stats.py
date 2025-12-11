import sys
import os
import pandas as pd

# Adicionar raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.repositories.db_manager import DatabaseManager

def check_recent_stats():
    db = DatabaseManager()
    
    query = """
    SELECT 
        g.date,
        g.home_team,
        g.home_score,
        g.away_score,
        gs.fgm,
        gs.fga,
        gs.fg3m,
        gs.efg_pct
    FROM games g
    JOIN game_stats gs ON g.game_id = gs.game_id AND g.home_team = gs.team_id
    WHERE g.date >= '2025-11-25' AND (g.home_team IN ('PHX', 'PHO') OR g.away_team IN ('PHX', 'PHO'))
    ORDER BY g.date DESC
    """
    
    try:
        with db.get_connection() as conn:
            df = pd.read_sql(query, conn)
            print("Recent Games Stats (PHX/PHO):")
            print(df)
            
            # Check for zeros
            zeros = df[(df['fgm'] == 0) | (df['fga'] == 0)]
            if not zeros.empty:
                print("\nWARNING: Found games with ZERO stats:")
                print(zeros)
            else:
                print("\nAll recent games have stats.")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_recent_stats()
