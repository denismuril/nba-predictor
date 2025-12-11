import sys
import os
import logging
from datetime import datetime

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.repositories.db_manager import get_db_manager
from data.scrapers.results_scraper import get_game_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_results():
    db = get_db_manager()
    
    # 1. Jogos Pendentes
    pending = db.get_pending_games()
    print("\n📋 JOGOS PENDENTES NO BANCO:")
    if pending.empty:
        print("Nenhum jogo pendente.")
    else:
        for _, row in pending.iterrows():
            print(f"ID: {row['id']} | {row['date']} | {row['home_team']} vs {row['away_team']}")

    # 2. Resultados da ESPN (Dia 24)
    print("\n📡 RESULTADOS ESPN (2024-11-24):")
    results = get_game_results(game_date="20241124")
    
    if not results:
        print("Nenhum resultado encontrado na ESPN.")
    else:
        for r in results:
            print(f"ID Gerado: {r['id']} | {r['date']} | {r['home_team']} ({r['home_score']}) vs {r['away_team']} ({r['away_score']})")

    # 3. Comparação
    print("\n🔍 ANÁLISE DE COMPATIBILIDADE:")
    if not pending.empty and results:
        for _, p_row in pending.iterrows():
            p_id = p_row['id']
            match = False
            for r in results:
                if r['id'] == p_id:
                    print(f"✅ MATCH: {p_id}")
                    match = True
                    break
            if not match:
                print(f"❌ SEM MATCH: {p_id}")
                # Tentar encontrar motivo
                for r in results:
                    if r['date'] == p_row['date']:
                        print(f"   Possível candidato: {r['id']} (Home: {r['home_team']} vs {p_row['home_team']})")

if __name__ == "__main__":
    debug_results()
