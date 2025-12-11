import sys
import os
import logging
import requests
from datetime import datetime

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.repositories.db_manager import get_db_manager
from data.scrapers.results_scraper import get_game_results, normalize_team_name
from config.constants import TEAM_ABBREV_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def force_update_24():
    print("🚀 Iniciando diagnóstico para 24/11/2024...")
    
    db = get_db_manager()
    
    # 1. Verificar o que tem no banco para o dia 24
    print("\n📊 JOGOS NO BANCO (2024-11-24):")
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, date, home_team, away_team, winner FROM predictions WHERE date = '2024-11-24'")
        games = cursor.fetchall()
        
    if not games:
        print("❌ Nenhum jogo encontrado no banco para 2024-11-24.")
    else:
        for g in games:
            status = "✅ Finalizado" if g[4] else "⏳ Pendente"
            print(f"   - [{status}] ID: {g[0]} | {g[2]} vs {g[3]}")

    # 2. Buscar na ESPN
    print("\n📡 BUSCANDO NA ESPN (20241124)...")
    results = get_game_results(game_date="20241124")
    
    print(f"\n📥 Resultados encontrados: {len(results)}")
    for r in results:
        print(f"   - ID ESPN: {r['id']} | {r['home_team']} ({r['home_score']}) vs {r['away_team']} ({r['away_score']})")
        
    # 3. Tentar Match Manual
    print("\n🔄 TENTANDO MATCH:")
    updates = 0
    for g in games:
        g_id = g[0]
        g_home = g[2]
        g_away = g[3]
        
        # Tentar encontrar pelo ID exato
        match = next((r for r in results if r['id'] == g_id), None)
        
        if match:
            print(f"   ✅ Match ID Exato: {g_id}")
            db.update_game_result(g_id, match['home_score'], match['away_score'])
            updates += 1
        else:
            print(f"   ⚠️  Sem match de ID para {g_id}")
            # Tentar match por times
            fuzzy_match = next((r for r in results if normalize_team_name(r['home_team']) == normalize_team_name(g_home)), None)
            if fuzzy_match:
                 print(f"      💡 Match por Home Team encontrado: {fuzzy_match['id']}")
                 # Atualizar mesmo com ID diferente? Melhor não arriscar sem confirmar.
                 # Mas podemos atualizar o ID no banco se for o caso.
            else:
                 print(f"      ❌ Nenhum match encontrado.")

    print(f"\n🏁 Atualizações realizadas: {updates}")

if __name__ == "__main__":
    force_update_24()
