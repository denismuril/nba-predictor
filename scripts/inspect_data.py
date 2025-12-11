import sqlite3
import pandas as pd
from pathlib import Path

# Caminho do banco
db_path = Path("data/nba_history.db")

def inspect_data():
    if not db_path.exists():
        print(f"❌ Banco de dados não encontrado em {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        
        # 1. Total de jogos
        total = pd.read_sql_query("SELECT COUNT(*) as count FROM predictions", conn).iloc[0]['count']
        print(f"📊 Total de jogos no banco: {total}")

        # 2. Jogos com vencedor definido (já aconteceram)
        finished = pd.read_sql_query("SELECT COUNT(*) as count FROM predictions WHERE winner IS NOT NULL", conn).iloc[0]['count']
        print(f"✅ Jogos finalizados: {finished}")

        # 3. Jogos finalizados mas sem estatísticas (ex: fgm é null ou 0)
        # Assumindo que se fgm é null ou 0, as stats estão faltando
        missing_stats_query = """
        SELECT * FROM predictions 
        WHERE winner IS NOT NULL 
        AND (fgm IS NULL OR fgm = 0)
        """
        missing_stats = pd.read_sql_query(missing_stats_query, conn)
        print(f"⚠️ Jogos finalizados sem estatísticas (FGM=0 ou NULL): {len(missing_stats)}")
        
        if not missing_stats.empty:
            print("\nExemplos de jogos sem stats:")
            print(missing_stats[['date', 'home_team', 'away_team', 'winner', 'fgm']].head())
            
            # Salvar IDs para uso posterior
            missing_stats[['id']].to_csv("missing_stats_ids.csv", index=False)
            print("\n💾 IDs salvos em missing_stats_ids.csv")

        conn.close()

    except Exception as e:
        print(f"❌ Erro ao ler banco: {e}")

if __name__ == "__main__":
    inspect_data()
