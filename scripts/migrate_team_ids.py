#!/usr/bin/env python
"""
Script de migração para normalizar nomes de times no database.

REFATORADO: Suporta SQLite e PostgreSQL via DatabaseManager.

Converte variações antigas (BKN, PHX, CHA) para os IDs canônicos (BRK, PHO, CHO).
"""
import sys
from pathlib import Path

# Adicionar diretório raiz ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import get_db_manager

# Mapeamento de conversão
TEAM_CONVERSIONS = {
    'BKN': 'BRK',  # Brooklyn Nets
    'PHX': 'PHO',  # Phoenix Suns
    'CHA': 'CHO',  # Charlotte Hornets
}

def migrate_team_names():
    """Migra nomes de times no database para IDs canônicos."""
    
    db = get_db_manager()
    print(f"📂 Database: {db.db_type.upper()}")
    
    print("="*60)
    print("MIGRAÇÃO DE NOMES DE TIMES")
    print("="*60)
    print(f"Conversões: {TEAM_CONVERSIONS}")
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Atualizar tabela games
        print("\n1. Atualizando tabela 'games'...")
        
        for old_id, new_id in TEAM_CONVERSIONS.items():
            # Home team
            cursor.execute(
                db._prepare_query("SELECT COUNT(*) FROM games WHERE home_team = ?"),
                (old_id,)
            )
            home_count = cursor.fetchone()[0]
            
            if home_count > 0:
                cursor.execute(
                    db._prepare_query("UPDATE games SET home_team = ? WHERE home_team = ?"),
                    (new_id, old_id)
                )
                print(f"   Home: {old_id} -> {new_id} ({home_count} jogos)")
            
            # Away team
            cursor.execute(
                db._prepare_query("SELECT COUNT(*) FROM games WHERE away_team = ?"),
                (old_id,)
            )
            away_count = cursor.fetchone()[0]
            
            if away_count > 0:
                cursor.execute(
                    db._prepare_query("UPDATE games SET away_team = ? WHERE away_team = ?"),
                    (new_id, old_id)
                )
                print(f"   Away: {old_id} -> {new_id} ({away_count} jogos)")
        
        # 2. Atualizar tabela game_stats
        print("\n2. Atualizando tabela 'game_stats'...")
        
        for old_id, new_id in TEAM_CONVERSIONS.items():
            cursor.execute(
                db._prepare_query("SELECT COUNT(*) FROM game_stats WHERE team_id = ?"),
                (old_id,)
            )
            stats_count = cursor.fetchone()[0]
            
            if stats_count > 0:
                cursor.execute(
                    db._prepare_query("UPDATE game_stats SET team_id = ? WHERE team_id = ?"),
                    (new_id, old_id)
                )
                print(f"   Stats: {old_id} -> {new_id} ({stats_count} registros)")
        
        # 3. Atualizar tabela predictions
        print("\n3. Atualizando tabela 'predictions'...")
        
        for old_id, new_id in TEAM_CONVERSIONS.items():
            cursor.execute(
                db._prepare_query("SELECT COUNT(*) FROM predictions WHERE home_team = ?"),
                (old_id,)
            )
            home_count = cursor.fetchone()[0]
            
            if home_count > 0:
                cursor.execute(
                    db._prepare_query("UPDATE predictions SET home_team = ? WHERE home_team = ?"),
                    (new_id, old_id)
                )
                print(f"   Home: {old_id} -> {new_id} ({home_count} predições)")
            
            cursor.execute(
                db._prepare_query("SELECT COUNT(*) FROM predictions WHERE away_team = ?"),
                (old_id,)
            )
            away_count = cursor.fetchone()[0]
            
            if away_count > 0:
                cursor.execute(
                    db._prepare_query("UPDATE predictions SET away_team = ? WHERE away_team = ?"),
                    (new_id, old_id)
                )
                print(f"   Away: {old_id} -> {new_id} ({away_count} predições)")
        
        # 4. Commit
        conn.commit()
        print("\n✅ Migração concluída com sucesso!")
        print("   Todas as alterações foram salvas no database.")
        
        # 5. Verificação
        print("\n" + "="*60)
        print("VERIFICAÇÃO PÓS-MIGRAÇÃO")
        print("="*60)
        
        cursor.execute(
            db._prepare_query("SELECT DISTINCT home_team FROM games ORDER BY home_team")
        )
        teams = [row[0] for row in cursor.fetchall()]
        print(f"\nTimes únicos no database (sample): {teams[:15]}")
        
        # Verificar se restou algum ID antigo
        old_ids_found = [team for team in teams if team in TEAM_CONVERSIONS.keys()]
        if old_ids_found:
            print(f"\n⚠️ ATENÇÃO: Ainda existem IDs antigos: {old_ids_found}")
        else:
            print(f"\n✅ Nenhum ID antigo encontrado - migração completa!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro durante migração: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
        
    finally:
        db.return_connection(conn)

if __name__ == "__main__":
    success = migrate_team_names()
    exit(0 if success else 1)
