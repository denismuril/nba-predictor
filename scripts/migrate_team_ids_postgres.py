#!/usr/bin/env python
"""
Script de migração para normalizar nomes de times no PostgreSQL.
Converte variações antigas (BKN, PHX, CHA) para os IDs canônicos (BRK, PHO, CHO).
"""
import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Carregar .env
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                try:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                except ValueError:
                    pass

# Mapeamento de conversão
TEAM_CONVERSIONS = {
    'BKN': 'BRK',  # Brooklyn Nets
    'PHX': 'PHO',  # Phoenix Suns
    'CHA': 'CHO',  # Charlotte Hornets
}


def migrate_postgres():
    """Migra nomes de times no PostgreSQL."""
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 não instalado!")
        print("   Instale com: pip install psycopg2-binary")
        return False
    
    # Configuração do PostgreSQL
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'nba_predictor_db'),
        'user': os.getenv('DB_USER', 'nba_admin'),
        'password': os.getenv('DB_PASS', 'password')
    }
    
    print("="*60)
    print("MIGRAÇÃO DE NOMES DE TIMES - PostgreSQL")
    print("="*60)
    print(f"\nConectando em: {db_config['user']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    print(f"Conversões: {TEAM_CONVERSIONS}\n")
    
    try:
        # Conectar
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        print("✅ Conectado ao PostgreSQL!\n")
        
        # 1. Atualizar tabela games
        print("1. Atualizando tabela 'games'...")
        
        for old_id, new_id in TEAM_CONVERSIONS.items():
            # Home team
            cursor.execute(
                "SELECT COUNT(*) FROM games WHERE home_team = %s",
                (old_id,)
            )
            home_count = cursor.fetchone()[0]
            
            if home_count > 0:
                cursor.execute(
                    "UPDATE games SET home_team = %s WHERE home_team = %s",
                    (new_id, old_id)
                )
                print(f"   Home: {old_id} -> {new_id} ({home_count} jogos)")
            
            # Away team
            cursor.execute(
                "SELECT COUNT(*) FROM games WHERE away_team = %s",
                (old_id,)
            )
            away_count = cursor.fetchone()[0]
            
            if away_count > 0:
                cursor.execute(
                    "UPDATE games SET away_team = %s WHERE away_team = %s",
                    (new_id, old_id)
                )
                print(f"   Away: {old_id} -> {new_id} ({away_count} jogos)")
        
        # 2. Atualizar tabela game_stats
        print("\n2. Atualizando tabela 'game_stats'...")
        print("   ⚠️ Estratégia: Deletar registros duplicados com IDs antigos")
        print("   (mantém apenas registros com IDs canônicos BRK, PHO, CHO)")
        
        for old_id, new_id in TEAM_CONVERSIONS.items():
            # Verificar se já existem registros com o novo ID
            cursor.execute(
                "SELECT COUNT(*) FROM game_stats WHERE team_id = %s",
                (new_id,)
            )
            new_id_count = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT COUNT(*) FROM game_stats WHERE team_id = %s",
                (old_id,)
            )
            old_id_count = cursor.fetchone()[0]
            
            if old_id_count > 0:
                if new_id_count > 0:
                    # Já existem registros com novo ID, deletar os antigos
                    cursor.execute(
                        "DELETE FROM game_stats WHERE team_id = %s",
                        (old_id,)
                    )
                    print(f"   🗑️  {old_id}: Deletados {old_id_count} registros duplicados (já existe {new_id})")
                else:
                    # Não há duplicatas, pode fazer UPDATE
                    cursor.execute(
                        "UPDATE game_stats SET team_id = %s WHERE team_id = %s",
                        (new_id, old_id)
                    )
                    print(f"   ✅ {old_id} -> {new_id}: {old_id_count} registros atualizados")

        
        # 3. Atualizar tabela predictions (se existir)
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='predictions')"
        )
        if cursor.fetchone()[0]:
            print("\n3. Atualizando tabela 'predictions'...")
            
            for old_id, new_id in TEAM_CONVERSIONS.items():
                cursor.execute(
                    "SELECT COUNT(*) FROM predictions WHERE home_team = %s",
                    (old_id,)
                )
                home_count = cursor.fetchone()[0]
                
                if home_count > 0:
                    cursor.execute(
                        "UPDATE predictions SET home_team = %s WHERE home_team = %s",
                        (new_id, old_id)
                    )
                    print(f"   Home: {old_id} -> {new_id} ({home_count} predições)")
                
                cursor.execute(
                    "SELECT COUNT(*) FROM predictions WHERE away_team = %s",
                    (old_id,)
                )
                away_count = cursor.fetchone()[0]
                
                if away_count > 0:
                    cursor.execute(
                        "UPDATE predictions SET away_team = %s WHERE away_team = %s",
                        (new_id, old_id)
                    )
                    print(f"   Away: {old_id} -> {new_id} ({away_count} predições)")
        
        # 4. Commit
        conn.commit()
        print("\n✅ Migração concluída com sucesso!")
        print("   Todas as alterações foram salvas no PostgreSQL.")
        
        # 5. Verificação
        print("\n" + "="*60)
        print("VERIFICAÇÃO PÓS-MIGRAÇÃO")
        print("="*60)
        
        cursor.execute(
            "SELECT DISTINCT home_team FROM games ORDER BY home_team LIMIT 15"
        )
        teams = [row[0] for row in cursor.fetchall()]
        print(f"\nTimes únicos no database (sample): {teams}")
        
        # Verificar se restou algum ID antigo
        old_ids_found = [team for team in teams if team in TEAM_CONVERSIONS.keys()]
        if old_ids_found:
            print(f"\n⚠️ ATENÇÃO: Ainda existem IDs antigos: {old_ids_found}")
        else:
            print(f"\n✅ Nenhum ID antigo encontrado - migração completa!")
        
        return True
        
    except psycopg2.Error as e:
        print(f"\n❌ Erro PostgreSQL: {e}")
        if conn:
            conn.rollback()
        return False
        
    except Exception as e:
        print(f"\n❌ Erro durante migração: {e}")
        if conn:
            conn.rollback()
        return False
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    success = migrate_postgres()
    exit(0 if success else 1)
