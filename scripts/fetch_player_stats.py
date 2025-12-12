"""
Script para executar scraper de player stats em background.
Salva os dados em data/nba_player_stats.csv para uso posterior.
"""
import sys
import pandas as pd
import os
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("=" * 70)
print("EXECUTANDO SCRAPER DE PLAYER STATS (RAPM/BPM/PIE)")
print("=" * 70)

try:
    from data.scrapers.stats_scraper import obter_player_stats
    
    print("\n🔄 Iniciando scraper...")
    # obter_player_stats retorna um DICT de DataFrames {'RAPM': df, 'BBALL_REF': df, ...}
    results = obter_player_stats()
    
    if isinstance(results, dict):
        print(f"\n✅ Scraper retornou chaves: {list(results.keys())}")
        
        # 1. Tentar obter RAPM (Prioridade)
        df_main = results.get('RAPM')
        if df_main is None or df_main.empty:
             df_main = results.get('BBALL_REF')
        
        # 2. Tentar obter Stats Básicos (PPG)
        df_basic = results.get('BASIC_STATS')
        
        if df_main is not None and not df_main.empty:
            # Se tivermos basic stats, fazer merge
            if df_basic is not None and not df_basic.empty:
                # Normalizar colunas para merge
                # df_main tem Player, Team, RAPM...
                # df_basic tem PLAYER, TEAM, PTS...
                
                df_basic.columns = [c.upper() for c in df_basic.columns]
                if 'PLAYER' in df_basic.columns:
                    df_basic = df_basic.rename(columns={'PLAYER': 'Player'})
                
                # Merge left on Player
                print(f"   Iniciando merge: RAPM ({len(df_main)}) + Basic ({len(df_basic)})")
                df_final = pd.merge(df_main, df_basic[['Player', 'PTS', 'MIN', 'GP']], on='Player', how='left')
            else:
                df_final = df_main
                if 'PTS' not in df_final.columns:
                    df_final['PTS'] = 0.0 # Fallback
            
            # Salvar
            csv_path = 'data/nba_player_stats.csv'
            # Garantir diretório data
            os.makedirs('data', exist_ok=True)
            
            df_final.to_csv(csv_path, index=False)
            print(f"\n💾 Dados salvos em: {csv_path}")
            print(f"   Total Jogadores: {len(df_final)}")
            print(f"   Colunas: {list(df_final.columns)}")
            
            # Salvar também o raw rapm para compatibilidade
            results.get('RAPM', df_main).to_csv('data/nba_rapm.csv', index=False)
            
        else:
            print("⚠️ Nenhuma tabela principal (RAPM/BBALL_REF) encontrada.")
            
    elif isinstance(results, pd.DataFrame):
        # Fallback caso mude a implementação retornando apenas DF
        print("\n✅ Scraper retornou DataFrame único")
        results.to_csv('data/nba_player_stats.csv', index=False)
        
    else:
        print("⚠️ Scraper não retornou dados válidos (None ou vazio)")
        
except Exception as e:
    print(f"\n❌ Erro ao executar scraper: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("SCRAPER FINALIZADO")
print("=" * 70)
