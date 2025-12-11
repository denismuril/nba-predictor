"""
Script para executar scraper de player stats em background.
Salva os dados em data/nba_rapm.csv para uso posterior.
"""
import sys
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("=" * 70)
print("EXECUTANDO SCRAPER DE PLAYER STATS (RAPM/BPM/PIE)")
print("=" * 70)

try:
    from data.scrapers.stats_scraper import obter_player_stats
    
    print("\n🔄 Iniciando scraper...")
    df_stats = obter_player_stats()
    
    if df_stats is not None and not df_stats.empty:
        print(f"\n✅ Scraper concluído com sucesso!")
        print(f"   Jogadores: {len(df_stats)}")
        print(f"   Colunas: {list(df_stats.columns)}")
        
        # Salvar em CSV para cache
        csv_path = 'data/nba_rapm.csv'
        df_stats.to_csv(csv_path, index=False)
        print(f"\n💾 Dados salvos em: {csv_path}")
        
        # Mostrar amostra
        print("\n📊 Amostra dos dados:")
        print(df_stats.head(10))
        
    else:
        print("⚠️ Scraper não retornou dados")
        
except Exception as e:
    print(f"\n❌ Erro ao executar scraper: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("SCRAPER FINALIZADO")
print("=" * 70)
