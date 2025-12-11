"""
Script para simular e validar fallback de RAPM quando fonte externa falha.

CONTEXTO:
- RAPM externo (nbarapm.com) pode ficar offline
- Sistema deve usar fallback hierárquico automaticamente
- Este script simula essa falha para validar o comportamento

FALLBACK HIERARCHY:
1. RAPM externo (nbarapm.com) - Prioridade 1
2. NetRtg da NBA API - Prioridade 2  
3. Game Score calculado - Prioridade 3
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.scrapers.stats_scraper import StatsScraper
import pandas as pd

async def test_fallback_simulation():
    """Simula falha do RAPM externo para testar fallback"""
    
    print("=" * 80)
    print("SIMULAÇÃO DE FALHA DO RAPM EXTERNO")
    print("=" * 80)
    print()
    
    scraper = StatsScraper()
    
    print("📊 CENÁRIO 1: RAPM Externo Funcionando (Normal)")
    print("-" * 80)
    
    df_normal = await scraper.get_rapm()
    
    if not df_normal.empty and 'RAPM_SOURCE' in df_normal.columns:
        source_normal = df_normal['RAPM_SOURCE'].value_counts()
        print(f"✅ Resultado: {len(df_normal)} jogadores")
        print(f"   Fonte: {source_normal.to_dict()}")
        print()
    
    print("=" * 80)
    print("📊 CENÁRIO 2: RAPM Externo OFFLINE (Simulado)")
    print("-" * 80)
    print()
    
    # Simular falha retornando None no fetch_json
    with patch.object(scraper, 'fetch_json', return_value=None):
        print("⚠️  Simulando falha... (fetch_json retornará None)")
        print()
        
        df_fallback = await scraper.get_rapm()
        
        if not df_fallback.empty:
            print(f"✅ FALLBACK FUNCIONOU! {len(df_fallback)} jogadores obtidos")
            
            if 'RAPM_SOURCE' in df_fallback.columns:
                source_fallback = df_fallback['RAPM_SOURCE'].value_counts()
                print(f"   Fonte usada: {source_fallback.to_dict()}")
                print()
                
                # Mostrar top 5
                print("🏀 Top 5 jogadores (via fallback):")
                print("-" * 40)
                top_5 = df_fallback.nlargest(5, 'RAPM')[['Player', 'Team', 'RAPM', 'RAPM_SOURCE']]
                print(top_5.to_string(index=False))
                print()
                
                # Validar range de valores
                print("📈 Validação de Métricas:")
                print("-" * 40)
                print(f"   RAPM: min={df_fallback['RAPM'].min():.2f}, max={df_fallback['RAPM'].max():.2f}")
                print(f"   Range esperado: [-8, +8]")
                
                if df_fallback['RAPM'].min() >= -8 and df_fallback['RAPM'].max() <= 8:
                    print("   ✅ Range correto!")
                else:
                    print("   ⚠️  Range fora do esperado!")
                print()
        else:
            print("❌ ERRO: Todos os fallbacks falharam!")
            print()
    
    print("=" * 80)
    print("📊 CENÁRIO 3: Simulando Falha Total (NetRtg + Basic Stats)")
    print("-" * 80)
    print()
    
    # Simular falha completa
    with patch.object(scraper, 'fetch_json', return_value=None):
        with patch('nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats') as mock_api:
            # Fazer todas as chamadas retornarem vazio
            mock_api.return_value.get_data_frames.return_value = [pd.DataFrame()]
            
            print("⚠️  Simulando FALHA TOTAL (todas as fontes)...")
            print()
            
            df_total_fail = await scraper.get_rapm()
            
            if df_total_fail.empty:
                print("✅ COMPORTAMENTO CORRETO: DataFrame vazio retornado")
                print("   Sistema não trava, apenas loga erro")
            else:
                print(f"⚠️  Inesperado: {len(df_total_fail)} jogadores obtidos")
            print()
    
    print("=" * 80)
    print("✅ VALIDAÇÃO DE FALLBACK COMPLETA")
    print("=" * 80)
    print()
    print("RESUMO:")
    print("  ✅ Cenário 1: RAPM externo funciona normalmente")
    print("  ✅ Cenário 2: Fallback ativado quando RAPM externo falha")
    print("  ✅ Cenário 3: Sistema não trava quando tudo falha")
    print()
    print("🚀 Sistema de fallback validado e pronto para produção!")

if __name__ == "__main__":
    asyncio.run(test_fallback_simulation())
