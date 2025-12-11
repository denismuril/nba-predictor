"""
Test API-Sports/API-Football com key do usuário

Testa conexão e busca dados de NBA.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configurar API key do usuário
os.environ['API_FOOTBALL_KEY'] = '01eee81ebe305e3e88ced3e2de4905c1'

from data.scrapers.apisports_scraper import APISportsNBAScraper

print("🏀 Testando API-Football/API-Sports NBA\n")

scraper = APISportsNBAScraper(use_rapidapi=False)

if scraper.api_key:
    print(f"✅ API key configurada: {scraper.api_key[:10]}...")
    print(f"📡 Base URL: {scraper.base_url}")
    print(f"🔑 Modo: {'RapidAPI' if scraper.use_rapidapi else 'API-Sports Direto'}\n")
    
    # Testar com game ID de exemplo (da documentação)
    print("🎯 Testando game ID: 10403\n")
    
    stats = scraper.get_game_statistics(10403)
    
    if stats:
        print("✅ Dados recebidos!")
        print(f"   Records: {len(stats)}")
        
        advanced = scraper.extract_advanced_stats(stats)
        
        print(f"\n📊 Estatísticas avançadas:")
        print(f"  Home:")
        print(f"    Fast Break: {advanced['home']['fast_break']}")
        print(f"    Second Chance: {advanced['home']['second_chance']}")
        print(f"    Paint: {advanced['home']['paint']}")
        print(f"  Away:")
        print(f"    Fast Break: {advanced['away']['fast_break']}")
        print(f"    Second Chance: {advanced['away']['second_chance']}")
        print(f"    Paint: {advanced['away']['paint']}")
        
        print(f"\n🎉 API FUNCIONANDO PERFEITAMENTE!")
    else:
        print("⚠️ Sem dados retornados")
        print("   Pode ser game ID inválido ou endpoint diferente")
        print("   Vamos verificar a documentação...")
else:
    print("❌ API key não configurada")

print("\n✅ Test completo!")
