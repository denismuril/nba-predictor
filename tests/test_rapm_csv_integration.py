import sys
import asyncio
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data.scrapers.stats_scraper import StatsScraper

async def test_rapm_integration():
    scraper = StatsScraper()
    rapm_data = await scraper.get_rapm()
    
    if not rapm_data.empty:
        print(f"✅ RAPM carregado com sucesso!")
        print(f"Fonte: {rapm_data['RAPM_SOURCE'].iloc[0] if 'RAPM_SOURCE' in rapm_data.columns else 'Desconhecida'}")
        print(f"Total de jogadores: {len(rapm_data)}")
        print(f"\nPrimeiras linhas:")
        print(rapm_data.head())
    else:
        print("❌ RAPM está vazio!")

if __name__ == "__main__":
    asyncio.run(test_rapm_integration())
