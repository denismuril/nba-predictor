import logging
import asyncio
from data.scrapers.async_scraper import AsyncScraper

logger = logging.getLogger(__name__)

class StandingsScraper(AsyncScraper):
    def __init__(self):
        super().__init__()
        self.url = "http://site.api.espn.com/apis/v2/sports/basketball/nba/standings"

    async def get_standings(self):
        """
        Busca standings da NBA via ESPN API (Async).
        Returns:
            dict: { 'Team Name': {'wins': 10, 'losses': 5}, ... }
        """
        logger.info("🔍 Buscando Standings (ESPN API)...")
        
        data = await self.fetch_json(self.url)
        
        if not data:
            logger.warning("⚠️  Falha ao buscar standings (JSON vazio).")
            return {}
            
        standings = {}
        try:
            children = data.get('children', [])
            
            for child in children:
                entries = child.get('standings', {}).get('entries', [])
                for entry in entries:
                    try:
                        team_name = entry['team']['displayName']
                        stats = entry.get('stats', [])
                        wins = 0
                        losses = 0
                        for stat in stats:
                            if stat['name'] == 'wins':
                                wins = int(stat['value'])
                            elif stat['name'] == 'losses':
                                losses = int(stat['value'])
                        
                        standings[team_name] = {'wins': wins, 'losses': losses}
                    except Exception as e:
                        continue
            
            if standings:
                logger.info(f"✅ Standings obtidos: {len(standings)} times")
                return standings
            
        except Exception as e:
            logger.error(f"❌ Erro parseando standings: {e}")
            
        return {}

# Manter função legada para compatibilidade se necessário (mas ideal é migrar tudo)
def obter_standings():
    """Wrapper síncrono legado"""
    scraper = StandingsScraper()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(scraper.get_standings())

