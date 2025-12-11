from nba_api.stats.endpoints import leaguegamelog
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_api():
    try:
        logger.info("Requesting LeagueGameLog for 2016-17...")
        # Adicionando timeout se possível, mas nba_api padrão não expõe fácil.
        # Vamos tentar uma request simples.
        log = leaguegamelog.LeagueGameLog(season='2016-17', player_or_team_abbreviation='T')
        
        print("Raw Response Keys:", log.get_dict().keys())
        
        # Tentar pegar data frames
        dfs = log.get_data_frames()
        print(f"DataFrames returned: {len(dfs)}")
        if len(dfs) > 0:
            print("Columns:", dfs[0].columns)
            print("First row:", dfs[0].head(1))
            
    except Exception as e:
        logger.error(f"API Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_api()
