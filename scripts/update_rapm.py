import sys
import logging
from pathlib import Path

# Adicionar raiz do projeto ao path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from data.scrapers.nba_rapm_scraper import NBARAPMScraper

# Configuração de Log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("UpdateRAPM")

def main():
    logger.info("🔄 Iniciando atualização manual do NBA RAPM...")
    
    scraper = NBARAPMScraper(output_dir=project_root / "data")
    success = scraper.scrape()
    
    if success:
        logger.info("✅ Atualização concluída com sucesso!")
        sys.exit(0)
    else:
        logger.error("❌ Falha na atualização.")
        sys.exit(1)

if __name__ == "__main__":
    main()
