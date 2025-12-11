import logging
import pandas as pd
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

class NBARAPMScraper:
    """
    Scraper para coletar dados de RAPM do site nbarapm.com.
    Usa o endpoint JSON ao invés de scraping HTML para maior confiabilidade.
    """
    
    JSON_URL = 'https://nbarapm.com/load/current_comp'
    
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent
        self.output_file = self.output_dir / "nba_rapm.csv"

    def scrape(self):
        logger.info(f"🚀 Iniciando raspagem do NBA RAPM via JSON: {self.JSON_URL}")
        
        try:
            # Fazer request ao endpoint JSON
            logger.info("📡 Fazendo request ao endpoint JSON...")
            response = requests.get(self.JSON_URL, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning("⚠️ Endpoint retornou dados vazios.")
                return False
            
            logger.info(f"✅ Recebidos {len(data)} registros do endpoint JSON.")
            
            # Converter para DataFrame
            df = pd.DataFrame(data)
            
            # Mapeamento de colunas (se necessário)
            # O JSON geralmente tem nomes como: player_name, team, rapm_timedecay, etc.
            logger.info(f"📋 Colunas disponíveis: {list(df.columns)[:10]}...")  # Primeiras 10
            
            # Salvar dados brutos
            df.to_csv(self.output_file, index=False)
            logger.info(f"💾 Dados salvos em: {self.output_file}")
            logger.info(f"📊 Total de jogadores: {len(df)}")
            
            # Log das primeiras linhas para debug
            logger.info(f"🔍 Primeiras colunas: {df.columns.tolist()[:5]}")
            
            return True
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro ao acessar endpoint JSON: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro fatal no scraper RAPM: {e}")
            return False

if __name__ == "__main__":
    # Configuração básica de log para execução direta
    logging.basicConfig(level=logging.INFO)
    scraper = NBARAPMScraper()
    success = scraper.scrape()
    if success:
        print("✅ Scraping concluído com sucesso!")
    else:
        print("❌ Falha no scraping.")
