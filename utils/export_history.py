"""
Exporta dados históricos do banco para CSV para evitar database locks.
"""
import pandas as pd
import logging
from pathlib import Path
from data.repositories.db_manager import get_db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_historical_data():
    """Exporta dados históricos do banco para CSV."""
    logger.info("📊 Exportando dados históricos do banco...")
    
    try:
        db = get_db_manager()
        df = db.get_comprehensive_history()
        
        if df is None or df.empty:
            logger.error("❌ Nenhum dado histórico encontrado no banco.")
            return False
        
        # Criar diretório se não existir
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Salvar CSV
        csv_path = data_dir / 'historical_games.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info(f"✅ {len(df)} jogos exportados para: {csv_path}")
        logger.info(f"   Colunas: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao exportar dados: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = export_historical_data()
    if success:
        print("\n✅ Dados exportados! Agora pode treinar o modelo com dados reais.")
    else:
        print("\n❌ Falha na exportação. Verifique o banco de dados.")
