# Exportador de resultados para CSV e Excel

import pandas as pd
import logging

logger = logging.getLogger("nba_predictor")


def exportar_para_csv(df: pd.DataFrame, path):
    """Salva o DataFrame em CSV.
    
    Args:
        df: DataFrame com os resultados.
        path: Caminho (str ou Path) onde o arquivo será salvo.
    """
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"✅ CSV exportado para {path}")
    except Exception as e:
        logger.error(f"❌ Falha ao exportar CSV: {e}")


def exportar_para_excel(df: pd.DataFrame, path):
    """Salva o DataFrame em Excel (xlsx).
    
    Args:
        df: DataFrame com os resultados.
        path: Caminho (str ou Path) onde o arquivo será salvo.
    """
    try:
        df.to_excel(path, index=False, engine="openpyxl")
        logger.info(f"✅ Excel exportado para {path}")
    except Exception as e:
        logger.error(f"❌ Falha ao exportar Excel: {e}")
