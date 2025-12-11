import pandas as pd
import logging

logger = logging.getLogger(__name__)

import os
from pathlib import Path

def _get_safe_path(filename):
    """Garante que o arquivo seja salvo em 'results/' e resolve conflitos de permissão."""
    # Garantir diretório results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Se filename já for um path, usar o nome
    name = Path(filename).name
    target_path = results_dir / name
    
    # Se não tiver permissão no arquivo alvo, tentar sufixo
    if target_path.exists() and not os.access(target_path, os.W_OK):
        import time
        timestamp = int(time.time())
        target_path = results_dir / f"{target_path.stem}_{timestamp}{target_path.suffix}"
        
    return str(target_path)

def exportar_para_csv(df, filename):
    try:
        safe_path = _get_safe_path(filename)
        df.to_csv(safe_path, index=False)
        logger.info(f"💾 CSV salvo: {safe_path}")
    except Exception as e:
        logger.error(f"❌ Erro ao salvar CSV: {e}")

def exportar_para_excel(df, filename):
    try:
        safe_path = _get_safe_path(filename)
        df.to_excel(safe_path, index=False)
        logger.info(f"💾 Excel salvo: {safe_path}")
    except Exception as e:
        logger.error(f"❌ Erro ao salvar Excel: {e}")
