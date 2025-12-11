"""
Generate Prepared Dataset for ML Training (V6 Compatible)

Pipeline consistente com train_ensemble_v6.py:
1. Carrega dados brutos
2. Aplica pipeline completo via load_historical_data
3. Salva em data/prepared_games.csv

GARANTIA: Sem data leakage - usa mesmos controles do V6.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from ml_pipeline.data_preparation import load_historical_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuração: Mesmas temporadas do V6
ML_SEASONS = ['2023-24', '2024-25', '2025-26']

def generate_dataset():
    """
    Gera dataset preparado usando pipeline V6.
    """
    logger.info("="*80)
    logger.info("🚀 GERAÇÃO DE DATASET - PIPELINE V6")
    logger.info("="*80)
    logger.info(f"📅 Temporadas: {ML_SEASONS}")
    
    # 1. Carregar dados brutos e aplicar pipeline completo (V13 Enhanced)
    logger.info("\n📥 PASSO 1: Carregando dados e aplicando Feature Engineering...")
    
    # load_historical_data já aplica todo o pipeline: Elo, Rolling, Advanced, Interactions, Pace Volatility
    df = load_historical_data(seasons=ML_SEASONS, apply_weights=False, raw=False)
    
    if df is None or df.empty:
        logger.error("❌ Falha ao carregar dados históricos (DataFrame vazio)")
        return
    
    logger.info(f"   ✅ {len(df)} jogos processados")
    logger.info(f"   📊 Shape final: {df.shape}")
    
    # 3. Salvar dataset preparado
    logger.info("\n💾 PASSO 3: Salvando dataset...")
    output_path = Path('data/prepared_games.csv')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    df.to_csv(output_path, index=False)
    
    logger.info("="*80)
    logger.info("✅ DATASET GERADO COM SUCESSO")
    logger.info("="*80)
    logger.info(f"📁 Arquivo: {output_path}")
    logger.info(f"📊 Shape: {df.shape[0]} jogos × {df.shape[1]} features")
    logger.info(f"🔒 Pipeline: V6 (Feature Engineering V2 - Leakage Free)")
    logger.info("="*80)

if __name__ == "__main__":
    generate_dataset()
