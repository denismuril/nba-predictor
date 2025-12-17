#!/usr/bin/env python3
"""
Prepara o cache de dados UMA VEZ - depois o ensemble_blending roda em segundos.

Uso:
    python scripts/prepare_data_cache.py

Isso pode demorar 2-4 horas, mas depois o cache fica pronto para uso rápido.
"""
import sys
import os

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("🚀 PREPARANDO CACHE DE DADOS (isso pode demorar)")
    logger.info("=" * 60)
    logger.info("Após terminar, o ensemble_blending vai rodar em segundos!")
    logger.info("")
    
    # 1. Carregar dados completos
    from ml_pipeline.data_preparation import (
        load_historical_data, add_rolling_features, add_advanced_features
    )
    
    logger.info("📦 Carregando dados históricos...")
    df = load_historical_data(
        seasons=['2023-24', '2024-25', '2025-26'],
        apply_weights=False
    )
    
    logger.info(f"   Dados brutos: {len(df)} jogos")
    
    # 2. Adicionar features
    logger.info("📊 Adicionando rolling features...")
    df = add_rolling_features(df)
    
    logger.info("📊 Adicionando advanced features...")
    df = add_advanced_features(df)
    
    logger.info(f"   Total: {len(df)} jogos, {len(df.columns)} colunas")
    
    # 3. Salvar no cache
    from ml_pipeline.data_cache import save_to_cache
    save_to_cache(df)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ CACHE PREPARADO COM SUCESSO!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Agora você pode rodar o ensemble_blending RÁPIDO:")
    logger.info("  python -c \"from ml_pipeline.ensemble_blending import train_ensemble_blending; train_ensemble_blending(optimize_hyperparams=False)\"")
    logger.info("")

if __name__ == "__main__":
    main()
