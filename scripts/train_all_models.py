#!/usr/bin/env python3
"""
Script para Treinar TODOS os Modelos - NBA Predictor

Executa sequencialmente o treinamento de:
1. Moneyline (Vencedor) - Ensemble V3
2. Spread (Handicap) - XGBoost
3. Totals (Over/Under) - XGBoost/RF

Usage:
    python scripts/train_all_models.py
"""

import sys
import os
import logging
import time

# Adicionar diretório raiz ao path
sys.path.insert(0, '/home/denis/nba-predictor')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_moneyline():
    logger.info("\n" + "="*60)
    logger.info("🏀 1. TREINANDO MODELO MONEYLINE (VENCEDOR) - V6")
    logger.info("="*60)
    try:
        from ml_pipeline.train_ensemble_v6 import train_ensemble_model_v6
        # V6 não precisa de argumentos, já tem config interna
        model, acc = train_ensemble_model_v6()
        logger.info(f"✅ Moneyline V6 concluído! Accuracy: {acc*100:.2f}%")
        return acc
    except Exception as e:
        logger.error(f"❌ Erro no Moneyline V6: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def train_spread():
    """
    NOTA: Modelo Spread desabilitado na limpeza de código (v21.7).
    O arquivo train_spread_real.py foi removido por ser obsoleto.
    O spread agora é calculado probabilisticamente a partir do Moneyline.
    """
    logger.info("\n" + "="*60)
    logger.info("📏 2. MODELO SPREAD (DESABILITADO)")
    logger.info("="*60)
    logger.info("ℹ️  Spread agora é calculado probabilisticamente (não precisa de modelo separado)")
    return None  # Não treina mais, usa cálculo probabilístico

def train_totals():
    logger.info("\n" + "="*60)
    logger.info("🔢 3. TREINANDO MODELO TOTALS (OVER/UNDER)")
    logger.info("="*60)
    try:
        from ml_pipeline.train_totals_model import train_totals_model
        model, mae, rmse = train_totals_model()
        logger.info(f"✅ Totals concluído! MAE: {mae:.2f}")
        return mae
    except Exception as e:
        logger.error(f"❌ Erro no Totals: {e}")
        return None

def main():
    start_time = time.time()
    logger.info("🚀 INICIANDO TREINAMENTO COMPLETO DO SISTEMA")
    
    # 1. Moneyline
    ml_acc = train_moneyline()
    
    # 2. Spread
    spread_mae = train_spread()
    
    # 3. Totals
    totals_mae = train_totals()
    
    elapsed = time.time() - start_time
    
    logger.info("\n" + "="*60)
    logger.info(f"🏁 TREINAMENTO COMPLETO FINALIZADO em {elapsed:.1f}s")
    logger.info("="*60)
    logger.info("RESUMO DOS RESULTADOS:")
    
    if ml_acc:
        logger.info(f"✅ Moneyline Accuracy: {ml_acc*100:.2f}%")
    else:
        logger.info(f"❌ Moneyline: Falhou")
        
    if spread_mae:
        logger.info(f"✅ Spread MAE: {spread_mae:.2f} pontos")
    else:
        logger.info("ℹ️  Spread: Desabilitado (cálculo probabilístico)")
        
    if totals_mae:
        logger.info(f"✅ Totals MAE: {totals_mae:.2f} pontos")
    else:
        logger.info(f"❌ Totals: Falhou")
        
    logger.info("="*60)
    
    return 0

if __name__ == "__main__":
    main()
