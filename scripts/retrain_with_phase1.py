#!/usr/bin/env python3
"""
Script Master para Retreinamento com Melhorias da Fase 1

Executa treinamento completo com:
- Top 5 Advanced Features (rest days, back-to-back, win streak, net rating trend, SOS)
- Sample weighting otimizado
- Hiperparâmetros atuais (ou otimizados se disponíveis)

Usage:
    python scripts/retrain_with_phase1.py [--optimize-first] [--exponential-weights]
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import argparse
import logging
import time

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_hyperparameter_optimization(model_type, n_trials=50):
    """Executa otimização de hiperparâmetros antes do treinamento."""
    logger.info(f"\n🔧 Otimizando hiperparâmetros para {model_type}...")
    
    if model_type == 'ensemble':
        from ml_pipeline.optimize_ensemble import run_optimization
        return run_optimization(n_trials=n_trials)
    elif model_type == 'spread':
        from ml_pipeline.optimize_hyperparameters import run_optimization
        return run_optimization(n_trials=n_trials)
    elif model_type == 'totals':
        from ml_pipeline.optimize_totals import run_optimization
        return run_optimization(n_trials=n_trials)
    else:
        logger.error(f"❌ Modelo desconhecido: {model_type}")
        return None

def train_all_models_with_improvements(optimize_first=False, use_exponential=False, n_trials=50):
    """Treina todos os modelos com as melhorias da Fase 1."""
    start_time = time.time()
    results = {}
    
    logger.info("="*80)
    logger.info("🚀 RETREINAMENTO COM MELHORIAS DA FASE 1")
    logger.info("="*80)
    logger.info(f"✨ Advanced Features: REST_DAYS, BACK_TO_BACK, WIN_STREAK, NET_RATING_TREND, SOS_10")
    logger.info(f"⚖️  Sample Weighting: {'Exponential Decay' if use_exponential else 'Step-Based'}")
    logger.info(f"🔧 Hyperparameter Optimization: {'ENABLED' if optimize_first else 'DISABLED'}")
    logger.info("="*80)
    
    # Otimização de hiperparâmetros (opcional)
    if optimize_first:
        logger.info("\n📊 FASE A: OTIMIZAÇÃO DE HIPERPARÂMETROS")
        logger.info(f"   Trials por modelo: {n_trials}")
        
        # Ensemble (Moneyline)
        logger.info("\n1️⃣ Otimizando Ensemble (Moneyline)...")
        run_hyperparameter_optimization('ensemble', n_trials=n_trials)
        
        # Spread
        logger.info("\n2️⃣ Otimizando Spread...")
        run_hyperparameter_optimization('spread', n_trials=n_trials)
        
        # Totals
        logger.info("\n3️⃣ Otimizando Totals...")
        run_hyperparameter_optimization('totals', n_trials=n_trials)
        
        logger.info("\n✅ Otimização de hiperparâmetros concluída!\n")
    
    # Treinamento dos modelos
    logger.info("\n📊 FASE B: TREINAMENTO DOS MODELOS")
    
    # 1. Moneyline (Ensemble V3)
    logger.info("\n" + "="*60)
    logger.info("🏀 1. TREINANDO MODELO MONEYLINE (VENCEDOR)")
    logger.info("="*60)
    try:
        from ml_pipeline.train_ensemble_v3 import train_ensemble_model_v3, ML_SEASONS
        model, acc, meta = train_ensemble_model_v3(use_sample_weights=True, seasons=ML_SEASONS)
        results['moneyline'] = {'accuracy': acc, 'status': 'success'}
        logger.info(f"✅ Moneyline concluído! Accuracy: {acc*100:.2f}%")
    except Exception as e:
        logger.error(f"❌ Erro no Moneyline: {e}")
        results['moneyline'] = {'accuracy': None, 'status': 'failed', 'error': str(e)}
    
    # 2. Spread
    logger.info("\n" + "="*60)
    logger.info("📏 2. TREINANDO MODELO SPREAD (HANDICAP)")
    logger.info("="*60)
    try:
        from ml_pipeline.train_spread_real import train_spread_model_real
        model, mae, rmse = train_spread_model_real()
        results['spread'] = {'mae': mae, 'rmse': rmse, 'status': 'success'}
        logger.info(f"✅ Spread concluído! MAE: {mae:.2f}")
    except Exception as e:
        logger.error(f"❌ Erro no Spread: {e}")
        results['spread'] = {'mae': None, 'status': 'failed', 'error': str(e)}
    
    # 3. Totals
    logger.info("\n" + "="*60)
    logger.info("🔢 3. TREINANDO MODELO TOTALS (OVER/UNDER)")
    logger.info("="*60)
    try:
        from ml_pipeline.train_totals_model import train_totals_model
        model, mae, rmse = train_totals_model()
        results['totals'] = {'mae': mae, 'rmse': rmse, 'status': 'success'}
        logger.info(f"✅ Totals concluído! MAE: {mae:.2f}")
    except Exception as e:
        logger.error(f"❌ Erro no Totals: {e}")
        results['totals'] = {'mae': None, 'status': 'failed', 'error': str(e)}
    
    # Resumo final
    elapsed = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info(f"🏁 RETREINAMENTO CONCLUÍDO em {elapsed:.1f}s")
    logger.info("="*80)
    logger.info("📊 COMPARAÇÃO COM BASELINE:")
    logger.info("="*80)
    
    # Moneyline
    if results['moneyline']['status'] == 'success':
        old_acc = 0.7455
        new_acc = results['moneyline']['accuracy']
        diff = (new_acc - old_acc) * 100
        arrow = "🎉" if diff > 0 else "⚠️"
        logger.info(f"{arrow} Moneyline: {old_acc*100:.2f}% → {new_acc*100:.2f}% ({diff:+.2f}%)")
    else:
        logger.info(f"❌ Moneyline: FALHOU")
    
    # Spread
    if results['spread']['status'] == 'success':
        old_mae = 7.80
        new_mae = results['spread']['mae']
        diff = old_mae - new_mae
        arrow = "🎉" if diff > 0 else "⚠️"
        logger.info(f"{arrow} Spread MAE: {old_mae:.2f} → {new_mae:.2f} ({diff:+.2f})")
    else:
        logger.info(f"❌ Spread: FALHOU")
    
    # Totals
    if results['totals']['status'] == 'success':
        old_mae = 11.87
        new_mae = results['totals']['mae']
        diff = old_mae - new_mae
        arrow = "🎉" if diff > 0 else "⚠️"
        logger.info(f"{arrow} Totals MAE: {old_mae:.2f} → {new_mae:.2f} ({diff:+.2f})")
    else:
        logger.info(f"❌ Totals: FALHOU")
    
    logger.info("="*80)
    
    # Salvar resultados
    import json
    from pathlib import Path
    results_file = Path('data/models/phase1_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n💾 Resultados salvos em: {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Retreinamento com Melhorias da Fase 1')
    parser.add_argument('--optimize-first', action='store_true', 
                       help='Executar otimização de hiperparâmetros antes (DEMORADO: ~30-60min)')
    parser.add_argument('--exponential-weights', action='store_true',
                       help='Usar exponential decay weighting ao invés de step-based')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Número de trials para otimização (padrão: 50)')
    
    args = parser.parse_args()
    
    if args.optimize_first:
        logger.info("⚠️  AVISO: Otimização de hiperparâmetros pode levar 30-60 minutos!")
        logger.info("    Para treinar sem otimizar, rode: python scripts/retrain_with_phase1.py")
    
    results = train_all_models_with_improvements(
        optimize_first=args.optimize_first,
        use_exponential=args.exponential_weights,
        n_trials=args.n_trials
    )
    
    return 0 if all(r['status'] == 'success' for r in results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
