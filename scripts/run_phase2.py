#!/usr/bin/env python3
"""
Executor Master da Fase 2

Roda todos os scripts da Fase 2 em sequência:
1. Walk-Forward Validation
2. Error Analysis
3. Gera relatório consolidado

Usage:
    python scripts/run_phase2.py
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import subprocess
import time
from pathlib import Path
import json

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path, description):
    """Executa um script Python e reporta o resultado."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 {description}")
    logger.info(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos timeout
        )
        
        elapsed = time.time() - start_time
        
        # Mostrar output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} concluído em {elapsed:.1f}s")
            return True, elapsed
        else:
            logger.error(f"❌ {description} falhou!")
            if result.stderr:
                logger.error(f"Erro: {result.stderr}")
            return False, elapsed
    
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} excedeu timeout de 10 minutos")
        return False, 600
    except Exception as e:
        logger.error(f"❌ Erro ao executar {description}: {e}")
        return False, 0

def generate_phase2_summary(results):
    """Gera resumo consolidado da Fase 2."""
    logger.info(f"\n{'='*80}")
    logger.info(f"📋 RESUMO DA FASE 2")
    logger.info(f"{'='*80}")
    
    total_time = sum(r['time'] for r in results.values())
    successful = sum(1 for r in results.values() if r['success'])
    
    logger.info(f"\n✅ Scripts executados: {successful}/{len(results)}")
    logger.info(f"⏱️  Tempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    logger.info(f"\n📊 Detalhes:")
    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        logger.info(f"   {status} {name}: {result['time']:.1f}s")
    
    # Verificar se existem relatórios gerados
    reports_found = []
    
    walk_forward_file = Path('data/models/walk_forward_ensemble_results.json')
    if walk_forward_file.exists():
        with open(walk_forward_file) as f:
            wf_data = json.load(f)
        reports_found.append(f"Walk-Forward: {len(wf_data)} folds")
    
    error_file = Path('data/models/error_analysis_report.json')
    if error_file.exists():
        with open(error_file) as f:
            err_data = json.load(f)
        reports_found.append(f"Error Analysis: {err_data.get('overall_accuracy', 0)*100:.2f}% accuracy")
    
    if reports_found:
        logger.info(f"\n📄 Relatórios gerados:")
        for report in reports_found:
            logger.info(f"   ✅ {report}")
    
    # Recomendações
    logger.info(f"\n💡 RECOMENDAÇÕES:")
    
    if successful == len(results):
        logger.info(f"   ✅ Todos os scripts rodaram com sucesso!")
        logger.info(f"   📊 Revise os relatórios em data/models/")
        logger.info(f"   🎯 Compare resultados com baseline (74.55%)")
        
        if walk_forward_file.exists():
            logger.info(f"   📈 Verifique se há drift temporal no walk-forward")
        
        if error_file.exists():
            logger.info(f"   🔍 Identifique padrões de erro no error analysis")
    else:
        logger.info(f"   ⚠️  Alguns scripts falharam - revise os erros acima")
    
    logger.info(f"{'='*80}")
    
    return successful == len(results)

def main():
    logger.info("="*80)
    logger.info("🎯 EXECUTOR MASTER - FASE 2")
    logger.info("="*80)
    logger.info("📊 Executará:")
    logger.info("   1. Walk-Forward Validation")
    logger.info("   2. Error Analysis")
    logger.info("="*80)
    
    start_time = time.time()
    results = {}
    
    # 1. Walk-Forward Validation
    success, elapsed = run_script(
        'scripts/walk_forward_validation.py',
        'Walk-Forward Validation'
    )
    results['walk_forward'] = {'success': success, 'time': elapsed}
    
    # 2. Error Analysis
    success, elapsed = run_script(
        'scripts/error_analysis.py',
        'Error Analysis'
    )
    results['error_analysis'] = {'success': success, 'time': elapsed}
    
    # Gerar resumo
    all_success = generate_phase2_summary(results)
    
    total_elapsed = time.time() - start_time
    logger.info(f"\n⏱️  Fase 2 completa em {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    
    # Salvar resumo em JSON
    summary_file = Path('data/models/phase2_summary.json')
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': total_elapsed,
        'scripts': results,
        'all_successful': all_success
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"💾 Resumo salvo: {summary_file}")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
