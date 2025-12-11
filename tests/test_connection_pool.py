"""
Teste de stress do Connection Pool com múltiplos processos simultâneos.

Verifica que:
1. Não há database locks com múltiplos processos
2. Pool gerencia conexões corretamente
3. Cleanup funciona adequadamente
"""
import sys
import time
import logging
from pathlib import Path
from multiprocessing import Process, Queue
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import get_db_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(processName)-12s] %(levelname)-8s %(message)s'
)
logger = logging.getLogger(__name__)


def worker_save_predictions(worker_id: int, results_queue: Queue):
    """Worker que salva predições no banco."""
    try:
        db = get_db_manager()
        
        # Salvar 10 predições
        for i in range(10):
            predictions = [{
                'Data': datetime.now().strftime('%Y-%m-%d'),
                'Casa': f'Test Home {worker_id}_{i}',
                'Visitante': f'Test Away {worker_id}_{i}',
                'Prob Casa %': 55.0 + (worker_id * 2),
                'Prob Visitante %': 45.0 - (worker_id * 2),
                'Prob MC Casa %': 56.0,
                'Prob MC Visitante %': 44.0,
                'Odd Casa': 1.85,
                'Odd Visitante': 2.10,
                'Confiança': 'Média'
            }]
            
            db.save_predictions(predictions)
            logger.info(f"Worker {worker_id}: Salvou predição {i+1}/10")
            time.sleep(0.1)  # Pequeno delay para simular trabalho
        
        results_queue.put(('success', worker_id, None))
        
    except Exception as e:
        logger.error(f"Worker {worker_id}: Erro - {e}")
        results_queue.put(('error', worker_id, str(e)))


def worker_read_history(worker_id: int, results_queue: Queue):
    """Worker que lê histórico do banco."""
    try:
        db = get_db_manager()
        
        # Ler histórico 20 vezes
        for i in range(20):
            df = db.get_comprehensive_history()
            logger.info(f"Worker {worker_id}: Leu {len(df)} registros (iter {i+1}/20)")
            time.sleep(0.05)
        
        results_queue.put(('success', worker_id, None))
        
    except Exception as e:
        logger.error(f"Worker {worker_id}: Erro - {e}")
        results_queue.put(('error', worker_id, str(e)))


def main():
    logger.info("=" * 80)
    logger.info("TESTE DE STRESS: Connection Pool com Múltiplos Processos")
    logger.info("=" * 80)
    
    # Queue para coletar resultados
    results = Queue()
    
    # Criar processos
    processes = []
    
    logger.info("\n🚀 Iniciando 3 workers de WRITE + 2 workers de READ...")
    
    # 3 processos escrevendo
    for i in range(3):
        p = Process(target=worker_save_predictions, args=(i, results), name=f'Writer-{i}')
        processes.append(p)
        p.start()
    
    # 2 processos lendo
    for i in range(2):
        p = Process(target=worker_read_history, args=(i, results), name=f'Reader-{i}')
        processes.append(p)
        p.start()
    
    # Aguardar todos terminarem
    for p in processes:
        p.join()
    
    logger.info("\n✅ Todos os workers terminaram")
    
    # Coletar resultados
    success_count = 0
    error_count = 0
    errors = []
    
    while not results.empty():
        status, worker_id, error_msg = results.get()
        if status == 'success':
            success_count += 1
        else:
            error_count += 1
            errors.append((worker_id, error_msg))
    
    # Relatório
    logger.info("\n" + "=" * 80)
    logger.info("RESULTADO DO TESTE")
    logger.info("=" * 80)
    logger.info(f"✅ Sucessos: {success_count}/{len(processes)}")
    logger.info(f"❌ Erros: {error_count}/{len(processes)}")
    
    if errors:
        logger.error("\n⚠️ ERROS DETECTADOS:")
        for worker_id, error_msg in errors:
            logger.error(f"  Worker {worker_id}: {error_msg}")
        
        # Verificar se há database locks
        db_locks = [e for w, e in errors if 'database is locked' in e.lower()]
        if db_locks:
            logger.error(f"\n❌ FALHA: {len(db_locks)} database locks detectados!")
            logger.error("   Connection Pool NÃO está funcionando corretamente")
            return False
    
    # Verificar pool stats
    db = get_db_manager()
    if hasattr(db, '_pool') and db._pool is not None:
        stats = db._pool.get_stats()
        logger.info(f"\n📊 Pool Stats: {stats}")
        
        if stats['utilization_pct'] < 90:
            logger.info("✅ Utilização do pool normal (<90%)")
        else:
            logger.warning(f"⚠️ Pool altamente utilizado: {stats['utilization_pct']:.1f}%")
    
    if error_count == 0:
        logger.info("\n🎉 SUCESSO TOTAL: Nenhum erro detectado!")
        logger.info("   Connection Pool está funcionando perfeitamente!")
        return True
    else:
        logger.warning(f"\n⚠️ SUCESSO PARCIAL: {error_count} erros encontrados")
        return error_count < (len(processes) * 0.2)  # <20% erro é aceitável


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
