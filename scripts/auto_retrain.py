#!/usr/bin/env python3
"""
Script de Retreinamento Automático - NBA Predictor

Verifica se há novos jogos suficientes desde o último treino e
retreina o modelo se necessário. Só substitui o modelo se a
accuracy melhorar ou ficar similar (delta < 2%).

Usage:
    python scripts/auto_retrain.py [--dry-run] [--threshold N]
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import json
import logging

# Adicionar diretório raiz ao path
sys.path.insert(0, '/home/denis/nba-predictor')

from data.repositories.db_manager import get_db_manager
from ml_pipeline.train_ensemble_v3 import train_ensemble_model_v3, ML_SEASONS

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_games_since_last_training():
    """
    Conta quantos jogos com resultado foram adicionados desde
    o último treinamento.
    
    Returns:
        Tuple: (new_games_count, last_training_date, total_games)
    """
    # Verificar metadados do último treinamento
    metadata_path = Path('/home/denis/nba-predictor/data/models/training_metadata.json')
    
    if not metadata_path.exists():
        logger.warning("⚠️  Metadados de treinamento não encontrados. Assumindo primeiro treino.")
        # Contar todos os jogos
        db = get_db_manager()
        df = db.get_comprehensive_history()
        total = len(df) if df is not None else 0
        return total, None, total
    
    # Carregar metadados
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    last_training_date = datetime.fromisoformat(metadata.get('timestamp', '2000-01-01'))
    last_total_games = metadata.get('total_games', 0)
    
    # Contar jogos atuais
    db = get_db_manager()
    df = db.get_comprehensive_history()
    current_total_games = len(df) if df is not None else 0
    
    new_games = max(0, current_total_games - last_total_games)
    
    logger.info(f"📊 Jogos desde último treino:")
    logger.info(f"   Último treino: {last_training_date.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Jogos no último treino: {last_total_games}")
    logger.info(f"   Jogos atuais: {current_total_games}")
    logger.info(f"   Novos jogos: {new_games}")
    
    return new_games, last_training_date, current_total_games


def should_retrain(new_games, threshold):
    """Determina se deve retreinar baseado no threshold"""
    should = new_games >= threshold
    
    if should:
        logger.info(f"✅ Threshold atingido ({new_games} >= {threshold}). Retreinamento necessário.")
    else:
        logger.info(f"ℹ️  Threshold não atingido ({new_games} < {threshold}). Retreinamento não necessário.")
    
    return should


def backup_current_model():
    """Faz backup do modelo atual antes de substituir"""
    model_path = Path('/home/denis/nba-predictor/data/models/ensemble_model.joblib')
    
    if not model_path.exists():
        logger.info("ℹ️  Nenhum modelo existente para fazer backup.")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = model_path.parent / f'ensemble_model_backup_{timestamp}.joblib'
    
    import shutil
    shutil.copy2(model_path, backup_path)
    
    logger.info(f"💾 Backup do modelo criado: {backup_path}")
    return backup_path


def compare_models(old_metadata, new_metadata, tolerance=0.02):
    """
    Compara modelos antigo e novo.
    
    Args:
        old_metadata: Dict com metadados do modelo antigo
        new_metadata: Dict com metadados do modelo novo
        tolerance: Tolerância de diferença de accuracy (default: 2%)
    
    Returns:
        bool: True se deve substituir, False caso contrário
    """
    if old_metadata is None:
        logger.info("✅ Nenhum modelo anterior. Usando novo modelo.")
        return True
    
    old_acc = old_metadata.get('accuracy', 0)
    new_acc = new_metadata.get('accuracy', 0)
    
    diff = new_acc - old_acc
    
    logger.info(f"📊 Comparação de modelos:")
    logger.info(f"   Modelo antigo: {old_acc*100:.2f}% accuracy")
    logger.info(f"   Modelo novo: {new_acc*100:.2f}% accuracy")
    logger.info(f"   Diferença: {diff*100:+.2f}%")
    
    # Substituir se melhorou ou ficou similar (dentro da tolerância)
    if diff >= -tolerance:
        logger.info(f"✅ Novo modelo igual ou melhor. Substituindo.")
        return True
    else:
        logger.warning(f"⚠️  Novo modelo pior ({diff*100:.2f}%). Mantendo modelo antigo.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Auto-retrain NBA Predictor ML model')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Simula sem treinar de fato')
    parser.add_argument('--threshold', type=int, default=10,
                       help='Mínimo de jogos novos para retreinar (default: 10)')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🤖 AUTO-RETRAIN NBA PREDICTOR")
    logger.info("="*80)
    logger.info(f"Modo: {'DRY-RUN (simulação)' if args.dry_run else 'PRODUÇÃO'}")
    logger.info(f"Threshold: {args.threshold} jogos")
    logger.info("")
    
    # 1. Verificar quantos jogos novos
    new_games, last_training_date, total_games = count_games_since_last_training()
    
    # 2. Decidir se retreina
    if not should_retrain(new_games, args.threshold):
        logger.info("🏁 Nenhuma ação necessária.")
        return 0
    
    if args.dry_run:
        logger.info("🎭 DRY-RUN: Treinamento seria executado agora.")
        return 0
    
    # 3. Fazer backup do modelo atual
    logger.info("")
    logger.info("📦 Preparando para retreinamento...")
    
    # Carregar metadados antigos
    metadata_path = Path('/home/denis/nba-predictor/data/models/training_metadata.json')
    old_metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            old_metadata = json.load(f)
    
    backup_path = backup_current_model()
    
    # 4. Treinar novo modelo
    logger.info("")
    logger.info("🎯 Iniciando treinamento do novo modelo...")
    
    try:
        model, accuracy, new_metadata = train_ensemble_model_v3(
            use_sample_weights=True,
            seasons=ML_SEASONS
        )
        
        if model is None:
            logger.error("❌ Falha no treinamento. Mantendo modelo antigo.")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 5. Comparar modelos
    logger.info("")
    if compare_models(old_metadata, new_metadata):
        logger.info("✅ Novo modelo aprovado e salvo!")
        logger.info(f"   Accuracy: {accuracy*100:.2f}%")
        logger.info(f"   Jogos de treino: {new_metadata['train_games']}")
        logger.info(f"   Temporadas: {', '.join(new_metadata['seasons'])}")
        result = 0
    else:
        logger.info("⚠️  Novo modelo rejeitado. Restaurando backup...")
        if backup_path and backup_path.exists():
            import shutil
            model_path = Path('/home/denis/nba-predictor/data/models/ensemble_model.joblib')
            shutil.copy2(backup_path, model_path)
            logger.info("✅ Modelo antigo restaurado.")
        result = 0  # Não é erro, apenas não substituiu
    
    logger.info("="*80)
    logger.info("🏁 FIM DO AUTO-RETRAIN")
    logger.info("="*80)
    
    return result


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Processo interrompido pelo usuário.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
