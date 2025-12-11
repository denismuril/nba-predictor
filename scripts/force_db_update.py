#!/usr/bin/env python3
"""
Script de Força Atualização do Banco de Dados V21.2

Este script força a atualização do schema do banco de dados adicionando
as colunas de métricas V21 (shooting_luck, rapm_penalty, fatigue_score)
e repopula o banco com as previsões mais recentes.

Uso:
    python3 scripts/force_db_update.py
"""
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.predict import predict_next_games
from data.repositories.db_manager import get_db_manager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def main():
    """Força atualização do banco de dados com schema V21.2"""
    logger.info("🔧 INICIANDO ATUALIZAÇÃO FORÇADA DO BANCO DE DADOS V21.2")
    logger.info("=" * 80)
    
    # Passo 1: Gerar previsões (com as novas métricas V21)
    logger.info("📊 Passo 1: Gerando previsões com métricas V21...")
    try:
        predictions_df = predict_next_games()
        logger.info(f"   ✅ {len(predictions_df)} previsões geradas")
        logger.info(f"   📋 Colunas disponíveis: {len(predictions_df.columns)}")
        
        # Verificar se métricas V21 estão presentes
        v21_metrics = [
            'home_shooting_luck', 'away_shooting_luck',
            'home_rapm_penalty', 'away_rapm_penalty', 'rapm_impact_diff'
        ]
        missing_metrics = [m for m in v21_metrics if m not in predictions_df.columns]
        
        if missing_metrics:
            logger.warning(f"   ⚠️  Métricas faltando: {missing_metrics}")
        else:
            logger.info(f"   ✅ Todas as métricas V21 presentes!")
            
    except Exception as e:
        logger.error(f"❌ Erro ao gerar previsões: {e}")
        return 1
    
    # Passo 2: Converter DataFrame para formato de dicionário
    logger.info("\\n🔄 Passo 2: Convertendo para formato de persistência...")
    try:
        predictions_list = []
        for _, row in predictions_df.iterrows():
            pred_dict = {
                'Data': row['date'],
                'Casa': row['home_team'],
                'Visitante': row['away_team'],
                'Prob Casa %': row.get('prob_home', 0),
                'Prob Visitante %': row.get('prob_away', 0),
                'Prob MC Casa %': row.get('prob_mc_home', 0),
                'Prob MC Visitante %': row.get('prob_mc_away', 0),
                'Odd Casa': row.get('odds_home', 0),
                'Odd Visitante': row.get('odds_away', 0),
                'Total Previsto': row.get('predicted_total', 0),
                'Previsão': row.get('prediction', 'N/A'),
                'Confiança': row.get('confidence', 'N/A'),
                'Spread Previsto': row.get('predicted_spread', 0),
                'home_injuries_list': row.get('home_injuries_list', ''),
                'away_injuries_list': row.get('away_injuries_list', ''),
                # MÉTRICAS V21
                'home_shooting_luck': row.get('home_shooting_luck', 0.0),
                'away_shooting_luck': row.get('away_shooting_luck', 0.0),
                'home_rapm_penalty': row.get('home_rapm_penalty', 0.0),
                'away_rapm_penalty': row.get('away_rapm_penalty', 0.0),
                'rapm_impact_diff': row.get('rapm_impact_diff', 0.0),
                'home_fatigue_score': row.get('home_fatigue_score', 0.0),
                'away_fatigue_score': row.get('away_fatigue_score', 0.0),
                'home_elo': row.get('home_elo', 0.0),
                'away_elo': row.get('away_elo', 0.0),
                'projected_pace_vegas': row.get('projected_pace_vegas', 0.0)
            }
            predictions_list.append(pred_dict)
        
        logger.info(f"   ✅ {len(predictions_list)} previsões convertidas")
        
    except Exception as e:
        logger.error(f"❌ Erro na conversão: {e}")
        return 1
    
    # Passo 3: Salvar no banco (dispara auto-migração)
    logger.info("\\n💾 Passo 3: Salvando no banco (AUTO-MIGRAÇÃO será ativada)...")
    try:
        db = get_db_manager()
        db.save_predictions(predictions_list)
        logger.info("   ✅ Previsões salvas com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro ao salvar no banco: {e}")
        return 1
    
    # Passo 4: Validação final
    logger.info("\\n🔍 Passo 4: Validando schema atualizado...")
    try:
        # Carregar previsões do banco para verificar
        saved_predictions = db.get_latest_predictions()
        
        if not saved_predictions.empty:
            logger.info(f"   ✅ {len(saved_predictions)} previsões recuperadas do banco")
            logger.info(f"   📋 Colunas no banco: {len(saved_predictions.columns)}")
            
            # Verificar novas colunas
            for metric in v21_metrics:
                if metric in saved_predictions.columns:
                    logger.info(f"   ✅ {metric}: OK")
                else:
                    logger.warning(f"   ❌ {metric}: FALTANDO")
        else:
            logger.warning("   ⚠️  Nenhuma previsão encontrada no banco")
            
    except Exception as e:
        logger.error(f"❌ Erro na validação: {e}")
        return 1
    
    logger.info("\\n" + "=" * 80)
    logger.info("✅ ATUALIZAÇÃO DO BANCO DE DADOS CONCLUÍDA COM SUCESSO!")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
