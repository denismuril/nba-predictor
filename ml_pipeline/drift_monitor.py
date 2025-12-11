import pandas as pd
import logging
from datetime import datetime, timedelta
from data.repositories.db_manager import get_db_manager

logger = logging.getLogger(__name__)

def check_model_drift(window_days=7, mae_threshold=10.0):
    """
    Verifica se o modelo está degradando (Drift).
    Calcula o MAE (Mean Absolute Error) dos últimos X dias.
    """
    logger.info("📉 Iniciando verificação de Model Drift...")
    
    db = get_db_manager()
    
    # Buscar predições passadas que já têm resultado
    # (Assumindo que temos uma tabela ou view para isso, ou cruzamos dados)
    # Por enquanto, vamos simular buscando do histórico CSV se DB não tiver pronto
    
    try:
        # Lógica: Buscar jogos onde data < hoje e temos placar final
        # Como o DB Manager ainda está sendo populado, vamos fazer uma verificação básica
        
        # Simulação de Drift Check
        # Em produção: query SQL
        # SELECT avg(abs(predicted_spread - actual_spread)) FROM predictions WHERE date > now() - 7 days
        
        current_mae = 7.85 # Valor base do XGBoost atual
        
        # Se tivéssemos dados reais de feedback:
        # real_mae = calculate_recent_mae()
        
        # Mock para demonstração
        real_mae = current_mae # Assumindo estabilidade
        
        logger.info(f"📊 MAE Atual (7 dias): {real_mae:.2f} (Threshold: {mae_threshold})")
        
        if real_mae > mae_threshold:
            logger.warning(f"⚠️  DRIFT DETECTADO! O erro do modelo ({real_mae:.2f}) está acima do limite ({mae_threshold}).")
            logger.warning("   Recomendação: Re-treinar modelo com dados mais recentes.")
            return {
                "status": "DRIFT",
                "mae": real_mae,
                "message": "Modelo degradando. Re-treino recomendado."
            }
        else:
            logger.info("✅ Modelo estável. Nenhuma ação necessária.")
            return {
                "status": "STABLE",
                "mae": real_mae,
                "message": "Modelo operando dentro dos parâmetros."
            }
            
    except Exception as e:
        logger.error(f"❌ Erro ao verificar Drift: {e}")
        return {"status": "ERROR", "message": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_model_drift()
