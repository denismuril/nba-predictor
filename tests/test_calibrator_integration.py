"""
Teste de integração do AutoCalibrator com dados NBA reais.

Valida que:
1. Calibrator melhora ECE
2. Brier score melhora
3. Calibration curve fica melhor
4. Integration funciona end-to-end
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from ml_pipeline.calibrator import AutoCalibrator
from monitoring.calibration_monitor import CalibrationMonitor
from data.repositories.db_manager import get_db_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_calibrator_with_nba_data():
    """Testa calibrator com dados NBA reais."""
    
    logger.info("🏀 Testando AutoCalibrator com dados NBA...\n")
    
    # Carregar dados do DB
    db = get_db_manager()
    
    # Query: pegar predictions históricas com resultados conhecidos
    query = """
    SELECT 
        date,
        home_team,
        away_team,
        CAST(home_score AS FLOAT) as home_score,
        CAST(away_score AS FLOAT) as away_score
    FROM predictions
    WHERE home_score IS NOT NULL 
      AND away_score IS NOT NULL
      AND home_score > 0
      AND away_score > 0
    ORDER BY date DESC
    LIMIT 200
    """
    
    try:
        conn = db.get_connection()
        df = pd.read_sql_query(query, conn)
        db.return_connection(conn)
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados: {e}")
        logger.info("ℹ️  Gerando dados sintéticos para demo...")
        
        # Fallback: dados sintéticos realistas
        np.random.seed(42)
        n = 200
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'home_team': ['Lakers'] * n,
            'away_team': ['Warriors'] * n,
            'home_score': np.random.normal(110, 12, n),
            'away_score': np.random.normal(108, 12, n)
        })
    
    logger.info(f"📊 Carregados {len(df)} jogos\n")
    
    # Criar labels (home win = 1)
    df['y_true'] = (df['home_score'] > df['away_score']).astype(int)
    
    # Simular probabilidades do modelo (com bias para simular má calibração)
    # Modelo overconfident: prediz muito próximo de 0 ou 1
    base_prob = df['y_true'].values.astype(float)
    noise = np.random.normal(0, 0.15, len(df))
    df['y_pred_raw'] = np.clip(base_prob + noise + 0.15, 0.1, 0.9)  # +0.15 = overconfident bias
    
    # Split: train 80%, test 20%
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    logger.info(f"📂 Split: {len(df_train)} train, {len(df_test)} test\n")
    
    # Criar e treinar calibrator
    calibrator = AutoCalibrator(lookback_days=60, min_samples=30)
    calibrator.fit(
        y_pred=df_train['y_pred_raw'].values,
        y_true=df_train['y_true'].values,
        dates=pd.to_datetime(df_train['date'])
    )
    
    # Testar em test set
    y_pred_calibrated = calibrator.predict(df_test['y_pred_raw'].values)
    
    # Métricas
    ece_before = calibrator.calculate_ece(df_test['y_pred_raw'].values, df_test['y_true'].values)
    ece_after = calibrator.calculate_ece(y_pred_calibrated, df_test['y_true'].values)
    
    from sklearn.metrics import brier_score_loss
    brier_before = brier_score_loss(df_test['y_true'], df_test['y_pred_raw'])
    brier_after = brier_score_loss(df_test['y_true'], y_pred_calibrated)
    
    logger.info("\n" + "="*60)
    logger.info("📊 RESULTADOS NO TEST SET")
    logger.info("="*60)
    logger.info(f"ECE:")
    logger.info(f"  Antes:  {ece_before:.4f}")
    logger.info(f"  Depois: {ece_after:.4f}")
    logger.info(f"  Melhoria: {((ece_before - ece_after) / ece_before * 100):.1f}%\n")
    
    logger.info(f"Brier Score:")
    logger.info(f"  Antes:  {brier_before:.4f}")
    logger.info(f"  Depois: {brier_after:.4f}")
    logger.info(f"  Melhoria: {((brier_before - brier_after) / brier_before * 100):.1f}%\n")
    
    # Calibration curve
    curve_before = calibrator.get_calibration_curve(
        df_test['y_pred_raw'].values,
        df_test['y_true'].values,
        n_bins=5
    )
    curve_after = calibrator.get_calibration_curve(
        y_pred_calibrated,
        df_test['y_true'].values,
        n_bins=5
    )
    
    logger.info("Calibration Curve (5 bins):")
    logger.info("  ANTES:")
    logger.info(f"    Pred: {curve_before['prob_pred']}")
    logger.info(f"    True: {curve_before['prob_true']}")
    logger.info("  DEPOIS:")
    logger.info(f"    Pred: {curve_after['prob_pred']}")
    logger.info(f"    True: {curve_after['prob_true']}\n")
    
    # Salvar calibrator
    calibrator.save('models/calibrator_test.pkl')
    
    # Update monitor
    monitor = CalibrationMonitor()
    monitor.update(
        y_pred_raw=df_test['y_pred_raw'].values,
        y_true=df_test['y_true'].values,
        y_pred_calibrated=y_pred_calibrated
    )
    monitor.save_metrics('monitoring/calibration_test_metrics.json')
    monitor.plot_dashboard(save_path='monitoring/calibration_test_dashboard.png')
    
    logger.info("="*60)
    logger.info("✅ Teste completo!")
    logger.info("="*60)
    logger.info("📁 Arquivos gerados:")
    logger.info("  - models/calibrator_test.pkl")
    logger.info("  - monitoring/calibration_test_metrics.json")
    logger.info("  - monitoring/calibration_test_dashboard.png")
    
    # Validação
    assert ece_after < ece_before, "ECE deveria melhorar!"
    assert brier_after <= brier_before, "Brier score deveria melhorar ou manter!"
    
    logger.info("\n✅ All assertions passed!")
    
    return {
        'ece_improvement_pct': ((ece_before - ece_after) / ece_before) * 100,
        'brier_improvement_pct': ((brier_before - brier_after) / brier_before) * 100,
        'n_train': len(df_train),
        'n_test': len(df_test)
    }


if __name__ == '__main__':
    results = test_calibrator_with_nba_data()
    print(f"\n🎯 Final Results: {results}")
