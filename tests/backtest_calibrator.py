"""
Backtesting do AutoCalibrator

Valida melhoria de calibração em dados históricos usando walk-forward.

Usage:
    python tests/backtest_calibrator.py --months 6
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import json

from ml_pipeline.calibrator import AutoCalibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibratorBacktest:
    """Backtesting framework para AutoCalibrator."""
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.results = []
    
    def load_historical_data(self, months: int = 6):
        """
        Carrega dados históricos para backtest.
        
        Args:
            months: Quantos meses olhar para trás
        
        Returns:
            DataFrame com predictions históricas
        """
        logger.info(f"📂 Carregando {months} meses de dados históricos...")
        
        try:
            from data.repositories.db_manager import get_db_manager
            db = get_db_manager()
            
            # Query predictions com resultados conhecidos
            cutoff = datetime.now() - timedelta(days=months * 30)
            
            query = f"""
            SELECT 
                date,
                home_team,
                away_team,
                predicted_prob_home as y_pred,
                CASE WHEN home_score > away_score THEN 1 ELSE 0 END as y_true
            FROM predictions
            WHERE date >= '{cutoff.strftime('%Y-%m-%d')}'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
              AND predicted_prob_home IS NOT NULL
            ORDER BY date
            """
            
            conn = db.get_connection()
            df = pd.read_sql_query(query, conn)
            db.return_connection(conn)
            
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"✅ {len(df)} jogos carregados")
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar dados reais: {e}")
            logger.info("📊 Gerando dados sintéticos para demo...")
            
            # Dados sintéticos
            np.random.seed(42)
            n = months * 30 * 10  # ~10 jogos/dia
            
            dates = pd.date_range(
                end=datetime.now(),
                periods=n,
                freq='1H'
            )
            
            # Simular probabilidades mal calibradas
            y_true = np.random.binomial(1, 0.55, n)
            y_pred = np.clip(
                y_true + np.random.normal(0.15, 0.12, n),  # Overconfident bias
                0.05, 0.95
            )
            
            df = pd.DataFrame({
                'date': dates,
                'home_team': ['Team A'] * n,
                'away_team': ['Team B'] * n,
                'y_pred': y_pred,
                'y_true': y_true
            })
            
            return df
    
    def walk_forward_backtest(self, df: pd.DataFrame, train_window: int = 60):
        """
        Walk-forward backtesting.
        
        Args:
            df: DataFrame com historical data
            train_window: Dias de janela de treinamento
        
        Returns:
            Results dictionary
        """
        logger.info("🚀 Iniciando walk-forward backtest...")
        
        # Ordenar por data
        df = df.sort_values('date').reset_index(drop=True)
        
        # Split em chunks
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # Criar windows
        current_date = min_date + timedelta(days=train_window)
        
        fold_results = []
        
        while current_date < max_date:
            # Train window
            train_end = current_date
            train_start = train_end - timedelta(days=train_window)
            
            # Test window (next 7 days)
            test_start = current_date
            test_end = test_start + timedelta(days=7)
            
            # Filter data
            train_df = df[
                (df['date'] >= train_start) &
                (df['date'] < train_end)
            ]
            
            test_df = df[
                (df['date'] >= test_start) &
                (df['date'] < test_end)
            ]
            
            if len(train_df) < 30 or len(test_df) < 5:
                # Skip folds com poucos dados
                current_date += timedelta(days=7)
                continue
            
            # Train calibrator
            calibrator = AutoCalibrator(lookback_days=train_window)
            calibrator.fit(
                y_pred=train_df['y_pred'].values,
                y_true=train_df['y_true'].values,
                dates=train_df['date'].values
            )
            
            # Test
            y_pred_raw = test_df['y_pred'].values
            y_true = test_df['y_true'].values
            y_pred_calibrated = calibrator.predict(y_pred_raw)
            
            # Metrics
            brier_raw = brier_score_loss(y_true, y_pred_raw)
            brier_cal = brier_score_loss(y_true, y_pred_calibrated)
            
            ece_raw = calibrator.calculate_ece(y_pred_raw, y_true)
            ece_cal = calibrator.calculate_ece(y_pred_calibrated, y_true)
            
            fold_results.append({
                'fold_start': test_start.isoformat(),
                'fold_end': test_end.isoformat(),
                'n_train': int(len(train_df)),
                'n_test': int(len(test_df)),
                'brier_raw': float(brier_raw),
                'brier_calibrated': float(brier_cal),
                'brier_improvement': float(((brier_raw - brier_cal) / brier_raw) * 100),
                'ece_raw': float(ece_raw),
                'ece_calibrated': float(ece_cal),
                'ece_improvement': float(((ece_raw - ece_cal) / ece_raw) * 100)
            })
            
            logger.info(
                f"  Fold {len(fold_results)}: {test_start.strftime('%Y-%m-%d')} | "
                f"Brier: {brier_raw:.4f}→{brier_cal:.4f} ({((brier_raw-brier_cal)/brier_raw)*100:+.1f}%)"
            )
            
            # Next window
            current_date += timedelta(days=7)
        
        return fold_results
    
    def generate_report(self, fold_results: list) -> dict:
        """Gera relatório de backtesting."""
        
        logger.info("\n" + "="*60)
        logger.info("📊 RELATÓRIO DE BACKTESTING")
        logger.info("="*60)
        
        df_results = pd.DataFrame(fold_results)
        
        report = {
            'n_folds': len(fold_results),
            'period_start': df_results['fold_start'].min(),  # Já é string
            'period_end': df_results['fold_end'].max(),  # Já é string
            'total_games_tested': int(df_results['n_test'].sum()),
            
            # Brier Score
            'brier_raw_avg': float(df_results['brier_raw'].mean()),
            'brier_calibrated_avg': float(df_results['brier_calibrated'].mean()),
            'brier_improvement_avg': float(df_results['brier_improvement'].mean()),
            'brier_improvement_median': float(df_results['brier_improvement'].median()),
            
            # ECE
            'ece_raw_avg': float(df_results['ece_raw'].mean()),
            'ece_calibrated_avg': float(df_results['ece_calibrated'].mean()),
            'ece_improvement_avg': float(df_results['ece_improvement'].mean()),
            'ece_improvement_median': float(df_results['ece_improvement'].median()),
            
            # Consistency
            'folds_improved': int((df_results['brier_improvement'] > 0).sum()),
            'folds_total': int(len(fold_results)),
            'improvement_rate': float(((df_results['brier_improvement'] > 0).sum() / len(fold_results)) * 100)
        }
        
        # Print report
        logger.info(f"\n📅 Período: {report['period_start']} a {report['period_end']}")
        logger.info(f"📊 {report['n_folds']} folds, {report['total_games_tested']} jogos testados\n")
        
        logger.info("🎯 Brier Score:")
        logger.info(f"  Raw (média): {report['brier_raw_avg']:.4f}")
        logger.info(f"  Calibrated (média): {report['brier_calibrated_avg']:.4f}")
        logger.info(f"  Melhoria média: {report['brier_improvement_avg']:.1f}%")
        logger.info(f"  Melhoria mediana: {report['brier_improvement_median']:.1f}%\n")
        
        logger.info("📏 ECE:")
        logger.info(f"  Raw (média): {report['ece_raw_avg']:.4f}")
        logger.info(f"  Calibrated (média): {report['ece_calibrated_avg']:.4f}")
        logger.info(f"  Melhoria média: {report['ece_improvement_avg']:.1f}%")
        logger.info(f"  Melhoria mediana: {report['ece_improvement_median']:.1f}%\n")
        
        logger.info("✅ Consistência:")
        logger.info(f"  Folds melhorados: {report['folds_improved']}/{report['folds_total']}")
        logger.info(f"  Taxa de melhoria: {report['improvement_rate']:.1f}%\n")
        
        logger.info("="*60)
        
        return report
    
    def plot_results(self, fold_results: list, save_path: str = 'reports/backtest_calibrator.png'):
        """Plota resultados do backtesting."""
        
        df = pd.DataFrame(fold_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Backtesting: AutoCalibrator Performance', fontsize=16, fontweight='bold')
        
        # Plot 1: Brier Score ao longo do tempo
        ax = axes[0, 0]
        ax.plot(df['fold_start'], df['brier_raw'], label='Raw', marker='o', alpha=0.7)
        ax.plot(df['fold_start'], df['brier_calibrated'], label='Calibrated', marker='s', alpha=0.7)
        ax.set_xlabel('Data')
        ax.set_ylabel('Brier Score')
        ax.set_title('Brier Score Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: ECE ao longo do tempo
        ax = axes[0, 1]
        ax.plot(df['fold_start'], df['ece_raw'], label='Raw', marker='o', alpha=0.7)
        ax.plot(df['fold_start'], df['ece_calibrated'], label='Calibrated', marker='s', alpha=0.7)
        ax.set_xlabel('Data')
        ax.set_ylabel('ECE')
        ax.set_title('ECE Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Brier Improvement %
        ax = axes[1, 0]
        ax.bar(range(len(df)), df['brier_improvement'], alpha=0.7, color='green')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Fold')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Brier Score Improvement per Fold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Distribution de melhorias
        ax = axes[1, 1]
        ax.hist(df['brier_improvement'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=df['brier_improvement'].median(), color='red', linestyle='--', 
                   label=f'Mediana: {df["brier_improvement"].median():.1f}%')
        ax.set_xlabel('Improvement (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Improvements')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        logger.info(f"📊 Gráficos salvos em: {save_path}")
        plt.close()


def run_backtest(months: int = 6, train_window: int = 60):
    """Executa backtesting completo."""
    
    logger.info("🚀 Iniciando Backtesting do Calibrator...")
    logger.info(f"   Período: {months} meses")
    logger.info(f"   Train window: {train_window} dias\n")
    
    # Criar backtester
    backtester = CalibratorBacktest(lookback_days=train_window)
    
    # Load data
    df = backtester.load_historical_data(months=months)
    
    if df.empty:
        logger.error("❌ Sem dados para backtest")
        return None
    
    # Walk-forward backtest
    fold_results = backtester.walk_forward_backtest(df, train_window=train_window)
    
    if not fold_results:
        logger.error("❌ Nenhum fold processado")
        return None
    
    # Generate report
    report = backtester.generate_report(fold_results)
    
    # Plot
    backtester.plot_results(fold_results)
    
    # Save report
    report_path = Path('reports/backtest_calibrator_report.json')
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n💾 Relatório salvo em: {report_path}")
    
    return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--months', type=int, default=6, help='Meses de histórico')
    parser.add_argument('--train-window', type=int, default=60, help='Dias de train window')
    
    args = parser.parse_args()
    
    report = run_backtest(months=args.months, train_window=args.train_window)
    
    if report:
        print(f"\n✅ Backtesting completo!")
        print(f"   Melhoria média: {report['brier_improvement_avg']:.1f}%")
        print(f"   Taxa de sucesso: {report['improvement_rate']:.1f}%")
