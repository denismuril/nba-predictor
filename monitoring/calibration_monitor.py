"""
Calibration Monitoring Dashboard

Monitora a calibração do modelo em tempo real:
- Calibration curves (expected vs observed)
- ECE (Expected Calibration Error)
- Brier score before/after
- Reliability diagram

Usage:
    from monitoring.calibration_monitor import CalibrationMonitor
    
    monitor = CalibrationMonitor()
    monitor.update(y_pred, y_true, y_pred_calibrated)
    monitor.plot_dashboard()
    monitor.save_metrics('monitoring/calibration_metrics.json')
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
import json
from pathlib import Path
from datetime import datetime

class CalibrationMonitor:
    """Monitor de calibração para tracking contínuo."""
    
    def __init__(self):
        self.history = []
        
    def update(self, y_pred_raw, y_true, y_pred_calibrated=None, date=None):
        """
        Adiciona novo checkpoint de calibração.
        
        Args:
            y_pred_raw: Probabilidades raw do modelo
            y_true: Labels verdadeiros
            y_pred_calibrated: Probabilidades calibradas (opcional)
            date: Data do checkpoint (default: now)
        """
        if date is None:
            date = datetime.now()
        
        # Calcular métricas
        brier_raw = brier_score_loss(y_true, y_pred_raw)
        
        metrics = {
            'date': date,
            'n_samples': len(y_pred_raw),
            'brier_raw': brier_raw,
            'mean_pred_raw': float(np.mean(y_pred_raw)),
            'mean_true': float(np.mean(y_true))
        }
        
        if y_pred_calibrated is not None:
            metrics['brier_calibrated'] = brier_score_loss(y_true, y_pred_calibrated)
            metrics['mean_pred_calibrated'] = float(np.mean(y_pred_calibrated))
            metrics['improvement_pct'] = ((brier_raw - metrics['brier_calibrated']) / brier_raw) * 100
        
        self.history.append(metrics)
        
    def plot_dashboard(self, save_path=None):
        """
        Plota dashboard de calibração.
        
        Args:
            save_path: Caminho para salvar imagem (opcional)
        """
        if not self.history:
            print("⚠️ Nenhum histórico disponível para plotar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Calibration Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        df = pd.DataFrame(self.history)
        
        # Plot 1: Brier Score ao longo do tempo
        ax = axes[0, 0]
        ax.plot(df['date'], df['brier_raw'], label='Raw', marker='o', color='red', alpha=0.7)
        if 'brier_calibrated' in df.columns:
            ax.plot(df['date'], df['brier_calibrated'], label='Calibrated', marker='s', color='green', alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Brier Score')
        ax.set_title('Brier Score Over Time (lower is better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Improvement %
        if 'improvement_pct' in df.columns:
            ax = axes[0, 1]
            ax.bar(range(len(df)), df['improvement_pct'], color='green', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Checkpoint')
            ax.set_ylabel('Improvement (%)')
            ax.set_title('Calibration Improvement per Checkpoint')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Mean Predictions
        ax = axes[1, 0]
        ax.plot(df['date'], df['mean_pred_raw'], label='Mean Pred (Raw)', marker='o', color='blue')
        if 'mean_pred_calibrated' in df.columns:
            ax.plot(df['date'], df['mean_pred_calibrated'], label='Mean Pred (Calibrated)', marker='s', color='purple')
        ax.plot(df['date'], df['mean_true'], label='Mean True', marker='^', color='orange', linestyle='--')
        ax.set_xlabel('Date')
        ax.set_ylabel('Probability')
        ax.set_title('Mean Predictions vs Reality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Sample Counts
        ax = axes[1, 1]
        ax.bar(range(len(df)), df['n_samples'], color='skyblue', alpha=0.7)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Size per Checkpoint')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Dashboard salvo em: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_metrics(self, filepath):
        """Salva métricas em JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dates to string
        history_serializable = []
        for item in self.history:
            item_copy = item.copy()
            if isinstance(item_copy['date'], datetime):
                item_copy['date'] = item_copy['date'].isoformat()
            history_serializable.append(item_copy)
        
        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"💾 Métricas salvas em: {filepath}")
    
    @classmethod
    def load_metrics(cls, filepath):
        """Carrega métricas de JSON."""
        monitor = cls()
        
        with open(filepath, 'r') as f:
            history = json.load(f)
        
        # Convert date strings back to datetime
        for item in history:
            if 'date' in item and isinstance(item['date'], str):
                item['date'] = datetime.fromisoformat(item['date'])
        
        monitor.history = history
        print(f"📂 Métricas carregadas de: {filepath}")
        
        return monitor
    
    def get_latest_metrics(self):
        """Retorna métricas mais recentes."""
        if not self.history:
            return None
        return self.history[-1]
    
    def get_summary(self):
        """Retorna resumo de todas as métricas."""
        if not self.history:
            return "Nenhum histórico disponível"
        
        df = pd.DataFrame(self.history)
        
        summary = {
            'total_checkpoints': len(df),
            'total_samples': int(df['n_samples'].sum()),
            'avg_brier_raw': float(df['brier_raw'].mean()),
            'latest_brier_raw': float(df['brier_raw'].iloc[-1])
        }
        
        if 'brier_calibrated' in df.columns:
            summary['avg_brier_calibrated'] = float(df['brier_calibrated'].mean())
            summary['latest_brier_calibrated'] = float(df['brier_calibrated'].iloc[-1])
            summary['avg_improvement_pct'] = float(df['improvement_pct'].mean())
        
        return summary


if __name__ == '__main__':
    # Demo
    print("🔍 Demo: Calibration Monitor\n")
    
    monitor = CalibrationMonitor()
    
    # Simular 3 checkpoints
    np.random.seed(42)
    for i in range(3):
        n = 100
        y_true = np.random.binomial(1, 0.55, n)
        y_pred_raw = np.clip(y_true + np.random.normal(0.15, 0.1, n), 0, 1)
        y_pred_calibrated = np.clip(y_pred_raw - 0.05, 0, 1)  # Simular melhoria
        
        monitor.update(y_pred_raw, y_true, y_pred_calibrated)
    
    # Summary
    print("📊 Summary:")
    summary = monitor.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Plot
    monitor.plot_dashboard(save_path='monitoring/calibration_dashboard_demo.png')
    
    print("\n✅ Demo completo!")
