"""
Auto-Calibration System para NBA Predictor

Usa Isotonic Regression para calibrar probabilidades do modelo,
melhorando a confiabilidade das previsões.

Usage:
    from ml_pipeline.calibrator import AutoCalibrator
    
    # Training
    calibrator = AutoCalibrator(lookback_days=30)
    calibrator.fit(y_pred, y_true, dates)
    calibrator.save('models/calibrator.pkl')
    
    # Prediction
    calibrator = AutoCalibrator.load('models/calibrator.pkl')
    probs_calibrated = calibrator.predict(probs_raw)
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class AutoCalibrator:
    """
    Calibrador automático usando Isotonic Regression.
    
    Melhora a calibração de probabilidades do modelo ML,
    alinhando previsões com frequências observadas.
    """
    
    def __init__(self, lookback_days=30, min_samples=50):
        """
        Args:
            lookback_days: Janela de dados recentes para calibração (dias)
            min_samples: Mínimo de amostras para calibrar
        """
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        self.fitted = False
        self.fit_date = None
        self.n_samples = 0
        
    def fit(self, y_pred, y_true, dates=None):
        """
        Fit calibrator nos dados.
        
        Args:
            y_pred: Probabilidades preditas (0-1)
            y_true: Labels verdadeiros (0 ou 1)
            dates: Datas dos jogos (opcional, para lookback)
        
        Returns:
            self
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        
        # Se dates fornecido, usar apenas últimos N dias
        if dates is not None:
            dates = pd.to_datetime(dates)
            cutoff_date = dates.max() - timedelta(days=self.lookback_days)
            mask = dates >= cutoff_date
            y_pred = y_pred[mask]
            y_true = y_true[mask]
            logger.info(f"📅 Usando últimos {self.lookback_days} dias: {mask.sum()} jogos")
        
        # Validação
        if len(y_pred) < self.min_samples:
            logger.warning(
                f"⚠️ Apenas {len(y_pred)} amostras (mínimo: {self.min_samples}). "
                f"Calibração pode ser instável!"
            )
        
        # Fit Isotonic Regression
        self.isotonic.fit(y_pred, y_true)
        self.fitted = True
        self.fit_date = datetime.now()
        self.n_samples = len(y_pred)
        
        # Metrics
        probs_calibrated = self.isotonic.predict(y_pred)
        brier_before = brier_score_loss(y_true, y_pred)
        brier_after = brier_score_loss(y_true, probs_calibrated)
        improvement = ((brier_before - brier_after) / brier_before) * 100
        
        logger.info(f"✅ Calibrator fitted com {self.n_samples} jogos")
        logger.info(f"   Brier score: {brier_before:.4f} → {brier_after:.4f} ({improvement:+.1f}%)")
        
        return self
    
    def predict(self, y_pred):
        """
        Retorna probabilidades calibradas.
        
        Args:
            y_pred: Probabilidades raw do modelo (0-1)
        
        Returns:
            probs_calibrated: Probabilidades calibradas (0-1)
        """
        if not self.fitted:
            logger.warning("⚠️ Calibrator não foi fitted! Retornando probabilidades raw.")
            return np.asarray(y_pred)
        
        y_pred = np.asarray(y_pred)
        probs_calibrated = self.isotonic.predict(y_pred)
        
        return probs_calibrated
    
    def get_calibration_curve(self, y_pred, y_true, n_bins=10):
        """
        Calcula curva de calibração para visualização.
        
        Args:
            y_pred: Probabilidades preditas
            y_true: Labels verdadeiros
            n_bins: Número de bins
        
        Returns:
            dict com 'prob_pred', 'prob_true', 'counts'
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        
        # Criar bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calcular frequência observada por bin
        prob_pred = []
        prob_true = []
        counts = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                prob_pred.append(y_pred[mask].mean())
                prob_true.append(y_true[mask].mean())
                counts.append(mask.sum())
            else:
                prob_pred.append(bins[i] + (bins[i+1] - bins[i]) / 2)
                prob_true.append(np.nan)
                counts.append(0)
        
        return {
            'prob_pred': np.array(prob_pred),
            'prob_true': np.array(prob_true),
            'counts': np.array(counts),
            'bins': bins
        }
    
    def calculate_ece(self, y_pred, y_true, n_bins=10):
        """
        Calcula Expected Calibration Error (ECE).
        
        Métrica padrão para avaliar calibração.
        
        Args:
            y_pred: Probabilidades preditas
            y_true: Labels verdadeiros
            n_bins: Número de bins
        
        Returns:
            ece: Expected Calibration Error (0-1, menor é melhor)
        """
        curve = self.get_calibration_curve(y_pred, y_true, n_bins)
        
        # ECE = weighted average of |prob_pred - prob_true|
        total = curve['counts'].sum()
        if total == 0:
            return np.nan
        
        ece = 0
        for i in range(n_bins):
            if curve['counts'][i] > 0 and not np.isnan(curve['prob_true'][i]):
                weight = curve['counts'][i] / total
                diff = abs(curve['prob_pred'][i] - curve['prob_true'][i])
                ece += weight * diff
        
        return ece
    
    def get_stats(self):
        """Retorna estatísticas do calibrator."""
        return {
            'fitted': self.fitted,
            'fit_date': self.fit_date,
            'n_samples': self.n_samples,
            'lookback_days': self.lookback_days,
            'min_samples': self.min_samples
        }
    
    def save(self, filepath):
        """
        Salva calibrator em arquivo.
        
        Args:
            filepath: Caminho completo para salvar (ex: 'models/calibrator.pkl')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"💾 Calibrator salvo em: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Carrega calibrator de arquivo.
        
        Args:
            filepath: Caminho do arquivo
        
        Returns:
            AutoCalibrator instance
        """
        with open(filepath, 'rb') as f:
            calibrator = pickle.load(f)
        
        logger.info(f"📂 Calibrator carregado de: {filepath}")
        logger.info(f"   Fitted em: {calibrator.fit_date}, {calibrator.n_samples} samples")
        
        return calibrator


def get_calibrator(filepath='models/calibrator.pkl', lookback_days=30):
    """
    Convenience function para carregar calibrator (ou criar novo se não existir).
    
    Args:
        filepath: Caminho do calibrator salvo
        lookback_days: Lookback se precisar criar novo
    
    Returns:
        AutoCalibrator instance
    """
    filepath = Path(filepath)
    
    if filepath.exists():
        try:
            return AutoCalibrator.load(filepath)
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar calibrator: {e}. Criando novo...")
    
    # Criar novo
    logger.info("🆕 Criando novo calibrator")
    return AutoCalibrator(lookback_days=lookback_days)


if __name__ == '__main__':
    # Demo
    print("🔍 Demo: Auto-Calibrator\n")
    
    # Simular dados
    np.random.seed(42)
    n_samples = 500
    
    # Modelo mal calibrado (overconfident)
    y_true = np.random.binomial(1, 0.6, n_samples)
    y_pred_raw = np.clip(y_true + np.random.normal(0.2, 0.15, n_samples), 0, 1)
    
    # Dates
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    
    # Fit calibrator
    calibrator = AutoCalibrator(lookback_days=60)
    calibrator.fit(y_pred_raw, y_true, dates)
    
    # Predict
    y_pred_calibrated = calibrator.predict(y_pred_raw)
    
    # Metrics
    ece_before = calibrator.calculate_ece(y_pred_raw, y_true)
    ece_after = calibrator.calculate_ece(y_pred_calibrated, y_true)
    
    print(f"\n📊 Resultados:")
    print(f"   ECE antes: {ece_before:.4f}")
    print(f"   ECE depois: {ece_after:.4f}")
    print(f"   Melhoria: {((ece_before - ece_after) / ece_before) * 100:.1f}%")
    
    # Curva de calibração
    curve_before = calibrator.get_calibration_curve(y_pred_raw, y_true, n_bins=5)
    curve_after = calibrator.get_calibration_curve(y_pred_calibrated, y_true, n_bins=5)
    
    print(f"\n📈 Calibration Curve (5 bins):")
    print(f"   ANTES  - Pred: {curve_before['prob_pred']}")
    print(f"          - True: {curve_before['prob_true']}")
    print(f"   DEPOIS - Pred: {curve_after['prob_pred']}")
    print(f"          - True: {curve_after['prob_true']}")
    
    print("\n✅ Demo completo!")
