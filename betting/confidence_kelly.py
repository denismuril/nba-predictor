"""
P3.1: Confidence-Adjusted Kelly Criterion (Modo Conservador)

Implementa Fractional Kelly com ajuste por confidence score.

Travas de Segurança:
    - Kelly/8 (0.125) para minimizar variância
    - Edge mínimo de 3% (filtro de ruído)
    - Max bet 3% da banca (hard cap)
    - Confidence mínimo 60%

Usage:
    from betting.confidence_kelly import ConfidenceKelly
    
    kelly = ConfidenceKelly(fraction=0.125)  # Kelly/8
    bet_size = kelly.calculate(prob, odds, confidence, bankroll=100)
"""
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class ConfidenceKelly:
    """
    Fractional Kelly Criterion ajustado por confidence score.
    
    Formula base (Kelly):
        f = (p*b - q) / b
        onde:
        - f = fração do bankroll a apostar
        - p = probabilidade de ganhar
        - q = 1 - p
        - b = odds decimais - 1
    
    Ajustes:
        1. Fractional Kelly: f_final = f * fraction
        2. Confidence adjustment: f_final *= confidence_score
        3. Max bet cap: f_final = min(f_final, max_bet_pct)
    """
    
    def __init__(
        self,
        fraction: float = 0.125,       # Kelly/8 (era 0.25 = Kelly/4)
        min_edge: float = 0.03,        # 3% edge mínimo (filtro rígido)
        max_bet_pct: float = 0.03,     # 3% máximo por aposta
        min_confidence: float = 0.6    # 60% confidence mínimo
    ):
        """
        Inicializa ConfidenceKelly com parâmetros conservadores.
        
        Args:
            fraction: Fração do Kelly (padrão: 0.125 = Kelly/8)
            min_edge: Edge mínimo para apostar (padrão: 3%)
            max_bet_pct: % máximo do bankroll por aposta (padrão: 3%)
            min_confidence: Confidence mínimo (padrão: 60%)
        
        Notas de Segurança:
            - Kelly/8 reduz risco de ruína para <0.1% em 1000 apostas
            - Edge 3% filtra "ruído" de probabilidades incertas
            - Hard cap 3% protege contra over-betting
        """
        self.fraction = fraction
        self.min_edge = min_edge
        self.max_bet_pct = max_bet_pct
        self.min_confidence = min_confidence
        
        logger.info("ConfidenceKelly inicializado (Modo Conservador):")
        logger.info(f"  📊 Kelly Fraction: {fraction} (Kelly/{int(1/fraction)})")
        logger.info(f"  🎯 Min edge: {min_edge:.1%}")
        logger.info(f"  🔒 Max bet: {max_bet_pct:.1%}")
        logger.info(f"  📈 Min confidence: {min_confidence:.1%}")
    
    def calculate_edge(self, prob: float, odds: float) -> float:
        """
        Calcula edge (vantagem matemática).
        
        Edge = prob * odds - 1
        
        Args:
            prob: Probabilidade de ganhar (0-1)
            odds: Odds decimais (ex: 2.0)
        
        Returns:
            Edge (positivo = +EV, negativo = -EV)
        """
        return prob * odds - 1
    
    def calculate_kelly(self, prob: float, odds: float) -> float:
        """
        Calcula Kelly Criterion puro.
        
        Args:
            prob: Probabilidade de ganhar (0-1)
            odds: Odds decimais
        
        Returns:
            Fração do bankroll (0-1)
        """
        if prob <= 0 or prob >= 1:
            return 0.0
        
        if odds <= 1:
            return 0.0
        
        b = odds - 1  # Net odds
        q = 1 - prob
        
        kelly = (prob * b - q) / b
        
        return max(0, kelly)  # Nunca negativo
    
    def calculate_confidence_score(
        self,
        calibration_ece: float,
        model_accuracy: float,
        sample_size: int
    ) -> float:
        """
        DEPRECATED: Esta função mistura heurísticas não relacionadas.
        
        Math-Fix: O critério de Kelly deve depender EXCLUSIVAMENTE
        da probabilidade do modelo vs odds oferecidas.
        
        Esta função é mantida para compatibilidade retroactiva mas
        não deve ser usada em novas implementações.
        """
        # Math-Fix: Aviso de depreciação
        warnings.warn(
            "calculate_confidence_score está DEPRECATED. "
            "O Kelly Criterion deve usar edge puro (prob vs odds). "
            "Use calculate() com confidence=1.0 para Kelly fracionado puro.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Mantida para compatibilidade retroactiva (legacy code)
        ece_score = max(0, 1 - (calibration_ece / 0.15))
        acc_score = max(0, (model_accuracy - 0.50) / 0.10)
        sample_score = min(1.0, sample_size / 100)
        
        confidence = (
            0.5 * ece_score +
            0.3 * acc_score +
            0.2 * sample_score
        )
        
        return np.clip(confidence, 0, 1)
    
    def calculate(
        self,
        prob: float,
        odds: float,
        confidence: float = 1.0,  # Math-Fix: Default 1.0 para Kelly puro
        bankroll: float = 1000.0,
        calibration_ece: Optional[float] = None,
        model_accuracy: Optional[float] = None,
        sample_size: Optional[int] = None
    ) -> Dict:
        """
        Calcula tamanho da aposta com Fractional Kelly.
        
        Math-Fix: O Kelly agora depende APENAS do edge matemático calculado
        como (prob * odds - 1). O parâmetro confidence é mantido para
        compatibilidade mas o default é 1.0 (sem ajuste).
        
        Args:
            prob: Probabilidade calibrada de ganhar
            odds: Odds decimais oferecidas
            confidence: Multiplicador opcional (default: 1.0 = sem ajuste)
            bankroll: Bankroll atual
            [DEPRECATED] calibration_ece, model_accuracy, sample_size:
                Estes parâmetros são ignorados. Use prob calibrada.
        
        Returns:
            Dict com bet_size, edge, kelly_full, kelly_fractional, etc.
        """
        # Math-Fix: Ignorar parâmetros de auto-confidence (deprecated)
        # O edge deve vir exclusivamente da probabilidade calibrada
        if any([calibration_ece is not None, model_accuracy is not None, 
                sample_size is not None]):
            logger.warning(
                "⚠️ Parâmetros calibration_ece/model_accuracy/sample_size são DEPRECATED "
                "e serão ignorados. Use probabilidades calibradas directamente."
            )
        
        # Calcular edge
        edge = self.calculate_edge(prob, odds)
        
        # Check 1: Edge positivo?
        if edge < self.min_edge:
            return {
                'bet_size': 0,
                'bet_pct': 0,
                'edge': edge,
                'kelly_full': 0,
                'kelly_fractional': 0,
                'confidence': confidence,
                'recommendation': 'SKIP',
                'reason': f'Edge baixo: {edge:.2%} < {self.min_edge:.2%}'
            }
        
        # Check 2: Confidence suficiente?
        if confidence < self.min_confidence:
            return {
                'bet_size': 0,
                'bet_pct': 0,
                'edge': edge,
                'kelly_full': 0,
                'kelly_fractional': 0,
                'confidence': confidence,
                'recommendation': 'SKIP',
                'reason': f'Confidence baixo: {confidence:.2%} < {self.min_confidence:.2%}'
            }
        
        # Calcular Kelly
        kelly_full = self.calculate_kelly(prob, odds)
        
        # Aplicar fraction
        kelly_frac = kelly_full * self.fraction
        
        # Aplicar confidence adjustment
        kelly_adjusted = kelly_frac * confidence
        
        # Aplicar max bet cap
        kelly_final = min(kelly_adjusted, self.max_bet_pct)
        
        # Calcular valores
        bet_pct = kelly_final
        bet_size = bankroll * bet_pct
        
        return {
            'bet_size': bet_size,
            'bet_pct': bet_pct,
            'edge': edge,
            'kelly_full': kelly_full,
            'kelly_fractional': kelly_frac,
            'kelly_adjusted': kelly_adjusted,
            'confidence': confidence,
            'recommendation': 'BET',
            'expected_value': bet_size * edge
        }


# Funções auxiliares implementadas
def get_confidence_from_calibrator(calibrator_path: str = 'models/calibrator.pkl') -> float:
    """
    Carrega confidence score do calibrator.
    
    Returns:
        Confidence score baseado em ECE do calibrator
    """
    try:
        from ml_pipeline.calibrator import AutoCalibrator
        from pathlib import Path
        
        if not Path(calibrator_path).exists():
            logger.warning(f"Calibrator não encontrado em {calibrator_path}")
            return 0.5  # Confidence neutro
        
        calibrator = AutoCalibrator.load(calibrator_path)
        
        if not calibrator.fitted:
            logger.warning("Calibrator não está fitted")
            return 0.5
        
        # Calcular confidence baseado em metrics
        stats = calibrator.get_stats()
        
        # ECE < 0.05 = excelente, ECE > 0.15 = ruim
        ece = stats.get('ece', 0.15)
        ece_score = max(0, 1 - (ece / 0.15))
        
        # Sample size confidence
        n_samples = stats.get('n_samples', 0)
        sample_score = min(1.0, n_samples / 100)
        
        # Weighted confidence
        confidence = 0.7 * ece_score + 0.3 * sample_score
        
        logger.info(f"Calibrator confidence: {confidence:.2%} (ECE={ece:.4f}, samples={n_samples})")
        
        return confidence
        
    except Exception as e:
        logger.error(f"Erro ao carregar calibrator: {e}")
        return 0.5


def backtest_kelly(
    predictions_df: pd.DataFrame,
    kelly_calculator: ConfidenceKelly,
    initial_bankroll: float = 1000
) -> Dict:
    """
    Backtest do Kelly Criterion em predictions históricas.
    
    Args:
        predictions_df: DataFrame com columns:
            - prob: Probabilidade calibrada
            - odds: Odds oferecidas
            - result: 1 se ganhou, 0 se perdeu
            - confidence: Confidence score
        kelly_calculator: Instância de ConfidenceKelly
        initial_bankroll: Bankroll inicial
    
    Returns:
        Dict com resultados do backtest
    """
    import pandas as pd
    import numpy as np
    
    logger.info("📊 Iniciando Backtest Kelly Criterion...")
    logger.info(f"   Bankroll inicial: ${initial_bankroll:.2f}")
    logger.info(f"   Games: {len(predictions_df)}\n")
    
    # Validar DataFrame
    required_cols = ['prob', 'odds', 'result', 'confidence']
    for col in required_cols:
        if col not in predictions_df.columns:
            raise ValueError(f"Column '{col}' não encontrada no DataFrame")
    
    # Initialize tracking
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bets_placed = []
    
    total_wagered = 0
    total_won = 0
    wins = 0
    losses = 0
    skipped = 0
    
    # Iterar por cada jogo
    for idx, row in predictions_df.iterrows():
        prob = row['prob']
        odds = row['odds']
        result = row['result']
        confidence = row['confidence']
        
        # Calcular Kelly
        recommendation = kelly_calculator.calculate(
            prob=prob,
            odds=odds,
            confidence=confidence,
            bankroll=bankroll
        )
        
        # Se recomendou apostar
        if recommendation['recommendation'] == 'BET':
            bet_size = recommendation['bet_size']
            
            # Aplicar aposta
            total_wagered += bet_size
            
            if result == 1:  # Ganhou
                profit = bet_size * (odds - 1)
                bankroll += profit
                total_won += (bet_size + profit)
                wins += 1
            else:  # Perdeu
                bankroll -= bet_size
                losses += 1
            
            # Registrar
            bets_placed.append({
                'index': idx,
                'prob': prob,
                'odds': odds,
                'confidence': confidence,
                'bet_size': bet_size,
                'bet_pct': recommendation['bet_pct'],
                'result': result,
                'profit': profit if result == 1 else -bet_size,
                'bankroll_after': bankroll
            })
        else:
            skipped += 1
        
        bankroll_history.append(bankroll)
    
    # Calcular métricas
    total_bets = wins + losses
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    
    total_profit = bankroll - initial_bankroll
    roi = (total_profit / initial_bankroll * 100) if initial_bankroll > 0 else 0
    
    # Sharpe Ratio (simplificado)
    if len(bets_placed) > 1:
        profits = [b['profit'] for b in bets_placed]
        sharpe = (np.mean(profits) / np.std(profits)) * np.sqrt(len(profits)) if np.std(profits) > 0 else 0
    else:
        sharpe = 0
    
    # Max Drawdown
    peak = initial_bankroll
    max_dd = 0
    for b in bankroll_history:
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    # Resultados
    results = {
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'total_profit': total_profit,
        'roi_pct': roi,
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'skipped': skipped,
        'win_rate': win_rate,
        'total_wagered': total_wagered,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'bankroll_history': bankroll_history,
        'bets': bets_placed
    }
    
    # Log summary
    logger.info("="*60)
    logger.info("📊 BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Initial Bankroll: ${initial_bankroll:.2f}")
    logger.info(f"Final Bankroll: ${bankroll:.2f}")
    logger.info(f"Total Profit: ${total_profit:.2f} ({roi:+.2f}%)")
    logger.info(f"\nBets: {total_bets} placed, {skipped} skipped")
    logger.info(f"Win Rate: {win_rate:.1f}% ({wins}W-{losses}L)")
    logger.info(f"Total Wagered: ${total_wagered:.2f}")
    logger.info(f"\nSharpe Ratio: {sharpe:.2f}")
    logger.info(f"Max Drawdown: {max_dd:.1f}%")
    logger.info("="*60)
    
    return results


if __name__ == '__main__':
    # Demo
    print("🎯 Demo: Confidence-Adjusted Kelly\n")
    
    kelly = ConfidenceKelly(fraction=0.125, min_edge=0.03, max_bet_pct=0.03)
    
    # Cenário 1: Boa aposta
    result1 = kelly.calculate(
        prob=0.65,           # 65% chance de ganhar
        odds=1.75,           # Odds 1.75
        confidence=0.85,     # 85% confidence
        bankroll=1000
    )
    
    print("Cenário 1: Boa aposta")
    print(f"  Edge: {result1['edge']:.2%}")
    print(f"  Kelly full: {result1['kelly_full']:.2%}")
    print(f"  Kelly frac: {result1['kelly_fractional']:.2%}")
    print(f"  Kelly adj: {result1['kelly_adjusted']:.2%}")
    print(f"  Bet: ${result1['bet_size']:.2f} ({result1['bet_pct']:.2%})")
    print(f"  EV: ${result1.get('expected_value', 0):.2f}")
    print(f"  → {result1['recommendation']}\n")
    
    # Cenário 2: Edge baixo
    result2 = kelly.calculate(
        prob=0.52,
        odds=1.90,
        confidence=0.80,
        bankroll=1000
    )
    
    print("Cenário 2: Edge baixo")
    print(f"  Edge: {result2['edge']:.2%}")
    print(f"  → {result2['recommendation']}: {result2.get('reason', '')}\n")
    
    # Cenário 3: Low confidence
    result3 = kelly.calculate(
        prob=0.70,
        odds=1.60,
        confidence=0.50,  # Low confidence
        bankroll=1000
    )
    
    print("Cenário 3: Low confidence")
    print(f"  Edge: {result3['edge']:.2%}")
    print(f"  Confidence: {result3['confidence']:.2%}")
    print(f"  → {result3['recommendation']}: {result3.get('reason', '')}\n")
    
    print("✅ Demo completo!")
