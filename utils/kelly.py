"""
Kelly Criterion for Optimal Bet Sizing in Sports Betting.

Implements both Full Kelly and Fractional Kelly (recommended for reduced volatility).
"""
from typing import Dict, Optional, Union, List, Any

def kelly_criterion_advanced(
    prob_win: float, 
    decimal_odds: float, 
    fractional: float = 0.25
) -> Dict[str, Union[float, bool]]:
    """
    Calcula a fração da banca a ser apostada usando o Critério de Kelly.
    """
    if prob_win <= 0 or prob_win >= 1:
        return {'kelly_full': 0, 'kelly_fractional': 0, 'ev': 0, 'should_bet': False}
    
    if decimal_odds <= 1:
        return {'kelly_full': 0, 'kelly_fractional': 0, 'ev': 0, 'should_bet': False}
    
    b = decimal_odds - 1
    p = prob_win
    q = 1 - p
    
    kelly_full = (b * p - q) / b
    ev = (p * b - q) * 100
    
    kelly_fractional_value = kelly_full * fractional
    should_bet = ev > 0 and kelly_full > 0
    
    # Limite 5%
    kelly_fractional_value = min(kelly_fractional_value, 0.05)
    
    return {
        'kelly_full': max(0, kelly_full),
        'kelly_fractional': max(0, kelly_fractional_value),
        'ev': ev,
        'should_bet': should_bet
    }

def calculate_kelly_criterion(prob_win, odds, bankroll=1000):
    """Wrapper simplificado para testes."""
    res = kelly_criterion_advanced(prob_win, odds, fractional=1.0)
    return bankroll * res['kelly_fractional']

def kelly_criterion_quarter(prob_win, odds, bankroll=1000):
    """Kelly Quarter (1/4 do Full Kelly)."""
    res = kelly_criterion_advanced(prob_win, odds, fractional=0.25)
    return bankroll * res['kelly_fractional']

def get_bet_recommendation(
    prob_home: float,
    prob_away: float,
    odd_home: float,
    odd_away: float,
    fractional: float = 0.25
) -> Dict[str, Union[str, float]]:
    """
    Retorna recomendação de aposta para um jogo.
    
    Args:
        prob_home: Probabilidade de vitória do time da casa (0-100).
        prob_away: Probabilidade de vitória do time visitante (0-100).
        odd_home: Odds do time da casa (decimal).
        odd_away: Odds do time visitante (decimal).
        fractional: Fração do Kelly a usar (padrão: 0.25 = Quarter Kelly).
        
    Returns:
        Dicionário com recomendação, stake percentual, EV e reasoning.
    """
    p_home = prob_home / 100
    p_away = prob_away / 100
    
    kelly_home = kelly_criterion_advanced(p_home, odd_home, fractional)
    kelly_away = kelly_criterion_advanced(p_away, odd_away, fractional)
    
    if kelly_home['should_bet'] and kelly_away['should_bet']:
        if kelly_home['ev'] > kelly_away['ev']:
            return _format_rec('HOME', kelly_home)
        else:
            return _format_rec('AWAY', kelly_away)
    elif kelly_home['should_bet']:
        return _format_rec('HOME', kelly_home)
    elif kelly_away['should_bet']:
        return _format_rec('AWAY', kelly_away)
    else:
        return {
            'recommendation': 'NO BET',
            'stake_pct': 0,
            'ev': max(kelly_home['ev'], kelly_away['ev']),
            'reasoning': 'Sem valor esperado positivo suficiente'
        }

def _format_rec(side: str, kelly_res: Dict[str, Any]) -> Dict[str, Union[str, float]]:
    side_pt = "Casa" if side == 'HOME' else "Visitante"
    return {
        'recommendation': side,
        'stake_pct': kelly_res['kelly_fractional'] * 100,
        'ev': kelly_res['ev'],
        'reasoning': f"Valor detectado na {side_pt} (EV: {kelly_res['ev']:.1f}%)"
    }

class PortfolioManager:
    """
    Gerencia o portfólio de apostas do dia para evitar exposição excessiva.
    """
    def __init__(self, max_exposure_per_game=0.05):
        self.max_exposure_per_game = max_exposure_per_game # Max 5% da banca em um único jogo

    def adjust_stakes_for_correlation(self, bets_list):
        """
        Ajusta os stakes se houver múltiplas apostas no mesmo jogo.
        
        Args:
            bets_list (list): Lista de dicts [{'game_id': 'LALvsBOS', 'stake_pct': 0.02, ...}]
            
        Returns:
            list: Lista com 'stake_pct' ajustado.
        """
        # Agrupar por jogo
        game_groups: Dict[str, List[Dict[str, Any]]] = {}
        for bet in bets_list:
            # Identificador único do jogo (Home vs Away)
            # Assumindo que o dict tem chaves 'Casa' e 'Visitante'
            if 'Casa' in bet and 'Visitante' in bet:
                game_id = f"{bet['Casa']} vs {bet['Visitante']}"
            else:
                game_id = "Unknown"
                
            if game_id not in game_groups:
                game_groups[game_id] = []
            game_groups[game_id].append(bet)
            
        adjusted_bets = []
        
        for game_id, bets in game_groups.items():
            total_stake = sum(b['Stake %'] for b in bets)
            
            # Se tivermos mais de uma aposta no mesmo jogo (ex: ML + Spread)
            if len(bets) > 1:
                # Fator de Correlação (Penalidade)
                # Se apostamos em 2 mercados correlacionados, reduzimos o tamanho total
                correlation_penalty = 0.7 # Reduzir em 30%
                
                for bet in bets:
                    original_stake = bet['Stake %']
                    # Ajuste proporcional
                    new_stake = original_stake * correlation_penalty
                    bet['Stake %'] = round(new_stake, 2)
                    bet['Reasoning'] = bet.get('Reasoning', '') + " (Ajustado por Correlação)"
                    adjusted_bets.append(bet)
            else:
                # Aposta única no jogo, verificar limite global por jogo
                if total_stake > (self.max_exposure_per_game * 100):
                    bet = bets[0]
                    bet['Stake %'] = self.max_exposure_per_game * 100
                    bet['Reasoning'] = bet.get('Reasoning', '') + " (Limitado por Max Exposure)"
                    adjusted_bets.append(bet)
                else:
                    adjusted_bets.append(bets[0])
                    
        return adjusted_bets
