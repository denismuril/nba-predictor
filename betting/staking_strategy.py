"""
Gestão de Banca Profissional - Staking Strategy (Modo Conservador)
====================================================================

Motor de position sizing para operação quantitativa com foco em preservação de capital.

Travas de Segurança Implementadas:
    - Kelly Criterion Fracionado (Kelly/8 = 0.125)
    - Hard Cap de 3% da banca por aposta
    - Edge mínimo de 3% sobre o mercado (filtro de ruído)
    - Detecção de correlação entre apostas
    - Proteção contra ruína financeira

Uso:
    from betting.staking_strategy import KellyCriterionStrategy
    
    strategy = KellyCriterionStrategy(bankroll=100.0)  # Banca conservadora
    result = strategy.calculate_optimal_stake(
        model_prob=0.55,
        market_odds=1.95,
        confidence=0.80
    )
    print(f"Apostar: R${result['stake_amount']:.2f}")

Autor: NBA Predictor Team
Data: 2025-12-15 (Atualizado para Modo Conservador)
"""

import logging
from typing import Dict, List, Optional
from betting.confidence_kelly import ConfidenceKelly

logger = logging.getLogger(__name__)


class KellyCriterionStrategy:
    """
    Estratégia profissional de staking usando Kelly Criterion (Modo Conservador).
    
    Travas de Segurança Financeira:
        1. Fractional Kelly (0.125 = Kelly/8) - Minimiza variância
        2. Hard Cap (3%) - Nunca arriscar mais que 3% da banca
        3. Min Edge (3%) - Filtro rígido de ruído vs mercado
        4. Detecção de Correlação - Reduz exposure em apostas relacionadas
    
    Filosofia:
        - Preservar capital é prioridade máxima
        - Kelly/8 reduz risco de ruína para ~0.1% em 1000 apostas
        - Edge 3% filtra apostas com incerteza alta
    """
    
    def __init__(
        self,
        bankroll: float = 100.0,           # Banca conservadora (era 1000.0)
        kelly_fraction: float = 0.125,      # Kelly/8 (era 0.25 = Kelly/4)
        hard_cap_pct: float = 0.03,         # 3% máximo por aposta
        min_edge_pct: float = 0.03,         # 3% edge mínimo (filtro rígido)
        min_confidence: float = 0.60        # 60% confidence mínimo
    ):
        """
        Inicializa estratégia de staking conservadora.
        
        Args:
            bankroll: Banca atual em R$ (padrão: R$100)
            kelly_fraction: Fração do Kelly (padrão: 0.125 = Kelly/8)
            hard_cap_pct: % máximo da banca por aposta (padrão: 3%)
            min_edge_pct: Edge mínimo para apostar (padrão: 3%)
            min_confidence: Confidence mínimo (padrão: 60%)
        
        Notas de Segurança:
            - Kelly/8 vs Kelly/4: Reduz variância em ~50%
            - Edge 3% vs 5%: Mais oportunidades, mas ainda conservador
            - Hard cap 3%: Proteção absoluta contra over-betting
        """
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.hard_cap_pct = hard_cap_pct
        self.min_edge_pct = min_edge_pct
        self.min_confidence = min_confidence
        
        # Usar ConfidenceKelly como engine interno
        self.kelly_engine = ConfidenceKelly(
            fraction=kelly_fraction,
            min_edge=min_edge_pct,
            max_bet_pct=hard_cap_pct,
            min_confidence=min_confidence
        )
        
        logger.info("🏦 KellyCriterionStrategy inicializada (Modo Conservador):")
        logger.info(f"   💰 Bankroll: R${bankroll:,.2f}")
        logger.info(f"   📊 Kelly Fraction: {kelly_fraction} (Kelly/{int(1/kelly_fraction)})")
        logger.info(f"   🔒 Hard Cap: {hard_cap_pct:.1%}")
        logger.info(f"   🎯 Min Edge: {min_edge_pct:.1%} (filtro de ruído)")
        logger.info(f"   📈 Min Confidence: {min_confidence:.1%}")
    
    def update_bankroll(self, new_bankroll: float):
        """
        Atualiza a banca atual.
        
        Args:
            new_bankroll: Nova banca em unidades monetárias
        """
        old_bankroll = self.bankroll
        self.bankroll = new_bankroll
        logger.info(f"💰 Bankroll atualizado: ${old_bankroll:,.2f} → ${new_bankroll:,.2f}")
    
    def calculate_optimal_stake(
        self,
        model_prob: float,
        market_odds: float,
        confidence: float = 1.0,
        game_id: Optional[str] = None,
        team: Optional[str] = None,
        market: Optional[str] = None
    ) -> Dict:
        """
        Calcula o stake ótimo para uma aposta.
        
        Args:
            model_prob: Probabilidade do modelo (0.0 a 1.0)
            market_odds: Odds decimais do mercado (ex: 1.95)
            confidence: Confidence score (0.0 a 1.0, default: 1.0)
            game_id: ID do jogo (opcional, para tracking)
            team: Time apostado (opcional, para tracking)
            market: Tipo de mercado (opcional, ex: 'Moneyline', 'Spread')
        
        Returns:
            Dict com:
                - stake_amount: Valor monetário a apostar
                - stake_pct: % da banca
                - edge: Edge matemático (EV)
                - kelly_full: Kelly completo
                - kelly_fraction: Kelly fracionado
                - recommendation: 'BET' ou 'SKIP'
                - reason: Razão para recomendação
                - correlation_alert: Alerta de correlação (se houver)
        """
        # Calcular usando engine Kelly
        kelly_result = self.kelly_engine.calculate(
            prob=model_prob,
            odds=market_odds,
            confidence=confidence,
            bankroll=self.bankroll
        )
        
        # Enriquecer resultado com metadata
        result = {
            'stake_amount': kelly_result['bet_size'],
            'stake_pct': kelly_result['bet_pct'] * 100,  # Converter para %
            'edge': kelly_result['edge'] * 100,  # Converter para %
            'kelly_full': kelly_result['kelly_full'] * 100,
            'kelly_fraction': kelly_result['kelly_fractional'] * 100,
            'recommendation': kelly_result['recommendation'],
            'reason': kelly_result.get('reason', ''),
            'expected_value': kelly_result.get('expected_value', 0),
            'confidence': confidence,
            'correlation_alert': None,
            # Metadata para tracking
            'game_id': game_id,
            'team': team,
            'market': market,
            'model_prob': model_prob * 100,
            'market_odds': market_odds
        }
        
        return result
    
    def adjust_for_correlation(self, bets_list: List[Dict]) -> List[Dict]:
        """
        Detecta correlação entre apostas e ajusta stakes.
        
        Lógica:
            - Se houver múltiplas apostas no mesmo jogo (mesmo game_id)
            - OU múltiplas apostas no mesmo time (mesmo team)
            - Reduz o stake de TODAS as apostas correlacionadas em 50%
        
        Args:
            bets_list: Lista de apostas calculadas (output de calculate_optimal_stake)
        
        Returns:
            Lista de apostas com stakes ajustados e alertas de correlação
        """
        if not bets_list or len(bets_list) < 2:
            return bets_list
        
        # Agrupar apostas por game_id e team
        game_groups = {}
        team_groups = {}
        
        for idx, bet in enumerate(bets_list):
            # Apenas processar apostas recomendadas
            if bet.get('recommendation') != 'BET':
                continue
            
            game_id = bet.get('game_id')
            team = bet.get('team')
            
            # Agrupar por jogo
            if game_id:
                if game_id not in game_groups:
                    game_groups[game_id] = []
                game_groups[game_id].append(idx)
            
            # Agrupar por time
            if team:
                if team not in team_groups:
                    team_groups[team] = []
                team_groups[team].append(idx)
        
        # Detectar correlações
        correlated_indices = set()
        correlation_reasons = {}
        
        # Correlação por jogo (ex: ML + Spread no mesmo jogo)
        for game_id, indices in game_groups.items():
            if len(indices) > 1:
                for idx in indices:
                    correlated_indices.add(idx)
                    correlation_reasons[idx] = f"Múltiplas apostas no jogo {game_id}"
        
        # Correlação por time (ex: Spread + Total em jogos diferentes do LAL)
        for team, indices in team_groups.items():
            if len(indices) > 1:
                for idx in indices:
                    correlated_indices.add(idx)
                    if idx in correlation_reasons:
                        correlation_reasons[idx] += f" + múltiplas apostas em {team}"
                    else:
                        correlation_reasons[idx] = f"Múltiplas apostas em {team}"
        
        # Aplicar ajuste de 50% nas apostas correlacionadas
        adjusted_bets = []
        for idx, bet in enumerate(bets_list):
            adjusted_bet = bet.copy()
            
            if idx in correlated_indices:
                # Reduzir stake em 50%
                original_stake = adjusted_bet['stake_amount']
                adjusted_bet['stake_amount'] = original_stake * 0.5
                adjusted_bet['stake_pct'] = adjusted_bet['stake_pct'] * 0.5
                adjusted_bet['correlation_alert'] = (
                    f"⚠️ Stake reduzido 50% (${original_stake:.2f} → ${adjusted_bet['stake_amount']:.2f}): "
                    f"{correlation_reasons[idx]}"
                )
                logger.warning(
                    f"⚠️ CORRELAÇÃO DETECTADA - {correlation_reasons[idx]}: "
                    f"Stake reduzido ${original_stake:.2f} → ${adjusted_bet['stake_amount']:.2f}"
                )
            
            adjusted_bets.append(adjusted_bet)
        
        return adjusted_bets


def format_stake_recommendation(stake_result: Dict) -> str:
    """
    Formata recomendação de stake para display.
    
    Args:
        stake_result: Output de calculate_optimal_stake()
    
    Returns:
        String formatada para console/log
    """
    if stake_result['recommendation'] == 'SKIP':
        return (
            f"❌ SKIP - {stake_result.get('team', 'N/A')} @ {stake_result.get('market_odds', 0):.2f}\n"
            f"   Razão: {stake_result.get('reason', 'N/A')}"
        )
    
    lines = [
        f"✅ APOSTAR - {stake_result.get('team', 'N/A')} {stake_result.get('market', 'N/A')} @ {stake_result.get('market_odds', 0):.2f}",
        f"   Edge: {stake_result['edge']:.1f}%",
        f"   Kelly: {stake_result['kelly_full']:.1f}% → Kelly/{int(1/(stake_result['kelly_fraction']/stake_result['kelly_full']))}: {stake_result['kelly_fraction']:.2f}%",
        f"   Stake Sugerido: ${stake_result['stake_amount']:.2f} ({stake_result['stake_pct']:.2f}% da banca)"
    ]
    
    if stake_result.get('correlation_alert'):
        lines.append(f"   {stake_result['correlation_alert']}")
    
    return "\n".join(lines)


# Exemplo de uso
if __name__ == '__main__':
    import sys
    
    print("=" * 70)
    print("🎯 DEMO: Sistema de Gestão de Banca Profissional")
    print("=" * 70)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Inicializar estratégia com banca conservadora de R$100
    strategy = KellyCriterionStrategy(bankroll=100.0)
    
    print("\n" + "=" * 70)
    print("📊 CENÁRIO 1: Aposta única com bom edge")
    print("=" * 70)
    
    result1 = strategy.calculate_optimal_stake(
        model_prob=0.555,  # 55.5% chance
        market_odds=1.95,  # Odds 1.95
        confidence=0.80,   # 80% confidence
        game_id="LAL_vs_BRK",
        team="LAL",
        market="Moneyline"
    )
    
    print(format_stake_recommendation(result1))
    
    print("\n" + "=" * 70)
    print("📊 CENÁRIO 2: Múltiplas apostas (CORRELAÇÃO)")
    print("=" * 70)
    
    # Criar lista de apostas no mesmo jogo
    bets = [
        strategy.calculate_optimal_stake(
            model_prob=0.555,
            market_odds=1.95,
            confidence=0.80,
            game_id="LAL_vs_BRK",
            team="LAL",
            market="Moneyline"
        ),
        strategy.calculate_optimal_stake(
            model_prob=0.58,
            market_odds=1.90,
            confidence=0.75,
            game_id="LAL_vs_BRK",
            team="LAL",
            market="Spread -3.5"
        )
    ]
    
    print("\n🔍 Antes de ajustar para correlação:")
    for bet in bets:
        print(f"   {bet['market']}: ${bet['stake_amount']:.2f}")
    
    # Ajustar para correlação
    adjusted_bets = strategy.adjust_for_correlation(bets)
    
    print("\n📉 Após ajustar para correlação:")
    for bet in adjusted_bets:
        print(format_stake_recommendation(bet))
    
    print("\n" + "=" * 70)
    print("📊 CENÁRIO 3: Edge baixo (SKIP)")
    print("=" * 70)
    
    result3 = strategy.calculate_optimal_stake(
        model_prob=0.52,  # 52% chance
        market_odds=1.90,  # Odds 1.90
        confidence=0.80,
        game_id="GSW_vs_PHX",
        team="GSW",
        market="Moneyline"
    )
    
    print(format_stake_recommendation(result3))
    
    print("\n" + "=" * 70)
    print("✅ Demo completo!")
    print("=" * 70)
