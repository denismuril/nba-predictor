"""
Testes Unitários - Staking Strategy
====================================

Testa o motor de gestão de banca profissional.

Execução:
    pytest tests/test_staking_strategy.py -v
"""

import pytest
import sys
from pathlib import Path

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from betting.staking_strategy import KellyCriterionStrategy, format_stake_recommendation


class TestKellyCriterionStrategy:
    """Testes para KellyCriterionStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Fixture: estratégia padrão com banca de $1000."""
        return KellyCriterionStrategy(
            bankroll=1000.0,
            kelly_fraction=0.25,
            hard_cap_pct=0.03,
            min_edge_pct=0.05,
            min_confidence=0.60
        )
    
    def test_initialization(self, strategy):
        """Testa inicialização correta."""
        assert strategy.bankroll == 1000.0
        assert strategy.kelly_fraction == 0.25
        assert strategy.hard_cap_pct == 0.03
        assert strategy.min_edge_pct == 0.05
        assert strategy.min_confidence == 0.60
    
    def test_update_bankroll(self, strategy):
        """Testa atualização de banca."""
        strategy.update_bankroll(1500.0)
        assert strategy.bankroll == 1500.0
    
    def test_good_edge_bet(self, strategy):
        """Testa aposta com edge positivo (8.5%)."""
        result = strategy.calculate_optimal_stake(
            model_prob=0.555,  # 55.5%
            market_odds=1.95,
            confidence=0.80
        )
        
        assert result['recommendation'] == 'BET'
        assert result['edge'] > 5.0  # Edge > 5%
        assert result['stake_amount'] > 0
        assert result['stake_pct'] <= 3.0  # Hard cap de 3%
    
    def test_low_edge_skip(self, strategy):
        """Testa skip quando edge é baixo (<5%)."""
        result = strategy.calculate_optimal_stake(
            model_prob=0.52,  # 52%
            market_odds=1.90,
            confidence=0.80
        )
        
        # Edge = 0.52 * 1.90 - 1 = -0.012 = 1.2% < 5%
        assert result['recommendation'] == 'SKIP'
        assert result['stake_amount'] == 0
        assert 'Edge baixo' in result['reason']
    
    def test_low_confidence_skip(self, strategy):
        """Testa skip quando confidence é baixo (<60%)."""
        result = strategy.calculate_optimal_stake(
            model_prob=0.70,
            market_odds=1.60,
            confidence=0.50  # 50% < 60%
        )
        
        assert result['recommendation'] == 'SKIP'
        assert result['stake_amount'] == 0
        assert 'Confidence baixo' in result['reason']
    
    def test_hard_cap_never_exceeded(self, strategy):
        """Testa que hard cap de 3% nunca é excedido."""
        # Cenário extremo: prob muito alta
        result = strategy.calculate_optimal_stake(
            model_prob=0.90,  # 90%
            market_odds=1.50,
            confidence=1.0
        )
        
        if result['recommendation'] == 'BET':
            assert result['stake_pct'] <= 3.0
            assert result['stake_amount'] <= 30.0  # 3% de $1000
    
    def test_kelly_fraction_applied(self, strategy):
        """Testa que fração de Kelly (0.25) é aplicada corretamente."""
        result = strategy.calculate_optimal_stake(
            model_prob=0.60,
            market_odds=1.85,
            confidence=1.0
        )
        
        if result['recommendation'] == 'BET':
            # Kelly fracionado deve ser ~25% do Kelly completo
            assert result['kelly_fraction'] < result['kelly_full']
            ratio = result['kelly_fraction'] / result['kelly_full']
            assert 0.20 <= ratio <= 0.30  # Aproximadamente 0.25
    
    def test_correlation_detection_same_game(self, strategy):
        """Testa detecção de correlação para apostas no mesmo jogo."""
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
        
        # Ambas devem ter stake positivo inicialmente
        assert bets[0]['recommendation'] == 'BET'
        assert bets[1]['recommendation'] == 'BET'
        original_stake_0 = bets[0]['stake_amount']
        original_stake_1 = bets[1]['stake_amount']
        
        # Ajustar para correlação
        adjusted = strategy.adjust_for_correlation(bets)
        
        # Stakes devem ser reduzidos em 50%
        assert adjusted[0]['stake_amount'] == pytest.approx(original_stake_0 * 0.5, rel=0.01)
        assert adjusted[1]['stake_amount'] == pytest.approx(original_stake_1 * 0.5, rel=0.01)
        
        # Deve ter alertas de correlação
        assert adjusted[0]['correlation_alert'] is not None
        assert adjusted[1]['correlation_alert'] is not None
        assert 'LAL_vs_BRK' in adjusted[0]['correlation_alert']
    
    def test_correlation_detection_same_team(self, strategy):
        """Testa detecção de correlação para múltiplas apostas no mesmo time."""
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
                game_id="LAL_vs_PHX",
                team="LAL",
                market="Total Over 220.5"
            )
        ]
        
        original_stake_0 = bets[0]['stake_amount']
        adjusted = strategy.adjust_for_correlation(bets)
        
        # Stakes devem ser reduzidos
        assert adjusted[0]['stake_amount'] < original_stake_0
        assert 'LAL' in adjusted[0]['correlation_alert']
    
    def test_no_correlation_different_games(self, strategy):
        """Testa que apostas em jogos diferentes não são correlacionadas."""
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
                game_id="GSW_vs_PHX",
                team="GSW",
                market="Moneyline"
            )
        ]
        
        original_stake_0 = bets[0]['stake_amount']
        adjusted = strategy.adjust_for_correlation(bets)
        
        # Stakes NÃO devem ser reduzidos (jogos diferentes, times diferentes)
        assert adjusted[0]['stake_amount'] == original_stake_0
        assert adjusted[0]['correlation_alert'] is None
    
    def test_skip_bets_not_adjusted(self, strategy):
        """Testa que apostas SKIP não são afetadas por correlação."""
        bets = [
            strategy.calculate_optimal_stake(
                model_prob=0.51,  # Edge muito baixo
                market_odds=1.90,
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
        
        adjusted = strategy.adjust_for_correlation(bets)
        
        # Primeira aposta é SKIP, não deve ter alerta
        assert adjusted[0]['recommendation'] == 'SKIP'
        assert adjusted[0]['correlation_alert'] is None


class TestFormatStakeRecommendation:
    """Testes para formatação de recomendações."""
    
    def test_format_bet_recommendation(self):
        """Testa formatação de recomendação BET."""
        result = {
            'recommendation': 'BET',
            'team': 'LAL',
            'market': 'Moneyline',
            'market_odds': 1.95,
            'edge': 8.5,
            'kelly_full': 17.2,
            'kelly_fraction': 4.3,
            'stake_amount': 43.00,
            'stake_pct': 4.3,
            'correlation_alert': None
        }
        
        formatted = format_stake_recommendation(result)
        
        assert 'APOSTAR' in formatted
        assert 'LAL' in formatted
        assert '1.95' in formatted
        assert '8.5%' in formatted
    
    def test_format_skip_recommendation(self):
        """Testa formatação de recomendação SKIP."""
        result = {
            'recommendation': 'SKIP',
            'team': 'BRK',
            'market_odds': 1.80,
            'reason': 'Edge baixo: 1.5% < 5.0%'
        }
        
        formatted = format_stake_recommendation(result)
        
        assert 'SKIP' in formatted
        assert 'Edge baixo' in formatted
    
    def test_format_with_correlation_alert(self):
        """Testa formatação com alerta de correlação."""
        result = {
            'recommendation': 'BET',
            'team': 'LAL',
            'market': 'Spread -3.5',
            'market_odds': 1.90,
            'edge': 6.8,
            'kelly_full': 14.0,
            'kelly_fraction': 3.5,
            'stake_amount': 17.50,
            'stake_pct': 1.75,
            'correlation_alert': '⚠️ Stake reduzido 50%: Múltiplas apostas no jogo LAL_vs_BRK'
        }
        
        formatted = format_stake_recommendation(result)
        
        assert 'APOSTAR' in formatted
        assert 'Stake reduzido' in formatted
        assert 'LAL_vs_BRK' in formatted


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
