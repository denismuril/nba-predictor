"""
Unit Tests for Kelly Criterion Concurrent Betting

Validates the concurrent Kelly formula implementation to ensure:
1. Correct variance adjustment with sqrt(n)
2. Max exposure limits enforced
3. Edge cases handled properly
4. No negative stakes suggested
"""
import pytest
import numpy as np
from ml_pipeline.betting_engine import BettingEngine


class TestKellyConcurrent:
    """Test suite for concurrent Kelly Criterion implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = BettingEngine(
            bankroll=1000,
            kelly_fraction=0.25,
            min_ev=0.02,
            max_total_exposure=0.15
        )
    
    def test_single_game_no_adjustment(self):
        """Single game should have no concurrent adjustment."""
        stake_single = self.engine.calculate_kelly_stake_concurrent(
            prob_win=0.55,
            decimal_odds=2.0,
            n_concurrent_bets=1
        )
        
        # Should be positive
        assert stake_single > 0
        # Should be reasonable (Kelly with 55% at 2.0 odds ≈ 10% * 0.25 = 2.5%)
        assert 0.01 < stake_single < 0.05
    
    def test_concurrent_adjustment_reduces_stake(self):
        """Multiple concurrent games should reduce individual stakes."""
        stake_1_game = self.engine.calculate_kelly_stake_concurrent(
            prob_win=0.55, decimal_odds=2.0, n_concurrent_bets=1
        )
        
        stake_10_games = self.engine.calculate_kelly_stake_concurrent(
            prob_win=0.55, decimal_odds=2.0, n_concurrent_bets=10
        )
        
        # 10 concurrent games should have much smaller stake
        assert stake_10_games < stake_1_game
        
        # Should be approximately sqrt(10) ≈ 3.16x smaller
        ratio = stake_1_game / stake_10_games
        assert 2.5 < ratio < 4.0
    
    def test_max_total_exposure_enforced(self):
        """Total exposure should never exceed max_total_exposure."""
        n_games = 20
        stake = self.engine.calculate_kelly_stake_concurrent(
            prob_win=0.90,  # High edge (unrealistic but tests limit)
            decimal_odds=3.0,
            n_concurrent_bets=n_games
        )
        
        total_exposure = stake * n_games
        
        # Total exposure should be <= 15%
        assert total_exposure <= 0.15
        
        # Each stake should be <= 15% / n_games
        assert stake <= 0.15 / n_games
    
    def test_individual_stake_cap(self):
        """No single stake should exceed 5% regardless of edge."""
        stake = self.engine.calculate_kelly_stake_concurrent(
            prob_win=0.99,  # Unrealistic but tests cap
            decimal_odds=10.0,
            n_concurrent_bets=1
        )
        
        assert stake <= 0.05
    
    def test_negative_ev_returns_zero(self):
        """Negative EV should return 0 stake."""
        stake = self.engine.calculate_kelly_stake_concurrent(
            prob_win=0.40,  # Losing edge
            decimal_odds=2.0,
            n_concurrent_bets=1
        )
        
        assert stake == 0.0
    
    def test_invalid_odds_returns_zero(self):
        """Odds <= 1 should return 0."""
        stake = self.engine.calculate_kelly_stake_concurrent(
            prob_win=0.55,
            decimal_odds=1.0,  # Invalid
            n_concurrent_bets=1
        )
        
        assert stake == 0.0
    
    def test_prob_clamping(self):
        """Probabilities outside [0,1] should be clamped."""
        # Should clamp to 0.99
        stake_high = self.engine.calculate_kelly_stake_concurrent(
            prob_win=1.5,  # Invalid
            decimal_odds=2.0,
            n_concurrent_bets=1
        )
        
        # Should clamp to 0.01
        stake_low = self.engine.calculate_kelly_stake_concurrent(
            prob_win=-0.1,  # Invalid
            decimal_odds=2.0,
            n_concurrent_bets=1
        )
        
        # Should not crash and return valid stakes
        assert stake_high >= 0
        assert stake_low == 0  # Should have negative EV after clamping
    
    def test_zero_concurrent_games_safe(self):
        """n_concurrent_bets=0 should be handled safely (treat as 1)."""
        stake = self.engine.calculate_kelly_stake_concurrent(
            prob_win=0.55,
            decimal_odds=2.0,
            n_concurrent_bets=0  # Edge case
        )
        
        # Should not crash and return positive stake
        assert stake >= 0
    
    def test_realistic_nba_scenario(self):
        """Test realistic NBA betting scenario."""
        # 10-game night, 55% confidence, odds 2.0 (even money)
        n_games = 10
        stake = self.engine.calculate_kelly_stake_concurrent(
            prob_win=0.55,
            decimal_odds=2.0,
            n_concurrent_bets=n_games
        )
        
        total_exposure = stake * n_games
        
        # Should have positive stake
        assert stake > 0
        
        # Total exposure should be reasonable (<15%)
        assert total_exposure < 0.15
        
        # Each bet should be small (<2%)
        assert stake < 0.02
        
        print(f"\nRealistic NBA Scenario:")
        print(f"  Individual stake: {stake*100:.2f}%")
        print(f"  Total exposure (10 games): {total_exposure*100:.2f}%")


class TestBettingEngineIntegration:
    """Integration tests for BettingEngine."""
    
    def test_analyze_bet_includes_concurrent_info(self):
        """analyze_bet should include concurrent game info."""
        engine = BettingEngine(bankroll=1000)
        
        game_info = {'home_team': 'LAL', 'away_team': 'GSW'}
        
        recommendation = engine.analyze_bet(
            game_info=game_info,
            model_prob=0.60,
            market_odds=2.0,
            bet_type='Moneyline',
            n_concurrent_bets=5
        )
        
        # Should include concurrent_games
        assert 'concurrent_games' in recommendation
        assert recommendation['concurrent_games'] == 5
        
        # Should include total_exposure_pct
        assert 'total_exposure_pct' in recommendation
        assert recommendation['total_exposure_pct'] > 0


def test_kelly_comparison_old_vs_new():
    """Compare old Kelly (no adjustment) vs new (concurrent)."""
    engine = BettingEngine(bankroll=1000, kelly_fraction=0.25)
    
    prob, odds = 0.55, 2.0
    
    # Old Kelly (legacy method, n=1)
    old_kelly = engine.calculate_kelly_stake(prob, odds)
    
    # New Kelly with 1 game (should be similar)
    new_kelly_1 = engine.calculate_kelly_stake_concurrent(prob, odds, n_concurrent_bets=1)
    
    # New Kelly with 10 games
    new_kelly_10 = engine.calculate_kelly_stake_concurrent(prob, odds, n_concurrent_bets=10)
    
    print(f"\nKelly Comparison:")
    print(f"  Old Kelly (legacy): {old_kelly*100:.2f}%")
    print(f"  New Kelly (1 game): {new_kelly_1*100:.2f}%")
    print(f"  New Kelly (10 games): {new_kelly_10*100:.2f}%")
    
    # Should be similar for single game
    assert abs(old_kelly - new_kelly_1) < 0.01
    
    # Should be much smaller for 10 games
    assert new_kelly_10 < old_kelly / 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
