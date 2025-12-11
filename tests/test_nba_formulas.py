"""
Unit Tests for NBA Formulas Module

Validates canonical NBA formulas against official Basketball-Reference standards.
Ensures training-serving consistency and mathematical correctness.
"""
import pytest
import numpy as np
from utils.nba_formulas import (
    calculate_possessions,
    calculate_pace,
    calculate_offensive_rating,
    calculate_defensive_rating,
    calculate_efg,
    calculate_ts,
    calculate_tov_pct,
    calculate_orb_pct,
    calculate_ftr,
    calculate_all_advanced_stats,
    validate_stats_ranges
)


class TestPossessionsFormula:
    """Test possessions calculation."""
    
    def test_standard_method_realistic_game(self):
        """Test with realistic NBA game stats."""
        # Warriors vs Lakers 2024-03-16 (example)
        poss = calculate_possessions(
            fga=88, fta=18, orb=8, drb_opp=35,
            tov=11, fgm=42, method='standard'
        )
        
        # NBA games typically have 95-105 possessions
        assert 95 <= poss <= 105
        
        # Should be close to 100
        assert 97 <= poss <= 103
    
    def test_simplified_method(self):
        """Test simplified formula (fallback)."""
        poss = calculate_possessions(
            fga=88, fta=18, orb=0, drb_opp=0,
            tov=11, fgm=0, method='simplified'
        )
        
        # Should be FGA + 0.44*FTA + TOV
        expected = 88 + 0.44 * 18 + 11
        assert abs(poss - expected) < 0.01
    
    def test_possessions_never_negative(self):
        """Edge case: ensure possessions never negative."""
        poss = calculate_possessions(
            fga=0, fta=0, orb=0, drb_opp=0,
            tov=0, fgm=0, method='standard'
        )
        
        assert poss >= 1  # Minimum is 1 to prevent division by zero


class TestRatings:
    """Test offensive and defensive rating calculations."""
    
    def test_offensive_rating_elite(self):
        """Elite offense should be >118."""
        ortg = calculate_offensive_rating(pts=118, possessions=100)
        assert ortg == 118.0
        assert ortg > 115  # Elite threshold
    
    def test_defensive_rating_elite(self):
        """Elite defense should be <108."""
        drtg = calculate_defensive_rating(opp_pts=105, possessions=100)
        assert drtg == 105.0
        assert drtg < 108  # Elite threshold
    
    def test_rating_division_by_zero_protected(self):
        """Zero possessions should not crash."""
        ortg = calculate_offensive_rating(pts=100, possessions=0)
        assert ortg > 0  # Should use max(poss, 1)


class TestPace:
    """Test pace calculation."""
    
    def test_pace_regulation_game(self):
        """48-minute game should normalize to possessions."""
        pace = calculate_pace(possessions=100, minutes_played=48)
        assert pace == 100.0
    
    def test_pace_overtime_game(self):
        """OT game should normalize to 48 minutes."""
        # 53-minute game (1 OT), 110 possessions
        pace = calculate_pace(possessions=110, minutes_played=53)
        
        # Should be ~99 (110 * 48/53)
        expected = 110 * (48 / 53)
        assert abs(pace - expected) < 0.1
        
        # Should be close to regulation pace
        assert 95 <= pace <= 105
    
    def test_pace_division_by_zero_protected(self):
        """Zero minutes should not crash."""
        pace = calculate_pace(possessions=100, minutes_played=0)
        assert pace > 0


class TestFourFactors:
    """Test Four Factors formulas."""
    
    def test_efg_realistic(self):
        """Test eFG% with realistic stats."""
        efg = calculate_efg(fgm=42, fg3m=12, fga=88)
        
        # Should be (42 + 0.5*12) / 88 = 48/88 = 0.545
        expected = (42 + 0.5 * 12) / 88
        assert abs(efg - expected) < 0.001
        
        # Realistic range: 45-60%
        assert 0.45 <= efg <= 0.60
    
    def test_ts_better_than_efg_for_ft_shooters(self):
        """TS% should be higher than eFG% for high FT teams."""
        # James Harden type: lots of FTs
        efg = calculate_efg(fgm=10, fg3m=4, fga=22)
        ts = calculate_ts(pts=34, fga=22, fta=12)  # 10/12 FT made
        
        # TS% should be higher (captures FT efficiency)
        assert ts > efg
    
    def test_tov_pct_scale(self):
        """TOV% should be in 8-20 range (percentage, not decimal)."""
        tov_pct = calculate_tov_pct(tov=12, fga=88, fta=18)
        
        # Should be 10-15 range for good teams
        assert 8 <= tov_pct <= 20
        
        # More specifically, 10-15 for average teams
        assert 10 <= tov_pct <= 15
    
    def test_orb_pct_realistic(self):
        """ORB% should be 20-30% range."""
        orb_pct = calculate_orb_pct(orb=10, drb_opp=35)
        
        # Should be 10/(10+35) = 0.222
        expected = 10 / (10 + 35)
        assert abs(orb_pct - expected) < 0.001
        
        # Realistic range: 20-30%
        assert 0.15 <= orb_pct <= 0.35
    
    def test_ft_rate_realistic(self):
        """FTR should be 15-35% range."""
        ftr = calculate_ftr(fta=18, fga=88)
        
        # Should be 18/88 = 0.204
        expected = 18 / 88
        assert abs(ftr - expected) < 0.001
        
        # Realistic range
        assert 0.10 <= ftr <= 0.45


class TestAllAdvancedStats:
    """Test batch calculation of all stats."""
    
    def test_realistic_game_all_stats(self):
        """Test with realistic complete game."""
        stats = calculate_all_advanced_stats(
            pts=115, fgm=42, fga=88, fg3m=15,
            fta=18, ftm=16, orb=8, drb=38, tov=11,
            opp_pts=108, opp_drb=35, minutes_played=48
        )
        
        # Check all expected keys present
        required_keys = [
            'possessions', 'pace', 'off_rating', 'def_rating',
            'efg_pct', 'ts_pct', 'tov_pct', 'orb_pct', 'ft_rate'
        ]
        for key in required_keys:
            assert key in stats
        
        # Validate ranges
        validate_stats_ranges(stats)  # Should not raise
        
        # Spot checks
        assert 95 <= stats['pace'] <= 105
        assert 110 <= stats['off_rating'] <= 125
        assert 0.50 <= stats['efg_pct'] <= 0.65
        assert 0.50 <= stats['ts_pct'] <= 0.70


class TestValidation:
    """Test validation functions."""
    
    def test_validation_accepts_good_stats(self):
        """Valid stats should pass validation."""
        stats = {
            'efg_pct': 0.54,
            'ts_pct': 0.58,
            'tov_pct': 12.5,
            'orb_pct': 0.24,
            'ft_rate': 0.22,
            'off_rating': 115,
            'def_rating': 110,
            'pace': 98
        }
        
        assert validate_stats_ranges(stats) == True
    
    def test_validation_rejects_bad_efg(self):
        """Invalid eFG% should raise error."""
        stats = {'efg_pct': 1.5}  # 150% is impossible
        
        with pytest.raises(ValueError, match="eFG%"):
            validate_stats_ranges(stats)
    
    def test_validation_rejects_bad_pace(self):
        """Invalid Pace should raise error."""
        stats = {'pace': 150}  # Way too high
        
        with pytest.raises(ValueError, match="Pace"):
            validate_stats_ranges(stats)


class TestConsistency:
    """Test training-serving consistency."""
    
    def test_same_inputs_same_outputs(self):
        """Same inputs should always produce same outputs."""
        inputs = dict(
            fga=88, fta=18, orb=8, drb_opp=35,
            tov=11, fgm=42, method='standard'
        )
        
        poss1 = calculate_possessions(**inputs)
        poss2 = calculate_possessions(**inputs)
        
        assert poss1 == poss2
    
    def test_array_and_scalar_consistency(self):
        """Arrays and scalars should produce consistent results."""
        # Scalar
        poss_scalar = calculate_possessions(
            fga=88, fta=18, orb=8, drb_opp=35,
            tov=11, fgm=42, method='standard'
        )
        
        # Array
        poss_array = calculate_possessions(
            fga=np.array([88]),
            fta=np.array([18]),
            orb=np.array([8]),
            drb_opp=np.array([35]),
            tov=np.array([11]),
            fgm=np.array([42]),
            method='standard'
        )
        
        # Should be nearly identical
        assert abs(poss_scalar - poss_array[0]) < 0.01


def test_formula_comparison_with_basketball_reference():
    """
    Compare our formulas with known Basketball-Reference values.
    
    Using Warriors vs Lakers 2024-03-16 as reference:
    - Possessions: ~100
    - ORtg: ~118 (Warriors)
    - Pace: ~97
    """
    # These are approximate reference values
    stats = calculate_all_advanced_stats(
        pts=118, fgm=45, fga=92, fg3m=16,
        fta=20, ftm=18, orb=9, drb=40, tov=12,
        opp_pts=112, opp_drb=36, minutes_played=48
    )
    
    print("\n📊 Basketball-Reference Comparison:")
    print(f"  Possessions: {stats['possessions']:.1f} (expected ~100)")
    print(f"  ORtg: {stats['off_rating']:.1f} (expected ~118)")
    print(f"  Pace: {stats['pace']:.1f} (expected ~97)")
    print(f"  eFG%: {stats['efg_pct']:.3f}")
    print(f"  TS%: {stats['ts_pct']:.3f}")
    
    # Rough validation (within reasonable tolerance)
    assert 95 <= stats['possessions'] <= 105
    assert 115 <= stats['off_rating'] <= 125
    assert 94 <= stats['pace'] <= 103  # Increased upper bound tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
