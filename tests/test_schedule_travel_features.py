"""
Unit Tests para Schedule Density & Travel Fatigue Features

Testa funcionalidade e edge cases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from ml_pipeline.advanced_features import add_schedule_density, add_travel_fatigue


class TestScheduleDensity:
    """Tests para schedule density feature."""
    
    def test_basic_calculation(self):
        """Test cálculo básico."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'home_team': ['LAL'] * 10,
            'away_team': ['GSW'] * 10,
            'home_fga': [85] * 10,
            'away_fga': [85] * 10,
        })
        
        result = add_schedule_density(df)
        
        assert 'schedule_density_home' in result.columns
        assert 'schedule_density_away' in result.columns
        assert 'schedule_density_gap' in result.columns
        
    def test_back_to_back_detection(self):
        """Test detecção de back-to-back."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-04']),
            'home_team': ['LAL', 'LAL', 'LAL'],
            'away_team': ['GSW', 'BOS', 'MIA'],
        })
        
        result = add_schedule_density(df)
        
        # Segundo jogo deve ser back-to-back
        assert result['back_to_back_home'].iloc[1] == 1
        
    def test_rest_days_calculation(self):
        """Test cálculo de rest days."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-04', '2024-01-06']),
            'home_team': ['LAL', 'LAL', 'LAL'],
            'away_team': ['GSW', 'BOS', 'MIA'],
        })
        
        result = add_schedule_density(df)
        
        # 3 dias entre jogos = 2 rest days
        assert result['rest_days_home'].iloc[1] == 2
        
    def test_no_date_column(self):
        """Test fallback quando sem data."""
        df = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['GSW'],
        })
        
        result = add_schedule_density(df)
        
        # Deve retornar valores neutros
        assert result['schedule_density_home'].iloc[0] == 0


class TestTravelFatigue:
    """Tests para travel fatigue feature."""
    
    def test_basic_calculation(self):
        """Test cálculo básico."""
        df = pd.DataFrame({
            'home_team': ['LAL', 'BOS'],
            'away_team': ['GSW', 'MIA'],
        })
        
        result = add_travel_fatigue(df)
        
        assert 'travel_fatigue_home' in result.columns
        assert 'travel_fatigue_away' in result.columns
        assert 'travel_fatigue_net' in result.columns
        
    def test_home_no_travel(self):
        """Test que home team não tem fadiga."""
        df = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['GSW'],
        })
        
        result = add_travel_fatigue(df)
        
        # Home sempre 0
        assert result['travel_fatigue_home'].iloc[0] == 0.0
        
    def test_cross_country_fatigue(self):
        """Test fadiga em viagens longas."""
        df = pd.DataFrame({
            'home_team': ['BOS'],  # Boston (East)
            'away_team': ['LAL'],  # LA (West) - ~4200km
        })
        
        result = add_travel_fatigue(df)
        
        # Deve ter fadiga significativa
        assert result['travel_fatigue_away'].iloc[0] < -0.1
        
    def test_local_game(self):
        """Test jogo local (mesma região)."""
        df = pd.DataFrame({
            'home_team': ['LAL'],
            'away_team': ['LAC'],  # Mesma cidade
        })
        
        result = add_travel_fatigue(df)
        
        # Fadiga muito baixa ou zero
        assert result['travel_fatigue_away'].iloc[0] >= -0.05


def test_integration():
    """Test integração schedule + travel."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'home_team': ['LAL', 'BOS', 'LAL', 'MIA', 'GSW'],
        'away_team': ['BOS', 'LAL', 'MIA', 'BOS', 'LAL'],
        'home_fga': [85] * 5,
        'away_fga': [85] * 5,
    })
    
    # Apply both
    df = add_schedule_density(df)
    df = add_travel_fatigue(df)
    
    # Verify all columns exist
    required_cols = [
        'schedule_density_home', 'schedule_density_away',
        'travel_fatigue_home', 'travel_fatigue_away'
    ]
    
    for col in required_cols:
        assert col in df.columns
    
    # Verify no NaNs
    assert df[required_cols].isna().sum().sum() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
