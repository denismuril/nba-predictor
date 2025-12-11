"""
Testes para Data Leakage Validation.

Valida que o framework detecta corretamente:
1. Rolling features com leakage (sem shift)
2. Rolling features corretas (com shift)
3. Primeiro valor NaN vs não-NaN
4. Temporal ordering
5. Edge cases
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.data_leakage_validator import (
    DataLeakageValidator,
    validate_no_leakage
)


class TestDataLeakageValidator(unittest.TestCase):
    """Testes para validação de data leakage."""
    
    def setUp(self):
        """Setup executado antes de cada teste."""
        self.validator = DataLeakageValidator(strict_mode=True)
        
        # Criar dataset de exemplo
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        teams = ['LAL', 'GSW', 'BOS', 'MIA', 'DEN']
        
        data = []
        for team in teams:
            for i, date in enumerate(dates[:10]):  # 10 jogos por time
                data.append({
                    'date': date,
                    'team': team,
                    'points': np.random.randint(90, 120),
                    'opp_points': np.random.randint(90, 120)
                })
        
        self.df = pd.DataFrame(data)
        self.df = self.df.sort_values(['team', 'date']).reset_index(drop=True)
    
    def test_correct_rolling_feature(self):
        """Rolling feature CORRETA (com shift) deve passar."""
        # Aplicar shift(1) ANTES do rolling
        self.df['rolling_5_points_correct'] = self.df.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        
        # Validar
        result = self.validator.validate_dataframe(self.df, team_col='team')
       
        
        # Deve passar
        self.assertTrue(result['valid'], "Rolling correta foi detectada como leakage!")
        self.assertEqual(len(result['errors']), 0)
        self.assertNotIn('rolling_5_points_correct', result['leakage_detected'])
    
    def test_leaky_rolling_feature(self):
        """Rolling feature COM LEAKAGE (sem shift) deve FALHAR."""
        # Aplicar rolling SEM shift(1) - INCORRETO!
        self.df['rolling_5_points_leaky'] = self.df.groupby('team')['points'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # Validar
        result = self.validator.validate_dataframe(self.df, team_col='team')
        
        # Deve falhar
        self.assertFalse(result['valid'], "Leakage NÃO foi detectado!")
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('rolling_5_points_leaky', result['leakage_detected'])
    
    def test_first_value_must_be_nan(self):
        """Primeiro valor de rolling feature DEVE ser NaN."""
        # Feature correta (com shift)
        self.df['rolling_correct'] = self.df.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        
        # Verificar que primeiro valor é NaN para cada team
        first_values = self.df.groupby('team')['rolling_correct'].first()
        
        # Todos devem ser NaN
        self.assertTrue(first_values.isna().all(), "Primeiro valor deveria ser NaN!")
    
    def test_multiple_features(self):
        """Validar múltiplas features de uma vez."""
        # Criar várias features
        self.df['rolling_5_correct'] = self.df.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        self.df['rolling_10_correct'] = self.df.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(10).mean()
        )
        self.df['rolling_leaky'] = self.df.groupby('team')['points'].transform(
            lambda x: x.rolling(5).mean()  # SEM SHIFT!
        )
        
        # Validar
        result = self.validator.validate_dataframe(self.df, team_col='team')
        
        # Deve detectar apenas a leaky
        self.assertEqual(len(result['leakage_detected']), 1)
        self.assertIn('rolling_leaky', result['leakage_detected'])
        self.assertNotIn('rolling_5_correct', result['leakage_detected'])
        self.assertNotIn('rolling_10_correct', result['leakage_detected'])
    
    def test_temporal_ordering(self):
        """DataFrame deve estar ordenado temporalmente."""
        # Criar DataFrame desordenado
        df_shuffled = self.df.sample(frac=1).reset_index(drop=True)
        
        # Adicionar rolling feature
        df_shuffled['rolling_5'] = df_shuffled.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        
        # Validar
        result = self.validator.validate_dataframe(df_shuffled, team_col='team')
        
        # Deve ter warning sobre ordenação
        self.assertGreater(len(result['warnings']), 0)
        warning_text = " ".join(result['warnings'])
        self.assertIn('ordenado', warning_text.lower())
    
    def test_convenience_function(self):
        """Testar função convenience validate_no_leakage()."""
        # Feature correta
        self.df['rolling_correct'] = self.df.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        
        # Deve retornar True
        is_valid = validate_no_leakage(self.df, team_col='team', strict=False)
        self.assertTrue(is_valid)
        
        # Feature com leakage
        self.df['rolling_leaky'] = self.df.groupby('team')['points'].transform(
            lambda x: x.rolling(5).mean()
        )
        
        # Deve retornar False
        is_valid = validate_no_leakage(self.df, team_col='team', strict=False)
        self.assertFalse(is_valid)
    
    def test_strict_mode_raises_on_leakage(self):
        """Strict mode deve raise ValueError em leakage."""
        # Feature com leakage
        self.df['rolling_leaky'] = self.df.groupby('team')['points'].transform(
            lambda x: x.rolling(5).mean()
        )
        
        # Strict mode deve raise
        with self.assertRaises(ValueError):
            validate_no_leakage(self.df, team_col='team', strict=True)
    
    def test_lag_features(self):
        """Validar lag features (deslocamento temporal)."""
        # Lag correto (usando shift)
        self.df['lag_1_correct'] = self.df.groupby('team')['points'].shift(1)
        
        # Primeiro valor DEVE ser NaN
        first_values = self.df.groupby('team')['lag_1_correct'].first()
        self.assertTrue(first_values.isna().all())
        
        # Validar
        result = self.validator.validate_dataframe(self.df, team_col='team')
        self.assertTrue(result['valid'])
    
    def test_edge_case_single_game_per_team(self):
        """Edge case: apenas 1 jogo por time."""
        df_single = self.df.groupby('team').head(1).copy()
        
        # Rolling feature
        df_single['rolling_5'] = df_single.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        
        # Todos devem ser NaN (apenas 1 jogo)
        self.assertTrue(df_single['rolling_5'].isna().all())
        
        # Validar
        result = self.validator.validate_dataframe(df_single, team_col='team')
        self.assertTrue(result['valid'])
    
    def test_mixed_teams_some_with_leakage(self):
        """Alguns teams com leakage, outros sem."""
        # Criar feature que vaza apenas para alguns teams
        def create_mixed_feature(group):
            team = group.name
            if team in ['LAL', 'GSW']:
                # COM LEAKAGE para estes times
                return group['points'].rolling(5).mean()
            else:
                # SEM LEAKAGE para outros
                return group['points'].shift(1).rolling(5).mean()
        
        self.df['rolling_mixed'] = self.df.groupby('team').apply(create_mixed_feature).reset_index(level=0, drop=True)
        
        # Validar
        result = self.validator.validate_dataframe(self.df, team_col='team')
        
        # Deve detectar leakage parcial
        self.assertFalse(result['valid'])
        self.assertIn('rolling_mixed', result['leakage_detected'])
    
    def test_validation_report(self):
        """Testar geração de relatório."""
        # Adicionar features
        self.df['rolling_correct'] = self.df.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(5).mean()
        )
        self.df['rolling_leaky'] = self.df.groupby('team')['points'].transform(
            lambda x: x.rolling(5).mean()
        )
        
        # Gerar relatório
        report = self.validator.create_validation_report(self.df)
        
        # Deve conter informações chave
        self.assertIn('VALIDATION REPORT', report)
        self.assertIn('rolling_leaky', report)
        self.assertIn('FAILED', report)


class TestRealWorldScenarios(unittest.TestCase):
    """Testes com cenários do mundo real."""
    
    def test_mlpipeline_data_preparation_style(self):
        """Simular exatamente o estilo usado em ml_pipeline/data_preparation.py."""
        # Setup similar ao data_preparation
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        home_teams = ['Lakers', 'Warriors'] * 50
        away_teams = ['Celtics', 'Heat'] * 50
        
        df = pd.DataFrame({
            'date': dates,
            'home_team': home_teams,
            'away_team': away_teams,
            'home_score': np.random.randint(90, 120, 100),
            'away_score': np.random.randint(90, 120, 100)
        })
        
        # Criar long format (como em data_preparation.py)
        home_df = df[['date', 'home_team', 'home_score']].copy()
        home_df.columns = ['date', 'team', 'points']
        
        away_df = df[['date', 'away_team', 'away_score']].copy()
        away_df.columns = ['date', 'team', 'points']
        
        long_df = pd.concat([home_df, away_df]).sort_values(['team', 'date'])
        
        # Aplicar rolling CORRETO (com shift)
        long_df['rolling_5_points'] = long_df.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        
        # Validar
        validator = DataLeakageValidator()
        result = validator.validate_dataframe(long_df, team_col='team')
        
        # Deve passar
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['leakage_detected']), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
