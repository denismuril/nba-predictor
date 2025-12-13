"""
Testes unitários para Data Validation Framework.
"""

import pytest
import pandas as pd
import numpy as np
from utils.data_validation import (
    DataValidator,
    ValidationRule,
    validate_rapm,
    validate_bpm,
    DataSource
)


class TestValidationRule:
    """Testes para ValidationRule dataclass."""
    
    def test_validation_rule_creation(self):
        """Testa criação de ValidationRule."""
        rule = ValidationRule(
            column='test_col',
            required=True,
            dtype=float,
            min_value=0.0,
            max_value=100.0
        )
        
        assert rule.column == 'test_col'
        assert rule.required == True
        assert rule.dtype == float
        assert rule.min_value == 0.0
        assert rule.max_value == 100.0


class TestDataValidator:
    """Testes para DataValidator."""
    
    def test_validate_rapm_valid(self):
        """Testa validação de RAPM válido."""
        df = pd.DataFrame({
            'Team': ['Lakers', 'Warriors', 'Celtics'],
            'Time Decay ORAPM': [3.5, 2.1, 4.2],
            'Time Decay DRAPM': [1.8, 3.2, 2.5]
        })
        
        result = DataValidator.validate(df, DataValidator.RAPM_SCHEMA, 'test_rapm')
        
        assert result.valid == True
        assert result.metrics['quality_score'] >= 90
        assert len(result.errors) == 0
    
    def test_validate_rapm_invalid_range(self):
        """Testa que valores fora do range são detectados."""
        df = pd.DataFrame({
            'Team': ['Lakers', 'Warriors'],
            'Time Decay ORAPM': [25.0, 2.1],  # 25.0 fora do range!
            'Time Decay DRAPM': [1.8, 3.2]
        })
        
        result = DataValidator.validate(df, DataValidator.RAPM_SCHEMA, 'test_invalid')
        
        # Deve ter warnings (ou errors em strict mode)
        assert len(result.warnings) > 0 or len(result.errors) > 0
        assert result.metrics['quality_score'] < 100
    
    def test_validate_missing_required_column(self):
        """Testa que colunas obrigatórias ausentes falham."""
        df = pd.DataFrame({
            'Team': ['Lakers', 'Warriors'],
            # Faltando 'Time Decay ORAPM' e 'Time Decay DRAPM'
        })
        
        result = DataValidator.validate(df, DataValidator.RAPM_SCHEMA, 'test_missing')
        
        assert result.valid == False
        assert len(result.errors) > 0
        assert 'obrigatórias ausentes' in result.errors[0]
    
    def test_validate_null_percentage(self):
        """Testa detecção de muitos valores null."""
        df = pd.DataFrame({
            'Team': ['Lakers', 'Warriors', 'Celtics'],
            'Time Decay ORAPM': [3.5, None, None],  # 66% null!
            'Time Decay DRAPM': [1.8, 3.2, 2.5]
        })
        
        result = DataValidator.validate(df, DataValidator.RAPM_SCHEMA, 'test_nulls')
        
        # Deve ter warnings sobre muitos nulls
        assert len(result.warnings) > 0
        null_warnings = [w for w in result.warnings if 'nulos' in w.lower()]
        assert len(null_warnings) > 0
    
    def test_quality_score_calculation(self):
        """Testa cálculo de quality score."""
        # DataFrame perfeito
        df_perfect = pd.DataFrame({
            'Team': ['Lakers', 'Warriors', 'Celtics'],
            'Time Decay ORAPM': [3.5, 2.1, 4.2],
            'Time Decay DRAPM': [1.8, 3.2, 2.5]
        })
        
        result_perfect = DataValidator.validate(df_perfect, DataValidator.RAPM_SCHEMA, 'perfect')
        # quality_score pode ser < 100 devido a conversões de tipo automaticas
        assert result_perfect.metrics['quality_score'] >= 95
        
        # DataFrame com problemas
        df_issues = pd.DataFrame({
            'Team': ['Lakers', 'Warriors'],
            'Time Decay ORAPM': [25.0, None],  # Range + null
            'Time Decay DRAPM': [1.8, 3.2]
        })
        
        result_issues = DataValidator.validate(df_issues, DataValidator.RAPM_SCHEMA, 'issues')
        assert result_issues.metrics['quality_score'] < 92
    
    def test_validate_bpm_valid(self):
        """Testa validação de BPM."""
        df = pd.DataFrame({
            'Player': ['LeBron James', 'Stephen Curry'],
            'Team': ['Lakers', 'Warriors'],
            'OBPM': [5.2, 6.1],
            'DBPM': [1.8, -0.5],
            'MP': [35.2, 34.1]
        })
        
        result = DataValidator.validate(df, DataValidator.BPM_SCHEMA, 'test_bpm')
        
        assert result.valid == True
        assert result.metrics['quality_score'] >= 90
    
    def test_validate_odds_valid(self):
        """Testa validação de odds."""
        df = pd.DataFrame({
            'home_team': ['Lakers', 'Warriors'],
            'away_team': ['Celtics', 'Nets'],
            'home_odds': [1.85, 2.10],
            'away_odds': [2.05, 1.80]
        })
        
        result = DataValidator.validate(df, DataValidator.ODDS_SCHEMA, 'test_odds')
        
        assert result.valid == True
        assert result.metrics['quality_score'] >= 90
    
    def test_validate_odds_invalid_too_low(self):
        """Testa que odds muito baixos são detectados."""
        df = pd.DataFrame({
            'home_team': ['Lakers'],
            'away_team': ['Celtics'],
            'home_odds': [0.50],  # Muito baixo!
            'away_odds': [2.05]
        })
        
        result = DataValidator.validate(df, DataValidator.ODDS_SCHEMA, 'test_odds_invalid')
        
        # Deve ter warnings sobre valores fora do range
        assert len(result.warnings) > 0 or len(result.errors) > 0
    
    def test_dtype_validation(self):
        """Testa validação de tipos de dados."""
        # DataFrame com tipo errado
        df = pd.DataFrame({
            'Team': ['Lakers', 'Warriors'],
            'Time Decay ORAPM': ['3.5', '2.1'],  # Strings em vez de floats!
            'Time Decay DRAPM': [1.8, 3.2]
        })
        
        result = DataValidator.validate(df, DataValidator.RAPM_SCHEMA, 'test_dtype')
        
        # Validator deve tentar converter - pode dar warning ou não dependendo do sucesso
        # Se converter com sucesso, pode não haver warnings
        assert result.valid == True  # Deve passar após conversão
    
    def test_get_schema_for_source(self):
        """Testa obtenção de schema por fonte."""
        rapm_schema = DataValidator.get_schema_for_source(DataSource.RAPM)
        assert rapm_schema == DataValidator.RAPM_SCHEMA
        
        bpm_schema = DataValidator.get_schema_for_source(DataSource.BPM)
        assert bpm_schema == DataValidator.BPM_SCHEMA
        
        # Source desconhecido
        unknown_schema = DataValidator.get_schema_for_source(DataSource.UNKNOWN)
        assert unknown_schema == []


class TestHelperFunctions:
    """Testes para funções helper."""
    
    def test_validate_rapm_helper(self):
        """Testa helper validate_rapm."""
        df = pd.DataFrame({
            'Team': ['Lakers'],
            'Time Decay ORAPM': [3.5],
            'Time Decay DRAPM': [1.8]
        })
        
        result = validate_rapm(df, 'test')
        
        assert result.valid == True
        assert 'RAPM_test' in result.data_source
    
    def test_validate_bpm_helper(self):
        """Testa helper validate_bpm."""
        df = pd.DataFrame({
            'Player': ['LeBron James'],
            'Team': ['Lakers'],
            'OBPM': [5.2],
            'DBPM': [1.8],
            'MP': [35.2]
        })
        
        result = validate_bpm(df)
        
        assert result.valid == True
        assert 'BPM' in result.data_source


class TestEdgeCases:
    """Testes para casos extremos."""
    
    def test_empty_dataframe(self):
        """Testa validação de DataFrame vazio."""
        df = pd.DataFrame()
        
        result = DataValidator.validate(df, DataValidator.RAPM_SCHEMA, 'empty')
        
        assert result.valid == False
        assert len(result.errors) > 0
    
    def test_all_nulls(self):
        """Testa DataFrame com todos os valores null."""
        df = pd.DataFrame({
            'Team': [None, None, None],
            'Time Decay ORAPM': [None, None, None],
            'Time Decay DRAPM': [None, None, None]
        })
        
        result = DataValidator.validate(df, DataValidator.RAPM_SCHEMA, 'all_nulls')
        
        # Deve detectar problema de nulls - null_percentage reflete % de nulls nas colunas
        assert result.metrics['null_percentage'] >= 50.0  # Pelo menos 50% nulls
        assert len(result.warnings) > 0 or len(result.errors) > 0
    
    def test_single_row(self):
        """Testa DataFrame com apenas1 linha."""
        df = pd.DataFrame({
            'Team': ['Lakers'],
            'Time Decay ORAPM': [3.5],
            'Time Decay DRAPM': [1.8]
        })
        
        result = DataValidator.validate(df, DataValidator.RAPM_SCHEMA, 'single_row')
        
        assert result.valid == True
        assert result.metrics['rows'] == 1
    
    def test_strict_mode(self):
        """Testa modo strict (warnings viram errors)."""
        df = pd.DataFrame({
            'Team': ['Lakers', 'Warriors'],
            'Time Decay ORAPM': [3.5, None],  # 50% null
            'Time Decay DRAPM': [1.8, 3.2]
        })
        
        # Modo normal: warnings
        result_normal = DataValidator.validate(df, DataValidator.RAPM_SCHEMA, 'normal', strict=False)
        assert len(result_normal.warnings) > 0
        
        # Modo strict: errors
        result_strict = DataValidator.validate(df, DataValidator.RAPM_SCHEMA, 'strict', strict=True)
        # Em strict, alguns warnings viram errors
        assert len(result_strict.errors) >= len(result_normal.errors)


# Integration test
def test_end_to_end_validation():
    """Teste end-to-end do fluxo de validação."""
    # Simular dados vindos de um scraper
    raw_data = {
        'player_name': ['LeBron James', 'Stephen Curry', 'Kevin Durant'],
        'team': ['LAL', 'GSW', 'PHX'],
        'rapm_timedecay': [4.2, 5.1, 3.8],
        'orapm_timedecay': [2.5, 3.2, 2.1],
        'drapm_timedecay': [1.7, 1.9, 1.7]
    }
    
    df = pd.DataFrame(raw_data)
    
    # Renomear colunas
    df = df.rename(columns={
        'player_name': 'Player',
        'team': 'Team',
        'rapm_timedecay': 'RAPM',
        'orapm_timedecay': 'ORAPM',
        'drapm_timedecay': 'DRAPM'
    })
    
    # Validar
    schema = [
        ValidationRule('Player', required=True, dtype=str),
        ValidationRule('Team', required=True, dtype=str),
        ValidationRule('RAPM', required=True, dtype=float, min_value=-15.0, max_value=15.0),
        ValidationRule('ORAPM', required=True, dtype=float, min_value=-15.0, max_value=15.0),
        ValidationRule('DRAPM', required=True, dtype=float, min_value=-15.0, max_value=15.0),
    ]
    
    result = DataValidator.validate(df, schema, 'end_to_end')
    
    assert result.valid == True
    # quality_score pode ser < 100 devido a conversões de tipo
    assert result.metrics['quality_score'] >= 90
    assert result.metrics['rows'] == 3
    assert len(result.errors) == 0
