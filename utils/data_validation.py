"""
Data Validation Framework para garantir qualidade de dados no pipeline.

Valida dados de todas as fontes (scrapers) antes de processar:
- RAPM, BPM, PIE, LEBRON
- Injury reports
- Odds
- Schedule data

Implementa:
- Schema validation
- Type checking
- Range validation
- Null percentage tracking
- Quality scoring (0-100)
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Fontes de dados conhecidas."""
    RAPM = "rapm"
    BPM = "bpm"
    PIE = "pie"
    LEBRON = "lebron"
    INJURY = "injury"
    ODDS = "odds"
    SCHEDULE = "schedule"
    GAME_STATS = "game_stats"
    UNKNOWN = "unknown"


@dataclass
class ValidationRule:
    """
    Regra de validação para uma coluna.
    
    Attributes:
        column: Nome da coluna
        required: Se a coluna é obrigatória
        dtype: Tipo de dado esperado (int, float, str, etc)
        min_value: Valor mínimo permitido (para numéricos)
        max_value: Valor máximo permitido (para numéricos)
        allowed_values: Lista de valores permitidos (para categóricos)
        not_null_pct: Percentual mínimo de valores não-null (0.0 a 1.0)
        pattern: Regex pattern para strings (opcional)
    """
    column: str
    required: bool = True
    dtype: Optional[type] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    not_null_pct: float = 0.95  # 95% dos valores devem ser não-null por padrão
    pattern: Optional[str] = None


@dataclass
class ValidationResult:
    """Resultado de validação de um DataFrame."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    data_source: str
    timestamp: str
    
    def __str__(self):
        """String representation para logging."""
        status = "✅ VÁLIDO" if self.valid else "❌ INVÁLIDO"
        return (
            f"{status} - {self.data_source}\n"
            f"  Quality Score: {self.metrics.get('quality_score', 0)}/100\n"
            f"  Errors: {len(self.errors)}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f"  Rows: {self.metrics.get('rows', 0)}"
        )


class DataValidator:
    """
    Validador universal de dados com schemas predefinidos.
    
    Usage:
        validator = DataValidator()
        result = validator.validate(df, DataValidator.RAPM_SCHEMA, 'rapm_external')
        
        if not result.valid:
            logger.error(f"Validation failed: {result.errors}")
    """
    
    # ===== SCHEMAS PREDEFINIDOS =====
    
    RAPM_SCHEMA = [
        ValidationRule('Team', required=True, dtype=str),
        ValidationRule('Time Decay ORAPM', required=True, dtype=float, 
                      min_value=-15.0, max_value=15.0, not_null_pct=0.9),
        ValidationRule('Time Decay DRAPM', required=True, dtype=float,
                      min_value=-15.0, max_value=15.0, not_null_pct=0.9),
    ]
    
    BPM_SCHEMA = [
        ValidationRule('Player', required=True, dtype=str, not_null_pct=1.0),
        ValidationRule('Team', required=True, dtype=str, not_null_pct=1.0),
        ValidationRule('OBPM', required=True, dtype=float, 
                      min_value=-15.0, max_value=15.0),
        ValidationRule('DBPM', required=True, dtype=float,
                      min_value=-10.0, max_value=10.0),
        ValidationRule('MP', required=True, dtype=float, min_value=0),
    ]
    
    PIE_SCHEMA = [
        ValidationRule('Player', required=True, dtype=str),
        ValidationRule('Team', required=True, dtype=str),
        ValidationRule('PIE', required=True, dtype=float,
                      min_value=0.0, max_value=100.0),  # PIE é percentual
    ]
    
    LEBRON_SCHEMA = [
        ValidationRule('Player', required=True, dtype=str),
        ValidationRule('Team', required=True, dtype=str),
        ValidationRule('LEBRON', required=True, dtype=float,
                      min_value=-10.0, max_value=10.0),
    ]
    
    INJURY_SCHEMA = [
        ValidationRule('Player', required=True, dtype=str),
        ValidationRule('Team', required=True, dtype=str),
        ValidationRule('Status', required=True, dtype=str,
                      allowed_values=['OUT', 'QUESTIONABLE', 'DOUBTFUL', 'PROBABLE', 'AVAILABLE']),
    ]
    
    ODDS_SCHEMA = [
        ValidationRule('home_team', required=True, dtype=str),
        ValidationRule('away_team', required=True, dtype=str),
        ValidationRule('home_odds', required=True, dtype=float,
                      min_value=1.01, max_value=50.0),
        ValidationRule('away_odds', required=True, dtype=float,
                      min_value=1.01, max_value=50.0),
    ]
    
    # ===== MÉTODOS DE VALIDAÇÃO =====
    
    @staticmethod
    def validate(
        df: pd.DataFrame, 
        schema: List[ValidationRule],
        data_source: str = "unknown",
        strict: bool = False
    ) -> ValidationResult:
        """
        Valida DataFrame contra schema.
        
        Args:
            df: DataFrame a validar
            schema: Lista de ValidationRules
            data_source: Nome da fonte de dados (para logging)
            strict: Se True, warnings se tornam errors
            
        Returns:
            ValidationResult com status, erros, warnings e métricas
        """
        errors = []
        warnings: List[str] = []
        
        # === 1. Validar colunas obrigatórias ===
        missing_cols = [
            rule.column for rule in schema 
            if rule.required and rule.column not in df.columns
        ]
        
        if missing_cols:
            errors.append(f"Colunas obrigatórias ausentes: {missing_cols}")
            # Se colunas críticas estão faltando, não faz sentido continuar
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                metrics={'quality_score': 0, 'rows': len(df), 'columns': len(df.columns)},
                data_source=data_source,
                timestamp=datetime.now().isoformat()
           )
        
        # === 2. Validar cada coluna presente ===
        for rule in schema:
            if rule.column not in df.columns:
                if rule.required:
                    # Já foi capturado acima
                    pass
                else:
                    warnings.append(f"Coluna opcional '{rule.column}' ausente")
                continue
            
            col = df[rule.column]
            
            # 2.1 Tipo de dado
            if rule.dtype:
                if not DataValidator._check_dtype(col, rule.dtype):
                    # Tentar conversão
                    try:
                        df[rule.column] = col.astype(rule.dtype)
                        col = df[rule.column]  # FIX: Atualizar referência para coluna convertida
                        warnings.append(f"'{rule.column}' convertido para {rule.dtype.__name__}")
                    except (ValueError, TypeError) as e:
                        errors.append(
                            f"'{rule.column}': tipo esperado {rule.dtype.__name__}, "
                            f"mas conversão falhou: {e}"
                       )
            
            # 2.2 Null values
            null_pct = col.isnull().sum() / len(col)
            if null_pct > (1 - rule.not_null_pct):
                msg = (
                    f"'{rule.column}': {null_pct*100:.1f}% nulos "
                    f"(máximo: {(1-rule.not_null_pct)*100:.1f}%)"
                )
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)
            
            # 2.3 Range validation (numéricos)
            if rule.dtype in [float, int, np.float64, np.int64]:
                valid_values = col[col.notna()]  # Ignorar NaNs
                
                if rule.min_value is not None:
                    invalid = valid_values < rule.min_value
                    if invalid.any():
                        count = invalid.sum()
                        msg = f"'{rule.column}': {count} valores < {rule.min_value}"
                        if strict:
                            errors.append(msg)
                        else:
                            warnings.append(msg)
                
                if rule.max_value is not None:
                    invalid = valid_values > rule.max_value
                    if invalid.any():
                        count = invalid.sum()
                        msg = f"'{rule.column}': {count} valores > {rule.max_value}"
                        if strict:
                            errors.append(msg)
                        else:
                            warnings.append(msg)
            
            # 2.4 Allowed values (categóricos)
            if rule.allowed_values:
                valid_values = col[col.notna()]
                invalid_mask = ~valid_values.isin(rule.allowed_values)
                if invalid_mask.any():
                    invalid_vals = valid_values[invalid_mask].unique()[:5]
                    msg = f"'{rule.column}': valores inválidos - {list(invalid_vals)}"
                    if strict:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
        
        # === 3. Quality Score ===
        quality_score = 100.0
        quality_score -= len(errors) * 15  # -15 pts por erro
        quality_score -= len(warnings) * 3  # -3 pts por warning
        quality_score = max(0, min(100, quality_score))  # Clamp 0-100
        
        # === 4. Métricas adicionais ===
        metrics = {
            'quality_score': round(quality_score, 2),
            'null_percentage': round(
                df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2
            ),
            'rows': len(df),
            'columns': len(df.columns),
            'validated_columns': len(schema),
            'errors_count': len(errors),
            'warnings_count': len(warnings)
        }
        
        # === 5. Resultado final ===
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            data_source=data_source,
            timestamp=datetime.now().isoformat()
        )
        
        # Log resultado
        if is_valid:
            logger.info(f"✅ Validation PASSED for {data_source}: Quality={quality_score:.1f}/100")
        else:
            logger.error(f"❌ Validation FAILED for {data_source}:")
            for error in errors:
                logger.error(f"   - {error}")
        
        if warnings:
            logger.warning(f"⚠️  Validation warnings for {data_source}:")
            for warning in warnings[:3]:  # Log apenas primeiros 3
                logger.warning(f"   - {warning}")
            if len(warnings) > 3:
                logger.warning(f"   ... e mais {len(warnings)-3} warnings")
        
        return result
    
    @staticmethod
    def _check_dtype(series: pd.Series, expected_dtype: type) -> bool:
        """Verifica se dtype da Series é compatível com esperado."""
        actual_dtype = series.dtype
        
        # Mapeamento de tipos numpy para Python
        dtype_mapping = {
            np.int64: int,
            np.float64: float,
            np.object_: str,
            object: str
        }
        
        actual = dtype_mapping.get(actual_dtype, actual_dtype)
        expected = dtype_mapping.get(expected_dtype, expected_dtype)
        
        return actual == expected or pd.api.types.is_dtype_equal(actual_dtype, expected_dtype)
    
    @staticmethod
    def get_schema_for_source(source: DataSource) -> List[ValidationRule]:
        """
        Retorna schema apropriado para fonte de dados.
        
        Args:
            source: Enum DataSource
            
        Returns:
            Lista de ValidationRules
        """
        schema_map = {
            DataSource.RAPM: DataValidator.RAPM_SCHEMA,
            DataSource.BPM: DataValidator.BPM_SCHEMA,
            DataSource.PIE: DataValidator.PIE_SCHEMA,
            DataSource.LEBRON: DataValidator.LEBRON_SCHEMA,
            DataSource.INJURY: DataValidator.INJURY_SCHEMA,
            DataSource.ODDS: DataValidator.ODDS_SCHEMA,
        }
        
        schema = schema_map.get(source)
        if schema is None:
            logger.warning(f"No predefined schema for {source}. Using minimal validation.")
            return []
        
        return schema


# ===== HELPER FUNCTIONS =====

def validate_rapm(df: pd.DataFrame, source_name: str = "rapm") -> ValidationResult:
    """Helper para validar dados RAPM."""
    return DataValidator.validate(df, DataValidator.RAPM_SCHEMA, f"RAPM_{source_name}")


def validate_bpm(df: pd.DataFrame) -> ValidationResult:
    """Helper para validar dados BPM."""
    return DataValidator.validate(df, DataValidator.BPM_SCHEMA, "BPM")


def validate_pie(df: pd.DataFrame) -> ValidationResult:
    """Helper para validar dados PIE."""
    return DataValidator.validate(df, DataValidator.PIE_SCHEMA, "PIE")


def validate_injury_report(df: pd.DataFrame) -> ValidationResult:
    """Helper para validar injury reports."""
    return DataValidator.validate(df, DataValidator.INJURY_SCHEMA, "INJURY")


def validate_odds(df: pd.DataFrame) -> ValidationResult:
    """Helper para validar odds data."""
    return DataValidator.validate(df, DataValidator.ODDS_SCHEMA, "ODDS")


# ===== EXEMPLO DE USO =====

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Teste 1: RAPM válido
    df_valid = pd.DataFrame({
        'Team': ['Lakers', 'Warriors', 'Celtics'],
        'Time Decay ORAPM': [3.5, 2.1, 4.2],
        'Time Decay DRAPM': [1.8, 3.2, 2.5]
    })
    
    result = validate_rapm(df_valid, "test")
    print(result)
    print()
    
    # Teste 2: RAP com erros
    df_invalid = pd.DataFrame({
        'Team': ['Lakers', 'Warriors'],
        'Time Decay ORAPM': [25.0, 2.1],  # 25.0 fora do range!
        'Time Decay DRAPM': [1.8, None]   # Null value
    })
    
    result = validate_rapm(df_invalid, "test_invalid")
    print(result)
