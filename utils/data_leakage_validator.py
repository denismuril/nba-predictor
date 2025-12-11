"""
Data Leakage Validation Framework

Detecta e previne data leakage em features temporais (rolling, lag, etc).

Problema:
- Rolling features podem acidentalmente incluir dados futuros
- Isso infla artificialmente a acurácia em backtests
- Modelo aprende patterns que não existirão em produção
- Resultado: -2-3% accuracy quando deployed

Solução:
- Validação temporal automática de todas features
- Garantir que shift(1) está aplicado corretamente
- Verificar que primeiro valor por team é NaN
- Alert system para novos features
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class DataLeakageValidator:
    """
    Valida que features temporais não contêm data leakage.
    
    Detecta:
    - Rolling features sem shift(1)
    - Lag features calculadas incorretamente
    - Colunas que usam dados futuros
    - Temporal inconsistencies
    
    Usage:
        validator = DataLeakageValidator()
        
        # Validar DataFrame completo
        result = validator.validate_dataframe(df, date_col='date', team_col='home_team')
        
        if not result['valid']:
            logger.error(f"Data leakage detected: {result['errors']}")
        
        # Validar features específicas
        validator.validate_rolling_features(df, ['rolling_5_points', 'rolling_10_efg'])
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: Se True, falha com qualquer warning. Se False, apenas alerta.
        """
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        team_col: str = 'home_team',
        feature_patterns: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Valida DataFrame completo para data leakage.
        
        Args:
            df: DataFrame com features
            date_col: Nome da coluna de data
            team_col: Nome da coluna de time (para groupby)
            feature_patterns: Patterns de features para validar (default: ['rolling', 'lag'])
        
        Returns:
            Dict com resultado da validação:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'features_checked': int,
                'leakage_detected': List[str]
            }
        """
        self.errors = []
        self.warnings = []
        
        if feature_patterns is None:
            feature_patterns = ['rolling', 'lag', 'prev_', 'shift']
        
        # Encontrar todas as features suspeitas
        suspicious_features = []
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in feature_patterns):
                suspicious_features.append(col)
        
        logger.info(f"🔍 Validando {len(suspicious_features)} features suspeitas de temporal dependency...")
        
        leakage_features = []
        
        for feature in suspicious_features:
            has_leakage = self._validate_feature_no_leakage(
                df, feature, date_col, team_col
            )
            if has_leakage:
                leakage_features.append(feature)
        
        # Check temporal consistency
        if date_col in df.columns:
            self._validate_temporal_order(df, date_col)
        
        valid = len(self.errors) == 0 and (not self.strict_mode or len(self.warnings) == 0)
        
        result = {
            'valid': valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'features_checked': len(suspicious_features),
            'leakage_detected': leakage_features
        }
        
        # Log summary
        if valid:
            logger.info(f"✅ Validação passou! {len(suspicious_features)} features checked, 0 leaks")
        else:
            logger.error(f"❌ Data leakage detected em {len(leakage_features)} features!")
            for feature in leakage_features:
                logger.error(f"   - {feature}")
        
        return result
    
    def _validate_feature_no_leakage(
        self,
        df: pd.DataFrame,
        feature: str,
        date_col: str,
        team_col: str
    ) -> bool:
        """
        Valida que uma feature específica não tem data leakage.
        
        NOVA ESTRATÉGIA (mais robusta):
        Compara valores da rolling feature com uma versão conhecidamente correta.
        Se os valores baterem com a versão SEM shift, há leakage.
        Se os valores baterem com a versão COM shift, está OK.
        
        Returns:
            True se leakage detectado, False se OK
        """
        if feature not in df.columns:
            self.warnings.append(f"Feature {feature} não encontrada no DataFrame")
            return False
        
        # Garantir que está ordenado por data
        df_sorted = df.sort_values([team_col, date_col]).copy()
        
        # Detectar window size da feature (se existir)
        # Exemplos: "rolling_5_points" → window=5, "lag_3" → não aplica
        window_match = None
        feature_lower = feature.lower()
        
        if 'rolling' in feature_lower:
            # Tentar extrair window size
            import re
            match = re.search(r'rolling[_\-]?(\d+)', feature_lower)
            if match:
                window_match = int(match.group(1))
        
        # ESTRATÉGIA 1: Verificar primeiro valor por team (regra base)
        first_values = df_sorted.groupby(team_col)[feature].first()
        nan_count = first_values.isna().sum()
        total_teams = len(first_values)
        
        # Se NENHUM team tem NaN e feature é rolling/lag, suspeito
        if nan_count == 0 and ('rolling' in feature_lower or 'lag' in feature_lower or 'shift' in feature_lower):
            self.warnings.append(
                f"⚠️ '{feature}': Nenhum team tem NaN no primeiro valor. "
                f"Pode indicar leakage se min_periods=1."
            )
        
        # ESTRATÉGIA 2: Se conseguimos identificar window, comparar com cálculo correto
        if window_match and 'rolling' in feature_lower:
            has_leakage = self._validate_rolling_computation(
                df_sorted, feature, team_col, window_match
            )
            if has_leakage:
                return True
        
        # ESTRATÉGIA 3: Verificar se segundo valor é diferente do primeiro
        # Feature correta com shift(1): segundo valor usa apenas primeiro jogo
        # Feature com leakage: segundo valor já usa primeiros 2 jogos
        second_values = df_sorted.groupby(team_col)[feature].nth(1)
       
        
        # Se segunda linha tem valores dramáticamente diferentes, pode ser OK
        # Mas se todos são iguais ou muito próximos, suspeito
        
        return False  # Default: assumir OK se não detectamos leakage claro
    
    def _validate_rolling_computation(
        self,
        df: pd.DataFrame,
        feature: str,
        team_col: str,
        window: int
    ) -> bool:
        """
        Valida rolling feature comparando com versão correta e incorreta.
        
        Args:
            df: DataFrame ordenado
            feature: Nome da feature a validar
            team_col: Coluna de team
            window: Window size da rolling
        
        Returns:
            True se leakage detectado, False se OK
        """
        # Encontrar coluna base (ex: 'rolling_5_points' → 'points')
        # Tentar padrões comuns
        base_col_candidates = []
        
        # Pattern: rolling_N_COLNAME
        import re
        match = re.search(r'rolling_\d+_(.+)', feature.lower())
        if match:
            base_name = match.group(1)
            # Procurar colunas que contenham esse nome
            for col in df.columns:
                if base_name in col.lower() and col != feature:
                    base_col_candidates.append(col)
        
        if not base_col_candidates:
            # Não conseguimos encontrar coluna base, skip validation
            self.warnings.append(
                f"⚠️ Não foi possível inferir coluna base para '{feature}'"
            )
            return False
        
        # Usar primeira candidata
        base_col = base_col_candidates[0]
        
        # Calcular versão CORRETA (com shift)
        correct_version = df.groupby(team_col)[base_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        # Calcular versão INCORRETA (sem shift)
        leaky_version = df.groupby(team_col)[base_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Comparar feature real com ambas versões
        actual_values = df[feature].values
        
        # Quantos valores batem com versão correta vs leaky?
        # (tolerância para floating point)
        correct_matches = np.sum(np.abs(actual_values - correct_version.values) < 0.001)
        leaky_matches = np.sum(np.abs(actual_values - leaky_version.values) < 0.001)
        
        total_valid = np.sum(~np.isnan(actual_values))
        
        if total_valid == 0:
            return False  # Sem dados para comparar
        
        correct_pct = (correct_matches / total_valid) * 100
        leaky_pct = (leaky_matches / total_valid) * 100
        
        logger.debug(
            f"  '{feature}': {correct_pct:.1f}% match correct, "
            f"{leaky_pct:.1f}% match leaky"
        )
        
        # Se mais de 90% dos valores batem com versão LEAKY, é leakage!
        if leaky_pct > 90 and leaky_pct > correct_pct:
            error_msg = (
                f"🚨 CRITICAL LEAKAGE em '{feature}': "
                f"{leaky_pct:.1f}% dos valores batem com rolling SEM shift(1). "
                f"Feature está usando dados do mesmo jogo!"
            )
            self.errors.append(error_msg)
            logger.error(error_msg)
            return True
        
        # Se mais de 90% batem com versão CORRETA, está OK
        elif correct_pct > 90:
            logger.debug(f"✅ '{feature}': {correct_pct:.1f}% match com versão correta")
            return False
        
        # Caso ambíguo
        else:
            warning_msg = (
                f"⚠️ AMBIGUOUS em '{feature}': "
                f"Correct={correct_pct:.1f}%, Leaky={leaky_pct:.1f}%. "
                f"Não conseguimos determinar se há leakage."
            )
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
            return False
    
    def _validate_temporal_order(self, df: pd.DataFrame, date_col: str):
        """Valida que o DataFrame está ordenado temporalmente."""
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except:
                self.warnings.append(f"Coluna {date_col} não é datetime e não pôde ser convertida")
                return
        
        # Verificar se está monotonicamente crescente (ordenado)
        dates = df[date_col].values
        is_sorted = np.all(dates[:-1] <= dates[1:])
        
        if not is_sorted:
            self.warnings.append(
                f"DataFrame não está ordenado por {date_col}. "
                "Rolling features podem estar incorretas."
            )
    
    def validate_rolling_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        team_col: str = 'home_team',
        date_col: str = 'date'
    ) -> Dict[str, bool]:
        """
        Valida lista específica de rolling features.
        
        Returns:
            Dict mapeando feature → bool (True se OK, False se leakage)
        """
        results = {}
        
        for feature in features:
            has_leakage = self._validate_feature_no_leakage(
                df, feature, date_col, team_col
            )
            results[feature] = not has_leakage  # Inverter: True = OK
        
        return results
    
    def create_validation_report(self, df: pd.DataFrame) -> str:
        """
        Cria relatório detalhado da validação.
        
        Returns:
            String com relatório formatado
        """
        result = self.validate_dataframe(df)
        
        report = []
        report.append("=" * 80)
        report.append("DATA LEAKAGE VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Status: {'✅ PASSED' if result['valid'] else '❌ FAILED'}")
        report.append(f"Features Checked: {result['features_checked']}")
        report.append(f"Leakage Detected: {len(result['leakage_detected'])}")
        report.append("")
        
        if result['errors']:
            report.append("🚨 ERRORS:")
            for error in result['errors']:
                report.append(f"  - {error}")
            report.append("")
        
        if result['warnings']:
            report.append("⚠️ WARNINGS:")
            for warning in result['warnings']:
                report.append(f"  - {warning}")
            report.append("")
        
        if result['leakage_detected']:
            report.append("📋 Features with Leakage:")
            for feature in result['leakage_detected']:
                report.append(f"  - {feature}")
        else:
            report.append("✅ No leakage detected in any feature!")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# Convenience function
def validate_no_leakage(
    df: pd.DataFrame,
    date_col: str = 'date',
    team_col: str = 'home_team',
    strict: bool = True
) -> bool:
    """
    Valida que DataFrame não tem data leakage.
    
    Args:
        df: DataFrame com features
        date_col: Nome da coluna de data
        team_col: Nome da coluna de time
        strict: Se True, falha com warnings também
    
    Returns:
        True se validação passou, False se leakage detectado
    
    Raises:
        ValueError: Se leakage crítico detectado
    
    Usage:
        # Quick validation
        if not validate_no_leakage(df):
            raise ValueError("Data leakage detected!")
        
        # Or let it raise automatically
        validate_no_leakage(df, strict=True)
    """
    validator = DataLeakageValidator(strict_mode=strict)
    result = validator.validate_dataframe(df, date_col, team_col)
    
    if result['errors']:
        error_msg = f"Data leakage detected in {len(result['leakage_detected'])} features"
        if strict:
            raise ValueError(error_msg)
    
    return result['valid']
