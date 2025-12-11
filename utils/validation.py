"""
Módulo de validação de dados para o NBA Predictor.
Fornece schemas e funções de validação para garantir integridade dos dados.
"""
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exceção customizada para erros de validação."""
    pass

def validate_game_schedule(game: Dict[str, Any]) -> bool:
    """
    Valida se um jogo do schedule tem os campos obrigatórios.
    
    Args:
        game: Dicionário com dados do jogo.
        
    Returns:
        bool: True se válido, False caso contrário.
        
    Raises:
        ValidationError: Se campos obrigatórios estiverem faltando.
    """
    required_fields = ['home', 'away']
    
    if not isinstance(game, dict):
        raise ValidationError(f"Jogo deve ser um dicionário, recebido: {type(game)}")
    
    missing_fields = [field for field in required_fields if field not in game]
    if missing_fields:
        raise ValidationError(
            f"Campos obrigatórios faltando no jogo: {missing_fields}. "
            f"Campos disponíveis: {list(game.keys())}"
        )
    
    if not game.get('home') or not game.get('away'):
        raise ValidationError("Campos 'home' e 'away' não podem estar vazios")
    
    return True

def validate_team_name(team_name: str) -> str:
    """
    Valida e normaliza o nome de um time.
    
    Args:
        team_name: Nome do time.
        
    Returns:
        str: Nome do time normalizado.
        
    Raises:
        ValidationError: Se o nome do time for inválido.
    """
    if not isinstance(team_name, str):
        raise ValidationError(f"Nome do time deve ser string, recebido: {type(team_name)}")
    
    team_name = team_name.strip()
    if not team_name:
        raise ValidationError("Nome do time não pode estar vazio")
    
    return team_name

def validate_date(date_str: str) -> str:
    """
    Valida formato de data (YYYY-MM-DD).
    
    Args:
        date_str: String de data.
        
    Returns:
        str: Data validada.
        
    Raises:
        ValidationError: Se a data for inválida.
    """
    if not isinstance(date_str, str):
        raise ValidationError(f"Data deve ser string, recebido: {type(date_str)}")
    
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise ValidationError(f"Formato de data inválido: {date_str}. Esperado: YYYY-MM-DD")

def validate_probability(prob: float, name: str = "probabilidade") -> float:
    """
    Valida se uma probabilidade está no range [0, 100].
    
    Args:
        prob: Valor da probabilidade.
        name: Nome da probabilidade (para mensagens de erro).
        
    Returns:
        float: Probabilidade validada.
        
    Raises:
        ValidationError: Se a probabilidade estiver fora do range.
    """
    if not isinstance(prob, (int, float)):
        raise ValidationError(f"{name} deve ser numérico, recebido: {type(prob)}")
    
    if prob < 0 or prob > 100:
        raise ValidationError(f"{name} deve estar entre 0 e 100, recebido: {prob}")
    
    return float(prob)

def validate_odds(odds: float, name: str = "odds") -> float:
    """
    Valida se as odds estão em um range razoável (decimal odds).
    
    Args:
        odds: Valor das odds.
        name: Nome das odds (para mensagens de erro).
        
    Returns:
        float: Odds validadas.
        
    Raises:
        ValidationError: Se as odds forem inválidas.
    """
    if not isinstance(odds, (int, float)):
        raise ValidationError(f"{name} deve ser numérico, recebido: {type(odds)}")
    
    if odds < 1.0:
        raise ValidationError(f"{name} deve ser >= 1.0 (decimal odds), recebido: {odds}")
    
    if odds > 100.0:
        logger.warning(f"⚠️  {name} muito alta: {odds}. Verificar se está em formato decimal.")
    
    return float(odds)

def validate_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida uma previsão completa.
    
    Args:
        prediction: Dicionário com dados da previsão.
        
    Returns:
        Dict: Previsão validada.
        
    Raises:
        ValidationError: Se a previsão for inválida.
    """
    required_fields = ['Casa', 'Visitante', 'Prob Casa %', 'Prob Visitante %']
    
    if not isinstance(prediction, dict):
        raise ValidationError(f"Previsão deve ser um dicionário, recebido: {type(prediction)}")
    
    missing_fields = [field for field in required_fields if field not in prediction]
    if missing_fields:
        raise ValidationError(f"Campos obrigatórios faltando: {missing_fields}")
    
    # Validar campos individuais
    validate_team_name(prediction['Casa'])
    validate_team_name(prediction['Visitante'])
    validate_probability(prediction['Prob Casa %'], "Prob Casa %")
    validate_probability(prediction['Prob Visitante %'], "Prob Visitante %")
    
    # Validar soma das probabilidades (deve ser ~100%)
    total_prob = prediction['Prob Casa %'] + prediction['Prob Visitante %']
    if abs(total_prob - 100.0) > 1.0:  # Tolerância de 1%
        logger.warning(
            f"⚠️  Soma das probabilidades não é 100%: {total_prob}% "
            f"(Casa: {prediction['Prob Casa %']}%, Visitante: {prediction['Prob Visitante %']}%)"
        )
    
    # Validar odds se presentes
    if 'Odd Casa' in prediction:
        validate_odds(prediction['Odd Casa'], "Odd Casa")
    if 'Odd Visitante' in prediction:
        validate_odds(prediction['Odd Visitante'], "Odd Visitante")
    
    return prediction

def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Obtém valor de um dicionário de forma segura, com logging de aviso.
    
    Args:
        dictionary: Dicionário.
        key: Chave a buscar.
        default: Valor padrão se a chave não existir.
        
    Returns:
        Valor da chave ou default.
    """
    if key not in dictionary:
        logger.debug(f"Chave '{key}' não encontrada no dicionário. Usando default: {default}")
    return dictionary.get(key, default)

