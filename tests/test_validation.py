"""
Testes unitários para o módulo de validação.
"""
import pytest
from utils.validation import (
    validate_game_schedule,
    validate_team_name,
    validate_date,
    validate_probability,
    validate_odds,
    validate_prediction,
    ValidationError,
    safe_get
)


class TestValidateGameSchedule:
    """Testes para validação de schedule de jogos."""
    
    def test_valid_game(self):
        """Testa jogo válido."""
        game = {'home': 'Lakers', 'away': 'Celtics'}
        assert validate_game_schedule(game) is True
    
    def test_missing_home(self):
        """Testa jogo sem time da casa."""
        game = {'away': 'Celtics'}
        with pytest.raises(ValidationError, match="Campos obrigatórios faltando"):
            validate_game_schedule(game)
    
    def test_missing_away(self):
        """Testa jogo sem time visitante."""
        game = {'home': 'Lakers'}
        with pytest.raises(ValidationError, match="Campos obrigatórios faltando"):
            validate_game_schedule(game)
    
    def test_empty_team_names(self):
        """Testa jogo com nomes vazios."""
        game = {'home': '', 'away': 'Celtics'}
        with pytest.raises(ValidationError, match="não podem estar vazios"):
            validate_game_schedule(game)
    
    def test_invalid_type(self):
        """Testa tipo inválido."""
        with pytest.raises(ValidationError, match="deve ser um dicionário"):
            validate_game_schedule("not a dict")


class TestValidateTeamName:
    """Testes para validação de nomes de times."""
    
    def test_valid_name(self):
        """Testa nome válido."""
        assert validate_team_name("Los Angeles Lakers") == "Los Angeles Lakers"
    
    def test_name_with_spaces(self):
        """Testa nome com espaços (deve ser normalizado)."""
        assert validate_team_name("  Lakers  ") == "Lakers"
    
    def test_empty_name(self):
        """Testa nome vazio."""
        with pytest.raises(ValidationError, match="não pode estar vazio"):
            validate_team_name("")
    
    def test_invalid_type(self):
        """Testa tipo inválido."""
        with pytest.raises(ValidationError, match="deve ser string"):
            validate_team_name(123)


class TestValidateDate:
    """Testes para validação de datas."""
    
    def test_valid_date(self):
        """Testa data válida."""
        assert validate_date("2025-11-25") == "2025-11-25"
    
    def test_invalid_format(self):
        """Testa formato inválido."""
        with pytest.raises(ValidationError, match="Formato de data inválido"):
            validate_date("25/11/2025")
    
    def test_invalid_date(self):
        """Testa data inválida."""
        with pytest.raises(ValidationError):
            validate_date("2025-13-45")
    
    def test_invalid_type(self):
        """Testa tipo inválido."""
        with pytest.raises(ValidationError, match="deve ser string"):
            validate_date(20251125)


class TestValidateProbability:
    """Testes para validação de probabilidades."""
    
    def test_valid_probability(self):
        """Testa probabilidade válida."""
        assert validate_probability(55.5) == 55.5
        assert validate_probability(0) == 0.0
        assert validate_probability(100) == 100.0
    
    def test_negative_probability(self):
        """Testa probabilidade negativa."""
        with pytest.raises(ValidationError, match="deve estar entre 0 e 100"):
            validate_probability(-5)
    
    def test_probability_over_100(self):
        """Testa probabilidade acima de 100."""
        with pytest.raises(ValidationError, match="deve estar entre 0 e 100"):
            validate_probability(150)
    
    def test_invalid_type(self):
        """Testa tipo inválido."""
        with pytest.raises(ValidationError, match="deve ser numérico"):
            validate_probability("55")


class TestValidateOdds:
    """Testes para validação de odds."""
    
    def test_valid_odds(self):
        """Testa odds válidas."""
        assert validate_odds(1.90) == 1.90
        assert validate_odds(2.50) == 2.50
    
    def test_odds_below_one(self):
        """Testa odds abaixo de 1.0."""
        with pytest.raises(ValidationError, match="deve ser >= 1.0"):
            validate_odds(0.5)
    
    def test_very_high_odds(self):
        """Testa odds muito altas (deve gerar warning mas passar)."""
        # Deve passar mas gerar warning no log
        assert validate_odds(150.0) == 150.0


class TestValidatePrediction:
    """Testes para validação de previsões completas."""
    
    def test_valid_prediction(self):
        """Testa previsão válida."""
        prediction = {
            'Casa': 'Lakers',
            'Visitante': 'Celtics',
            'Prob Casa %': 55.0,
            'Prob Visitante %': 45.0,
            'Odd Casa': 1.90,
            'Odd Visitante': 2.10
        }
        result = validate_prediction(prediction)
        assert result == prediction
    
    def test_missing_required_field(self):
        """Testa previsão com campo faltando."""
        prediction = {
            'Casa': 'Lakers',
            'Prob Casa %': 55.0,
            'Prob Visitante %': 45.0
        }
        with pytest.raises(ValidationError, match="Campos obrigatórios faltando"):
            validate_prediction(prediction)
    
    def test_invalid_probabilities_sum(self):
        """Testa previsão com soma de probabilidades incorreta."""
        prediction = {
            'Casa': 'Lakers',
            'Visitante': 'Celtics',
            'Prob Casa %': 60.0,
            'Prob Visitante %': 50.0  # Soma = 110%
        }
        # Deve passar mas gerar warning
        result = validate_prediction(prediction)
        assert result is not None


class TestSafeGet:
    """Testes para função safe_get."""
    
    def test_existing_key(self):
        """Testa chave existente."""
        d = {'key': 'value'}
        assert safe_get(d, 'key') == 'value'
    
    def test_missing_key(self):
        """Testa chave faltando."""
        d = {'key': 'value'}
        assert safe_get(d, 'missing', 'default') == 'default'
    
    def test_no_default(self):
        """Testa sem default."""
        d = {'key': 'value'}
        assert safe_get(d, 'missing') is None


@pytest.mark.parametrize("prob,odds,expected_valid", [
    (55.0, 1.90, True),
    (50.0, 2.00, True),
    (0.0, 1.90, True),   # Probabilidade zero É válida (range [0, 100])
    (100.0, 1.90, True),  # Probabilidade 100% É válida (range [0, 100])
    (-1.0, 1.90, False),  # Prob negativa é inválida
    (101.0, 1.90, False),  # Prob > 100 é inválida
    (55.0, 0.5, False),  # Odds < 1.0 inválidas
])
def test_validation_combinations(prob, odds, expected_valid):
    """Testa combinações de probabilidade e odds."""
    if expected_valid:
        validate_probability(prob)
        validate_odds(odds)
    else:
        # Testar probabilidade inválida
        if prob < 0 or prob > 100:
            with pytest.raises(ValidationError):
                validate_probability(prob)
        # Testar odds inválidas
        if odds < 1.0:
            with pytest.raises(ValidationError):
                validate_odds(odds)

