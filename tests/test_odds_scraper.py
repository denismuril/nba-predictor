"""
Testes unitários para Odds Scraper.
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from data.scrapers.odds_scraper import (
    OddsValidator,
    TheOddsAPIClient,
    obter_odds,
    get_odds_for_game
)
from exceptions.odds_exceptions import OddsUnavailableError

# Mock para OddsPediaScraper se não puder ser importado
try:
    from data.scrapers.odds_web_scraper import OddsPediaScraper
except ImportError:
    OddsPediaScraper = MagicMock()


class TestOddsValidator:
    """Testes para validação de odds."""
    
    def test_validate_odds_value_valid(self):
        """Testa que odds válidos passam."""
        assert OddsValidator.validate_odds_value(1.50) == True
        assert OddsValidator.validate_odds_value(2.00) == True
        assert OddsValidator.validate_odds_value(5.50) == True
    
    def test_validate_odds_value_too_low(self):
        """Testa que odds muito baixos falham."""
        assert OddsValidator.validate_odds_value(0.50) == False
        assert OddsValidator.validate_odds_value(1.00) == False
    
    def test_validate_odds_value_too_high(self):
        """Testa que odds muito altos falham."""
        assert OddsValidator.validate_odds_value(100.0) == False
    
    def test_validate_odds_value_non_numeric(self):
        """Testa que valores não-numéricos falham."""
        assert OddsValidator.validate_odds_value("1.50") == False
        assert OddsValidator.validate_odds_value(None) == False
    
    def test_validate_game_odds_valid(self):
        """Testa validação de odds de jogo."""
        # Odds típicos com vigorish de ~5%
        assert OddsValidator.validate_game_odds(1.90, 1.90) == True
        assert OddsValidator.validate_game_odds(1.80, 2.10) == True
    
    def test_validate_game_odds_no_vigorish(self):
        """Testa que odds sem vigorish falham (soma < 1.0)."""
        # Soma de probabilities = 1/2.2 + 1/2.2 = 0.91 < 1.0
        assert OddsValidator.validate_game_odds(2.20, 2.20) == False
    
    def test_validate_game_odds_excessive_vigorish(self):
        """Test que vigorish excessivo falha."""
        # Vigorish > 30% é suspeito
        assert OddsValidator.validate_game_odds(1.30, 1.30) == False
    
    def test_normalize_and_validate(self):
        """Testa normalização e validação de dict."""
        input_dict = {
            'Lakers vs Warriors': {
                'home_odds': 1.85,
                'away_odds': 2.00,  # Vigorish ~5% (válido)
                'home_team': 'Lakers',
                'away_team': 'Warriors',
                'source': 'test'
            },
            'Heat vs Celtics': {
                'home_odds': 0.50,  # INVÁLIDO (<1.01)
                'away_odds': 2.00,
                'home_team': 'Heat',
                'away_team': 'Celtics',
                'source': 'test'
            }
        }

        validated = OddsValidator.normalize_and_validate(input_dict)

        # Apenas o primeiro jogo deve passar
        assert len(validated) == 1
        assert 'Lakers vs Warriors' in validated
        # Verifica se calculou fair odds (indicativo de validação bem sucedida)
        assert 'fair_home_odds' in validated['Lakers vs Warriors']
        assert validated['Lakers vs Warriors']['vigorish_pct'] > 0


class TestTheOddsAPIClient:
    """Testes para cliente da TheOddsAPI."""
    
    def test_init_with_api_key(self):
        """Testa inicialização com API key."""
        client = TheOddsAPIClient(api_key='test_key')
        assert client.api_key == 'test_key'
    
    def test_init_without_api_key_raises(self):
        """Testa que sem API key lança erro."""
        with patch.dict('os.environ', {}, clear=True):
            client = TheOddsAPIClient()
            
            with pytest.raises(ValueError, match="ODDS_API_KEY"):
                client.fetch_odds()
    
    @patch('requests.Session.get')
    def test_fetch_odds_success(self, mock_get):
        """Testa fetch bem-sucedido."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                'home_team': 'Los Angeles Lakers',
                'away_team': 'Golden State Warriors',
                'commence_time': '2025-11-28T19:00:00Z',
                'bookmakers': [
                    {
                        'key': 'draftkings',
                        'title': 'DraftKings',
                        'markets': [
                            {
                                'key': 'h2h',
                                'outcomes': [
                                    {'name': 'Los Angeles Lakers', 'price': 1.85},
                                    {'name': 'Golden State Warriors', 'price': 2.05}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
        mock_response.headers = {'x-requests-remaining': '450'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = TheOddsAPIClient(api_key='test_key')
        odds = client.fetch_odds()
        
        # Verificações
        assert len(odds) == 1
        game_key = 'Los Angeles Lakers vs Golden State Warriors'
        assert game_key in odds
        assert odds[game_key]['home_odds'] == 1.85
        assert odds[game_key]['away_odds'] == 2.05
    
    @patch('requests.Session.get')
    def test_fetch_odds_api_error(self, mock_get):
        """Testa comportamento em erro de API."""
        mock_get.side_effect = requests.exceptions.RequestException("API down")
        
        client = TheOddsAPIClient(api_key='test_key')
        
        with pytest.raises(Exception):
            client.fetch_odds()


class TestObterOdds:
    """Testes para função principal obter_odds."""
    
    @pytest.fixture(autouse=True)
    def mock_cache_instance(self):
        """Mock da instância de cache interna para garantir cache miss."""
        with patch('utils.cache._cache') as mock_c:
            mock_c.get.return_value = None  # Always miss (força execução)
            yield mock_c

    @pytest.fixture
    def mock_oddspedia(self):
        # Patch na classe importada dentro de odds_scraper
        with patch('data.scrapers.odds_scraper.OddsPediaScraper') as mock:
            yield mock

    @pytest.fixture
    def mock_client_class(self):
        with patch('data.scrapers.odds_scraper.TheOddsAPIClient') as mock:
            yield mock
    
    @pytest.fixture(autouse=True)
    def force_oddspedia_available(self):
        """Força ODDSPEDIA_AVAILABLE = True durante os testes."""
        with patch('data.scrapers.odds_scraper.ODDSPEDIA_AVAILABLE', True):
            yield

    def test_obter_odds_oddspedia_success(self, mock_oddspedia, mock_client_class):
        """Testa que OddsPedia é tentado primeiro (TIER 1)."""
        # Configurar mock
        mock_instance = mock_oddspedia.return_value
        mock_instance.fetch_odds.return_value = {
            "TeamA vs TeamB": {
                "home_team": "TeamA", "away_team": "TeamB",
                "home_odds": 1.5, "away_odds": 2.5,
                "source": "oddspedia_scraper"
            }
        }
        
        # Executar
        resultado = obter_odds()
        
        # Verificar
        assert len(resultado) == 1
        assert resultado["TeamA vs TeamB"]["source"] == "oddspedia_scraper"
        mock_oddspedia.assert_called_once()
        # TheOddsAPI NÃO deve ser chamado
        mock_client_class.assert_not_called()

    def test_obter_odds_theoddsapi_success(self, mock_oddspedia, mock_client_class):
        """Testa fallback para TheOddsAPI quando OddsPedia falha."""
        # Configurar OddsPedia para falhar/retornar vazio
        mock_oddspedia_instance = mock_oddspedia.return_value
        mock_oddspedia_instance.fetch_odds.side_effect = Exception("Scraper error")
        
        # Configurar TheOddsAPI para sucesso
        mock_client = mock_client_class.return_value
        mock_client.fetch_odds.return_value = {
            "TeamA vs TeamB": {
                "home_team": "TeamA", "away_team": "TeamB",
                "home_odds": 1.95, "away_odds": 1.95,
                "source": "theoddsapi_test" 
            }
        }
        
        # Executar
        resultado = obter_odds()
        
        # Verificar
        assert len(resultado) == 1
        assert resultado["TeamA vs TeamB"]["source"] == "theoddsapi_test"
        
        # Ambos devem ter sido chamados
        mock_oddspedia.assert_called_once()
        mock_client_class.assert_called_once()
    
    def test_obter_odds_fallback_to_default(self, mock_oddspedia, mock_client_class):
        """Testa fallback para default quando tudo falha."""
        # OddsPedia falha
        mock_oddspedia.return_value.fetch_odds.side_effect = Exception("Scraper fail")
        
        # TheOddsAPI falha
        mock_client_class.return_value.fetch_odds.side_effect = Exception("API fail")
        
        # Mockar outros fallbacks também (SportsDataIO, RapidAPI, OddsAPI.io)
        from data.scrapers.odds_scraper import SportsDataIOClient, RapidAPIFootballClient, OddsAPIioClient
        with patch.object(SportsDataIOClient, 'fetch_odds', side_effect=Exception("SD fail")), \
             patch.object(RapidAPIFootballClient, 'fetch_odds', side_effect=Exception("Rapid fail")), \
             patch.object(OddsAPIioClient, 'fetch_odds', side_effect=Exception("OddsIO fail")):
            
            # Executar - deve lançar OddsUnavailableError (Tier 6)
            with pytest.raises(OddsUnavailableError):
                obter_odds()
    
    def test_force_source_default(self):
        """Testa forçar source que não existe lança exceção (agora todas as fontes falham)."""
        with pytest.raises(OddsUnavailableError):
            obter_odds(force_source='nonexistent_source')


class TestGetOddsForGame:
    """Testes para get_odds_for_game."""
    
    def test_get_odds_from_cache(self):
        """Testa obtenção de odds do cache."""
        cache = {
            'Lakers vs Warriors': {
                'home_odds': 1.85,
                'away_odds': 2.05,
                'source': 'theoddsapi'
            }
        }
        
        odds = get_odds_for_game('Lakers', 'Warriors', odds_cache=cache)
        
        assert odds['home_odds'] == 1.85
        assert odds['away_odds'] == 2.05
        assert odds['source'] == 'theoddsapi'
    
    def test_get_odds_raises_when_not_in_cache(self):
        """Testa que lança exceção quando não está no cache (não usa default 1.90)."""
        cache = {}
        
        with pytest.raises(OddsUnavailableError):
            get_odds_for_game('Lakers', 'Warriors', odds_cache=cache)
    
    def test_get_odds_raises_without_cache(self):
        """Testa que lança exceção sem cache (não usa default 1.90)."""
        with pytest.raises(OddsUnavailableError):
            get_odds_for_game('Lakers', 'Warriors')


# Integration test (requer API key real - skip por padrão)
@pytest.mark.skip(reason="Requer ODDS_API_KEY real")
def test_obter_odds_integration():
    """Teste de integração com API real."""
    odds = obter_odds()
    
    # Deve retornar alguma coisa
    assert isinstance(odds, dict)
    
    # Se retornar jogos, validar formato
    if odds:
        for game_key, game_odds in odds.items():
            assert 'home_odds' in game_odds
            assert 'away_odds' in game_odds
            assert game_odds['home_odds'] > 1.01
            assert game_odds['away_odds'] > 1.01
