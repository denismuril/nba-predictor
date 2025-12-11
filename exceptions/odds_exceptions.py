"""
Custom exceptions for odds fetching.
"""


class OddsException(Exception):
    """Base exception for odds-related errors."""
    pass


class OddsUnavailableError(OddsException):
    """
    Raised when all odds sources fail and no real odds are available.
    
    This exception indicates that the system could not fetch odds from any
    of the configured APIs (TheOddsAPI, SportsDataIO, RapidAPI, etc.).
    The system will NOT fall back to simulated/fake odds.
    """
    
    def __init__(self, message="All odds sources failed - no real odds available"):
        self.message = message
        super().__init__(self.message)


class OddsAPIKeyMissingError(OddsException):
    """Raised when required API key is not configured."""
    
    def __init__(self, api_name):
        self.api_name = api_name
        self.message = f"{api_name} API key not configured in .env"
        super().__init__(self.message)


class OddsValidationError(OddsException):
    """Raised when odds data fails validation."""
    
    def __init__(self, message, odds_data=None):
        self.message = message
        self.odds_data = odds_data
        super().__init__(self.message)
