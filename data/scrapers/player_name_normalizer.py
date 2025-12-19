"""
Player Name Normalizer for NBA Predictor.

This module provides utilities to normalize player names from various sources
(e.g., Action Network, ScoresAndOdds) to match the canonical names in our database.

Uses fuzzy matching against the nba_player_stats.csv reference file to handle:
- Name variations (e.g., "LeBron James" vs "Lebron James")
- Abbreviated names (e.g., "J. Smith" vs "John Smith")
- Nicknames and alternative spellings

v26.2: Initial implementation for Player Props integration.
v26.4: Upgraded to thefuzz with rapidfuzz backend for better matching.
"""

import logging
import os
import re
from typing import Optional, Dict, List
import pandas as pd

# Try to import thefuzz (preferred), fallback to difflib
try:
    from thefuzz import fuzz
    THEFUZZ_AVAILABLE = True
except ImportError:
    THEFUZZ_AVAILABLE = False
    from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Cache for player names to avoid repeated file reads
_PLAYER_CACHE: Optional[pd.DataFrame] = None
_NAME_TO_CANONICAL: Dict[str, str] = {}


def _load_player_reference() -> pd.DataFrame:
    """
    Loads the player reference data from nba_player_stats.csv.
    
    Returns:
        DataFrame with player data including canonical names
        
    Raises:
        FileNotFoundError: If reference file doesn't exist
    """
    global _PLAYER_CACHE
    
    if _PLAYER_CACHE is not None:
        return _PLAYER_CACHE
    
    # Determine project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    csv_path = os.path.join(project_root, "data", "nba_player_stats.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Player stats file not found: {csv_path}")
    
    _PLAYER_CACHE = pd.read_csv(csv_path)
    
    method = "thefuzz+rapidfuzz" if THEFUZZ_AVAILABLE else "difflib"
    logger.info(f"✅ Loaded {len(_PLAYER_CACHE)} player references (using {method})")
    
    return _PLAYER_CACHE


def _similarity_score(name1: str, name2: str) -> float:
    """
    Calculate similarity score between two names.
    
    Uses thefuzz token_set_ratio if available (handles word reordering),
    falls back to difflib SequenceMatcher.
    
    Handles abbreviated names like "D. Daniels" vs "Dyson Daniels".
    
    Args:
        name1: First name
        name2: Second name
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Normalize for comparison
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    
    # Exact match
    if n1 == n2:
        return 1.0
    
    # Check for abbreviated first name pattern (e.g., "D. Daniels" vs "Dyson Daniels")
    abbrev_pattern = r'^([a-z])\.?\s+(.+)$'
    
    match1 = re.match(abbrev_pattern, n1)
    match2 = re.match(abbrev_pattern, n2)
    
    # If one is abbreviated, check if initial matches and last name matches
    if match1 and not match2:
        initial1, lastname1 = match1.groups()
        parts2 = n2.split()
        if len(parts2) >= 2:
            first2 = parts2[0]
            lastname2 = ' '.join(parts2[1:])
            if first2.startswith(initial1):
                if THEFUZZ_AVAILABLE:
                    lastname_score = fuzz.ratio(lastname1, lastname2) / 100.0
                else:
                    lastname_score = SequenceMatcher(None, lastname1, lastname2).ratio()
                if lastname_score > 0.85:
                    return 0.92  # High confidence match for abbreviated names
    
    elif match2 and not match1:
        initial2, lastname2 = match2.groups()
        parts1 = n1.split()
        if len(parts1) >= 2:
            first1 = parts1[0]
            lastname1 = ' '.join(parts1[1:])
            if first1.startswith(initial2):
                if THEFUZZ_AVAILABLE:
                    lastname_score = fuzz.ratio(lastname1, lastname2) / 100.0
                else:
                    lastname_score = SequenceMatcher(None, lastname1, lastname2).ratio()
                if lastname_score > 0.85:
                    return 0.92
    
    # Use thefuzz or fallback to difflib
    if THEFUZZ_AVAILABLE:
        # token_set_ratio handles word reordering and partial matches well
        # e.g., "LeBron James" vs "James LeBron" still matches
        return fuzz.token_set_ratio(n1, n2) / 100.0
    else:
        return SequenceMatcher(None, n1, n2).ratio()


def normalize_player_name(raw_name: str, min_similarity: float = 0.80) -> Optional[str]:
    """
    Normalizes a player name to match canonical database format.
    
    Args:
        raw_name: Raw player name from external source
        min_similarity: Minimum similarity threshold (0.0-1.0)
        
    Returns:
        Canonical player name if match found, None otherwise
        
    Examples:
        >>> normalize_player_name("LeBron James")
        'LeBron James'
        >>> normalize_player_name("Lebron james")
        'LeBron James'
        >>> normalize_player_name("L. James")
        'LeBron James'
    """
    # Reject empty or None
    if not raw_name or not isinstance(raw_name, str):
        return None
    
    raw_name = raw_name.strip()
    
    # Reject names too short (likely team abbreviations like HOU, LAL, ATL)
    if len(raw_name) < 4:
        logger.debug(f"Skipping short name (likely team abbrev): '{raw_name}'")
        return None
    
    # Reject if looks like a team abbreviation (all caps, 2-3 chars)
    if raw_name.isupper() and len(raw_name) <= 3:
        logger.debug(f"Skipping team abbreviation: '{raw_name}'")
        return None
    
    # Check cache first
    if raw_name in _NAME_TO_CANONICAL:
        return _NAME_TO_CANONICAL[raw_name]
    
    try:
        players_df = _load_player_reference()
    except FileNotFoundError as e:
        logger.error(f"❌ Cannot normalize name: {e}")
        return None
    
    # Ensure Player column exists
    if "Player" not in players_df.columns:
        logger.error("❌ 'Player' column not found in nba_player_stats.csv")
        return None
    
    best_match = None
    best_score = 0.0
    
    # Find best matching player
    for player_name in players_df["Player"].dropna().unique():
        score = _similarity_score(raw_name, player_name)
        
        if score > best_score:
            best_score = score
            best_match = player_name
    
    # Only return match if similarity is above threshold
    if best_score >= min_similarity:
        _NAME_TO_CANONICAL[raw_name] = best_match
        if best_score < 1.0:
            logger.debug(f"🔄 Normalized '{raw_name}' → '{best_match}' (score: {best_score:.2f})")
        return best_match
    
    logger.warning(f"⚠️ No match found for '{raw_name}' (best: '{best_match}', score: {best_score:.2f})")
    return None


def batch_normalize_names(raw_names: List[str], min_similarity: float = 0.80) -> Dict[str, Optional[str]]:
    """
    Normalizes a batch of player names.
    
    Args:
        raw_names: List of raw player names
        min_similarity: Minimum similarity threshold
        
    Returns:
        Dict mapping raw names to canonical names (or None if no match)
    """
    return {name: normalize_player_name(name, min_similarity) for name in raw_names}


def clear_cache():
    """Clears the player name cache. Useful for testing or after database updates."""
    global _PLAYER_CACHE, _NAME_TO_CANONICAL
    _PLAYER_CACHE = None
    _NAME_TO_CANONICAL = {}
    logger.info("🔄 Player name cache cleared")


def get_fuzzy_backend() -> str:
    """Returns the name of the fuzzy matching backend in use."""
    return "thefuzz+rapidfuzz" if THEFUZZ_AVAILABLE else "difflib"
