"""
Breaking News Filter (LLM Lite)
================================
Real-time news monitoring for last-minute lineup changes.

Monitors news sources for injury updates and adjusts prediction confidence.

Tiers:
1. Keyword-based NLP (free, fast)
2. Optional LLM integration for complex analysis

Usage:
    from ml_pipeline.news_filter import NewsFilter, filter_predictions_by_news
    filter = NewsFilter()
    adjustments = filter.check_breaking_news(['LAL', 'BOS'])

Author: NBA Predictor v24.0 - Quant Edge
"""
import os
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================
# Critical keywords and their confidence impact
INJURY_KEYWORDS = {
    # Definitive out
    'out': -1.0,
    'ruled out': -1.0,
    'will not play': -1.0,
    'sidelined': -0.9,
    'miss': -0.8,
    'rest': -0.8,
    'load management': -0.7,
    'dnp': -1.0,
    
    # Likely out
    'doubtful': -0.6,
    'unlikely': -0.5,
    
    # Questionable
    'questionable': -0.3,
    'game-time decision': -0.4,
    'gtd': -0.4,
    'day-to-day': -0.3,
    
    # Positive
    'cleared': 0.3,
    'will play': 0.3,
    'active': 0.2,
    'returning': 0.4,
    'back in lineup': 0.4,
    'upgraded': 0.3,
}

# High-impact player patterns (stars whose absence significantly affects spread)
HIGH_IMPACT_PATTERNS = [
    'lebron', 'giannis', 'jokic', 'embiid', 'curry', 'doncic', 'tatum',
    'durant', 'morant', 'booker', 'lillard', 'edwards', 'brown', 'irving',
    'wembanyama', 'gilgeous-alexander', 'mitchell', 'brunson', 'harden',
    'leonard', 'george', 'butler', 'adebayo', 'davis', 'randle', 'fox'
]

# Team abbreviation mapping
TEAM_KEYWORDS = {
    'lakers': 'LAL', 'celtics': 'BOS', 'warriors': 'GSW', 'nets': 'BRK',
    'nuggets': 'DEN', 'bucks': 'MIL', 'suns': 'PHX', 'heat': 'MIA',
    'sixers': 'PHI', '76ers': 'PHI', 'mavericks': 'DAL', 'grizzlies': 'MEM',
    'cavaliers': 'CLE', 'thunder': 'OKC', 'kings': 'SAC', 'hawks': 'ATL',
    'clippers': 'LAC', 'wolves': 'MIN', 'timberwolves': 'MIN', 'spurs': 'SAS',
    'knicks': 'NYK', 'raptors': 'TOR', 'jazz': 'UTA', 'pistons': 'DET',
    'pacers': 'IND', 'bulls': 'CHI', 'hornets': 'CHA', 'magic': 'ORL',
    'rockets': 'HOU', 'pelicans': 'NOP', 'blazers': 'POR', 'wizards': 'WAS',
}


@dataclass
class NewsImpact:
    """Represents the impact of a news item."""
    team: str
    player: str
    status: str
    confidence_impact: float
    is_high_impact: bool
    source: str
    headline: str
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# NEWS FILTER CLASS
# =============================================================================
class NewsFilter:
    """
    Filter for breaking NBA news that impacts predictions.
    
    Uses keyword-based NLP for fast, free analysis.
    Optionally integrates with LLM for complex cases.
    """
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_api_key: Optional[str] = None,
        cache_ttl_minutes: int = 15
    ):
        """
        Initialize NewsFilter.
        
        Args:
            use_llm: Whether to use LLM for complex analysis
            llm_api_key: API key for LLM service
            cache_ttl_minutes: How long to cache news results
        """
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key or os.getenv('OPENAI_API_KEY')
        self.cache_ttl_minutes = cache_ttl_minutes
        
        self._cache: Dict[str, Tuple[datetime, List[NewsImpact]]] = {}
        
    def analyze_headline(self, headline: str) -> Optional[NewsImpact]:
        """
        Analyze a single headline for injury/availability info.
        
        Args:
            headline: News headline text
            
        Returns:
            NewsImpact if relevant news detected, None otherwise
        """
        headline_lower = headline.lower()
        
        # Find team
        team_found = None
        for keyword, abbrev in TEAM_KEYWORDS.items():
            if keyword in headline_lower:
                team_found = abbrev
                break
        
        # Also check for team abbreviations directly
        if not team_found:
            for abbrev in ['LAL', 'BOS', 'GSW', 'MIA', 'PHI', 'DEN', 'MIL']:
                if abbrev.lower() in headline_lower or abbrev in headline:
                    team_found = abbrev
                    break
        
        if not team_found:
            return None
        
        # Find player and impact keywords
        player_found = None
        is_high_impact = False
        for player in HIGH_IMPACT_PATTERNS:
            if player in headline_lower:
                player_found = player.title()
                is_high_impact = True
                break
        
        # Find status keywords
        status = None
        confidence_impact = 0.0
        for keyword, impact in INJURY_KEYWORDS.items():
            if keyword in headline_lower:
                status = keyword.upper()
                confidence_impact = impact
                break
        
        if not status:
            return None
        
        # Amplify impact for high-impact players
        if is_high_impact:
            confidence_impact *= 1.5
            # Cap at -1.0 for negative impacts
            confidence_impact = max(-1.0, min(1.0, confidence_impact))
        
        return NewsImpact(
            team=team_found,
            player=player_found or "Unknown",
            status=status,
            confidence_impact=confidence_impact,
            is_high_impact=is_high_impact,
            source="keyword_analysis",
            headline=headline
        )
    
    def analyze_batch(self, headlines: List[str]) -> List[NewsImpact]:
        """
        Analyze multiple headlines.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            List of NewsImpact objects
        """
        impacts = []
        for headline in headlines:
            impact = self.analyze_headline(headline)
            if impact:
                impacts.append(impact)
        return impacts
    
    def fetch_rss_news(self, max_items: int = 20) -> List[str]:
        """
        Fetch latest NBA injury news from RSS feeds.
        
        Returns:
            List of headline strings
        """
        headlines = []
        
        try:
            import feedparser
            
            # Rotowire injury feed
            feeds = [
                'https://www.rotowire.com/rss/injury.htm',
                'https://www.espn.com/espn/rss/nba/news',
            ]
            
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:max_items]:
                        title = entry.get('title', '')
                        if self._is_nba_relevant(title):
                            headlines.append(title)
                except Exception as e:
                    logger.debug(f"Error parsing feed {feed_url}: {e}")
            
            logger.info(f"📰 Fetched {len(headlines)} NBA headlines from RSS")
            
        except ImportError:
            logger.warning("⚠️ feedparser not installed, skipping RSS")
        
        return headlines
    
    def _is_nba_relevant(self, text: str) -> bool:
        """Check if text is NBA-related."""
        text_lower = text.lower()
        nba_indicators = ['nba', 'basketball'] + list(TEAM_KEYWORDS.keys())
        return any(ind in text_lower for ind in nba_indicators)
    
    def check_breaking_news(
        self, 
        teams: List[str],
        use_cache: bool = True
    ) -> Dict[str, List[NewsImpact]]:
        """
        Check for breaking news affecting specific teams.
        
        Args:
            teams: List of team abbreviations to check
            use_cache: Whether to use cached results
            
        Returns:
            Dict mapping team to list of NewsImpacts
        """
        results: Dict[str, List[NewsImpact]] = {team: [] for team in teams}
        
        # Check cache
        cache_key = '_'.join(sorted(teams))
        if use_cache and cache_key in self._cache:
            cached_time, cached_results = self._cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                logger.debug("Using cached news results")
                return {team: [r for r in cached_results if r.team == team] for team in teams}
        
        # Fetch fresh news
        headlines = self.fetch_rss_news()
        
        # Analyze headlines
        all_impacts = self.analyze_batch(headlines)
        
        # Filter by requested teams
        for impact in all_impacts:
            if impact.team in results:
                results[impact.team].append(impact)
        
        # Cache results
        self._cache[cache_key] = (datetime.now(), all_impacts)
        
        return results
    
    def get_confidence_adjustment(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Get confidence adjustment for a game.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            
        Returns:
            Dict with 'home_adj', 'away_adj', 'net_adj'
        """
        news = self.check_breaking_news([home_team, away_team])
        
        home_adj = sum(n.confidence_impact for n in news.get(home_team, []))
        away_adj = sum(n.confidence_impact for n in news.get(away_team, []))
        
        # Cap adjustments
        home_adj = max(-1.0, min(0.5, home_adj))
        away_adj = max(-1.0, min(0.5, away_adj))
        
        return {
            'home_adj': home_adj,
            'away_adj': away_adj,
            'net_adj': home_adj - away_adj,  # Positive = news favors home
            'home_news': news.get(home_team, []),
            'away_news': news.get(away_team, []),
        }


# =============================================================================
# PREDICTION FILTERING
# =============================================================================
def filter_predictions_by_news(
    predictions: List[Dict],
    zero_on_star_out: bool = True
) -> List[Dict]:
    """
    Filter/adjust predictions based on breaking news.
    
    Args:
        predictions: List of prediction dicts with 'home_team', 'away_team', 'confidence'
        zero_on_star_out: If True, zero confidence when star is OUT
        
    Returns:
        Adjusted predictions list
    """
    filter_instance = NewsFilter()
    adjusted = []
    
    for pred in predictions:
        home = pred.get('home_team', '')
        away = pred.get('away_team', '')
        confidence = pred.get('confidence', 1.0)
        
        adjustment = filter_instance.get_confidence_adjustment(home, away)
        
        # Apply adjustment
        new_confidence = confidence
        
        # Check for star OUT on either side
        for news_list in [adjustment['home_news'], adjustment['away_news']]:
            for news in news_list:
                if zero_on_star_out and news.is_high_impact and news.confidence_impact <= -0.8:
                    logger.warning(f"🚨 STAR OUT: {news.player} ({news.team}) - Zeroing confidence")
                    new_confidence = 0.0
                    break
        
        # Apply net adjustment if not zeroed
        if new_confidence > 0:
            new_confidence = max(0, min(1.0, confidence * (1 + adjustment['net_adj'] * 0.2)))
        
        adjusted_pred = pred.copy()
        adjusted_pred['confidence'] = new_confidence
        adjusted_pred['news_adjustment'] = adjustment['net_adj']
        adjusted.append(adjusted_pred)
    
    return adjusted


# =============================================================================
# CLI TEST
# =============================================================================
if __name__ == "__main__":
    print("🧪 Testing News Filter...")
    
    # Test headlines
    test_headlines = [
        "LeBron James ruled OUT for Lakers game tonight",
        "Jayson Tatum questionable with ankle injury",
        "Stephen Curry cleared to play tonight",
        "Giannis Antetokounmpo will not play due to rest",
        "Random non-NBA news headline",
        "Lakers vs Celtics preview",
        "Denver Nuggets announce Jokic is active for tonight",
    ]
    
    filter_instance = NewsFilter()
    
    print("\n📰 Analyzing Headlines:")
    print("-" * 60)
    
    for headline in test_headlines:
        impact = filter_instance.analyze_headline(headline)
        if impact:
            emoji = "🔴" if impact.confidence_impact < -0.5 else (
                "🟡" if impact.confidence_impact < 0 else "🟢"
            )
            print(f"{emoji} {impact.team}: {impact.player} - {impact.status}")
            print(f"   Impact: {impact.confidence_impact:+.2f}")
            print(f"   High-Impact Player: {impact.is_high_impact}")
        else:
            print(f"⚪ Not relevant: {headline[:50]}...")
        print()
    
    # Test prediction filtering
    print("\n🎯 Testing Prediction Filtering:")
    print("-" * 60)
    
    test_predictions = [
        {'home_team': 'LAL', 'away_team': 'BOS', 'confidence': 0.65},
        {'home_team': 'DEN', 'away_team': 'MIA', 'confidence': 0.72},
    ]
    
    # Mock with cached analysis
    filter_instance._cache['LAL_BOS'] = (
        datetime.now(),
        [NewsImpact(
            team='LAL', player='LeBron', status='OUT',
            confidence_impact=-1.0, is_high_impact=True,
            source='test', headline='LeBron OUT'
        )]
    )
    
    adjusted = filter_predictions_by_news(test_predictions)
    
    for orig, adj in zip(test_predictions, adjusted):
        print(f"{orig['home_team']} vs {orig['away_team']}")
        print(f"   Original Confidence: {orig['confidence']:.2f}")
        print(f"   Adjusted Confidence: {adj['confidence']:.2f}")
        print()
    
    print("✅ Test completed!")
