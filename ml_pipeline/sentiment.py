"""
News Sentiment Analyzer - NLP para Notícias de Lesões
======================================================
Analisa tweets/notícias para extrair sentimento que impacta jogos.

Hierarquia:
1. Léxico baseado em keywords (rápido, sem custo)
2. GPT-4 (futuro, mais preciso)

Autor: NBA Predictor System
Data: 2025-12-05
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """
    Analisador de sentimento para notícias esportivas.
    
    Foco: Detectar impacto de lesões e disponibilidade de jogadores.
    
    Scores:
        -1.0 = Muito negativo (jogador importante OUT)
        0.0 = Neutro
        +1.0 = Muito positivo (jogador retornando)
    """
    
    # Léxico de palavras negativas (indicam ausência/lesão)
    NEGATIVE_WORDS = {
        # Críticos (-1.0)
        'out': -1.0,
        'ruled out': -1.0,
        'will not play': -1.0,
        'season-ending': -1.0,
        'surgery': -0.9,
        'torn': -0.9,
        'fracture': -0.9,
        
        # Moderados (-0.5 a -0.8)
        'doubtful': -0.7,
        'unlikely': -0.7,
        'sidelined': -0.6,
        'injury': -0.5,
        'questionable': -0.4,
        'day-to-day': -0.3,
        'limited': -0.3,
        'rest': -0.2,
    }
    
    # Léxico de palavras positivas (indicam retorno/saúde)
    POSITIVE_WORDS = {
        # Muito positivos (+0.8 a +1.0)
        'return': 0.8,
        'returning': 0.8,
        'cleared': 0.9,
        'will play': 1.0,
        'available': 0.7,
        'active': 0.6,
        
        # Moderados (+0.3 a +0.5)
        'healthy': 0.5,
        'recovered': 0.6,
        'expected to play': 0.7,
        'probable': 0.4,
        'upgraded': 0.5,
        'back': 0.4,
    }
    
    # Mapeamento de jogadores para times (top stars)
    # Isso pode ser carregado de um CSV externo
    STAR_PLAYERS = {
        'lebron': 'LAL', 'lebron james': 'LAL',
        'stephen curry': 'GSW', 'curry': 'GSW', 'steph': 'GSW',
        'kevin durant': 'PHO', 'durant': 'PHO', 'kd': 'PHO',
        'giannis': 'MIL', 'antetokounmpo': 'MIL',
        'luka': 'DAL', 'doncic': 'DAL',
        'nikola jokic': 'DEN', 'jokic': 'DEN',
        'jayson tatum': 'BOS', 'tatum': 'BOS',
        'anthony davis': 'LAL', 'ad': 'LAL',
        'shai': 'OKC', 'shai gilgeous': 'OKC',
        'anthony edwards': 'MIN', 'ant': 'MIN',
        'devin booker': 'PHO', 'booker': 'PHO',
        'ja morant': 'MEM', 'morant': 'MEM',
        'damian lillard': 'MIL', 'dame': 'MIL',
        'jimmy butler': 'MIA', 'jimmy': 'MIA',
        'kawhi': 'LAC', 'kawhi leonard': 'LAC',
        'paul george': 'PHI', 'pg': 'PHI',
        'joel embiid': 'PHI', 'embiid': 'PHI',
    }

    def __init__(self, use_llm: bool = False, llm_api_key: str = None):
        """
        Args:
            use_llm: Se True, usa GPT-4 para análise (mais preciso, mais lento)
            llm_api_key: API key para OpenAI (se use_llm=True)
        """
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key
        
        if use_llm and not llm_api_key:
            logger.warning("⚠️ LLM solicitado mas API key não fornecida. Usando léxico.")
            self.use_llm = False
    
    def analyze_text(self, text: str) -> Tuple[float, Dict]:
        """
        Analisa um texto e retorna score de sentimento.
        
        Args:
            text: Texto para analisar (tweet, manchete, etc.)
            
        Returns:
            Tuple[score, metadata]:
                - score: -1.0 a +1.0
                - metadata: {'matched_words': [...], 'team': str, 'player': str}
        """
        if not text:
            return 0.0, {}
        
        text_lower = text.lower()
        
        # Detectar jogador e time
        player, team = self._extract_player_team(text_lower)
        
        # Calcular score
        if self.use_llm:
            score = self._analyze_with_llm(text)
        else:
            score = self._analyze_with_lexicon(text_lower)
        
        metadata = {
            'player': player,
            'team': team,
            'matched_words': self._get_matched_words(text_lower),
            'raw_text': text[:100] + '...' if len(text) > 100 else text
        }
        
        return score, metadata
    
    def _analyze_with_lexicon(self, text: str) -> float:
        """Análise baseada em léxico de palavras-chave."""
        scores = []
        
        # Verificar palavras negativas
        for word, score in self.NEGATIVE_WORDS.items():
            if word in text:
                scores.append(score)
                
        # Verificar palavras positivas
        for word, score in self.POSITIVE_WORDS.items():
            if word in text:
                scores.append(score)
        
        if not scores:
            return 0.0
        
        # Retornar média ponderada (mais forte domina)
        return sum(scores) / len(scores)
    
    def _analyze_with_llm(self, text: str) -> float:
        """
        Análise usando GPT-4 (mais precisa).
        
        TODO: Implementar quando necessário.
        """
        # Placeholder para futura integração com OpenAI
        logger.info("🤖 LLM analysis not implemented. Falling back to lexicon.")
        return self._analyze_with_lexicon(text.lower())
    
    def _extract_player_team(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extrai jogador e time mencionados no texto."""
        for player, team in self.STAR_PLAYERS.items():
            if player in text:
                return player.title(), team
        return None, None
    
    def _get_matched_words(self, text: str) -> List[str]:
        """Retorna palavras-chave encontradas no texto."""
        matched = []
        for word in list(self.NEGATIVE_WORDS.keys()) + list(self.POSITIVE_WORDS.keys()):
            if word in text:
                matched.append(word)
        return matched
    
    def analyze_tweets(self, tweets: List[Dict]) -> Dict[str, float]:
        """
        Analisa lista de tweets e retorna sentimento por time.
        
        Args:
            tweets: Lista de dicts com 'text' (do twitter_scraper.py)
            
        Returns:
            Dict {team_abbr: avg_sentiment_score}
        """
        team_scores = defaultdict(list)
        
        for tweet in tweets:
            text = tweet.get('text', '')
            score, metadata = self.analyze_text(text)
            
            team = metadata.get('team')
            if team:
                team_scores[team].append(score)
        
        # Calcular média por time
        result = {}
        for team, scores in team_scores.items():
            result[team] = sum(scores) / len(scores) if scores else 0.0
        
        return result
    
    def get_team_sentiment(self, team_abbr: str, tweets: List[Dict] = None) -> float:
        """
        Obtém sentimento para um time específico.
        
        Args:
            team_abbr: Sigla do time (ex: 'BOS', 'LAL')
            tweets: Lista de tweets (se None, busca do twitter_scraper)
            
        Returns:
            Score de sentimento (-1.0 a +1.0)
        """
        if tweets is None:
            # Buscar tweets frescos
            try:
                from data.scrapers.twitter_scraper import fetch_latest_injury_tweets
                tweets = fetch_latest_injury_tweets()
            except Exception as e:
                logger.warning(f"⚠️ Erro ao buscar tweets: {e}")
                return 0.0  # Neutro em caso de falha
        
        team_sentiments = self.analyze_tweets(tweets)
        return team_sentiments.get(team_abbr, 0.0)


def get_sentiment_for_game(home_team: str, away_team: str) -> Dict[str, float]:
    """
    Helper: Obtém sentimento para um jogo específico.
    
    Args:
        home_team: Sigla do time da casa
        away_team: Sigla do time visitante
        
    Returns:
        Dict com {'home_sentiment': float, 'away_sentiment': float, 'sentiment_diff': float}
    """
    analyzer = NewsSentimentAnalyzer()
    
    try:
        from data.scrapers.twitter_scraper import fetch_latest_injury_tweets
        tweets = fetch_latest_injury_tweets()
    except Exception:
        tweets = []
    
    home_sentiment = analyzer.get_team_sentiment(home_team, tweets)
    away_sentiment = analyzer.get_team_sentiment(away_team, tweets)
    
    return {
        'home_sentiment': home_sentiment,
        'away_sentiment': away_sentiment,
        'sentiment_diff': home_sentiment - away_sentiment
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🧠 Testando News Sentiment Analyzer\n")
    
    analyzer = NewsSentimentAnalyzer()
    
    # Testar textos
    test_texts = [
        "LeBron James ruled out for tonight's game with ankle injury",
        "Stephen Curry expected to play tonight after rest day",
        "Joel Embiid questionable with knee soreness",
        "Kevin Durant cleared to return after missing 5 games",
        "Random tweet about nothing related",
    ]
    
    print("📊 Análise de Textos de Exemplo:\n")
    for text in test_texts:
        score, meta = analyzer.analyze_text(text)
        emoji = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
        print(f"{emoji} Score: {score:+.2f}")
        print(f"   Texto: {text}")
        print(f"   Time: {meta.get('team', 'N/A')}, Player: {meta.get('player', 'N/A')}")
        print(f"   Keywords: {meta.get('matched_words', [])}")
        print()
    
    # Testar análise de tweets (mock)
    mock_tweets = [
        {'text': 'Breaking: Jayson Tatum will not play tonight'},
        {'text': 'Anthony Davis cleared to return after injury'},
        {'text': 'Luka Doncic questionable for tomorrow'},
    ]
    
    print("📊 Análise por Time:\n")
    sentiments = analyzer.analyze_tweets(mock_tweets)
    for team, score in sentiments.items():
        print(f"   {team}: {score:+.2f}")
    
    print("\n✅ Teste completo!")
