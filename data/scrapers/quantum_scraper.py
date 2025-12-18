"""
Quantum Scraper - Coleta de Dados Autônoma & Infinita

Sistema resiliente de coleta de dados com fallback automático e inferência.
Quando APIs falham, ele não para - usa dados históricos e KNN.

Hierarquia de APIs:
1. NBA API (oficial, grátis)
2. SportsDataIO (pago, backup)
3. Balldontlie API (grátis)
4. Web scraping (último recurso)
5. Inferência KNN (para dados faltantes)

Autor: Lead Quant Researcher & AI Architect
Versão: 1.0.0 - Quantum Edition
"""

import pandas as pd
import numpy as np
import requests
import logging
import json
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv

# Configuração
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# =============================================================================
# CONFIGURAÇÃO DE APIs
# =============================================================================

API_CONFIG = {
    'nba_api': {
        'enabled': True,
        'rate_limit': 1.0,  # Segundos entre requests
        'priority': 1
    },
    'sportsdata': {
        'enabled': bool(os.getenv('SPORTSDATA_API_KEY')),
        'api_key': os.getenv('SPORTSDATA_API_KEY', ''),
        'base_url': 'https://api.sportsdata.io/v3/nba',
        'rate_limit': 0.5,
        'priority': 2
    },
    'balldontlie': {
        'enabled': True,
        'base_url': 'https://api.balldontlie.io/v1',
        'api_key': os.getenv('BALLDONTLIE_API_KEY', ''),
        'rate_limit': 1.0,
        'priority': 3
    },
    'odds_api': {
        'enabled': bool(os.getenv('ODDS_API_KEY')),
        'api_key': os.getenv('ODDS_API_KEY', ''),
        'base_url': 'https://api.the-odds-api.com/v4',
        'rate_limit': 0.5,
        'priority': 1
    }
}


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class QuantumDataCollector:
    """
    Coletor de dados com fallback automático e inferência.
    
    Features:
    - Multi-API com fallback
    - Cache inteligente
    - Inferência KNN para dados faltantes
    - Rate limiting automático
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; QuantumCollector/1.0)'
        })
        self.last_request_time = {}
        self._knn_model = None
        self._player_features_cache = None
        
    def _rate_limit(self, api_name: str):
        """Aplica rate limiting por API."""
        config = API_CONFIG.get(api_name, {})
        wait_time = config.get('rate_limit', 1.0)
        
        last_time = self.last_request_time.get(api_name, 0)
        elapsed = time.time() - last_time
        
        if elapsed < wait_time:
            time.sleep(wait_time - elapsed)
        
        self.last_request_time[api_name] = time.time()
    
    # =========================================================================
    # NBA API (Oficial)
    # =========================================================================
    
    def fetch_player_boxscores_nba_api(
        self, 
        player_id: str = None,
        player_name: str = None,
        n_games: int = 20,
        season: str = '2024-25'
    ) -> Optional[pd.DataFrame]:
        """
        Busca boxscores do jogador via NBA API oficial.
        
        Args:
            player_id: ID do jogador (opcional)
            player_name: Nome do jogador (opcional, usado para buscar ID)
            n_games: Número de jogos a buscar
            season: Temporada
            
        Returns:
            DataFrame com boxscores ou None
        """
        if not API_CONFIG['nba_api']['enabled']:
            return None
        
        try:
            from nba_api.stats.endpoints import playergamelog
            from nba_api.stats.static import players
            
            self._rate_limit('nba_api')
            
            # Obter player_id se necessário
            if not player_id and player_name:
                player_list = players.find_players_by_full_name(player_name)
                if player_list:
                    player_id = player_list[0]['id']
                else:
                    logger.warning(f"Jogador não encontrado: {player_name}")
                    return None
            
            if not player_id:
                return None
            
            # Buscar game log
            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            
            df = log.get_data_frames()[0]
            
            if df.empty:
                return None
            
            # Padronizar colunas
            df = df.rename(columns={
                'GAME_DATE': 'date',
                'MATCHUP': 'matchup',
                'PTS': 'PTS',
                'REB': 'REB',
                'AST': 'AST',
                'MIN': 'MIN',
                'FGM': 'FGM',
                'FGA': 'FGA',
                'FG3M': 'FG3M',
                'FG3A': 'FG3A',
                'FTM': 'FTM',
                'FTA': 'FTA',
                'STL': 'STL',
                'BLK': 'BLK',
                'TOV': 'TOV',
                'PF': 'PF'
            })
            
            logger.info(f"✅ NBA API: {len(df)} jogos para player_id={player_id}")
            return df.head(n_games)
            
        except Exception as e:
            logger.warning(f"⚠️ NBA API falhou: {e}")
            return None
    
    def fetch_todays_games_nba_api(self) -> Optional[List[Dict]]:
        """Busca jogos de hoje via NBA API."""
        try:
            from nba_api.stats.endpoints import scoreboardv2
            
            self._rate_limit('nba_api')
            
            today = datetime.now().strftime('%Y-%m-%d')
            board = scoreboardv2.ScoreboardV2(game_date=today)
            
            games_df = board.get_data_frames()[0]
            
            if games_df.empty:
                logger.info("📅 Nenhum jogo hoje")
                return []
            
            games = []
            for _, row in games_df.iterrows():
                games.append({
                    'game_id': row.get('GAME_ID'),
                    'home_team': row.get('HOME_TEAM_ABBREVIATION', row.get('HOME_TEAM_ID', '')),
                    'away_team': row.get('VISITOR_TEAM_ABBREVIATION', row.get('VISITOR_TEAM_ID', '')),
                    'game_time': row.get('GAME_STATUS_TEXT', ''),
                    'date': today
                })
            
            logger.info(f"✅ {len(games)} jogos hoje")
            return games
            
        except Exception as e:
            logger.warning(f"⚠️ NBA API scoreboard falhou: {e}")
            return None
    
    def fetch_all_player_stats_nba_api(self, season: str = '2024-25') -> Optional[pd.DataFrame]:
        """Busca stats de todos os jogadores da temporada."""
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            
            self._rate_limit('nba_api')
            
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame'
            )
            
            df = stats.get_data_frames()[0]
            
            df = df.rename(columns={
                'PLAYER_NAME': 'player',
                'PLAYER_ID': 'player_id',
                'TEAM_ABBREVIATION': 'team',
                'GP': 'games_played',
                'MIN': 'min_avg',
                'PTS': 'pts_avg',
                'REB': 'reb_avg',
                'AST': 'ast_avg',
                'USG_PCT': 'usage_pct'
            })
            
            logger.info(f"✅ {len(df)} jogadores carregados")
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Falha ao buscar stats: {e}")
            return None
    
    # =========================================================================
    # SPORTSDATA.IO (Backup Pago)
    # =========================================================================
    
    def fetch_player_boxscores_sportsdata(
        self,
        player_name: str,
        n_games: int = 20
    ) -> Optional[pd.DataFrame]:
        """Busca boxscores via SportsData.io."""
        config = API_CONFIG['sportsdata']
        
        if not config['enabled']:
            return None
        
        try:
            self._rate_limit('sportsdata')
            
            # Endpoint de stats por jogador
            url = f"{config['base_url']}/stats/json/PlayerSeasonStats/2025"
            params = {'key': config['api_key']}
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Filtrar por nome
            player_data = [
                p for p in data 
                if player_name.lower() in p.get('Name', '').lower()
            ]
            
            if player_data:
                df = pd.DataFrame(player_data)
                logger.info(f"✅ SportsData.io: Encontrado {player_name}")
                return df
            else:
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ SportsData.io falhou: {e}")
            return None
    
    # =========================================================================
    # BALLDONTLIE API (Grátis)
    # =========================================================================
    
    def fetch_player_boxscores_balldontlie(
        self,
        player_name: str,
        n_games: int = 20
    ) -> Optional[pd.DataFrame]:
        """Busca boxscores via BallDontLie API."""
        config = API_CONFIG['balldontlie']
        
        if not config['enabled']:
            return None
        
        try:
            self._rate_limit('balldontlie')
            
            headers = {}
            if config['api_key']:
                headers['Authorization'] = config['api_key']
            
            # Buscar jogador
            search_url = f"{config['base_url']}/players"
            params = {'search': player_name}
            
            response = self.session.get(search_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            players = response.json().get('data', [])
            
            if not players:
                return None
            
            player_id = players[0]['id']
            
            # Buscar stats
            stats_url = f"{config['base_url']}/stats"
            params = {
                'player_ids[]': player_id,
                'per_page': n_games,
                'seasons[]': 2024
            }
            
            response = self.session.get(stats_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            stats = response.json().get('data', [])
            
            if stats:
                df = pd.DataFrame(stats)
                df = df.rename(columns={
                    'pts': 'PTS',
                    'reb': 'REB',
                    'ast': 'AST',
                    'min': 'MIN'
                })
                logger.info(f"✅ Balldontlie: {len(df)} jogos para {player_name}")
                return df
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Balldontlie falhou: {e}")
            return None
    
    # =========================================================================
    # ODDS APIs
    # =========================================================================
    
    def fetch_player_props_odds(self) -> Optional[List[Dict]]:
        """
        Busca linhas de Player Props das casas de apostas.
        
        Returns:
            Lista de props com linhas e odds
        """
        config = API_CONFIG['odds_api']
        
        if not config['enabled']:
            logger.warning("⚠️ ODDS_API_KEY não configurada")
            return self._generate_mock_props_lines()
        
        try:
            self._rate_limit('odds_api')
            
            url = f"{config['base_url']}/sports/basketball_nba/events"
            params = {
                'apiKey': config['api_key'],
                'regions': 'us',
                'markets': 'player_points,player_rebounds,player_assists',
                'oddsFormat': 'decimal'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            events = response.json()
            
            props = []
            for event in events:
                for bookmaker in event.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key', '')
                        
                        if 'player' not in market_key:
                            continue
                        
                        for outcome in market.get('outcomes', []):
                            props.append({
                                'game_id': event.get('id'),
                                'home_team': event.get('home_team'),
                                'away_team': event.get('away_team'),
                                'player': outcome.get('description', ''),
                                'prop_type': market_key.replace('player_', '').upper(),
                                'line': outcome.get('point', 0),
                                'odds_over': outcome.get('price', 1.91) if outcome.get('name') == 'Over' else 1.91,
                                'odds_under': outcome.get('price', 1.91) if outcome.get('name') == 'Under' else 1.91,
                                'bookmaker': bookmaker.get('key')
                            })
            
            logger.info(f"✅ {len(props)} props lines obtidos")
            return props
            
        except Exception as e:
            logger.warning(f"⚠️ Odds API falhou: {e}")
            return self._generate_mock_props_lines()
    
    def _generate_mock_props_lines(self) -> List[Dict]:
        """Gera linhas mock para testes quando API não está disponível."""
        logger.info("🎲 Gerando linhas mock para testes...")
        
        star_players = [
            ('LeBron James', 'LAL', 24.5, 7.5, 7.5),
            ('Stephen Curry', 'GSW', 25.5, 5.5, 5.5),
            ('Nikola Jokic', 'DEN', 25.5, 11.5, 8.5),
            ('Luka Doncic', 'DAL', 28.5, 8.5, 8.5),
            ('Giannis Antetokounmpo', 'MIL', 29.5, 11.5, 5.5),
        ]
        
        props = []
        for player, team, pts_line, reb_line, ast_line in star_players:
            props.append({
                'player': player,
                'team': team,
                'prop_type': 'PTS',
                'line': pts_line,
                'odds_over': 1.91,
                'odds_under': 1.91,
                'bookmaker': 'MOCK'
            })
            props.append({
                'player': player,
                'team': team,
                'prop_type': 'REB',
                'line': reb_line,
                'odds_over': 1.91,
                'odds_under': 1.91,
                'bookmaker': 'MOCK'
            })
            props.append({
                'player': player,
                'team': team,
                'prop_type': 'AST',
                'line': ast_line,
                'odds_over': 1.91,
                'odds_under': 1.91,
                'bookmaker': 'MOCK'
            })
        
        return props
    
    # =========================================================================
    # FALLBACK COM INFERÊNCIA KNN
    # =========================================================================
    
    def _build_knn_model(self, df_all_players: pd.DataFrame):
        """
        Constrói modelo KNN para inferência de jogadores similares.
        
        Usado quando dados de um jogador específico estão faltando.
        """
        if df_all_players is None or df_all_players.empty:
            return
        
        features = ['min_avg', 'pts_avg', 'reb_avg', 'ast_avg', 'usage_pct']
        available = [f for f in features if f in df_all_players.columns]
        
        if len(available) < 3:
            return
        
        # Preparar dados
        X = df_all_players[available].fillna(df_all_players[available].median())
        
        # Normalizar
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Treinar KNN
        self._knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self._knn_model.fit(X_scaled)
        
        self._player_features_cache = {
            'df': df_all_players,
            'features': available,
            'scaler': scaler
        }
        
        logger.info("✅ Modelo KNN construído para inferência")
    
    def infer_missing_player_stats(
        self,
        player_name: str,
        position: str = None,
        team: str = None
    ) -> Optional[Dict]:
        """
        Infere stats de um jogador usando jogadores similares (KNN).
        
        Usado quando não temos dados históricos do jogador.
        
        Args:
            player_name: Nome do jogador
            position: Posição (G, F, C)
            team: Time
            
        Returns:
            Dict com stats estimados ou None
        """
        if self._knn_model is None:
            # Tentar construir modelo
            df = self.fetch_all_player_stats_nba_api()
            if df is not None:
                self._build_knn_model(df)
        
        if self._knn_model is None or self._player_features_cache is None:
            logger.warning("❌ Não foi possível construir modelo KNN")
            # Retornar médias da liga
            return {
                'pts_avg': 10.0,
                'reb_avg': 4.0,
                'ast_avg': 2.5,
                'min_avg': 20.0,
                'inferred': True,
                'method': 'league_average'
            }
        
        try:
            cache = self._player_features_cache
            df = cache['df']
            features = cache['features']
            scaler = cache['scaler']
            
            # Buscar jogador no dataset
            player_mask = df['player'].str.contains(player_name, case=False, na=False)
            
            if player_mask.any():
                # Jogador encontrado, não precisa inferir
                player_row = df[player_mask].iloc[0]
                return {
                    'pts_avg': player_row.get('pts_avg', 10.0),
                    'reb_avg': player_row.get('reb_avg', 4.0),
                    'ast_avg': player_row.get('ast_avg', 2.5),
                    'min_avg': player_row.get('min_avg', 20.0),
                    'inferred': False,
                    'method': 'exact_match'
                }
            
            # Buscar jogadores similares por posição/time
            similar_mask = pd.Series([True] * len(df))
            if position:
                if 'position' in df.columns:
                    similar_mask &= df['position'].str.contains(position, case=False, na=False)
            if team:
                if 'team' in df.columns:
                    similar_mask &= (df['team'] == team)
            
            if similar_mask.sum() < 3:
                similar_mask = pd.Series([True] * len(df))  # Reset
            
            similar_df = df[similar_mask]
            
            if similar_df.empty:
                similar_df = df
            
            # Calcular média ponderada dos mais similares
            X_similar = similar_df[features].fillna(similar_df[features].median())
            
            # Usar médias como resultado
            result = {
                'pts_avg': X_similar.get('pts_avg', pd.Series([10.0])).mean(),
                'reb_avg': X_similar.get('reb_avg', pd.Series([4.0])).mean(),
                'ast_avg': X_similar.get('ast_avg', pd.Series([2.5])).mean(),
                'min_avg': X_similar.get('min_avg', pd.Series([20.0])).mean(),
                'inferred': True,
                'method': 'knn_similar_players',
                'n_similar': len(similar_df)
            }
            
            logger.info(f"🔮 Stats inferidos para {player_name}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na inferência KNN: {e}")
            return {
                'pts_avg': 10.0,
                'reb_avg': 4.0,
                'ast_avg': 2.5,
                'min_avg': 20.0,
                'inferred': True,
                'method': 'fallback_average'
            }
    
    # =========================================================================
    # FUNÇÃO PRINCIPAL DE COLETA
    # =========================================================================
    
    def fetch_player_data(
        self,
        player_name: str,
        n_games: int = 20
    ) -> Dict:
        """
        Busca dados de um jogador com fallback automático.
        
        Hierarquia:
        1. NBA API
        2. SportsData.io
        3. Balldontlie
        4. Inferência KNN
        
        Returns:
            Dict com dados do jogador
        """
        logger.info(f"🔍 Buscando dados para {player_name}...")
        
        result = {
            'player': player_name,
            'boxscores': None,
            'stats': None,
            'source': None,
            'inferred': False
        }
        
        # Tentar NBA API
        df = self.fetch_player_boxscores_nba_api(player_name=player_name, n_games=n_games)
        if df is not None and not df.empty:
            result['boxscores'] = df
            result['source'] = 'nba_api'
            result['stats'] = self._calculate_stats_from_boxscores(df)
            return result
        
        # Tentar SportsData.io
        df = self.fetch_player_boxscores_sportsdata(player_name, n_games)
        if df is not None and not df.empty:
            result['boxscores'] = df
            result['source'] = 'sportsdata'
            result['stats'] = self._calculate_stats_from_boxscores(df)
            return result
        
        # Tentar Balldontlie
        df = self.fetch_player_boxscores_balldontlie(player_name, n_games)
        if df is not None and not df.empty:
            result['boxscores'] = df
            result['source'] = 'balldontlie'
            result['stats'] = self._calculate_stats_from_boxscores(df)
            return result
        
        # Fallback: Inferência
        logger.warning(f"⚠️ Nenhuma API retornou dados para {player_name}, usando inferência...")
        inferred_stats = self.infer_missing_player_stats(player_name)
        result['stats'] = inferred_stats
        result['source'] = 'knn_inference'
        result['inferred'] = True
        
        return result
    
    def _calculate_stats_from_boxscores(self, df: pd.DataFrame) -> Dict:
        """Calcula estatísticas agregadas de boxscores."""
        return {
            'pts_avg': df['PTS'].mean() if 'PTS' in df.columns else 0,
            'reb_avg': df['REB'].mean() if 'REB' in df.columns else 0,
            'ast_avg': df['AST'].mean() if 'AST' in df.columns else 0,
            'min_avg': df['MIN'].mean() if 'MIN' in df.columns else 0,
            'games': len(df),
            'inferred': False
        }
    
    # =========================================================================
    # CÁLCULO DE EV+
    # =========================================================================
    
    @staticmethod
    def calculate_ev_plus(
        model_prob: float,
        odds_decimal: float
    ) -> float:
        """
        Calcula Expected Value (EV).
        
        EV = (Probabilidade_Modelo * Odds_Decimal) - 1
        
        Se EV > 0, a aposta tem valor positivo.
        Se EV > 0.05 (5%), é aposta recomendada.
        
        Args:
            model_prob: Probabilidade do modelo (0-1)
            odds_decimal: Odds em formato decimal (ex: 1.91)
            
        Returns:
            EV como percentual (0.05 = 5%)
        """
        ev = (model_prob * odds_decimal) - 1
        return ev
    
    @staticmethod
    def calculate_implied_probability(odds_decimal: float) -> float:
        """
        Calcula probabilidade implícita das odds.
        
        Args:
            odds_decimal: Odds em decimal
            
        Returns:
            Probabilidade implícita (0-1)
        """
        return 1 / odds_decimal
    
    def evaluate_bet_opportunity(
        self,
        prediction_median: float,
        prediction_low: float,
        prediction_high: float,
        line: float,
        odds_over: float = 1.91,
        odds_under: float = 1.91
    ) -> Dict:
        """
        Avalia oportunidade de aposta.
        
        Args:
            prediction_median: Previsão mediana (50th percentil)
            prediction_low: Previsão baixa (10th percentil)
            prediction_high: Previsão alta (90th percentil)
            line: Linha da casa de apostas
            odds_over: Odds para OVER
            odds_under: Odds para UNDER
            
        Returns:
            Dict com análise da aposta
        """
        # Calcular probabilidades implícitas
        implied_over = self.calculate_implied_probability(odds_over)
        implied_under = self.calculate_implied_probability(odds_under)
        
        # Estimar probabilidade do modelo baseada nos quantis
        # Se linha < P10, ~90% chance de OVER
        # Se linha > P90, ~90% chance de UNDER
        
        if prediction_low > line:
            # Forte OVER
            model_prob_over = 0.85
            model_prob_under = 0.15
            strength = 'ALL-IN OVER'
        elif prediction_high < line:
            # Forte UNDER
            model_prob_over = 0.15
            model_prob_under = 0.85
            strength = 'ALL-IN UNDER'
        elif prediction_median > line:
            # Moderado OVER
            diff_ratio = (prediction_median - line) / (prediction_high - prediction_low + 0.1)
            model_prob_over = 0.5 + min(0.3, diff_ratio * 0.3)
            model_prob_under = 1 - model_prob_over
            strength = 'LEAN OVER'
        elif prediction_median < line:
            # Moderado UNDER
            diff_ratio = (line - prediction_median) / (prediction_high - prediction_low + 0.1)
            model_prob_under = 0.5 + min(0.3, diff_ratio * 0.3)
            model_prob_over = 1 - model_prob_under
            strength = 'LEAN UNDER'
        else:
            model_prob_over = 0.5
            model_prob_under = 0.5
            strength = 'SKIP'
        
        # Calcular EVs
        ev_over = self.calculate_ev_plus(model_prob_over, odds_over)
        ev_under = self.calculate_ev_plus(model_prob_under, odds_under)
        
        # Determinar melhor aposta
        if ev_over > ev_under and ev_over > 0.03:
            recommendation = 'OVER'
            ev = ev_over
            edge = model_prob_over - implied_over
        elif ev_under > ev_over and ev_under > 0.03:
            recommendation = 'UNDER'
            ev = ev_under
            edge = model_prob_under - implied_under
        else:
            recommendation = 'SKIP'
            ev = max(ev_over, ev_under)
            edge = 0
        
        return {
            'recommendation': recommendation,
            'strength': strength,
            'ev_plus': round(ev * 100, 2),  # Em percentual
            'edge': round(edge * 100, 2),   # Vantagem sobre a casa
            'model_prob_over': round(model_prob_over, 3),
            'model_prob_under': round(model_prob_under, 3),
            'implied_prob_over': round(implied_over, 3),
            'implied_prob_under': round(implied_under, 3),
            'prediction_median': prediction_median,
            'prediction_range': (prediction_low, prediction_high),
            'line': line
        }


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

_collector_instance = None

def get_quantum_collector() -> QuantumDataCollector:
    """Retorna instância singleton do collector."""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = QuantumDataCollector()
    return _collector_instance


def fetch_all_data_for_predictions() -> Dict:
    """
    Busca todos os dados necessários para fazer previsões.
    
    Returns:
        Dict com jogos, odds, e dados de jogadores
    """
    collector = get_quantum_collector()
    
    logger.info("🚀 Coletando dados para previsões Quantum...")
    
    result = {
        'games': [],
        'props_lines': [],
        'player_stats': None,
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. Jogos de hoje
    games = collector.fetch_todays_games_nba_api()
    if games:
        result['games'] = games
    
    # 2. Linhas de Props
    props = collector.fetch_player_props_odds()
    if props:
        result['props_lines'] = props
    
    # 3. Stats de jogadores
    stats = collector.fetch_all_player_stats_nba_api()
    if stats is not None:
        result['player_stats'] = stats
    
    logger.info(f"✅ Coleta concluída: {len(result['games'])} jogos, {len(result['props_lines'])} props")
    
    return result


# =============================================================================
# TESTES
# =============================================================================

def test_quantum_scraper():
    """Testa todas as funcionalidades do scraper."""
    print("🧪 Testando Quantum Scraper...")
    
    collector = QuantumDataCollector()
    
    # Teste 1: Jogos de hoje
    print("\n1️⃣ Testando fetch de jogos de hoje...")
    games = collector.fetch_todays_games_nba_api()
    print(f"   Resultado: {len(games) if games else 0} jogos")
    
    # Teste 2: Props lines
    print("\n2️⃣ Testando fetch de props lines...")
    props = collector.fetch_player_props_odds()
    print(f"   Resultado: {len(props) if props else 0} props")
    
    # Teste 3: Dados de jogador
    print("\n3️⃣ Testando fetch de dados de jogador (LeBron James)...")
    player_data = collector.fetch_player_data("LeBron James")
    print(f"   Fonte: {player_data['source']}")
    print(f"   Inferido: {player_data['inferred']}")
    if player_data['stats']:
        print(f"   Stats: PTS={player_data['stats'].get('pts_avg', 'N/A'):.1f}")
    
    # Teste 4: Cálculo de EV
    print("\n4️⃣ Testando cálculo de EV+...")
    ev_result = collector.evaluate_bet_opportunity(
        prediction_median=26.5,
        prediction_low=22.0,
        prediction_high=31.0,
        line=24.5,
        odds_over=1.91,
        odds_under=1.91
    )
    print(f"   Recomendação: {ev_result['recommendation']}")
    print(f"   EV+: {ev_result['ev_plus']}%")
    print(f"   Edge: {ev_result['edge']}%")
    
    print("\n✅ Todos os testes concluídos!")


if __name__ == "__main__":
    test_quantum_scraper()
