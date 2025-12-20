"""
NBA Stats Client - Interface moderna para nba_api

Este módulo fornece acesso profissional aos dados da NBA usando a biblioteca nba_api.
Implementa cache local (SQLite/Parquet) e rate limiting respeitoso.

v27.0: Implementação inicial com:
- Box Scores históricos
- Standings (classificação)
- Team Game Logs
- Cache SQLite com TTL
- Rate limiting (3s entre requests)
- Compatibilidade asyncio

Uso:
    from data.ingestion.stats_client import NBAStatsClient
    
    client = NBAStatsClient()
    box_score = await client.get_box_scores("0022400123")
    standings = await client.get_standings("2024-25")
"""

import asyncio
import logging
import sqlite3
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import wraps

import pandas as pd

# NBA API imports com fallback
try:
    from nba_api.stats.endpoints import (
        boxscoretraditionalv2,
        leaguestandings,
        teamgamelog,
        playergamelog,
        commonteamroster,
        teamdetails
    )
    from nba_api.stats.static import teams, players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configurações
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DB = CACHE_DIR / "nba_stats.db"
RATE_LIMIT_SECONDS = 3.0
CACHE_TTL_HOURS = 24  # Box scores expiram em 24h
STANDINGS_TTL_HOURS = 6  # Standings mais voláteis


def rate_limited(func):
    """Decorator para rate limiting entre chamadas à API."""
    last_call = [0.0]
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        elapsed = time.time() - last_call[0]
        if elapsed < RATE_LIMIT_SECONDS:
            await asyncio.sleep(RATE_LIMIT_SECONDS - elapsed)
        result = await func(*args, **kwargs)
        last_call[0] = time.time()
        return result
    
    return wrapper


class NBAStatsClient:
    """
    Cliente moderno para a NBA Stats API.
    
    Características:
    - Cache SQLite local para evitar requests repetidos
    - Rate limiting respeitoso (3s entre chamadas)
    - Compatibilidade com asyncio
    - Fallback gracioso quando API indisponível
    
    Exemplo:
        >>> client = NBAStatsClient()
        >>> standings = await client.get_standings("2024-25")
        >>> box = await client.get_box_scores("0022400123")
    """
    
    def __init__(self, cache_db: Path = None):
        """
        Inicializa o cliente.
        
        Args:
            cache_db: Caminho para o banco SQLite de cache (opcional)
        """
        self.cache_db = cache_db or CACHE_DB
        self._ensure_cache_dir()
        self._init_cache_db()
        
        if not NBA_API_AVAILABLE:
            logger.warning("⚠️ nba_api não instalada. Instale com: pip install nba_api")
    
    def _ensure_cache_dir(self):
        """Garante que o diretório de cache existe."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def _init_cache_db(self):
        """Inicializa o banco SQLite de cache."""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stats_cache (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ttl_hours INTEGER DEFAULT 24
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON stats_cache(created_at)
            """)
            conn.commit()
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Busca dados do cache se ainda válidos.
        
        Args:
            key: Chave única para o item
            
        Returns:
            Dados cacheados ou None se expirado/inexistente
        """
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    """
                    SELECT data, created_at, ttl_hours 
                    FROM stats_cache 
                    WHERE key = ?
                    """, 
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    data, created_at, ttl_hours = row
                    created = datetime.fromisoformat(created_at)
                    if datetime.now() - created < timedelta(hours=ttl_hours):
                        logger.debug(f"📦 Cache hit: {key}")
                        return json.loads(data)
                    else:
                        # Cache expirado, limpar
                        conn.execute("DELETE FROM stats_cache WHERE key = ?", (key,))
                        conn.commit()
        except Exception as e:
            logger.warning(f"⚠️ Erro lendo cache: {e}")
        
        return None
    
    def _save_to_cache(self, key: str, data: Dict[str, Any], ttl_hours: int = 24):
        """
        Salva dados no cache.
        
        Args:
            key: Chave única para o item
            data: Dados a cachear (serializáveis em JSON)
            ttl_hours: Tempo de vida em horas
        """
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO stats_cache (key, data, created_at, ttl_hours)
                    VALUES (?, ?, ?, ?)
                    """,
                    (key, json.dumps(data, default=str), datetime.now().isoformat(), ttl_hours)
                )
                conn.commit()
                logger.debug(f"💾 Cached: {key}")
        except Exception as e:
            logger.warning(f"⚠️ Erro salvando cache: {e}")
    
    def clear_expired_cache(self):
        """Remove entradas expiradas do cache."""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute("""
                    DELETE FROM stats_cache 
                    WHERE datetime(created_at, '+' || ttl_hours || ' hours') < datetime('now')
                """)
                deleted = conn.total_changes
                conn.commit()
                if deleted > 0:
                    logger.info(f"🧹 Cache limpo: {deleted} entradas expiradas removidas")
        except Exception as e:
            logger.warning(f"⚠️ Erro limpando cache: {e}")
    
    # =========================================================================
    # MÉTODOS DE DADOS
    # =========================================================================
    
    @rate_limited
    async def get_box_scores(self, game_id: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Obtém box score detalhado de um jogo.
        
        Args:
            game_id: ID do jogo NBA (ex: "0022400123")
            
        Returns:
            Dict com DataFrames:
            - 'player_stats': Estatísticas por jogador
            - 'team_stats': Estatísticas por time
            
        Exemplo:
            >>> box = await client.get_box_scores("0022400123")
            >>> print(box['player_stats'].head())
        """
        if not NBA_API_AVAILABLE:
            logger.error("❌ nba_api não disponível")
            return None
        
        cache_key = f"box_score_{game_id}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return {
                'player_stats': pd.DataFrame(cached.get('player_stats', [])),
                'team_stats': pd.DataFrame(cached.get('team_stats', []))
            }
        
        try:
            logger.info(f"🏀 Buscando box score: {game_id}")
            
            # Executa em thread separada para não bloquear asyncio
            loop = asyncio.get_event_loop()
            box = await loop.run_in_executor(
                None,
                lambda: boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            )
            
            player_stats = box.player_stats.get_data_frame()
            team_stats = box.team_stats.get_data_frame()
            
            # Cache os resultados
            cache_data = {
                'player_stats': player_stats.to_dict(orient='records'),
                'team_stats': team_stats.to_dict(orient='records')
            }
            self._save_to_cache(cache_key, cache_data, ttl_hours=CACHE_TTL_HOURS)
            
            return {
                'player_stats': player_stats,
                'team_stats': team_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Erro buscando box score {game_id}: {e}")
            return None
    
    @rate_limited
    async def get_standings(self, season: str = "2024-25") -> Optional[pd.DataFrame]:
        """
        Obtém tabela de classificação da temporada.
        
        Args:
            season: Temporada no formato "YYYY-YY" (ex: "2024-25")
            
        Returns:
            DataFrame com colunas:
            - TeamID, TeamCity, TeamName, TeamSlug
            - Conference, ConferenceRecord
            - PlayoffRank, Division
            - Record (W-L), WinPCT
            - HOME, ROAD, L10, strCurrentStreak
            
        Exemplo:
            >>> standings = await client.get_standings("2024-25")
            >>> print(standings[['TeamName', 'Record', 'WinPCT']])
        """
        if not NBA_API_AVAILABLE:
            logger.error("❌ nba_api não disponível")
            return None
        
        cache_key = f"standings_{season}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)
        
        try:
            logger.info(f"📊 Buscando standings: {season}")
            
            loop = asyncio.get_event_loop()
            standings = await loop.run_in_executor(
                None,
                lambda: leaguestandings.LeagueStandings(
                    season=season,
                    league_id="00"  # NBA
                )
            )
            
            df = standings.get_data_frames()[0]
            
            # Cache com TTL menor (standings mudam frequentemente)
            self._save_to_cache(cache_key, df.to_dict(orient='records'), 
                               ttl_hours=STANDINGS_TTL_HOURS)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro buscando standings {season}: {e}")
            return None
    
    @rate_limited
    async def get_team_game_log(
        self, 
        team_id: int, 
        season: str = "2024-25",
        season_type: str = "Regular Season"
    ) -> Optional[pd.DataFrame]:
        """
        Obtém histórico de jogos de um time.
        
        Args:
            team_id: ID do time NBA (ex: 1610612747 para Lakers)
            season: Temporada no formato "YYYY-YY"
            season_type: "Regular Season", "Playoffs", etc.
            
        Returns:
            DataFrame com colunas:
            - Game_ID, GAME_DATE, MATCHUP
            - WL (Win/Loss), W, L
            - PTS, FGM, FGA, FG_PCT
            - FG3M, FG3A, FG3_PCT
            - REB, AST, STL, BLK, TOV
            - PLUS_MINUS
            
        Exemplo:
            >>> lakers_id = 1610612747
            >>> log = await client.get_team_game_log(lakers_id, "2024-25")
            >>> print(log[['GAME_DATE', 'MATCHUP', 'WL', 'PTS']])
        """
        if not NBA_API_AVAILABLE:
            logger.error("❌ nba_api não disponível")
            return None
        
        cache_key = f"team_log_{team_id}_{season}_{season_type.replace(' ', '_')}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)
        
        try:
            logger.info(f"📋 Buscando game log: time {team_id}, {season}")
            
            loop = asyncio.get_event_loop()
            log = await loop.run_in_executor(
                None,
                lambda: teamgamelog.TeamGameLog(
                    team_id=team_id,
                    season=season,
                    season_type_all_star=season_type
                )
            )
            
            df = log.get_data_frames()[0]
            
            self._save_to_cache(cache_key, df.to_dict(orient='records'), 
                               ttl_hours=CACHE_TTL_HOURS)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro buscando game log {team_id}: {e}")
            return None
    
    @rate_limited
    async def get_player_game_log(
        self,
        player_id: int,
        season: str = "2024-25",
        season_type: str = "Regular Season"
    ) -> Optional[pd.DataFrame]:
        """
        Obtém histórico de jogos de um jogador.
        
        Args:
            player_id: ID do jogador NBA
            season: Temporada no formato "YYYY-YY"
            season_type: "Regular Season", "Playoffs", etc.
            
        Returns:
            DataFrame com estatísticas por jogo
        """
        if not NBA_API_AVAILABLE:
            logger.error("❌ nba_api não disponível")
            return None
        
        cache_key = f"player_log_{player_id}_{season}_{season_type.replace(' ', '_')}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)
        
        try:
            logger.info(f"🏃 Buscando player log: {player_id}, {season}")
            
            loop = asyncio.get_event_loop()
            log = await loop.run_in_executor(
                None,
                lambda: playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star=season_type
                )
            )
            
            df = log.get_data_frames()[0]
            
            self._save_to_cache(cache_key, df.to_dict(orient='records'),
                               ttl_hours=CACHE_TTL_HOURS)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro buscando player log {player_id}: {e}")
            return None
    
    @rate_limited
    async def get_team_roster(
        self,
        team_id: int,
        season: str = "2024-25"
    ) -> Optional[pd.DataFrame]:
        """
        Obtém elenco atual de um time.
        
        Args:
            team_id: ID do time NBA
            season: Temporada
            
        Returns:
            DataFrame com jogadores do elenco
        """
        if not NBA_API_AVAILABLE:
            logger.error("❌ nba_api não disponível")
            return None
        
        cache_key = f"roster_{team_id}_{season}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)
        
        try:
            logger.info(f"👥 Buscando roster: time {team_id}, {season}")
            
            loop = asyncio.get_event_loop()
            roster = await loop.run_in_executor(
                None,
                lambda: commonteamroster.CommonTeamRoster(
                    team_id=team_id,
                    season=season
                )
            )
            
            df = roster.get_data_frames()[0]
            
            self._save_to_cache(cache_key, df.to_dict(orient='records'),
                               ttl_hours=CACHE_TTL_HOURS)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro buscando roster {team_id}: {e}")
            return None
    
    # =========================================================================
    # MÉTODOS AUXILIARES
    # =========================================================================
    
    def get_all_teams(self) -> List[Dict[str, Any]]:
        """
        Retorna lista de todos os times NBA.
        
        Returns:
            Lista de dicts com id, full_name, abbreviation, etc.
        """
        if not NBA_API_AVAILABLE:
            return []
        
        return teams.get_teams()
    
    def get_team_id(self, team_name: str) -> Optional[int]:
        """
        Busca ID de um time pelo nome ou abreviação.
        
        Args:
            team_name: Nome completo ou abreviação (ex: "Lakers" ou "LAL")
            
        Returns:
            ID do time ou None se não encontrado
        """
        if not NBA_API_AVAILABLE:
            return None
        
        all_teams = self.get_all_teams()
        team_name_lower = team_name.lower()
        
        for team in all_teams:
            if (team_name_lower in team['full_name'].lower() or
                team_name_lower == team['abbreviation'].lower() or
                team_name_lower == team['nickname'].lower()):
                return team['id']
        
        return None
    
    def get_all_players(self, only_active: bool = True) -> List[Dict[str, Any]]:
        """
        Retorna lista de jogadores.
        
        Args:
            only_active: Se True, retorna apenas jogadores ativos
            
        Returns:
            Lista de dicts com id, full_name, etc.
        """
        if not NBA_API_AVAILABLE:
            return []
        
        if only_active:
            return players.get_active_players()
        return players.get_players()
    
    def get_player_id(self, player_name: str) -> Optional[int]:
        """
        Busca ID de um jogador pelo nome.
        
        Args:
            player_name: Nome do jogador (ex: "LeBron James")
            
        Returns:
            ID do jogador ou None
        """
        if not NBA_API_AVAILABLE:
            return None
        
        all_players = self.get_all_players()
        player_name_lower = player_name.lower()
        
        for player in all_players:
            if player_name_lower in player['full_name'].lower():
                return player['id']
        
        return None
    
    async def get_yesterday_games(self) -> Optional[pd.DataFrame]:
        """
        Obtém jogos de ontem para atualização diária.
        
        Returns:
            DataFrame com jogos de ontem ou None
        """
        # Implementação simplificada - busca standings para identificar jogos recentes
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"📅 Buscando jogos de {yesterday}")
        
        # Para implementação completa, usar scoreboard endpoint
        # Por ora, retorna None e o sistema usa fallback
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verifica saúde do cliente.
        
        Returns:
            Dict com status e métricas
        """
        status = {
            "nba_api_available": NBA_API_AVAILABLE,
            "cache_db_exists": self.cache_db.exists(),
            "cache_dir": str(CACHE_DIR),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.cache_db.exists():
            try:
                with sqlite3.connect(self.cache_db) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM stats_cache")
                    status["cached_items"] = cursor.fetchone()[0]
            except Exception as e:
                status["cache_error"] = str(e)
        
        return status


# =============================================================================
# FUNÇÕES AUXILIARES PARA USO SÍNCRONO
# =============================================================================

def get_standings_sync(season: str = "2024-25") -> Optional[pd.DataFrame]:
    """Versão síncrona de get_standings para uso em scripts."""
    client = NBAStatsClient()
    return asyncio.run(client.get_standings(season))


def get_box_score_sync(game_id: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Versão síncrona de get_box_scores para uso em scripts."""
    client = NBAStatsClient()
    return asyncio.run(client.get_box_scores(game_id))


def get_team_log_sync(
    team_id: int, 
    season: str = "2024-25"
) -> Optional[pd.DataFrame]:
    """Versão síncrona de get_team_game_log para uso em scripts."""
    client = NBAStatsClient()
    return asyncio.run(client.get_team_game_log(team_id, season))


# =============================================================================
# CLI PARA TESTES
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        client = NBAStatsClient()
        
        print("\n" + "="*60)
        print("🏀 NBA Stats Client - Teste")
        print("="*60)
        
        # Health check
        health = client.health_check()
        print(f"\n📊 Health Check:")
        for k, v in health.items():
            print(f"   {k}: {v}")
        
        # Listar times
        teams_list = client.get_all_teams()
        print(f"\n🏟️ Times NBA: {len(teams_list)}")
        
        # Buscar standings
        print("\n📊 Buscando standings 2024-25...")
        standings = await client.get_standings("2024-25")
        if standings is not None:
            print(standings[['TeamName', 'Record', 'WinPCT']].head(10).to_string())
        else:
            print("   ⚠️ Não foi possível buscar standings")
        
        # Buscar game log de um time
        lakers_id = client.get_team_id("Lakers")
        if lakers_id:
            print(f"\n📋 Game log Lakers (ID: {lakers_id})...")
            log = await client.get_team_game_log(lakers_id, "2024-25")
            if log is not None:
                print(log[['GAME_DATE', 'MATCHUP', 'WL', 'PTS']].head(5).to_string())
        
        print("\n✅ Teste concluído!")
    
    asyncio.run(main())
