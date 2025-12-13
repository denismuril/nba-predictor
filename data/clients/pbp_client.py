"""
PBPStats Client - Cliente para Métricas Limpas (Sem Garbage Time)

Usa a biblioteca pbpstats para obter estatísticas filtradas por momentos
competitivos, excluindo "Garbage Time" (finais de jogos já decididos).

Documentação: https://pbpstats.readthedocs.io/
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class PBPClient:
    """
    Cliente para acessar dados de posse da NBA via pbpstats.
    
    Filtra automaticamente Garbage Time para obter métricas "limpas"
    como OffRtg, DefRtg e Pace que representam apenas minutos competitivos.
    
    Attributes:
        cache_dir: Diretório para salvar JSONs localmente (evita rate limiting)
        settings: Configuração do cliente pbpstats
    
    Example:
        >>> client = PBPClient()
        >>> df = client.get_clean_stats("2024-25")
        >>> print(df[['team_id', 'off_rtg', 'def_rtg', 'pace']].head())
    """
    
    def __init__(self, cache_dir: str = "data/cache/pbp"):
        """
        Inicializa o cliente PBPStats.
        
        Args:
            cache_dir: Diretório para cache local de JSONs. Recomendado para
                       evitar bloqueios por rate limiting (erro 429).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar subdiretórios necessários para o pbpstats
        # A biblioteca às vezes falha em criar recursivamente
        for subdir in ["schedule", "games", "possessions", "boxscore"]:
            (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Configuração do cliente pbpstats
        self.settings = {
            "dir": str(self.cache_dir),
            "Possessions": {"source": "web", "data_provider": "stats_nba"},
            "Boxscore": {"source": "web", "data_provider": "stats_nba"},
            "Games": {"source": "web", "data_provider": "data_nba"},
        }
        
        self._client = None
        logger.info(f"📊 PBPClient inicializado com cache em: {self.cache_dir}")
    
    @property
    def client(self):
        """Lazy loading do client pbpstats."""
        if self._client is None:
            try:
                from pbpstats.client import Client
                self._client = Client(self.settings)
            except ImportError:
                logger.error("❌ pbpstats não instalado. Execute: pip install pbpstats")
                raise
        return self._client
    
    def _with_retry(self, func, *args, **kwargs):
        """
        Wrapper com Exponential Backoff para chamadas à API.
        
        Usa tenacity se disponível, fallback para retry manual caso contrário.
        Evita erros 429 (Too Many Requests) da API da NBA.
        """
        try:
            from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
            import requests
            
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=30),
                retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
                reraise=True
            )
            def _call():
                return func(*args, **kwargs)
            
            return _call()
            
        except ImportError:
            # Fallback manual se tenacity não estiver disponível
            import time
            for attempt in range(3):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == 2:
                        raise
                    wait_time = 2 ** (attempt + 1)  # 2, 4, 8 segundos
                    logger.warning(f"⚠️ Tentativa {attempt + 1}/3 falhou: {e}. Aguardando {wait_time}s...")
                    time.sleep(wait_time)
    
    def get_clean_stats(self, season_year: str) -> pd.DataFrame:
        """
        Obtém estatísticas limpas (sem Garbage Time) para uma temporada.
        
        Garbage Time é definido como: últimos 5 minutos com diferença > 15 pontos.
        Filtramos essas possessões para obter métricas mais representativas.
        
        Args:
            season_year: Temporada no formato "2024-25" ou "2023-24"
            
        Returns:
            DataFrame com colunas:
            - game_id: ID do jogo NBA
            - team_id: ID do time
            - team_abbrev: Sigla do time (ex: BOS, LAL)
            - off_rtg: Offensive Rating (pts/100 posses) sem Garbage Time
            - def_rtg: Defensive Rating (pts/100 posses) sem Garbage Time
            - pace: Pace (posses/48min) sem Garbage Time
            - possessions: Número de posses competitivas
            
        Raises:
            Exception: Se falhar após 3 tentativas de retry
        """
        logger.info(f"📊 Buscando dados PBPStats para temporada {season_year}...")
        
        records = []
        
        try:
            # Buscar todos os jogos finalizados da temporada
            def _get_season():
                season = self.client.Season("nba", season_year, "Regular Season")
                return list(season.games.final_games)
            
            final_games = self._with_retry(_get_season)
            logger.info(f"   Encontrados {len(final_games)} jogos finalizados")
            
            for game_info in final_games[:10]:  # Limitar para teste inicial
                try:
                    game_id = game_info['game_id']
                    game_data = self._process_game(game_id)
                    if game_data:
                        records.extend(game_data)
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao processar jogo {game_info.get('game_id', 'unknown')}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Erro ao buscar temporada {season_year}: {e}")
            raise
        
        if not records:
            logger.warning("⚠️ Nenhum dado de posse encontrado. Retornando DataFrame vazio.")
            return pd.DataFrame(columns=['game_id', 'team_id', 'team_abbrev', 'off_rtg', 'def_rtg', 'pace', 'possessions'])
        
        df = pd.DataFrame(records)
        logger.info(f"✅ Processados {len(df)} registros de {df['game_id'].nunique()} jogos")
        
        return df
    
    def _process_game(self, game_id: str) -> List[Dict[str, Any]]:
        """
        Processa um jogo individual, filtrando Garbage Time.
        
        Garbage Time: Últimos 5 minutos do 4º período (ou OT) com margem > 15 pontos.
        
        Args:
            game_id: ID do jogo no formato NBA (ex: "0022400123")
            
        Returns:
            Lista de dicts com stats por time (home/away)
        """
        try:
            # Configurar para buscar possessions
            settings_game = {
                "dir": str(self.cache_dir),
                "Possessions": {"source": "web", "data_provider": "stats_nba"},
            }
            
            from pbpstats.client import Client
            client = Client(settings_game)
            
            def _get_game():
                return client.Game(game_id)
            
            game = self._with_retry(_get_game)
            
            # Filtrar possessions sem Garbage Time
            clean_possessions = []
            for poss in game.possessions.items:
                # Filtrar Garbage Time: últimos 5 min, margem > 15
                if hasattr(poss, 'period') and hasattr(poss, 'time') and hasattr(poss, 'score_margin'):
                    period = poss.period
                    time_remaining = poss.time if hasattr(poss, 'time') else 0
                    margin = abs(poss.score_margin) if hasattr(poss, 'score_margin') else 0
                    
                    # Garbage Time: Q4 (ou OT), < 5min restantes, margem > 15
                    is_garbage = (period >= 4 and time_remaining < 300 and margin > 15)
                    
                    if not is_garbage:
                        clean_possessions.append(poss)
                else:
                    # Se não tiver atributos, incluir por padrão
                    clean_possessions.append(poss)
            
            # Calcular métricas por time
            team_stats = self._calculate_team_stats(game_id, clean_possessions)
            
            return team_stats
            
        except Exception as e:
            logger.debug(f"Erro ao processar jogo {game_id}: {e}")
            return []
    
    def _calculate_team_stats(self, game_id: str, possessions: list) -> List[Dict[str, Any]]:
        """
        Calcula OffRtg, DefRtg e Pace a partir das posses filtradas.
        
        Args:
            game_id: ID do jogo
            possessions: Lista de posses (já filtradas)
            
        Returns:
            Lista com dict de stats para cada time
        """
        if not possessions:
            return []
        
        # Agrupar por time ofensivo
        team_offense = {}  # team_id -> {pts: X, poss: N}
        team_defense = {}  # team_id -> {pts_allowed: X, poss: N}
        
        for poss in possessions:
            try:
                off_team = poss.offense_team_id if hasattr(poss, 'offense_team_id') else None
                def_team = poss.defense_team_id if hasattr(poss, 'defense_team_id') else None
                pts = poss.points if hasattr(poss, 'points') else 0
                
                if off_team:
                    if off_team not in team_offense:
                        team_offense[off_team] = {'pts': 0, 'poss': 0}
                    team_offense[off_team]['pts'] += pts
                    team_offense[off_team]['poss'] += 1
                
                if def_team:
                    if def_team not in team_defense:
                        team_defense[def_team] = {'pts_allowed': 0, 'poss': 0}
                    team_defense[def_team]['pts_allowed'] += pts
                    team_defense[def_team]['poss'] += 1
                    
            except Exception:
                continue
        
        # Calcular métricas
        results = []
        all_teams = set(team_offense.keys()) | set(team_defense.keys())
        total_poss = len(possessions)
        
        for team_id in all_teams:
            off_data = team_offense.get(team_id, {'pts': 0, 'poss': 0})
            def_data = team_defense.get(team_id, {'pts_allowed': 0, 'poss': 0})
            
            # OffRtg = (Pts / Posses) * 100
            off_rtg = (off_data['pts'] / off_data['poss'] * 100) if off_data['poss'] > 0 else np.nan
            
            # DefRtg = (Pts Allowed / Posses) * 100
            def_rtg = (def_data['pts_allowed'] / def_data['poss'] * 100) if def_data['poss'] > 0 else np.nan
            
            # Pace aproximado = Posses por 48 min (assumindo ~48 min de jogo limpo)
            # Para cálculo preciso precisaríamos de minutos jogados
            pace = (total_poss / 2) * (48 / 48)  # Simplificado
            
            results.append({
                'game_id': game_id,
                'team_id': team_id,
                'team_abbrev': '',  # Pode ser preenchido com lookup
                'off_rtg': round(off_rtg, 1) if not np.isnan(off_rtg) else None,
                'def_rtg': round(def_rtg, 1) if not np.isnan(def_rtg) else None,
                'pace': round(pace, 1),
                'possessions': off_data['poss']
            })
        
        return results
    
    def get_lineup_data(self, team_id: int, season_year: str = "2024-25") -> pd.DataFrame:
        """
        Obtém dados de performance de lineups de 5 jogadores.
        
        Para uso futuro no recurso StatLineShift (detecção de rotações eficientes).
        
        Args:
            team_id: ID do time NBA (ex: 1610612738 para Boston Celtics)
            season_year: Temporada no formato "2024-25"
            
        Returns:
            DataFrame com colunas:
            - lineup_ids: Tuple com IDs dos 5 jogadores
            - minutes: Minutos jogados pelo lineup
            - net_rating: NetRtg do lineup (OffRtg - DefRtg)
            - plus_minus: Plus/Minus do lineup
            
        Note:
            Este método é placeholder para implementação futura.
        """
        logger.info(f"📊 Buscando dados de lineup para time {team_id}...")
        
        # TODO: Implementar lógica de lineup quando necessário
        # A API pbpstats suporta isso via game.possessions.items[x].lineup
        
        logger.warning("⚠️ get_lineup_data ainda não implementado. Retornando DataFrame vazio.")
        
        return pd.DataFrame(columns=['lineup_ids', 'minutes', 'net_rating', 'plus_minus'])


# Singleton para uso global (evita reinstanciar)
_pbp_client_instance = None


def get_pbp_client(cache_dir: str = "data/cache/pbp") -> PBPClient:
    """
    Retorna instância singleton do PBPClient.
    
    Args:
        cache_dir: Diretório de cache
        
    Returns:
        Instância do PBPClient
    """
    global _pbp_client_instance
    if _pbp_client_instance is None:
        _pbp_client_instance = PBPClient(cache_dir)
    return _pbp_client_instance
