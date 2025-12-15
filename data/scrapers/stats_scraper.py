import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from config.constants import TEAM_ABBREV_MAP  # Legacy - será removido
from data.scrapers.async_scraper import AsyncScraper
from functools import partial
from utils.data_validation import validate_rapm, DataValidator, ValidationRule
from utils.team_normalization import normalize_team  # NEW: Centralized normalization

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
EXCEL_BPM = DATA_DIR / "nba_bpm.xlsx"
EXCEL_RAPM = DATA_DIR / "nba_rapm.xlsx"
EXCEL_LEBRON = DATA_DIR / "nba_lebron.xlsx"
EXCEL_PIE = DATA_DIR / "nba_pie.xlsx"

class StatsScraper(AsyncScraper):
    def __init__(self):
        super().__init__(max_concurrent_requests=5)

    async def get_stats(self):
        """
        Wrapper para buscar todas as estatísticas (RAPM, BPM, etc).
        Mantém compatibilidade com chamadas antigas.
        """
        return await self.get_all_stats()

    async def get_rapm(self):
        """
        Busca métricas RAPM com fallback hierárquico robusto + VALIDAÇÃO.
        
        Hierarquia de fallback:
        1. CSV Local (nba_rapm.csv) - Gerado pelo scraper Selenium
        2. RAPM externo (nbarapm.com JSON) - Prioridade 2
        3. BPM da NBA Official API - Prioridade 3
        4. Game Score calculado - Prioridade 4
        
        Returns:
            DataFrame com colunas: Player, Team, RAPM, ORAPM, DRAPM, RAPM_SOURCE
        """
        # PRIORIDADE 1: CSV Local (Scraper Selenium)
        csv_path = DATA_DIR / "nba_rapm.csv"
        if csv_path.exists():
            logger.info(f"🔍 [P1] Carregando RAPM do CSV local: {csv_path}")
            try:
                df = pd.read_csv(csv_path)
                # Mapeamento de colunas do CSV do scraper para o padrão interno
                # O scraper retorna headers como 'Name', 'Team', 'time decay rapm', etc. ou similar
                # Vamos inspecionar/adaptar dinamicamente ou forçar nomes se soubermos
                
                # Normalização de nomes de colunas comuns
                df.columns = [c.lower().strip() for c in df.columns]
                
                # Mapeamento flexível
                col_map = {
                    'name': 'Player', 
                    'player': 'Player',
                    'team': 'Team',
                    'time decay rapm': 'RAPM',
                    'rapm': 'RAPM',
                    'time decay orapm': 'ORAPM',
                    'orapm': 'ORAPM',
                    'time decay drapm': 'DRAPM',
                    'drapm': 'DRAPM'
                }
                
                # Se o CSV vier com nomes exatos do site (ex: "time decay rapm"), o lower() resolveu
                # Se vier com "Col_X", precisaremos de lógica mais esperta ou assumir ordem.
                # O scraper tenta pegar headers.
                
                df = df.rename(columns=col_map)
                
                # Verificar colunas essenciais
                if 'Player' in df.columns and 'Team' in df.columns:
                    # Se faltar RAPM explícito mas tiver colunas numéricas, tentar inferir?
                    # Por segurança, só aceitamos se tiver mapeado corretamente ou se tivermos certeza.
                    
                    # Garantir colunas numéricas
                    for col in ['RAPM', 'ORAPM', 'DRAPM']:
                        if col not in df.columns:
                            df[col] = 0.0
                        else:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                            
                    df['RAPM_SOURCE'] = 'LOCAL_CSV'
                    
                    # Validação
                    rapm_schema = [
                        ValidationRule('Player', required=True, dtype=str),
                        ValidationRule('Team', required=True, dtype=str),
                        ValidationRule('RAPM', required=False, dtype=float, min_value=-20.0, max_value=20.0)
                    ]
                    validation = DataValidator.validate(df, rapm_schema, "RAPM_LOCAL")
                    
                    if validation.valid or validation.metrics['quality_score'] >= 70:
                        logger.info(f"✅ [P1] RAPM local validado: {len(df)} jogadores")
                        return df[['Player', 'Team', 'RAPM', 'ORAPM', 'DRAPM', 'RAPM_SOURCE']]
            except Exception as e:
                logger.warning(f"⚠️ [P1] Erro lendo CSV local: {e}")

        # PRIORIDADE 2: RAPM Externo (JSON)
        url = "https://nbarapm.com/load/current_comp"
        logger.info(f"🔍 [P2] Tentando RAPM externo (JSON): {url}")
        
        data = await self.fetch_json(url)
        
        if data:
            try:
                df = pd.DataFrame(data)
                col_map = {
                    'player_name': 'Player', 'team': 'Team',
                    'rapm_timedecay': 'RAPM', 'orapm_timedecay': 'ORAPM',
                    'drapm_timedecay': 'DRAPM'
                }
                cols_to_rename = {k: v for k, v in col_map.items() if k in df.columns}
                df = df.rename(columns=cols_to_rename)
                
                required_cols = ['Player', 'Team', 'RAPM', 'ORAPM', 'DRAPM']
                available_cols = [c for c in required_cols if c in df.columns]
                
                if available_cols and len(df) > 0:
                    final_df = df[available_cols].copy()
                    for col in ['RAPM', 'ORAPM', 'DRAPM']:
                        if col in final_df.columns:
                            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
                    
                    final_df['RAPM_SOURCE'] = 'EXTERNAL_JSON'
                    
                    rapm_schema = [
                        ValidationRule('Player', required=True, dtype=str),
                        ValidationRule('Team', required=True, dtype=str),
                        ValidationRule('RAPM', required=False, dtype=float, min_value=-15.0, max_value=15.0)
                    ]
                    validation = DataValidator.validate(final_df, rapm_schema, "RAPM_EXTERNAL")
                    
                    if validation.valid or validation.metrics['quality_score'] >= 70:
                        logger.info(f"✅ [P2] RAPM externo validado: {len(final_df)} jogadores")
                        return final_df
            except Exception as e:
                logger.warning(f"⚠️ [P2] Erro processando RAPM externo: {e}")
        
        logger.warning("⚠️ [P1/P2] RAPM externo/local falhou. Iniciando fallback...")
        
        # PRIORIDADE 2: BPM da NBA Official API
        logger.info("🔍 [P2] Tentando fallback: BPM da NBA Official API...")
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            
            # Buscar stats avançados (contém BPM)
            stats_response = leaguedashplayerstats.LeagueDashPlayerStats(
                season='2025-26',
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Advanced'
            )
            bpm_df = stats_response.get_data_frames()[0]
            
            if bpm_df is not None and not bpm_df.empty and 'PLAYER_NAME' in bpm_df.columns:
                # Mapear colunas (BPM geralmente não vem na NBA API padrão)
                # Vamos tentar usar PIE ou NetRtg como proxy se disponível
                df = bpm_df[['PLAYER_NAME', 'TEAM_ABBREVIATION']].copy()
                df = df.rename(columns={'PLAYER_NAME': 'Player', 'TEAM_ABBREVIATION': 'Team'})
                
                # Verificar se temos métricas avançadas disponíveis
                if 'NET_RATING' in bpm_df.columns:
                    # Usar NetRtg como proxy (normalizado)
                    df['RAPM'] = pd.to_numeric(bpm_df['NET_RATING'], errors='coerce').fillna(0)
                    df['RAPM'] = df['RAPM'].clip(-8, 8)  # Clip para range seguro
                    # REMOVIDO: Splits arbitrários (ORAPM/DRAPM) que eram invenção
                    df['ORAPM'] = 0.0 
                    df['DRAPM'] = 0.0
                    df['RAPM_SOURCE'] = 'NET_RTG_NBA'
                    logger.info(f"✅ [P2] Fallback BPM (NetRtg): {len(df)} jogadores")
                    return df
                else:
                    logger.warning("⚠️  [P2] BPM/NetRtg não disponível na NBA API")
        except Exception as e:
            logger.warning(f"⚠️  [P2] Erro no fallback BPM: {e}")
        
        # PRIORIDADE 3: Game Score Calculado (último recurso)
        logger.warning("⚠️  [P2] BPM fallback falhou. Usando último recurso: Game Score calculado")
        logger.info("🔍 [P3] Calculando Game Score como fallback final...")
        
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            
            # Buscar stats básicos para calcular Game Score
            basic_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season='2025-26',
                per_mode_detailed='PerGame'
            ).get_data_frames()[0]
            
            if basic_stats is not None and not basic_stats.empty:
                df = self._calculate_local_metrics(basic_stats)
                if not df.empty:
                    logger.info(f"✅ [P3] Game Score calculado: {len(df)} jogadores")
                    return df
        except Exception as e:
            logger.error(f"❌ [P3] Erro ao calcular Game Score: {e}")
        
        # Fallback total: retornar DataFrame vazio
        logger.error("❌ TODOS OS FALLBACKS FALHARAM! Retornando DataFrame vazio.")
        return pd.DataFrame()
    
    def _calculate_local_metrics(self, df_basic_stats):
        """
        Calcula métricas locais (Game Score + Efficiency) como último fallback para RAPM.
        
        Game Score Formula (John Hollinger):
        PTS + 0.4*FG - 0.7*FGA - 0.4*(FTA - FTM) + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
        
        Args:
            df_basic_stats: DataFrame com stats básicos (PTS, FG, REB, AST, etc.)
        
        Returns:
            DataFrame com colunas: Player, Team, RAPM, ORAPM, DRAPM, RAPM_SOURCE='GAME_SCORE'
        """
        try:
            df = df_basic_stats.copy()
            
            # Mapear nomes de colunas da NBA API
            required_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'FGM', 'FGA', 
                           'FTM', 'FTA', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
            
            if not all(col in df.columns for col in required_cols):
                logger.error(f"❌ Colunas faltando para calcular Game Score: {set(required_cols) - set(df.columns)}")
                return pd.DataFrame()
            
            # Assumir OREB e DREB reais se existirem, senão falhar (sem estimativa fake)
            if 'OREB' not in df.columns or 'DREB' not in df.columns:
                 # Tentar inferir se REB existe mas OREB/DREB não (muito raro na API oficial)
                 # Se não tiver, retornar vazio para não inventar dados
                 logger.error("❌ OREB/DREB faltando para Game Score. Abortando para não usar dados fake.")
                 return pd.DataFrame()

            # Calcular Game Score
            df['GAME_SCORE'] = (
                df['PTS'] + 
                0.4 * df['FGM'] - 
                0.7 * df['FGA'] - 
                0.4 * (df['FTA'] - df['FTM']) + 
                0.7 * df['OREB'] + 
                0.3 * df['DREB'] + 
                df['STL'] + 
                0.7 * df['AST'] + 
                0.7 * df['BLK'] - 
                0.4 * df['PF'] - 
                df['TOV']
            )
            
            # Normalizar Game Score para escala similar a RAPM (-8 a +8)
            # Usar Z-Score e depois escalar
            mean_gs = df['GAME_SCORE'].mean()
            std_gs = df['GAME_SCORE'].std()
            
            if std_gs > 0:
                df['RAPM_NORMALIZED'] = ((df['GAME_SCORE'] - mean_gs) / std_gs) * 3  # Escala ~-9 a +9
                df['RAPM_NORMALIZED'] = df['RAPM_NORMALIZED'].clip(-8, 8)
            else:
                df['RAPM_NORMALIZED'] = 0
            
            # Criar DataFrame final
            result = pd.DataFrame({
                'Player': df['PLAYER_NAME'],
                'Team': df['TEAM_ABBREVIATION'],
                'RAPM': df['RAPM_NORMALIZED'],
                'ORAPM': 0.0,  # Sem split fake
                'DRAPM': 0.0,  # Sem split fake
                'RAPM_SOURCE': 'GAME_SCORE'
            })
            
            logger.info(f"📊 Game Score stats: média={mean_gs:.2f}, desvio={std_gs:.2f}")
            logger.info(f"📊 RAPM normalizado: min={result['RAPM'].min():.2f}, max={result['RAPM'].max():.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao calcular métricas locais: {e}")
            return pd.DataFrame()

    async def get_bball_ref(self):
        """Async fetch of Basketball Reference Advanced Stats with Selenium fallback."""
        url = "https://www.basketball-reference.com/leagues/NBA_2026_advanced.html"
        logger.info("🔍 Buscando LEBRON/BPM (Basketball-Reference)...")
        
        # Tentar método assíncrono primeiro
        html = await self.fetch_text(url)
        
        if html:
            try:
                loop = asyncio.get_event_loop()
                dfs = await loop.run_in_executor(
                    None, partial(pd.read_html, html, match="Advanced")
                )
                
                if dfs:
                    df = dfs[0]
                    if 'Player' in df.columns:
                        df = df[df['Player'] != 'Player'].dropna(subset=['Player'])
                    
                    cols_to_numeric = ['OBPM', 'DBPM', 'BPM', 'VORP', 'PER', 'WS']
                    for col in cols_to_numeric:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                    logger.info(f"✅ Bball-Ref obtido (async): {len(df)} linhas")
                    return df
            except Exception as e:
                logger.warning(f"⚠️ Erro parsing Bball-Ref async: {e}")
        
        # Fallback: Selenium
        logger.info("🔄 Tentando fallback Selenium para BBRef...")
        try:
            from data.scrapers.bbref_selenium import get_bbref_with_selenium
            
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, get_bbref_with_selenium, "2026")
            
            if df is not None and len(df) > 0:
                logger.info(f"✅ Bball-Ref obtido (Selenium): {len(df)} linhas")
                return df
        except ImportError:
            logger.warning("⚠️ Selenium não disponível para fallback")
        except Exception as e:
            logger.warning(f"⚠️ Erro no fallback Selenium: {e}")
            
        return None

    async def get_nba_official_async(self):
        """Wrapper for synchronous NBA API call"""
        logger.info("🔍 Buscando Stats Oficiais (NBA.com)...")
        
        def _fetch():
            try:
                from nba_api.stats.endpoints import leaguedashteamstats
                try:
                    stats = leaguedashteamstats.LeagueDashTeamStats(season='2025-26', per_mode_detailed='PerGame', measure_type_nullable='Advanced')
                except:
                    stats = leaguedashteamstats.LeagueDashTeamStats(season='2025-26', per_mode_detailed='PerGame')
                return stats.get_data_frames()[0]
            except Exception as e:
                logger.warning(f"⚠️  Erro nba_api: {e}")
                return None

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, _fetch)
        
        if df is not None:
            logger.info(f"✅ NBA API (Team Stats) obtida: {len(df)} linhas")
        return df

    async def get_basic_stats(self):
        """Fetch basic player stats (PTS, REB, AST) from NBA API"""
        logger.info("🔍 Buscando Stats Básicos (PTS, REB, AST)...")
        
        def _fetch():
            try:
                from nba_api.stats.endpoints import leaguedashplayerstats
                stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', per_mode_detailed='PerGame')
                return stats.get_data_frames()[0]
            except Exception as e:
                logger.warning(f"⚠️  Erro nba_api basic stats: {e}")
                return None

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, _fetch)
        
        if df is not None:
            logger.info(f"✅ Basic Stats obtidos: {len(df)} jogadores")
            # Renomear colunas para padrão
            df = df.rename(columns={
                'PLAYER_NAME': 'PLAYER',
                'TEAM_ABBREVIATION': 'TEAM'
            })
        return df

    async def get_all_stats(self):
        """Fetch all stats in parallel"""
        logger.info("\n" + "="*80)
        logger.info("MÓDULO 4: PLAYER STATS (ASYNC FETCH)")
        logger.info("="*80)
        
        tasks = [
            self.get_rapm(),
            self.get_bball_ref(),
            self.get_nba_official_async(),
            self.get_basic_stats()  # NEW
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        dfs = {}
        if isinstance(results[0], pd.DataFrame) and not results[0].empty:
            dfs['RAPM'] = results[0]
        if isinstance(results[1], pd.DataFrame) and not results[1].empty:
            dfs['BBALL_REF'] = results[1]
        if isinstance(results[2], pd.DataFrame) and not results[2].empty:
            dfs['NBA_OFFICIAL'] = results[2]
        if isinstance(results[3], pd.DataFrame) and not results[3].empty:
            dfs['BASIC_STATS'] = results[3]  # NEW
            
        return dfs

def carregar_excel_stats():
    """Carrega stats dos arquivos Excel (Mantido para fallback)"""
    logger.info("🔍 Tentando carregar Stats dos Excels...")
    dfs = {}
    try:
        if EXCEL_BPM.exists(): dfs['bpm'] = pd.read_excel(EXCEL_BPM, sheet_name="Worksheet")
        if EXCEL_RAPM.exists(): dfs['rapm'] = pd.read_excel(EXCEL_RAPM, sheet_name="Planilha3")
        if EXCEL_LEBRON.exists(): dfs['lebron'] = pd.read_excel(EXCEL_LEBRON, sheet_name=0)
        if EXCEL_PIE.exists(): dfs['pie'] = pd.read_excel(EXCEL_PIE, sheet_name=0)
    except Exception as e:
        logger.warning(f"⚠️  Erro carregando Excel: {e}")
    return dfs

# Cache global
_STATS_CACHE = None

def obter_player_stats():
    """
    Função síncrona wrapper para manter compatibilidade.
    Executa o scraper async e faz fallback para Excel.
    """
    global _STATS_CACHE
    if _STATS_CACHE is not None:
        return _STATS_CACHE

    scraper = StatsScraper()
    
    # Run async loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Se já existe loop rodando (ex: streamlit), usa nest_asyncio ou run_coroutine_threadsafe
            import nest_asyncio
            nest_asyncio.apply()
            df = loop.run_until_complete(scraper.get_stats())
        else:
            df = loop.run_until_complete(scraper.get_stats())
            
        _STATS_CACHE = df
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro fatal no scraper: {e}")
        return pd.DataFrame()

def obter_team_stats_splits():
    # ... (código original mantido para compatibilidade, ou poderia ser migrado também)
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        home = leaguedashteamstats.LeagueDashTeamStats(season='2025-26', location_nullable='Home').get_data_frames()[0]
        away = leaguedashteamstats.LeagueDashTeamStats(season='2025-26', location_nullable='Road').get_data_frames()[0]
        
        splits = {}
        for _, row in home.iterrows():
            t = row['TEAM_ABBREVIATION']
            if t not in splits: splits[t] = {}
            splits[t]['Home'] = {'PTS': row['PTS'], 'PLUS_MINUS': row['PLUS_MINUS']}
            
        for _, row in away.iterrows():
            t = row['TEAM_ABBREVIATION']
            if t not in splits: splits[t] = {}
            splits[t]['Away'] = {'PTS': row['PTS'], 'PLUS_MINUS': row['PLUS_MINUS']}
            
        return splits
    except:
        return {}

def get_shot_quality_data():
    """
    Calcula métricas de Shot Quality (Qualidade de Arremesso) baseadas na localização dos chutes.
    
    Usa dados da NBA API (LeagueDashTeamShotLocations) para calcular o Expected eFG% (xEFG)
    de cada time, comparando a frequência de arremessos em cada zona com a eficiência média da liga.
    
    Returns:
        Dict: {
            'LAL': {'sq_score': 1.02, 'xEFG': 0.545, 'actual_EFG': 0.530},
            ...
        }
    """
    try:
        from nba_api.stats.endpoints import leaguedashteamshotlocations
        logger.info("🔍 Buscando dados de Shot Locations (Shot Quality)...")
        
        # Buscar dados de arremesso por zona
        shot_locs = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
            season='2025-26',
            distance_range='By Zone'
        ).get_data_frames()[0]
        
        if shot_locs.empty:
            logger.warning("⚠️ Shot Locations retornou vazio")
            return {}
        
        # Flatten columns se MultiIndex
        if isinstance(shot_locs.columns, pd.MultiIndex):
            shot_locs.columns = ['_'.join(col).strip() for col in shot_locs.columns]
        
        # Zonas típicas e seus valores esperados de eFG%
        # Baseado em médias históricas da NBA
        ZONE_EXPECTED_EFG = {
            'Restricted Area': 0.63,      # Layups/dunks
            'In The Paint (Non-RA)': 0.40,  # Floaters
            'Mid-Range': 0.40,              # Midrange jumpers
            'Left Corner 3': 0.40,          # Corner 3
            'Right Corner 3': 0.40,         # Corner 3
            'Above the Break 3': 0.37,      # Top of key 3
            'Backcourt': 0.10               # Heaves
        }
        
        result = {}
        
        # Procurar colunas de team
        team_col = None
        for col in shot_locs.columns:
            if 'TEAM' in col.upper() and 'ABBREVIATION' in col.upper():
                team_col = col
                break
            elif col.upper() == 'TEAM_ABBREVIATION':
                team_col = col
                break
        
        if not team_col:
            # Fallback: usar primeira coluna como identificador
            team_col = shot_locs.columns[0]
        
        # Procurar colunas de FGM/FGA por zona
        fgm_cols = [c for c in shot_locs.columns if 'FGM' in c.upper()]
        fga_cols = [c for c in shot_locs.columns if 'FGA' in c.upper()]
        
        if not fgm_cols or not fga_cols:
            logger.warning("⚠️ Colunas FGM/FGA não encontradas no formato esperado")
            return {}
        
        for _, row in shot_locs.iterrows():
            team = str(row[team_col])[:3].upper()  # Normalizar para 3 letras
            
            total_fga = 0
            weighted_xefg = 0
            actual_fgm = 0
            
            # Calcular xEFG ponderado por zona
            for fgm_col, fga_col in zip(fgm_cols, fga_cols):
                try:
                    fgm = float(row.get(fgm_col, 0) or 0)
                    fga = float(row.get(fga_col, 0) or 0)
                    
                    if fga > 0:
                        # Identificar zona pelo nome da coluna
                        zone_efg = 0.45  # Default
                        for zone, efg in ZONE_EXPECTED_EFG.items():
                            if zone.lower().replace(' ', '') in fgm_col.lower().replace(' ', ''):
                                zone_efg = efg
                                break
                        
                        total_fga += fga
                        weighted_xefg += fga * zone_efg
                        actual_fgm += fgm
                except (ValueError, TypeError):
                    continue
            
            if total_fga > 0:
                xefg = weighted_xefg / total_fga
                actual_efg = actual_fgm / total_fga
                sq_score = actual_efg / xefg if xefg > 0 else 1.0
                
                result[team] = {
                    'sq_score': round(sq_score, 3),
                    'xEFG': round(xefg, 3),
                    'actual_EFG': round(actual_efg, 3)
                }
        
        if result:
            logger.info(f"✅ Shot Quality calculado para {len(result)} times")
        else:
            logger.warning("⚠️ Shot Quality: Nenhum time processado")
            
        return result

    except Exception as e:
        logger.warning(f"⚠️ Erro ao buscar Shot Quality data: {e}")
        return {} 

def obter_game_log(team_abbr, season='2025-26'):
    try:
        from nba_api.stats.endpoints import teamgamelog
        from nba_api.stats.static import teams
        nba_teams = teams.get_teams()
        team_id = next((t['id'] for t in nba_teams if t['abbreviation'] == team_abbr), None)
        if team_id:
            return teamgamelog.TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
    except:
        pass
    return None
