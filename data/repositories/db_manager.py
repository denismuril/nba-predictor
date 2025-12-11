import sqlite3
import logging
import time
import os
import functools
from datetime import datetime
from pathlib import Path
import pandas as pd
import warnings

# Suprimir warning do pandas sobre conexões DBAPI2
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
import shutil
import uuid
from dotenv import load_dotenv
from functools import lru_cache
from config.constants import TEAM_ABBREV_MAP, TEAMS_MAP
from utils.connection_pool import ConnectionPool
from utils.team_normalization import normalize_team

# Tentar importar psycopg2 para suporte a Postgres
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

logger = logging.getLogger(__name__)
load_dotenv()

def retry_on_lock(max_retries=10, initial_delay=1.0):
    """
    Decorator para tentar novamente operações de banco de dados quando ocorre erro de lock.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e):
                        if attempt < max_retries - 1:
                            logger.warning(f"⚠️  Database locked in {func.__name__}. Retry {attempt + 1}/{max_retries} in {delay:.2f}s...")
                            time.sleep(delay)
                            delay *= 2
                        else:
                            logger.error(f"❌ Database locked after {max_retries} retries in {func.__name__}")
                            raise
                    else:
                        raise
                except Exception as e:
                    logger.error(f"❌ Unexpected error in {func.__name__}: {e}")
                    raise
        return wrapper
    return decorator

class DatabaseManager:
    def __init__(self, db_path="nba_history.db"):
        self.db_type = os.getenv('DB_TYPE', 'sqlite').lower()
        
        # Config SQLite
        if not Path(db_path).is_absolute():
            base_dir = Path(__file__).parent.parent.parent
            self.db_path = base_dir / "data" / db_path
        else:
            self.db_path = Path(db_path)
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Config Postgres
        self.pg_host = os.getenv('DB_HOST', 'localhost')
        self.pg_port = os.getenv('DB_PORT', '5432')
        self.pg_name = os.getenv('DB_NAME', 'nba_predictor_db')
        self.pg_user = os.getenv('DB_USER', 'nba_admin')
        self.pg_pass = os.getenv('DB_PASS', 'password')
        
        if self.db_type == 'postgres' and not HAS_POSTGRES:
            logger.warning("⚠️ DB_TYPE=postgres mas psycopg2 não instalado. Voltando para SQLite.")
            self.db_type = 'sqlite'

        logger.info(f"📂 Banco de dados: {self.db_type.upper()} ({self.db_path if self.db_type == 'sqlite' else self.pg_name})")
            
        self._initialized = False
        
        # Connection Pool para SQLite
        self._pool = None
        if self.db_type == 'sqlite':
            self._pool = ConnectionPool(
                db_path=str(self.db_path),
                pool_size=5,
                timeout=30,
                busy_timeout=30000
            )
            logger.info(f"✅ Connection Pool ativado: {self._pool.get_stats()}")
    
    @staticmethod
    def _normalize_team_id(team_name):
        """Normaliza nome de time para ID padronizado de 3 letras."""
        result = normalize_team(team_name)
        if result is None and team_name:
            logger.debug(f"⚠️ ID de time não encontrado para: '{team_name}'")
        return result

    def get_connection(self):
        """Obtém conexão do pool (SQLite) ou cria nova (Postgres)."""
        if self.db_type == 'postgres':
            try:
                conn = psycopg2.connect(
                    host=self.pg_host,
                    port=self.pg_port,
                    dbname=self.pg_name,
                    user=self.pg_user,
                    password=self.pg_pass
                )
                return conn
            except Exception as e:
                logger.error(f"❌ Erro conexão Postgres: {e}")
                raise
        else:
            if self._pool is None:
                raise RuntimeError("Connection pool não inicializado!")
            return self._pool.get_connection()
    
    def return_connection(self, conn):
        """Retorna conexão ao pool (SQLite) ou fecha (Postgres)."""
        if self.db_type == 'postgres':
            conn.close()
        else:
            if self._pool is not None:
                self._pool.return_connection(conn)
            else:
                conn.close()
    
    def close_pool(self):
        """Fecha o connection pool."""
        if self._pool is not None:
            stats = self._pool.get_stats()
            logger.info(f"🔄 Fechando connection pool: {stats}")
            self._pool.close()
            self._pool = None
    
    def __del__(self):
        self.close_pool()

    def _prepare_query(self, query):
        """Adapta query para o dialeto correto"""
        if self.db_type == 'postgres':
            query = query.replace('?', '%s')
            query = query.replace('DATETIME', 'TIMESTAMP')
            query = query.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')
            if 'INSERT OR IGNORE' in query:
                query = query.replace('INSERT OR IGNORE', 'INSERT')
                query += " ON CONFLICT DO NOTHING"
        return query

    @retry_on_lock()
    def init_db(self):
        """Inicializa o esquema do banco de dados (Versão Grão-Mestre Normalizada)."""
        if self._initialized:
            return

        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Tabela de Jogos (Games) - Apenas metadados e placar
            schema_games = '''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                season TEXT,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                winner TEXT,
                status TEXT DEFAULT 'Final',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            '''
            
            # 2. Tabela de Estatísticas (Game Stats) - Detalhes por time
            schema_stats = '''
            CREATE TABLE IF NOT EXISTS game_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                team_id TEXT,
                is_home BOOLEAN,
                pts INTEGER,
                fgm INTEGER, fga INTEGER, fg_pct REAL,
                fg3m INTEGER, fg3a INTEGER, fg3_pct REAL,
                ftm INTEGER, fta INTEGER, ft_pct REAL,
                oreb INTEGER, dreb INTEGER, reb INTEGER,
                ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER, pf INTEGER,
                plus_minus INTEGER,
                off_rating REAL, def_rating REAL,
                efg_pct REAL, ts_pct REAL, pace REAL, pie REAL,
                FOREIGN KEY(game_id) REFERENCES games(game_id),
                UNIQUE(game_id, team_id)
            )
            '''
            
            # 3. Tabela de Previsões (Predictions) - Separada dos dados históricos
            schema_predictions = '''
            CREATE TABLE IF NOT EXISTS predictions (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                home_team TEXT,
                away_team TEXT,
                prob_home REAL,
                prob_away REAL,
                prob_mc_home REAL,
                prob_mc_away REAL,
                odd_home REAL,
                odd_away REAL,
                prediction TEXT,
                confidence TEXT,
                predicted_spread REAL,
                predicted_total REAL,
                ci_lower REAL,
                ci_upper REAL,
                model_version TEXT,
                home_injuries_list TEXT,
                away_injuries_list TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            '''
            
            # 4. Tabela de Apostas (Bets) - Rastreamento completo de apostas e P&L
            schema_bets = '''
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bet_date TEXT NOT NULL,
                game_id TEXT NOT NULL,
                home_team TEXT,
                away_team TEXT,
                side TEXT NOT NULL,
                bet_type TEXT NOT NULL,
                line REAL,
                opening_odds REAL,
                bet_odds REAL NOT NULL,
                closing_odds REAL,
                stake_pct REAL NOT NULL,
                stake_amount REAL NOT NULL,
                bankroll_at_bet REAL,
                model_prob REAL,
                ev_pct REAL,
                kelly_fraction REAL,
                result TEXT DEFAULT 'PENDING',
                payout REAL DEFAULT 0,
                profit REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                settled_at TEXT
            )
            '''

            try:
                cursor.execute(self._prepare_query(schema_games))
                cursor.execute(self._prepare_query(schema_stats))
                cursor.execute(self._prepare_query(schema_predictions))
                cursor.execute(self._prepare_query(schema_bets))
                
                # Índices para performance
                cursor.execute(self._prepare_query("CREATE INDEX IF NOT EXISTS idx_games_date ON games(date)"))
                cursor.execute(self._prepare_query("CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team, away_team)"))
                cursor.execute(self._prepare_query("CREATE INDEX IF NOT EXISTS idx_stats_game_id ON game_stats(game_id)"))
                cursor.execute(self._prepare_query("CREATE INDEX IF NOT EXISTS idx_stats_team_id ON game_stats(team_id)"))
                cursor.execute(self._prepare_query("CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(bet_date)"))
                cursor.execute(self._prepare_query("CREATE INDEX IF NOT EXISTS idx_bets_game_id ON bets(game_id)"))
                cursor.execute(self._prepare_query("CREATE INDEX IF NOT EXISTS idx_bets_result ON bets(result)"))
                
                if self.db_type == 'postgres':
                    conn.commit()
                
                logger.info("✅ Esquema de Banco de Dados 'Grão-Mestre' inicializado.")
                
            except Exception as e:
                logger.error(f"❌ Erro init_db schema: {e}")
                if self.db_type == 'postgres': conn.rollback()
            
            self._initialized = True

    @retry_on_lock()
    def save_predictions(self, predictions):
        """Salva previsões com AUTO-MIGRAÇÃO para métricas V21."""
        if not predictions:
            return

        self.init_db()
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # ==================== SELF-HEALING SCHEMA V21.2 ====================
            # Definir novas colunas para métricas avançadas
            new_cols_v21 = [
                ('home_shooting_luck', 'REAL'),
                ('away_shooting_luck', 'REAL'),
                ('home_rapm_penalty', 'REAL'),
                ('away_rapm_penalty', 'REAL'),
                ('rapm_impact_diff', 'REAL'),
                ('home_fatigue_score', 'REAL'),
                ('away_fatigue_score', 'REAL'),
                ('home_elo', 'REAL'),
                ('away_elo', 'REAL'),
                ('projected_pace_vegas', 'REAL')
            ]
            
            # Auto-Migração: Adicionar colunas se não existirem
            if self.db_type == 'sqlite':
                cursor.execute("PRAGMA table_info(predictions)")
                existing_cols = [row[1] for row in cursor.fetchall()]
                
                for col_name, col_type in new_cols_v21:
                    if col_name not in existing_cols:
                        logger.warning(f"🔧 AUTO-MIGRAÇÃO: Criando coluna {col_name} ({col_type})...")
                        alter_query = f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type} DEFAULT 0"
                        cursor.execute(alter_query)
                        logger.info(f"   ✅ Coluna {col_name} adicionada")
                
                conn.commit()
                
            else:  # Postgres
                # Verificar colunas existentes
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'predictions'
                """)
                existing_cols = [row[0] for row in cursor.fetchall()]
                
                for col_name, col_type in new_cols_v21:
                    if col_name not in existing_cols:
                        logger.warning(f"🔧 AUTO-MIGRAÇÃO (Postgres): Criando coluna {col_name}...")
                        # Postgres usa DOUBLE PRECISION em vez de REAL
                        pg_type = 'DOUBLE PRECISION' if col_type == 'REAL' else col_type
                        alter_query = f"ALTER TABLE predictions ADD COLUMN {col_name} {pg_type} DEFAULT 0"
                        cursor.execute(alter_query)
                        logger.info(f"   ✅ Coluna {col_name} adicionada")
                
                conn.commit()
            
            # ==================== INSERT EXPANDIDO COM MÉTRICAS V21 ====================
            if self.db_type == 'sqlite':
                query = '''
                INSERT OR REPLACE INTO predictions (
                    game_id, date, home_team, away_team, 
                    prob_home, prob_away, prob_mc_home, prob_mc_away,
                    odd_home, odd_away, prediction, confidence,
                    predicted_spread, predicted_total, ci_lower, ci_upper,
                    model_version,
                    odds_home, odds_away, total_line, odds_source,
                    home_injuries_list, away_injuries_list,
                    home_shooting_luck, away_shooting_luck,
                    home_rapm_penalty, away_rapm_penalty, rapm_impact_diff,
                    home_fatigue_score, away_fatigue_score,
                    home_elo, away_elo, projected_pace_vegas
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                cursor.executemany(query, [
                    (
                        f"{p['Data']}_{p['Casa']}_{p['Visitante']}".replace(" ", ""), 
                        p['Data'], p['Casa'], p['Visitante'],
                        p['Prob Casa %'], p['Prob Visitante %'], 
                        p.get('Prob MC Casa %', 0), p.get('Prob MC Visitante %', 0),
                        p.get('Odd Casa', p.get('odds_home', 0)), p.get('Odd Visitante', p.get('odds_away', 0)),
                        p.get('Previsão', 'N/A'), p.get('Confiança', 'N/A'),
                        p.get('Spread Previsto', 0), p.get('Total Previsto', p.get('total_line', 0)),
                        p.get('ci_lower', 0), p.get('ci_upper', 0),
                        'v21.2',
                        p.get('Odd Casa', p.get('odds_home', 0)), p.get('Odd Visitante', p.get('odds_away', 0)), 
                        p.get('Total Previsto', p.get('total_line', 0)), 
                        p.get('odds_source', 'simulated' if p.get('Odd Casa', 0) > 0 else 'none'),
                        p.get('home_injuries_list', ''), p.get('away_injuries_list', ''),
                        # MÉTRICAS V21
                        p.get('home_shooting_luck', 0.0), p.get('away_shooting_luck', 0.0),
                        p.get('home_rapm_penalty', 0.0), p.get('away_rapm_penalty', 0.0),
                        p.get('rapm_impact_diff', 0.0),
                        p.get('home_fatigue_score', 0.0), p.get('away_fatigue_score', 0.0),
                        p.get('home_elo', 0.0), p.get('away_elo', 0.0),
                        p.get('projected_pace_vegas', 0.0)
                    ) for p in predictions
                ])
            else:
                # Postgres UPSERT
                query = '''
                INSERT INTO predictions (
                    game_id, date, home_team, away_team, 
                    prob_home, prob_away, prob_mc_home, prob_mc_away,
                    odd_home, odd_away, prediction, confidence,
                    predicted_spread, predicted_total, ci_lower, ci_upper,
                    model_version,
                    odds_home, odds_away, total_line, odds_source,
                    home_injuries_list, away_injuries_list,
                    home_shooting_luck, away_shooting_luck,
                    home_rapm_penalty, away_rapm_penalty, rapm_impact_diff,
                    home_fatigue_score, away_fatigue_score,
                    home_elo, away_elo, projected_pace_vegas
                ) VALUES %s
                ON CONFLICT (game_id) DO UPDATE SET
                    prob_home = EXCLUDED.prob_home,
                    prob_away = EXCLUDED.prob_away,
                    prob_mc_home = EXCLUDED.prob_mc_home,
                    prob_mc_away = EXCLUDED.prob_mc_away,
                    odd_home = EXCLUDED.odd_home,
                    odd_away = EXCLUDED.odd_away,
                    prediction = EXCLUDED.prediction,
                    confidence = EXCLUDED.confidence,
                    predicted_spread = EXCLUDED.predicted_spread,
                    predicted_total = EXCLUDED.predicted_total,
                    ci_lower = EXCLUDED.ci_lower,
                    ci_upper = EXCLUDED.ci_upper,
                    model_version = EXCLUDED.model_version,
                    odds_home = EXCLUDED.odds_home,
                    odds_away = EXCLUDED.odds_away,
                    total_line = EXCLUDED.total_line,
                    odds_source = EXCLUDED.odds_source,
                    home_injuries_list = EXCLUDED.home_injuries_list,
                    away_injuries_list = EXCLUDED.away_injuries_list,
                    home_shooting_luck = EXCLUDED.home_shooting_luck,
                    away_shooting_luck = EXCLUDED.away_shooting_luck,
                    home_rapm_penalty = EXCLUDED.home_rapm_penalty,
                    away_rapm_penalty = EXCLUDED.away_rapm_penalty,
                    rapm_impact_diff = EXCLUDED.rapm_impact_diff,
                    home_fatigue_score = EXCLUDED.home_fatigue_score,
                    away_fatigue_score = EXCLUDED.away_fatigue_score,
                    home_elo = EXCLUDED.home_elo,
                    away_elo = EXCLUDED.away_elo,
                    projected_pace_vegas = EXCLUDED.projected_pace_vegas,
                    created_at = CURRENT_TIMESTAMP
                '''
                values = [
                    (
                        f"{p['Data']}_{p['Casa']}_{p['Visitante']}".replace(" ", ""), 
                        p['Data'], p['Casa'], p['Visitante'],
                        p['Prob Casa %'], p['Prob Visitante %'], 
                        p.get('Prob MC Casa %', 0), p.get('Prob MC Visitante %', 0),
                        p.get('Odd Casa', p.get('odds_home', 0)), p.get('Odd Visitante', p.get('odds_away', 0)),
                        p.get('Previsão', 'N/A'), p.get('Confiança', 'N/A'),
                        p.get('Spread Previsto', 0), p.get('Total Previsto', p.get('total_line', 0)),
                        p.get('ci_lower', 0), p.get('ci_upper', 0),
                        'v21.2',
                        p.get('Odd Casa', p.get('odds_home', 0)), p.get('Odd Visitante', p.get('odds_away', 0)), 
                        p.get('Total Previsto', p.get('total_line', 0)), 
                        p.get('odds_source', 'simulated' if p.get('Odd Casa', 0) > 0 else 'none'),
                        p.get('home_injuries_list', ''), p.get('away_injuries_list', ''),
                        # MÉTRICAS V21
                        p.get('home_shooting_luck', 0.0), p.get('away_shooting_luck', 0.0),
                        p.get('home_rapm_penalty', 0.0), p.get('away_rapm_penalty', 0.0),
                        p.get('rapm_impact_diff', 0.0),
                        p.get('home_fatigue_score', 0.0), p.get('away_fatigue_score', 0.0),
                        p.get('home_elo', 0.0), p.get('away_elo', 0.0),
                        p.get('projected_pace_vegas', 0.0)
                    ) for p in predictions
                ]
                execute_values(cursor, query, values)
            
            conn.commit()
            logger.info(f"💾 {len(predictions)} previsões V21.2 salvas (com métricas avançadas)")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erro ao salvar previsões V21: {e}")
            raise
        finally:
            self.return_connection(conn)


    @retry_on_lock()
    def update_game_score(self, game_id, home_score, away_score):
        """Atualiza o placar de um jogo finalizado."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Tentar diferentes formatos de game_id
            # Formato 1: YYYY-MM-DD_TEAM_TEAM (padrão)
            # Formato 2: YYYYMMDD_TEAM_TEAM
            # Formato 3: Usar data, home_team e away_team separadamente
            
            # Extrair componentes do game_id
            parts = game_id.split('_')
            if len(parts) >= 3:
                date_part = parts[0]
                home_team = parts[1]
                away_team = parts[2]
                
                # Verificar se jogo já existe na tabela games
                # Tentar match por data e times (mais robusto)
                query_check = """
                SELECT game_id FROM games 
                WHERE date = ? AND home_team = ? AND away_team = ?
                LIMIT 1
                """
                cursor.execute(self._prepare_query(query_check), (date_part, home_team, away_team))
                result = cursor.fetchone()
                
                if result:
                    actual_game_id = result[0]
                    
                    # Determinar vencedor
                    winner = 'HOME' if home_score > away_score else 'AWAY'
                    
                    # Atualizar placar e vencedor
                    query = """
                    UPDATE games 
                    SET home_score = ?, away_score = ?, status = 'finished', winner = ?
                    WHERE game_id = ?
                    """
                    cursor.execute(self._prepare_query(query), (home_score, away_score, winner, actual_game_id))
                    conn.commit()
                    logger.info(f"✅ Jogo {actual_game_id} atualizado: {home_score}-{away_score} ({winner})")
                    return True
                else:
                    logger.warning(f"⚠️ Jogo não encontrado: {date_part} {home_team} vs {away_team}")
                    return False
            else:
                logger.warning(f"⚠️ Formato de game_id inválido: {game_id}")
                return False
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erro ao atualizar jogo {game_id}: {e}")
            return False
        finally:
            self.return_connection(conn)


    @retry_on_lock()
    def insert_game_stats(self, game_data, home_stats, away_stats):
        """
        Insere um jogo completo e suas estatísticas.
        """
        self.init_db()
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if self.db_type == 'sqlite':
                cursor.execute('BEGIN TRANSACTION')
            
            # 1. Inserir Jogo
            game_values = (
                game_data['id'], game_data['date'], game_data.get('season', '2024-25'),
                game_data['home_team'], game_data['away_team'],
                game_data['home_score'], game_data['away_score'],
                game_data['winner'], 'Final'
            )

            # 1. Inserir Jogo (Upsert compatível)
            if self.db_type == 'postgres':
                game_query = '''
                INSERT INTO games (
                    game_id, date, season, home_team, away_team, 
                    home_score, away_score, winner, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id) DO UPDATE SET
                    date = EXCLUDED.date,
                    season = EXCLUDED.season,
                    home_team = EXCLUDED.home_team,
                    away_team = EXCLUDED.away_team,
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    winner = EXCLUDED.winner,
                    status = EXCLUDED.status
                '''
            else:
                game_query = '''
                INSERT OR REPLACE INTO games (
                    game_id, date, season, home_team, away_team, 
                    home_score, away_score, winner, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
            
            cursor.execute(game_query, game_values)
            
            # 2. Inserir Stats (Casa)
            if home_stats:
                self._insert_stats_record(cursor, game_data['id'], game_data['home_team'], True, home_stats)
                
            # 3. Inserir Stats (Visitante)
            if away_stats:
                self._insert_stats_record(cursor, game_data['id'], game_data['away_team'], False, away_stats)
                
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erro ao inserir stats do jogo {game_data.get('id')}: {e}")
            raise
        finally:
            self.return_connection(conn)

    @retry_on_lock()
    def bulk_insert_games(self, games_list):
        """
        Insere múltiplos jogos e suas estatísticas de uma vez (Bulk Insert).
        
        Args:
            games_list: Lista de tuplas (game_data, home_stats, away_stats)
        """
        if not games_list:
            return

        self.init_db()
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if self.db_type == 'sqlite':
                cursor.execute('BEGIN TRANSACTION')
            
            # Preparar dados
            games_values = []
            stats_values = []
            
            for game_data, home_stats, away_stats in games_list:
                # Game
                games_values.append((
                    game_data['id'], game_data['date'], game_data.get('season', '2024-25'),
                    game_data['home_team'], game_data['away_team'],
                    game_data['home_score'], game_data['away_score'],
                    game_data['winner'], 'Final'
                ))
                
                # Home Stats
                if home_stats:
                    stats_values.append(self._prepare_stats_record(game_data['id'], game_data['home_team'], True, home_stats))
                
                # Away Stats
                if away_stats:
                    stats_values.append(self._prepare_stats_record(game_data['id'], game_data['away_team'], False, away_stats))

            # Bulk Insert Games
            if self.db_type == 'postgres':
                # Postgres: execute_values para performance máxima
                from psycopg2.extras import execute_values
                
                game_query = '''
                INSERT INTO games (
                    game_id, date, season, home_team, away_team, 
                    home_score, away_score, winner, status
                ) VALUES %s
                ON CONFLICT (game_id) DO UPDATE SET
                    date = EXCLUDED.date,
                    season = EXCLUDED.season,
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    winner = EXCLUDED.winner
                '''
                execute_values(cursor, game_query, games_values)
                
                # Bulk Insert Stats
                if stats_values:
                    # Obter colunas dinamicamente do primeiro registro
                    columns = list(stats_values[0].keys())
                    cols_str = ', '.join(columns)
                    
                    # Converter dicts para tuplas na ordem das colunas
                    stats_tuples = [[s[c] for c in columns] for s in stats_values]
                    
                    stats_query = f'''
                    INSERT INTO game_stats ({cols_str}) VALUES %s
                    ON CONFLICT (game_id, team_id) DO UPDATE SET
                        pts = EXCLUDED.pts,
                        fgm = EXCLUDED.fgm,
                        fga = EXCLUDED.fga,
                        fg_pct = EXCLUDED.fg_pct,
                        fg3m = EXCLUDED.fg3m,
                        fg3a = EXCLUDED.fg3a,
                        fg3_pct = EXCLUDED.fg3_pct,
                        ftm = EXCLUDED.ftm,
                        fta = EXCLUDED.fta,
                        ft_pct = EXCLUDED.ft_pct,
                        oreb = EXCLUDED.oreb,
                        dreb = EXCLUDED.dreb,
                        reb = EXCLUDED.reb,
                        ast = EXCLUDED.ast,
                        stl = EXCLUDED.stl,
                        blk = EXCLUDED.blk,
                        tov = EXCLUDED.tov,
                        pf = EXCLUDED.pf,
                        plus_minus = EXCLUDED.plus_minus,
                        off_rating = EXCLUDED.off_rating,
                        def_rating = EXCLUDED.def_rating,
                        efg_pct = EXCLUDED.efg_pct,
                        ts_pct = EXCLUDED.ts_pct,
                        pace = EXCLUDED.pace,
                        pie = EXCLUDED.pie
                    '''
                    execute_values(cursor, stats_query, stats_tuples)
                    
            else:
                # SQLite: executemany
                game_query = '''
                INSERT OR REPLACE INTO games (
                    game_id, date, season, home_team, away_team, 
                    home_score, away_score, winner, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                cursor.executemany(game_query, games_values)
                
                if stats_values:
                    columns = list(stats_values[0].keys())
                    cols_str = ', '.join(columns)
                    placeholders = ', '.join(['?'] * len(columns))
                    
                    stats_tuples = [[s[c] for c in columns] for s in stats_values]
                    
                    stats_query = f'''
                    INSERT OR REPLACE INTO game_stats ({cols_str}) VALUES ({placeholders})
                    '''
                    cursor.executemany(stats_query, stats_tuples)

            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erro no Bulk Insert: {e}")
            raise
        finally:
            self.return_connection(conn)

    def _prepare_stats_record(self, game_id, team_id, is_home, stats):
        """Helper para preparar dicionário de stats para bulk insert."""
        record = {
            'game_id': game_id,
            'team_id': self._normalize_team_id(team_id),
            'is_home': is_home,
            'pts': stats.get('PTS', 0),
            'fgm': stats.get('FGM', 0),
            'fga': stats.get('FGA', 0),
            'fg_pct': stats.get('FG_PCT', 0.0),
            'fg3m': stats.get('FG3M', 0),
            'fg3a': stats.get('FG3A', 0),
            'fg3_pct': stats.get('FG3_PCT', 0.0),
            'ftm': stats.get('FTM', 0),
            'fta': stats.get('FTA', 0),
            'ft_pct': stats.get('FT_PCT', 0.0),
            'oreb': stats.get('OREB', 0),
            'dreb': stats.get('DREB', 0),
            'reb': stats.get('REB', 0),
            'ast': stats.get('AST', 0),
            'stl': stats.get('STL', 0),
            'blk': stats.get('BLK', 0),
            'tov': stats.get('TOV', 0),
            'pf': stats.get('PF', 0),
            'plus_minus': stats.get('PLUS_MINUS', 0),
            'off_rating': stats.get('OFF_RATING', 0.0),
            'def_rating': stats.get('DEF_RATING', 0.0),
            'pace': stats.get('PACE', 0.0)
        }
        return record

    def _insert_stats_record(self, cursor, game_id, team_id, is_home, stats):
        """Helper para inserir registro de stats"""
        # Calcular eFG% se não vier da API ou vier zerado
        efg_pct = stats.get('EFG_PCT', 0)
        if efg_pct == 0 or efg_pct is None:
            fgm = stats.get('FGM', 0)
            fg3m = stats.get('FG3M', 0)
            fga = stats.get('FGA', 0)
            if fga > 0:
                efg_pct = (fgm + 0.5 * fg3m) / fga
        
        # Calcular TS% se não vier da API ou vier zerado
        ts_pct = stats.get('TS_PCT', 0)
        if ts_pct == 0 or ts_pct is None:
            pts = stats.get('PTS', 0)
            fga = stats.get('FGA', 0)
            fta = stats.get('FTA', 0)
            if fga + 0.44 * fta > 0:
                ts_pct = pts / (2 * (fga + 0.44 * fta))
        
        cols = [
            'game_id', 'team_id', 'is_home', 'pts', 
            'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct',
            'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb',
            'ast', 'stl', 'blk', 'tov', 'pf', 'plus_minus',
            'off_rating', 'def_rating', 'efg_pct', 'ts_pct', 'pace', 'pie'
        ]
        
        vals = [
            game_id, team_id, is_home, stats.get('PTS', 0),
            stats.get('FGM', 0), stats.get('FGA', 0), stats.get('FG_PCT', 0),
            stats.get('FG3M', 0), stats.get('FG3A', 0), stats.get('FG3_PCT', 0),
            stats.get('FTM', 0), stats.get('FTA', 0), stats.get('FT_PCT', 0),
            stats.get('OREB', 0), stats.get('DREB', 0), stats.get('REB', 0),
            stats.get('AST', 0), stats.get('STL', 0), stats.get('BLK', 0),
            stats.get('TOV', 0), stats.get('PF', 0), stats.get('PLUS_MINUS', 0),
            stats.get('OFF_RATING', 0), stats.get('DEF_RATING', 0),
            efg_pct, ts_pct,  # ← Usar valores calculados
            stats.get('PACE', 0), stats.get('PIE', 0)
        ]
        
        del_query = "DELETE FROM game_stats WHERE game_id = ? AND team_id = ?"
        cursor.execute(self._prepare_query(del_query), (game_id, team_id))
        
        placeholders = ', '.join(['?' for _ in cols])
        query = f"INSERT INTO game_stats ({', '.join(cols)}) VALUES ({placeholders})"
        cursor.execute(self._prepare_query(query), vals)

    @retry_on_lock()
    def get_history(self):
        """
        Retorna histórico unificado (compatibilidade com código legado).
        Reconstrói o formato 'flat' antigo fazendo join de games e game_stats.
        """
        conn = self.get_connection()
        
        # Colunas esperadas para garantir que o DF nunca venha sem schema
        expected_cols = [
            'date', 'home_team', 'away_team', 'home_score', 'away_score', 'winner', 'id',
            'pts', 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
            'ast', 'stl', 'blk', 'pf',
            'home_off_rating', 'home_def_rating', 'home_efg_pct', 'home_ts_pct', 'home_pace', 'home_pie',
            'opp_pts', 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 
            'opp_fta', 'opp_ftm', 'opp_ast', 'opp_stl', 'opp_blk', 'opp_pf',
            'away_off_rating', 'away_def_rating', 'away_efg_pct', 'away_ts_pct', 'away_pace', 'away_pie',
            'correct'
        ]
        
        try:
            # Query complexa para pivotar home/away stats em uma única linha
            query = '''
            SELECT 
                g.date, g.home_team, g.away_team, g.home_score, g.away_score, g.winner,
                g.game_id as id,
                
                -- Home Stats
                sh.pts as pts, sh.fgm, sh.fga, sh.fg3m, sh.tov, sh.oreb, sh.dreb, sh.fta, sh.ftm,
                sh.ast, sh.stl, sh.blk, sh.pf,
                sh.off_rating as home_off_rating, sh.def_rating as home_def_rating, 
                sh.efg_pct as home_efg_pct, sh.ts_pct as home_ts_pct, sh.pace as home_pace, sh.pie as home_pie,
                
                -- Away Stats (Opponent)
                sa.pts as opp_pts, sa.fgm as opp_fgm, sa.fga as opp_fga, sa.fg3m as opp_fg3m, 
                sa.tov as opp_tov, sa.oreb as opp_oreb, sa.dreb as opp_dreb, 
                sa.fta as opp_fta, sa.ftm as opp_ftm,
                sa.ast as opp_ast, sa.stl as opp_stl, sa.blk as opp_blk, sa.pf as opp_pf,
                sa.off_rating as away_off_rating, sa.def_rating as away_def_rating,
                sa.efg_pct as away_efg_pct, sa.ts_pct as away_ts_pct, sa.pace as away_pace, sa.pie as away_pie
                
            FROM games g
            LEFT JOIN game_stats sh ON g.game_id = sh.game_id AND sh.is_home = TRUE
            LEFT JOIN game_stats sa ON g.game_id = sa.game_id AND sa.is_home = FALSE
            ORDER BY g.date DESC
            '''
            
            df = pd.read_sql_query(self._prepare_query(query), conn)
            
            # Adicionar colunas de compatibilidade que podem estar faltando
            if not df.empty:
                df['correct'] = 0 # Placeholder
            else:
                return pd.DataFrame(columns=expected_cols)
                
            return df
        except Exception as e:
            logger.error(f"Erro get_history: {e}")
            return pd.DataFrame(columns=expected_cols)
        finally:
            self.return_connection(conn)

    @retry_on_lock()
    def get_latest_predictions(self, date_str=None):
        """Retorna previsões salvas na tabela predictions."""
        conn = self.get_connection()
        try:
            query = "SELECT * FROM predictions"
            params = []
            
            if date_str:
                query += " WHERE date = ?"
                params.append(date_str)
            
            query += " ORDER BY date DESC, prob_home DESC"
            
            df = pd.read_sql_query(self._prepare_query(query), conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Erro get_latest_predictions: {e}")
            return pd.DataFrame()
        finally:
            self.return_connection(conn)

    def get_comprehensive_history(self):
        """Alias para get_history para compatibilidade."""
        return self.get_history()

    @retry_on_lock()
    def get_pending_games(self):
        """Retorna jogos pendentes da tabela de previsões."""
        conn = self.get_connection()
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            query = "SELECT * FROM predictions WHERE date >= ? ORDER BY date ASC"
            df = pd.read_sql_query(self._prepare_query(query), conn, params=(today,))
            return df
        finally:
            self.return_connection(conn)

    def update_pending_results(self):
        """
        Atualiza resultados. 
        """
        logger.info("ℹ️ update_pending_results agora é responsabilidade do fetch_historical_data (pipeline de ingestão).")
        return 0

# Singleton helper
_db_instance = None

def get_db_manager():
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
