"""
AsyncDataManager - DataManager Assíncrono com SQLAlchemy
=========================================================
Substitui o DatabaseManager síncrono atual para alta performance.
Suporta PostgreSQL (produção) e SQLite (desenvolvimento).

Autor: NBA Predictor v22.0
"""

from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, Index
from sqlalchemy import select, insert, update, delete, func
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import logging

# Garantir que .env seja carregado do diretório correto
from pathlib import Path
from dotenv import load_dotenv

# Encontrar o diretório raiz do projeto (2 níveis acima de infrastructure/)
_project_root = Path(__file__).parent.parent
_env_path = _project_root / '.env'
load_dotenv(_env_path)

logger = logging.getLogger(__name__)

# Base para modelos SQLAlchemy
Base = declarative_base()


# ============= MODELOS ORM =============

class Game(Base):
    """Modelo para jogos da NBA"""
    __tablename__ = 'games'
    
    game_id = Column(String(50), primary_key=True)
    date = Column(String(10), index=True)
    season = Column(String(10))
    home_team = Column(String(20), index=True)
    away_team = Column(String(20), index=True)
    home_score = Column(Integer)
    away_score = Column(Integer)
    winner = Column(String(10))
    status = Column(String(20), default='Final')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_games_date_teams', 'date', 'home_team', 'away_team'),
    )


class GameStats(Base):
    """Modelo para estatísticas de jogos"""
    __tablename__ = 'game_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String(50), index=True)
    team_id = Column(String(10), index=True)
    is_home = Column(Boolean)
    pts = Column(Integer)
    fgm = Column(Integer)
    fga = Column(Integer)
    fg_pct = Column(Float)
    fg3m = Column(Integer)
    fg3a = Column(Integer)
    fg3_pct = Column(Float)
    ftm = Column(Integer)
    fta = Column(Integer)
    ft_pct = Column(Float)
    oreb = Column(Integer)
    dreb = Column(Integer)
    reb = Column(Integer)
    ast = Column(Integer)
    stl = Column(Integer)
    blk = Column(Integer)
    tov = Column(Integer)
    pf = Column(Integer)
    plus_minus = Column(Integer)
    off_rating = Column(Float)
    def_rating = Column(Float)
    efg_pct = Column(Float)
    ts_pct = Column(Float)
    pace = Column(Float)
    pie = Column(Float)
    
    __table_args__ = (
        Index('idx_stats_game_team', 'game_id', 'team_id', unique=True),
    )


class Prediction(Base):
    """Modelo para previsões"""
    __tablename__ = 'predictions'
    
    game_id = Column(String(50), primary_key=True)
    date = Column(String(10), index=True)
    home_team = Column(String(20))
    away_team = Column(String(20))
    prob_home = Column(Float)
    prob_away = Column(Float)
    prob_mc_home = Column(Float)
    prob_mc_away = Column(Float)
    odd_home = Column(Float)
    odd_away = Column(Float)
    prediction = Column(String(50))
    confidence = Column(String(20))
    predicted_spread = Column(Float)
    predicted_total = Column(Float)
    ci_lower = Column(Float)
    ci_upper = Column(Float)
    model_version = Column(String(20))
    home_injuries_list = Column(Text)
    away_injuries_list = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Bet(Base):
    """Modelo para apostas"""
    __tablename__ = 'bets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bet_date = Column(String(10), index=True)
    game_id = Column(String(50), index=True)
    home_team = Column(String(20))
    away_team = Column(String(20))
    side = Column(String(10))
    bet_type = Column(String(20))
    line = Column(Float)
    opening_odds = Column(Float)
    bet_odds = Column(Float)
    closing_odds = Column(Float)
    stake_pct = Column(Float)
    stake_amount = Column(Float)
    bankroll_at_bet = Column(Float)
    model_prob = Column(Float)
    ev_pct = Column(Float)
    kelly_fraction = Column(Float)
    result = Column(String(10), default='PENDING', index=True)
    payout = Column(Float, default=0)
    profit = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(String(30))


class FeatureRecord(Base):
    """Modelo para Feature Store"""
    __tablename__ = 'feature_store'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(50), index=True)
    entity_type = Column(String(10))
    feature_name = Column(String(100), index=True)
    feature_value = Column(Float)
    valid_from = Column(DateTime, index=True)
    valid_to = Column(DateTime, nullable=True)
    version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_feature_lookup', 'entity_id', 'feature_name', 'valid_from'),
    )


# ============= ASYNC DATA MANAGER =============

class AsyncDataManager:
    """
    DataManager assíncrono para alta performance.
    
    Características:
    - Connection pooling com SQLAlchemy Async
    - Suporte a PostgreSQL (produção) e SQLite (dev)
    - Context managers para transações seguras
    - Métodos bulk para alta performance
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Inicializa o DataManager.
        
        Args:
            db_url: URL de conexão. Se None, usa variáveis de ambiente.
        """
        self.db_url = db_url or self._build_url()
        
        # Configurar parâmetros específicos para asyncpg (desabilitar SSL por padrão)
        connect_args = {}
        if 'postgresql' in self.db_url:
            connect_args = {'ssl': False}  # Desabilitar SSL para conexões locais
        
        self.engine: AsyncEngine = create_async_engine(
            self.db_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            echo=os.getenv('DEBUG', 'false').lower() == 'true',
            connect_args=connect_args
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        self._initialized = False
        
        logger.info(f"📂 AsyncDataManager inicializado: {self._get_db_type()}")
    
    def _build_url(self) -> str:
        """Monta URL de conexão baseado em variáveis de ambiente"""
        from urllib.parse import quote_plus
        
        db_type = os.getenv('DB_TYPE', 'sqlite').lower()
        db_host = os.getenv('DB_HOST', 'localhost')
        
        # Debug: mostrar valores
        logger.debug(f"🔍 DB_TYPE={db_type}, DB_HOST={db_host}")
        
        if db_type == 'postgres':
            # URL-encode a senha para tratar caracteres especiais como @
            db_pass = quote_plus(os.getenv('DB_PASS', 'password'))
            db_user = quote_plus(os.getenv('DB_USER', 'nba_admin'))
            
            url = (
                f"postgresql+asyncpg://"
                f"{db_user}:{db_pass}@"
                f"{db_host}:"
                f"{os.getenv('DB_PORT', '5432')}/"
                f"{os.getenv('DB_NAME', 'nba_predictor_db')}"
            )
            # Mostrar URL (com senha mascarada)
            safe_url = url.replace(db_pass, '****')
            logger.info(f"🔗 URL PostgreSQL: {safe_url}")
            return url
        else:
            db_path = os.getenv('DB_PATH', 'data/nba_games.db')
            return f"sqlite+aiosqlite:///{db_path}"
    
    def _get_db_type(self) -> str:
        """Retorna tipo do banco de dados"""
        if 'postgresql' in self.db_url:
            return 'PostgreSQL'
        return 'SQLite'
    
    async def init_db(self):
        """Inicializa esquema do banco de dados"""
        if self._initialized:
            return
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        self._initialized = True
        logger.info("✅ Esquema do banco de dados inicializado")
    
    @asynccontextmanager
    async def get_session(self):
        """Context manager para sessões de banco de dados"""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"❌ Erro na transação: {e}")
                raise
    
    # ============= MÉTODOS CRUD GAMES =============
    
    async def insert_game(self, game_data: Dict[str, Any]) -> bool:
        """Insere um jogo no banco de dados"""
        async with self.get_session() as session:
            game = Game(**game_data)
            session.add(game)
            return True
    
    async def bulk_insert_games(self, games: List[Dict[str, Any]]) -> int:
        """Insere múltiplos jogos em lote"""
        if not games:
            return 0
        
        await self.init_db()
        
        async with self.get_session() as session:
            for game_data in games:
                if 'postgresql' in self.db_url:
                    # Usar insert do dialeto PostgreSQL para ter on_conflict_do_update
                    from sqlalchemy.dialects.postgresql import insert as pg_insert
                    stmt = pg_insert(Game).values(**game_data)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['game_id'],
                        set_={
                            'home_score': stmt.excluded.home_score,
                            'away_score': stmt.excluded.away_score,
                            'winner': stmt.excluded.winner,
                            'status': stmt.excluded.status
                        }
                    )
                else:
                    # SQLite: usar insert genérico
                    stmt = insert(Game).values(**game_data)
                
                await session.execute(stmt)
        
        logger.info(f"💾 {len(games)} jogos inseridos/atualizados")
        return len(games)
    
    async def get_games_by_date(self, date: str) -> List[Game]:
        """Busca jogos por data"""
        async with self.get_session() as session:
            result = await session.execute(
                select(Game).where(Game.date == date)
            )
            return result.scalars().all()
    
    async def get_history(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Busca histórico de jogos com estatísticas.
        Compatível com interface do DatabaseManager legado.
        """
        async with self.get_session() as session:
            result = await session.execute(
                select(Game, GameStats)
                .outerjoin(GameStats, Game.game_id == GameStats.game_id)
                .order_by(Game.date.desc())
                .limit(limit)
            )
            
            rows = []
            for game, stats in result.all():
                row = {
                    'game_id': game.game_id,
                    'date': game.date,
                    'home_team': game.home_team,
                    'away_team': game.away_team,
                    'home_score': game.home_score,
                    'away_score': game.away_score,
                    'winner': game.winner
                }
                if stats:
                    row.update({
                        'pts': stats.pts,
                        'fg_pct': stats.fg_pct,
                        'off_rating': stats.off_rating,
                        'def_rating': stats.def_rating
                    })
                rows.append(row)
            
            return rows
    
    # ============= MÉTODOS CRUD PREDICTIONS =============
    
    async def save_predictions(self, predictions: List[Dict[str, Any]]) -> int:
        """Salva previsões no banco de dados"""
        if not predictions:
            return 0
        
        await self.init_db()
        
        async with self.get_session() as session:
            for pred in predictions:
                game_id = f"{pred['Data']}_{pred['Casa']}_{pred['Visitante']}".replace(" ", "")
                
                prediction = Prediction(
                    game_id=game_id,
                    date=pred['Data'],
                    home_team=pred['Casa'],
                    away_team=pred['Visitante'],
                    prob_home=pred.get('Prob Casa %', 0),
                    prob_away=pred.get('Prob Visitante %', 0),
                    prob_mc_home=pred.get('Prob MC Casa %', 0),
                    prob_mc_away=pred.get('Prob MC Visitante %', 0),
                    odd_home=pred.get('Odd Casa', 0),
                    odd_away=pred.get('Odd Visitante', 0),
                    prediction=pred.get('Previsão', 'N/A'),
                    confidence=pred.get('Confiança', 'N/A'),
                    predicted_spread=pred.get('Spread Previsto', 0),
                    predicted_total=pred.get('Total Previsto', 0),
                    ci_lower=pred.get('ci_lower', 0),
                    ci_upper=pred.get('ci_upper', 0),
                    model_version='v22.0',
                    home_injuries_list=pred.get('home_injuries_list', ''),
                    away_injuries_list=pred.get('away_injuries_list', '')
                )
                
                await session.merge(prediction)
        
        logger.info(f"💾 {len(predictions)} previsões salvas")
        return len(predictions)
    
    async def get_predictions_by_date(self, date: str) -> List[Prediction]:
        """Busca previsões por data"""
        async with self.get_session() as session:
            result = await session.execute(
                select(Prediction).where(Prediction.date == date)
            )
            return result.scalars().all()
    
    # ============= MÉTODOS CRUD BETS =============
    
    async def save_bet(self, bet_data: Dict[str, Any]) -> int:
        """Salva uma aposta"""
        async with self.get_session() as session:
            bet = Bet(**bet_data)
            session.add(bet)
            await session.flush()
            return bet.id
    
    async def get_pending_bets(self) -> List[Bet]:
        """Busca apostas pendentes"""
        async with self.get_session() as session:
            result = await session.execute(
                select(Bet).where(Bet.result == 'PENDING')
            )
            return result.scalars().all()
    
    async def settle_bet(self, bet_id: int, result: str, payout: float, profit: float):
        """Liquida uma aposta"""
        async with self.get_session() as session:
            await session.execute(
                update(Bet)
                .where(Bet.id == bet_id)
                .values(
                    result=result,
                    payout=payout,
                    profit=profit,
                    settled_at=datetime.utcnow().isoformat()
                )
            )
    
    # ============= MÉTODOS FEATURE STORE =============
    
    async def save_feature(self, entity_id: str, entity_type: str,
                           feature_name: str, value: float,
                           valid_from: datetime, version: str = '1.0'):
        """Salva uma feature no Feature Store"""
        async with self.get_session() as session:
            record = FeatureRecord(
                entity_id=entity_id,
                entity_type=entity_type,
                feature_name=feature_name,
                feature_value=value,
                valid_from=valid_from,
                version=version
            )
            session.add(record)
    
    async def get_feature(self, entity_id: str, feature_name: str,
                          as_of: datetime) -> Optional[float]:
        """
        Busca feature com point-in-time correctness.
        Retorna o valor mais recente válido para a data.
        """
        async with self.get_session() as session:
            result = await session.execute(
                select(FeatureRecord.feature_value)
                .where(FeatureRecord.entity_id == entity_id)
                .where(FeatureRecord.feature_name == feature_name)
                .where(FeatureRecord.valid_from <= as_of)
                .order_by(FeatureRecord.valid_from.desc())
                .limit(1)
            )
            row = result.scalar()
            return row if row is not None else None
    
    # ============= MÉTODOS UTILITÁRIOS =============
    
    async def count_games(self) -> int:
        """Conta total de jogos"""
        async with self.get_session() as session:
            result = await session.execute(select(func.count(Game.game_id)))
            return result.scalar() or 0
    
    async def count_predictions(self) -> int:
        """Conta total de previsões"""
        async with self.get_session() as session:
            result = await session.execute(select(func.count(Prediction.game_id)))
            return result.scalar() or 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica saúde da conexão"""
        try:
            async with self.get_session() as session:
                await session.execute(select(1))
            
            return {
                'status': 'healthy',
                'db_type': self._get_db_type(),
                'games_count': await self.count_games(),
                'predictions_count': await self.count_predictions()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def close(self):
        """Fecha conexões"""
        await self.engine.dispose()
        logger.info("🔒 AsyncDataManager fechado")


# ============= SINGLETON HELPER =============

_db_instance: Optional[AsyncDataManager] = None


async def get_async_db() -> AsyncDataManager:
    """
    Retorna instância singleton do AsyncDataManager.
    
    Uso:
        db = await get_async_db()
        games = await db.get_games_by_date('2024-12-14')
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = AsyncDataManager()
        await _db_instance.init_db()
    return _db_instance


async def reset_db_instance():
    """Reseta singleton (útil para testes)"""
    global _db_instance
    if _db_instance:
        await _db_instance.close()
    _db_instance = None
