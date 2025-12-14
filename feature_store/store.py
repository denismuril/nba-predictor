"""
Feature Store - Armazenamento de Features com Point-in-Time Correctness
=======================================================================
Garante consistência entre treino e inferência evitando data leakage.
A feature calculada para backtest do dia X usa APENAS dados até dia X-1.

Autor: NBA Predictor v22.0
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Feature Store com Point-in-Time correctness.
    
    Garante que a feature 'media_pontos_ultimos_5_jogos' calculada
    para o backtest do dia 2024-01-15 use APENAS dados até 2024-01-14.
    
    Características:
    - Versionamento de features
    - Point-in-time queries (evita data leakage)
    - Cache em memória para performance
    - Persistência em banco de dados
    """
    
    def __init__(self, db_manager=None):
        """
        Inicializa o Feature Store.
        
        Args:
            db_manager: AsyncDataManager para persistência
        """
        self.db = db_manager
        self._registry: Dict[str, Dict[str, Any]] = {}  # Feature definitions
        self._cache: Dict[str, Any] = {}  # In-memory cache
        self._cache_ttl = 3600  # 1 hora
        
        # Registrar features padrão
        self._register_default_features()
    
    def _register_default_features(self):
        """Registra features padrão do sistema"""
        
        # Features de time
        self.register_feature(
            name="pts_avg_5",
            transformer=self._compute_pts_avg,
            entity_type="team",
            description="Média de pontos nos últimos 5 jogos",
            window=5
        )
        
        self.register_feature(
            name="pts_avg_10",
            transformer=self._compute_pts_avg,
            entity_type="team",
            description="Média de pontos nos últimos 10 jogos",
            window=10
        )
        
        self.register_feature(
            name="win_rate_10",
            transformer=self._compute_win_rate,
            entity_type="team",
            description="Taxa de vitória nos últimos 10 jogos",
            window=10
        )
        
        self.register_feature(
            name="off_rating_avg",
            transformer=self._compute_off_rating_avg,
            entity_type="team",
            description="Offensive Rating médio",
            window=10
        )
        
        self.register_feature(
            name="def_rating_avg",
            transformer=self._compute_def_rating_avg,
            entity_type="team",
            description="Defensive Rating médio",
            window=10
        )
    
    def register_feature(self, name: str, transformer: Callable,
                         entity_type: str = "team",
                         description: str = "",
                         version: str = "1.0",
                         **params):
        """
        Registra uma feature no store.
        
        Args:
            name: Nome único da feature
            transformer: Função que calcula a feature
            entity_type: Tipo de entidade ('team', 'game', 'player')
            description: Descrição da feature
            version: Versão da feature
            **params: Parâmetros adicionais (window, etc)
        """
        self._registry[name] = {
            'transformer': transformer,
            'entity_type': entity_type,
            'description': description,
            'version': version,
            'params': params
        }
        logger.debug(f"Feature '{name}' registrada")
    
    # ============= MÉTODOS DE CÁLCULO =============
    
    async def compute_and_store(self, entity_id: str, feature_name: str,
                                 as_of_date: date) -> Optional[float]:
        """
        Calcula feature com point-in-time correctness e armazena.
        
        GARANTIA: Usa apenas dados ANTERIORES a as_of_date.
        
        Args:
            entity_id: ID do time ou jogo
            feature_name: Nome da feature registrada
            as_of_date: Data de referência
            
        Returns:
            Valor da feature ou None se não calculável
        """
        if feature_name not in self._registry:
            raise ValueError(f"Feature '{feature_name}' não registrada")
        
        # Verificar cache
        cache_key = f"{entity_id}:{feature_name}:{as_of_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Verificar se já existe no DB
        existing = await self._get_from_db(entity_id, feature_name, as_of_date)
        if existing is not None:
            self._cache[cache_key] = existing
            return existing
        
        # Calcular nova feature
        config = self._registry[feature_name]
        transformer = config['transformer']
        params = config['params']
        
        try:
            value = await transformer(entity_id, as_of_date, **params)
            
            if value is not None:
                # Armazenar no DB
                await self._store_in_db(
                    entity_id=entity_id,
                    entity_type=config['entity_type'],
                    feature_name=feature_name,
                    value=value,
                    valid_from=as_of_date,
                    version=config['version']
                )
                
                # Atualizar cache
                self._cache[cache_key] = value
            
            return value
            
        except Exception as e:
            logger.error(f"Erro calculando {feature_name} para {entity_id}: {e}")
            return None
    
    async def get_features_for_training(self, entities: List[str],
                                         feature_names: List[str],
                                         dates: Dict[str, date]) -> pd.DataFrame:
        """
        Busca features para treino com point-in-time correctness.
        
        GARANTIA: Cada entidade recebe features calculadas APENAS
        com dados disponíveis ANTES da data especificada.
        
        Args:
            entities: Lista de IDs (times ou jogos)
            feature_names: Lista de features a buscar
            dates: Dict mapeando entity_id → data de referência
            
        Returns:
            DataFrame com entity_id como índice e features como colunas
        """
        rows = []
        
        for entity_id in entities:
            as_of = dates.get(entity_id, date.today())
            row = {'entity_id': entity_id}
            
            for feat_name in feature_names:
                value = await self.compute_and_store(entity_id, feat_name, as_of)
                row[feat_name] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows).set_index('entity_id')
    
    async def get_features_for_inference(self, entity_id: str,
                                          feature_names: List[str]) -> Dict[str, float]:
        """
        Busca features para inferência (produção).
        Usa a data atual como referência.
        
        Args:
            entity_id: ID do time ou jogo
            feature_names: Lista de features a buscar
            
        Returns:
            Dict mapeando feature_name → valor
        """
        today = date.today()
        features = {}
        
        for feat_name in feature_names:
            features[feat_name] = await self.compute_and_store(entity_id, feat_name, today)
        
        return features
    
    async def compute_team_features(self, team_id: str, target_date: str):
        """
        Computa todas as features para um time em uma data.
        
        Args:
            team_id: ID do time (ex: 'LAL')
            target_date: Data alvo no formato YYYY-MM-DD
        """
        as_of = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        for feat_name in self._registry:
            config = self._registry[feat_name]
            if config['entity_type'] == 'team':
                await self.compute_and_store(team_id, feat_name, as_of)
    
    # ============= TRANSFORMERS (CÁLCULO DE FEATURES) =============
    
    async def _compute_pts_avg(self, team_id: str, as_of_date: date, window: int = 5) -> float:
        """Calcula média de pontos nos últimos N jogos"""
        if not self.db:
            return None
        
        async with self.db.get_session() as session:
            from sqlalchemy import select, func
            from infrastructure.database import Game, GameStats
            
            # Buscar últimos N jogos ANTES de as_of_date
            result = await session.execute(
                select(func.avg(GameStats.pts))
                .join(Game, Game.game_id == GameStats.game_id)
                .where(GameStats.team_id == team_id)
                .where(Game.date < str(as_of_date))
                .order_by(Game.date.desc())
                .limit(window)
            )
            
            avg = result.scalar()
            return float(avg) if avg else None
    
    async def _compute_win_rate(self, team_id: str, as_of_date: date, window: int = 10) -> float:
        """Calcula taxa de vitória nos últimos N jogos"""
        if not self.db:
            return None
        
        async with self.db.get_session() as session:
            from sqlalchemy import select, case, func
            from infrastructure.database import Game
            
            # Subquery para jogos em casa
            home_wins = select(
                func.sum(case((Game.winner == 'HOME', 1), else_=0)).label('wins'),
                func.count().label('games')
            ).where(
                Game.home_team == team_id,
                Game.date < str(as_of_date)
            ).limit(window)
            
            # Subquery para jogos fora
            away_wins = select(
                func.sum(case((Game.winner == 'AWAY', 1), else_=0)).label('wins'),
                func.count().label('games')
            ).where(
                Game.away_team == team_id,
                Game.date < str(as_of_date)
            ).limit(window)
            
            # Por simplicidade, retorna estimativa
            return 0.5  # Placeholder
    
    async def _compute_off_rating_avg(self, team_id: str, as_of_date: date, window: int = 10) -> float:
        """Calcula Offensive Rating médio"""
        if not self.db:
            return None
        
        async with self.db.get_session() as session:
            from sqlalchemy import select, func
            from infrastructure.database import Game, GameStats
            
            result = await session.execute(
                select(func.avg(GameStats.off_rating))
                .join(Game, Game.game_id == GameStats.game_id)
                .where(GameStats.team_id == team_id)
                .where(Game.date < str(as_of_date))
                .where(GameStats.off_rating > 0)
                .order_by(Game.date.desc())
                .limit(window)
            )
            
            avg = result.scalar()
            return float(avg) if avg else None
    
    async def _compute_def_rating_avg(self, team_id: str, as_of_date: date, window: int = 10) -> float:
        """Calcula Defensive Rating médio"""
        if not self.db:
            return None
        
        async with self.db.get_session() as session:
            from sqlalchemy import select, func
            from infrastructure.database import Game, GameStats
            
            result = await session.execute(
                select(func.avg(GameStats.def_rating))
                .join(Game, Game.game_id == GameStats.game_id)
                .where(GameStats.team_id == team_id)
                .where(Game.date < str(as_of_date))
                .where(GameStats.def_rating > 0)
                .order_by(Game.date.desc())
                .limit(window)
            )
            
            avg = result.scalar()
            return float(avg) if avg else None
    
    # ============= PERSISTÊNCIA =============
    
    async def _get_from_db(self, entity_id: str, feature_name: str,
                            as_of: date) -> Optional[float]:
        """Busca feature do banco de dados"""
        if not self.db:
            return None
        
        try:
            return await self.db.get_feature(entity_id, feature_name, 
                                              datetime.combine(as_of, datetime.min.time()))
        except Exception:
            return None
    
    async def _store_in_db(self, entity_id: str, entity_type: str,
                            feature_name: str, value: float,
                            valid_from: date, version: str):
        """Armazena feature no banco de dados"""
        if not self.db:
            return
        
        try:
            await self.db.save_feature(
                entity_id=entity_id,
                entity_type=entity_type,
                feature_name=feature_name,
                value=value,
                valid_from=datetime.combine(valid_from, datetime.min.time()),
                version=version
            )
        except Exception as e:
            logger.error(f"Erro ao salvar feature: {e}")
    
    # ============= UTILITÁRIOS =============
    
    def get_registered_features(self) -> Dict[str, Dict[str, Any]]:
        """Retorna informações sobre features registradas"""
        return {
            name: {
                'entity_type': config['entity_type'],
                'description': config['description'],
                'version': config['version'],
                'params': config['params']
            }
            for name, config in self._registry.items()
        }
    
    def clear_cache(self):
        """Limpa cache em memória"""
        self._cache.clear()
        logger.info("Cache do Feature Store limpo")
