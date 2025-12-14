"""
NBA Predictor Orchestrator v23.0 (Enterprise Edition)
=====================================================
Gerencia o fluxo diário de execução com infraestrutura enterprise:
1. Coleta de Dados (Async + Paralelo)
2. Validação Pydantic
3. Cache Redis
4. Treinamento (Opcional)
5. Previsão
6. Dashboard

NOVIDADES v23.0:
- AsyncDataManager (PostgreSQL)
- RedisCache com TTLs
- Prefect Flows nativos
- asyncio.gather para paralelização
- Circuit Breaker integrado
- Eliminação de subprocess.run

Autor: NBA Predictor Enterprise
"""
import sys
import logging
import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Setup paths
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

# Configuração de Logs Enterprise
try:
    from infrastructure.logging_config import setup_structured_logging
    setup_structured_logging(level="INFO", json_file="logs/orchestrator.jsonl")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/orchestrator.log"),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger("Orchestrator")

# Versão do sistema
VERSION = "23.0"


class EnterpriseOrchestrator:
    """
    Orquestrador Enterprise com infraestrutura async.
    
    Características:
    - AsyncDataManager para PostgreSQL
    - RedisCache para cache distribuído
    - Prefect flows para ETL
    - Circuit Breaker para resiliência
    - Rate Limiter para APIs
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.db = None
        self.redis = None
        self.stats = {
            'games_processed': 0,
            'odds_fetched': 0,
            'injuries_fetched': 0,
            'predictions_generated': 0,
            'errors': []
        }
        
        # Circuit Breaker Registry
        try:
            from infrastructure.circuit_breaker import CircuitBreakerRegistry
            self.circuit_breaker = CircuitBreakerRegistry
            self.circuit_breaker_enabled = True
        except ImportError:
            self.circuit_breaker = None
            self.circuit_breaker_enabled = False
    
    async def initialize(self):
        """Inicializa conexões assíncronas."""
        logger.info("🚀 Inicializando Enterprise Orchestrator v23.0...")
        
        # 1. Banco de Dados Async
        try:
            from infrastructure.database import get_async_db
            self.db = await get_async_db()
            health = await self.db.health_check()
            logger.info(f"✅ Database: {health['db_type']} ({health.get('games_count', 0)} jogos)")
        except Exception as e:
            logger.warning(f"⚠️ Database fallback: {e}")
            # Fallback para db_manager legado
            from data.repositories.db_manager import get_db_manager
            self.db = get_db_manager()
        
        # 2. Redis Cache
        try:
            from infrastructure.redis_cache import get_redis
            self.redis = await get_redis()
            health = await self.redis.health_check()
            if health['status'] == 'healthy':
                logger.info(f"✅ Redis: v{health.get('redis_version', 'N/A')} ({health.get('keys_count', 0)} keys)")
            else:
                logger.info("ℹ️ Redis não disponível - usando cache local")
        except Exception as e:
            logger.info(f"ℹ️ Redis não disponível: {e}")
            self.redis = None
        
        # 3. Rate Limiter
        try:
            from infrastructure.rate_limiter import get_rate_limiter
            self.rate_limiter = await get_rate_limiter()
            logger.info("✅ Rate Limiter inicializado")
        except ImportError:
            self.rate_limiter = None
    
    async def run_step(self, step_name: str, coro, *args, **kwargs):
        """
        Executa uma etapa assíncrona com tratamento de erro e medição de tempo.
        
        Args:
            step_name: Nome da etapa para logging
            coro: Coroutine a executar
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"🚀 Iniciando Etapa: {step_name}")
        logger.info(f"{'='*50}")
        
        start = time.time()
        try:
            result = await coro(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"✅ Etapa '{step_name}' concluída em {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"❌ Falha na etapa '{step_name}' após {duration:.2f}s: {e}", exc_info=True)
            self.stats['errors'].append({'step': step_name, 'error': str(e)})
            
            # Etapas críticas levantam exceção
            if step_name in ['Data Collection', 'Prediction Generation']:
                raise
            return None
    
    # ============= ETAPAS DO PIPELINE =============
    
    async def step_data_collection(self) -> Dict[str, Any]:
        """
        Coleta dados usando Prefect Flow ou fallback direto.
        
        Usa o daily_data_flow do Prefect que:
        - Busca jogos agendados
        - Valida com Pydantic
        - Busca odds e lesões
        - Armazena no DB e Redis
        """
        logger.info("📡 Coletando dados com ETL Enterprise...")
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Tentar usar Prefect Flow
            from etl.flows.daily_data_flow import daily_data_flow
            
            result = await daily_data_flow(date=today, skip_features=False)
            
            self.stats['games_processed'] = result.get('games_valid', 0)
            self.stats['odds_fetched'] = result.get('odds_valid', 0)
            self.stats['injuries_fetched'] = result.get('injuries_count', 0)
            
            logger.info(f"📊 ETL Stats: {result['games_valid']} jogos, {result['odds_valid']} odds, {result['injuries_count']} lesões")
            
            return result
            
        except ImportError:
            logger.warning("⚠️ Prefect não disponível - usando coleta direta")
            return await self._fallback_data_collection(today)
        except Exception as e:
            logger.warning(f"⚠️ ETL Flow falhou: {e} - usando fallback")
            return await self._fallback_data_collection(today)
    
    async def _fallback_data_collection(self, date: str) -> Dict[str, Any]:
        """Coleta direta sem Prefect (fallback)."""
        result = {'games_valid': 0, 'odds_valid': 0, 'injuries_count': 0}
        
        # Paralelizar coleta
        async def fetch_games():
            try:
                from data.scrapers.schedule_scraper import get_todays_games
                games = await asyncio.to_thread(get_todays_games, date)
                return games or []
            except Exception as e:
                logger.warning(f"Erro ao buscar jogos: {e}")
                return []
        
        async def fetch_odds():
            try:
                from data.scrapers.odds_scraper import obter_odds
                odds = await asyncio.to_thread(obter_odds)
                return odds or {}
            except Exception as e:
                logger.warning(f"Erro ao buscar odds: {e}")
                return {}
        
        async def fetch_injuries():
            try:
                from data.scrapers.injury_scraper_v2 import get_all_injuries
                injuries = await asyncio.to_thread(get_all_injuries)
                return injuries or {}
            except Exception as e:
                logger.warning(f"Erro ao buscar lesões: {e}")
                return {}
        
        # Executar em paralelo com asyncio.gather
        games, odds, injuries = await asyncio.gather(
            fetch_games(),
            fetch_odds(),
            fetch_injuries(),
            return_exceptions=True
        )
        
        # Processar resultados
        if isinstance(games, list):
            result['games_valid'] = len(games)
            self.stats['games_processed'] = len(games)
        
        if isinstance(odds, dict):
            result['odds_valid'] = len(odds)
            self.stats['odds_fetched'] = len(odds)
            # Cachear no Redis se disponível
            if self.redis:
                for game_id, odd_data in odds.items():
                    await self.redis.set_odds(game_id, odd_data)
        
        if isinstance(injuries, dict):
            result['injuries_count'] = sum(len(v) for v in injuries.values())
            self.stats['injuries_fetched'] = result['injuries_count']
            # Cachear no Redis se disponível
            if self.redis:
                for team_id, team_injuries in injuries.items():
                    await self.redis.set_injuries(team_id, team_injuries)
        
        return result
    
    async def step_twitter_collection(self) -> bool:
        """Coleta tweets de insiders para sentiment analysis."""
        logger.info("🐦 Coletando tweets de lesão (Woj/Shams)...")
        
        try:
            from data.scrapers.twitter_scraper import fetch_latest_injury_tweets
            from ml_pipeline.sentiment import NewsSentimentAnalyzer
            
            tweets = await asyncio.to_thread(fetch_latest_injury_tweets)
            
            if tweets:
                analyzer = NewsSentimentAnalyzer()
                sentiments = analyzer.analyze_tweets(tweets)
                logger.info(f"📰 {len(tweets)} tweets coletados, {len(sentiments)} times afetados")
                
                # Salvar em cache
                cache_file = Path('data/cache/sentiment_cache.json')
                cache_file.parent.mkdir(exist_ok=True)
                import json
                with open(cache_file, 'w') as f:
                    json.dump({'tweets': len(tweets), 'sentiments': sentiments}, f)
            else:
                logger.info("✅ Nenhum tweet crítico encontrado")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Twitter collection failed (não crítico): {e}")
            return True  # Não bloqueia pipeline
    
    async def step_model_training(self, force: bool = False) -> bool:
        """
        Treina modelos se necessário.
        
        Verifica data do último treino e retreina se:
        - force=True
        - Último treino foi em dia diferente
        """
        last_train_file = Path("logs/last_train_date.txt")
        should_train = force
        
        if not should_train and last_train_file.exists():
            last_date = last_train_file.read_text().strip()
            if last_date != datetime.now().strftime("%Y-%m-%d"):
                should_train = True
        else:
            should_train = True
        
        if should_train:
            logger.info("🧠 Iniciando retreino dos modelos V18/V6...")
            
            try:
                # Treinar modelos de forma assíncrona
                from ml_pipeline.train_all import train_all_models
                
                await asyncio.to_thread(train_all_models)
                
                # Atualizar data de treino
                last_train_file.parent.mkdir(exist_ok=True)
                last_train_file.write_text(datetime.now().strftime("%Y-%m-%d"))
                
                logger.info("✅ Modelos treinados com sucesso")
                return True
                
            except ImportError:
                logger.warning("⚠️ train_all não disponível - pulando treino")
            except Exception as e:
                logger.error(f"❌ Erro no treino: {e}")
                return False
        else:
            logger.info("⏭️ Pulando treinamento (já treinado hoje)")
        
        return True
    
    async def step_prediction(self) -> List[Dict[str, Any]]:
        """
        Gera previsões para os jogos de hoje usando modelos ML.
        
        Usa o pipeline de inferência diretamente, sem subprocess.
        """
        logger.info("🔮 Gerando previsões com ML (V18/V6)...")
        
        try:
            from interfaces.cli import run_predictions
            
            predictions = await asyncio.to_thread(run_predictions, use_ml=True)
            
            if predictions:
                self.stats['predictions_generated'] = len(predictions)
                logger.info(f"✅ {len(predictions)} previsões geradas")
                
                # Salvar no banco async se disponível
                if self.db and hasattr(self.db, 'save_predictions'):
                    await self.db.save_predictions(predictions)
                
                # Cachear no Redis
                if self.redis:
                    today = datetime.now().strftime('%Y-%m-%d')
                    await self.redis.set_predictions(today, predictions)
            
            return predictions or []
            
        except ImportError:
            logger.warning("⚠️ CLI não disponível - usando main.py direto")
            # Fallback para execução direta
            try:
                from core.prediction_engine import PredictionEngine
                engine = PredictionEngine()
                predictions = await asyncio.to_thread(engine.generate_predictions)
                self.stats['predictions_generated'] = len(predictions) if predictions else 0
                return predictions or []
            except Exception as e:
                logger.error(f"❌ Erro na predição: {e}")
                return []
    
    async def step_fetch_odds(self) -> Dict[str, Any]:
        """
        Busca odds reais com fallback multi-fonte.
        
        Hierarquia:
        1. Cache Redis (se < 5 min)
        2. OddsPedia (scraping gratuito)
        3. TheOddsAPI (pago)
        4. Fair Odds calculadas
        """
        logger.info("💰 Buscando odds reais...")
        
        # Verificar cache Redis primeiro
        if self.redis:
            try:
                from data.scrapers.schedule_scraper import get_todays_games
                games = await asyncio.to_thread(get_todays_games)
                
                if games:
                    game_ids = [f"{g.get('date')}_{g.get('home_team')}_{g.get('away_team')}" for g in games]
                    cached_odds = await self.redis.get_all_odds(game_ids)
                    
                    if len(cached_odds) >= len(games) * 0.8:  # 80% cached
                        logger.info(f"✅ {len(cached_odds)} odds do cache Redis")
                        return cached_odds
            except Exception as e:
                logger.debug(f"Cache check falhou: {e}")
        
        # Buscar odds frescos
        try:
            from data.scrapers.odds_scraper import obter_odds
            
            # Rate limiting se disponível
            if self.rate_limiter:
                await self.rate_limiter.wait_and_acquire('oddspedia', max_wait=30)
            
            odds_data = await asyncio.to_thread(obter_odds)
            
            if odds_data:
                logger.info(f"✅ {len(odds_data)} jogos com odds encontrados")
                
                # Cachear no Redis
                if self.redis:
                    for game_id, odds in odds_data.items():
                        await self.redis.set_odds(game_id, odds)
                
                return odds_data
            else:
                logger.warning("⚠️ Nenhuma odd encontrada - usando Fair Odds")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Erro ao buscar odds: {e}")
            return {}
    
    async def step_generate_player_props(self) -> bool:
        """Gera player props baseado nos jogos de hoje."""
        logger.info("⛹️ Gerando Player Props...")
        
        try:
            from scripts.generate_player_props_quick import generate_props
            
            result = await asyncio.to_thread(generate_props)
            logger.info("✅ Player props gerados com sucesso")
            return True
            
        except ImportError:
            logger.warning("⚠️ generate_player_props_quick não disponível")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Erro ao gerar player props (não crítico): {e}")
            return True
    
    async def step_update_feature_store(self) -> bool:
        """Atualiza Feature Store com features do dia."""
        logger.info("📊 Atualizando Feature Store...")
        
        try:
            from feature_store import FeatureStore
            
            if self.db:
                fs = FeatureStore(self.db)
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Top times para computar features
                teams = ['LAL', 'BOS', 'GSW', 'MIA', 'PHX', 'DEN', 'MIL', 'PHI',
                         'NYK', 'BKN', 'LAC', 'DAL', 'MEM', 'SAC', 'MIN', 'CLE']
                
                for team in teams:
                    await fs.compute_team_features(team, today)
                
                logger.info(f"✅ Features atualizadas para {len(teams)} times")
                return True
                
        except ImportError:
            logger.info("ℹ️ Feature Store não disponível")
        except Exception as e:
            logger.warning(f"⚠️ Erro na Feature Store: {e}")
        
        return True
    
    async def step_send_summary(self) -> bool:
        """Envia resumo do pipeline via Telegram."""
        import os
        
        try:
            from telegram import Bot
            
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            admin_id = os.getenv('TELEGRAM_ADMIN_ID')
            
            if not token or not admin_id:
                logger.info("Telegram não configurado - skip summary")
                return True
            
            bot = Bot(token)
            
            duration = time.time() - self.start_time
            errors_count = len(self.stats['errors'])
            
            msg = (
                f"✅ **Pipeline Daily v{VERSION}**\n\n"
                f"📅 Jogos: {self.stats['games_processed']}\n"
                f"💰 Odds: {self.stats['odds_fetched']}\n"
                f"🏥 Lesões: {self.stats['injuries_fetched']}\n"
                f"🔮 Previsões: {self.stats['predictions_generated']}\n"
                f"⚠️ Erros: {errors_count}\n"
                f"⏱️ Duração: {duration:.1f}s"
            )
            
            await bot.send_message(admin_id, msg, parse_mode='Markdown')
            logger.info("📱 Resumo enviado para admin")
            
            return True
            
        except Exception as e:
            logger.debug(f"Erro enviando resumo: {e}")
            return True
    
    # ============= FLUXO PRINCIPAL =============
    
    async def run_daily_flow(self):
        """
        Executa o fluxo diário completo v23.0.
        
        Etapas:
        1. Inicialização (DB + Redis)
        2. Twitter Sentiment
        3. Data Collection (Prefect ETL)
        4. Model Training (se necessário)
        5. Fetch Odds
        6. Prediction Generation
        7. Player Props
        8. Feature Store Update
        9. Send Summary
        """
        logger.info(f"🎬 Iniciando Enterprise Orchestrator v{VERSION}")
        
        try:
            # Inicialização
            await self.initialize()
            
            # Pipeline principal
            await self.run_step("Twitter Sentiment", self.step_twitter_collection)
            await self.run_step("Data Collection (ETL)", self.step_data_collection)
            await self.run_step("Model Training", self.step_model_training)
            await self.run_step("Fetch Real Odds", self.step_fetch_odds)
            await self.run_step("Prediction Generation", self.step_prediction)
            await self.run_step("Player Props", self.step_generate_player_props)
            await self.run_step("Feature Store Update", self.step_update_feature_store)
            await self.run_step("Send Summary", self.step_send_summary)
            
            total_time = time.time() - self.start_time
            logger.info(f"\n🎉 Fluxo v{VERSION} concluído com sucesso em {total_time:.1f}s")
            
            # Estatísticas finais
            logger.info(f"📊 Stats: {self.stats['games_processed']} jogos, "
                       f"{self.stats['predictions_generated']} previsões, "
                       f"{len(self.stats['errors'])} erros")
            
        except Exception as e:
            logger.critical(f"🔥 Falha crítica no pipeline: {e}", exc_info=True)
            sys.exit(1)
        finally:
            # Cleanup
            if self.db and hasattr(self.db, 'close'):
                await self.db.close()
            if self.redis:
                await self.redis.disconnect()


# ============= COMPATIBILIDADE LEGADA =============

class PipelineOrchestrator:
    """Wrapper de compatibilidade para código legado."""
    
    def __init__(self):
        self._enterprise = EnterpriseOrchestrator()
    
    def run_daily_flow(self):
        """Executa fluxo via asyncio."""
        asyncio.run(self._enterprise.run_daily_flow())


# ============= ENTRY POINT =============

async def main():
    """Entry point assíncrono."""
    orchestrator = EnterpriseOrchestrator()
    await orchestrator.run_daily_flow()


if __name__ == "__main__":
    # Suporte a Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
