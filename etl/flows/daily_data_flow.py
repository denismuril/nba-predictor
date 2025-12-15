"""
Daily Data Flow - Pipeline Diário de Dados com Prefect
=======================================================
Orquestra a coleta, validação e armazenamento de dados da NBA.
Substitui os scripts manuais (run_daily.sh, fix_*.py).

Auto-Cura: Dados inválidos vão para DLQ, pipeline continua.

Autor: NBA Predictor v22.0

Uso:
    # Rodar localmente
    python -m etl.flows.daily_data_flow
    
    # Rodar via Prefect
    prefect deployment run "NBA Daily Data Pipeline/daily"
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

# Prefect imports
try:
    from prefect import flow, task, get_run_logger
    from prefect.tasks import task_input_hash
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    # Fallback decorators
    def flow(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    def task(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    def get_run_logger():
        return logging.getLogger(__name__)
    
    task_input_hash = None

logger = logging.getLogger(__name__)


# ============= TASKS =============

@task(retries=3, retry_delay_seconds=60)
async def fetch_games(date: str) -> List[Dict[str, Any]]:
    """
    Task: Busca jogos agendados para a data.
    
    Args:
        date: Data no formato YYYY-MM-DD
        
    Returns:
        Lista de jogos encontrados
    """
    log = get_run_logger() if PREFECT_AVAILABLE else logger
    log.info(f"📅 Buscando jogos para {date}...")
    
    try:
        from data.scrapers.schedule_scraper import obter_schedule
        
        games = obter_schedule(date)
        
        if not games:
            log.warning(f"⚠️ Nenhum jogo encontrado para {date}")
            return []
        
        log.info(f"✅ {len(games)} jogos encontrados")
        return games
        
    except Exception as e:
        log.error(f"❌ Erro ao buscar jogos: {e}")
        raise


@task(retries=3, retry_delay_seconds=30)
async def validate_games(games: List[Dict[str, Any]]) -> tuple:
    """
    Task: Valida jogos com Pydantic e separa válidos de inválidos.
    
    Returns:
        Tupla (jogos_válidos, jogos_inválidos)
    """
    log = get_run_logger() if PREFECT_AVAILABLE else logger
    
    from etl.schemas import GameSchema
    from etl.dead_letter_queue import get_dlq
    from datetime import datetime
    
    dlq = await get_dlq()
    valid_games = []
    invalid_count = 0
    
    for game in games:
        try:
            # Normalizar dados primeiro
            date_val = game.get('date', game.get('Data', datetime.now().strftime('%Y-%m-%d')))
            home_team = game.get('home_team', game.get('home', game.get('Casa', '')))
            away_team = game.get('away_team', game.get('away', game.get('Visitante', '')))
            
            # Gerar game_id com valores normalizados
            game_id = game.get('game_id') or f"{date_val}_{home_team}_{away_team}"
            
            normalized = {
                'game_id': game_id,
                'date': date_val,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': game.get('home_score'),
                'away_score': game.get('away_score'),
                'status': game.get('status', 'scheduled'),
                'season': game.get('season', '2024-25')
            }
            
            # Validar com Pydantic
            validated = GameSchema(**normalized)
            valid_games.append(validated.dict())
            
        except Exception as e:
            invalid_count += 1
            await dlq.push(game, str(e), source="games_validation")
            continue
    
    log.info(f"✅ {len(valid_games)} jogos válidos, {invalid_count} inválidos → DLQ")
    return valid_games, invalid_count


@task(retries=3, retry_delay_seconds=60)
async def fetch_odds(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Task: Busca odds para os jogos.
    
    Usa fallback multi-fonte:
    1. OddsPedia (scraping gratuito)
    2. TheOddsAPI (pago)
    3. SportsDataIO (pago)
    
    NOTE: APIs de odds estão desabilitadas temporariamente (keys expiradas)
    """
    log = get_run_logger() if PREFECT_AVAILABLE else logger
    log.info(f"💰 Buscando odds para {len(games)} jogos...")
    
    try:
        from data.scrapers.odds_scraper import obter_odds
        
        odds_data = await asyncio.to_thread(obter_odds)
        
        if not odds_data:
            log.warning("⚠️ Nenhuma odd encontrada")
            return []
        
        # Converter dict para lista
        odds_list = list(odds_data.values()) if isinstance(odds_data, dict) else odds_data
        log.info(f"✅ Odds coletadas para {len(odds_list)} jogos")
        return odds_list
        
    except Exception as e:
        log.error(f"❌ Erro ao buscar odds: {e}")
        return []


@task(retries=3, retry_delay_seconds=30)
async def validate_odds(odds: List[Dict[str, Any]]) -> tuple:
    """
    Task: Valida odds com Pydantic.
    """
    log = get_run_logger() if PREFECT_AVAILABLE else logger
    
    from etl.schemas import OddsSchema
    from etl.dead_letter_queue import get_dlq
    
    dlq = await get_dlq()
    valid_odds = []
    invalid_count = 0
    
    for odd in odds:
        try:
            validated = OddsSchema(**odd)
            valid_odds.append(validated.dict())
        except Exception as e:
            invalid_count += 1
            await dlq.push(odd, str(e), source="odds_validation")
    
    log.info(f"✅ {len(valid_odds)} odds válidas, {invalid_count} inválidas → DLQ")
    return valid_odds, invalid_count


@task(retries=2)
async def fetch_injuries() -> Dict[str, List[Dict[str, Any]]]:
    """
    Task: Busca relatório de lesões.
    """
    log = get_run_logger() if PREFECT_AVAILABLE else logger
    log.info("🏥 Buscando relatório de lesões...")
    
    try:
        from data.scrapers.injury_scraper_v2 import InjuryManager
        
        manager = InjuryManager()
        injuries_list = await asyncio.to_thread(manager.get_latest_injuries)
        
        # Agrupa por time
        injuries_by_team = {}
        for injury in injuries_list:
            team = getattr(injury, 'team', None) or injury.get('team', 'UNKNOWN')
            if team not in injuries_by_team:
                injuries_by_team[team] = []
            injuries_by_team[team].append(
                injury.dict() if hasattr(injury, 'dict') else injury
            )
        
        total = sum(len(v) for v in injuries_by_team.values())
        log.info(f"✅ {total} lesões encontradas em {len(injuries_by_team)} times")
        return injuries_by_team
        
    except Exception as e:
        log.warning(f"⚠️ Erro ao buscar lesões: {e}")
        return {}


@task(retries=2)
async def store_data(games: List[Dict], odds: List[Dict], injuries: Dict):
    """
    Task: Armazena dados no banco e cache.
    """
    log = get_run_logger() if PREFECT_AVAILABLE else logger
    log.info("💾 Armazenando dados...")
    
    try:
        from infrastructure.database import get_async_db
        from infrastructure.redis_cache import get_redis
        
        db = await get_async_db()
        redis = await get_redis()
        
        # Armazenar jogos
        if games:
            await db.bulk_insert_games(games)
            log.info(f"  └─ {len(games)} jogos salvos no DB")
        
        # Armazenar odds no cache
        for odd in odds:
            game_id = odd.get('game_id')
            if game_id:
                await redis.set_odds(game_id, odd)
        log.info(f"  └─ {len(odds)} odds cacheadas no Redis")
        
        # Armazenar lesões no cache
        for team_id, team_injuries in injuries.items():
            await redis.set_injuries(team_id, team_injuries)
        log.info(f"  └─ {len(injuries)} times com lesões cacheados")
        
    except Exception as e:
        log.error(f"❌ Erro ao armazenar dados: {e}")
        raise


@task
async def update_feature_store(date: str):
    """
    Task: Atualiza Feature Store com features do dia.
    """
    log = get_run_logger() if PREFECT_AVAILABLE else logger
    log.info("📊 Atualizando Feature Store...")
    
    try:
        from feature_store import FeatureStore
        from infrastructure.database import get_async_db
        
        db = await get_async_db()
        fs = FeatureStore(db)
        
        # Computar features para o dia
        teams = ['LAL', 'BOS', 'GSW', 'MIA', 'PHX', 'DEN', 'MIL', 'PHI']  # Top times
        
        for team in teams:
            await fs.compute_team_features(team, date)
        
        log.info(f"✅ Features atualizadas para {len(teams)} times")
        
    except Exception as e:
        log.warning(f"⚠️ Erro na Feature Store: {e}")


@task
async def send_summary(date: str, stats: Dict[str, Any]):
    """
    Task: Envia resumo do pipeline via Telegram.
    """
    log = get_run_logger() if PREFECT_AVAILABLE else logger
    
    import os
    
    try:
        from telegram import Bot
        
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        admin_id = os.getenv('TELEGRAM_ADMIN_ID')
        
        if not token or not admin_id:
            log.info("Telegram não configurado - skip summary")
            return
        
        bot = Bot(token)
        
        msg = (
            f"✅ **Pipeline Daily - {date}**\n\n"
            f"📅 Jogos: {stats.get('games_valid', 0)}\n"
            f"💰 Odds: {stats.get('odds_valid', 0)}\n"
            f"🏥 Lesões: {stats.get('injuries_count', 0)}\n"
            f"⚠️ DLQ: {stats.get('dlq_count', 0)}\n"
            f"⏱️ Duração: {stats.get('duration', 0):.1f}s"
        )
        
        await bot.send_message(admin_id, msg, parse_mode='Markdown')
        log.info("📱 Resumo enviado para admin")
        
    except Exception as e:
        log.debug(f"Erro enviando resumo: {e}")


# ============= FLOW PRINCIPAL =============

@flow(name="NBA Daily Data Pipeline", log_prints=True)
async def daily_data_flow(date: str = None, skip_features: bool = False):
    """
    Flow Principal: Pipeline de coleta diária de dados.
    
    Etapas:
    1. Buscar jogos agendados
    2. Validar jogos (inválidos → DLQ)
    3. Buscar odds
    4. Validar odds (inválidas → DLQ)
    5. Buscar lesões
    6. Armazenar dados (DB + Cache)
    7. Atualizar Feature Store
    8. Enviar resumo
    
    Args:
        date: Data no formato YYYY-MM-DD. Se None, usa hoje.
        skip_features: Se True, pula atualização de features.
        
    Returns:
        Dict com estatísticas do pipeline
    """
    log = get_run_logger() if PREFECT_AVAILABLE else logger
    start_time = datetime.now()
    
    # Usar data de hoje se não especificada
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    log.info(f"🏀 ============ NBA Daily Pipeline - {date} ============")
    
    # Stats
    stats = {
        'date': date,
        'games_valid': 0,
        'games_invalid': 0,
        'odds_valid': 0,
        'odds_invalid': 0,
        'injuries_count': 0,
        'dlq_count': 0,
        'success': False,
        'duration': 0
    }
    
    try:
        # Step 1: Buscar jogos
        games = await fetch_games(date)
        
        # Step 2: Validar jogos
        valid_games, invalid_games = await validate_games(games)
        stats['games_valid'] = len(valid_games)
        stats['games_invalid'] = invalid_games
        
        # Step 3: Buscar odds
        odds = await fetch_odds(valid_games)
        
        # Step 4: Validar odds
        valid_odds, invalid_odds = await validate_odds(odds)
        stats['odds_valid'] = len(valid_odds)
        stats['odds_invalid'] = invalid_odds
        
        # Step 5: Buscar lesões
        injuries = await fetch_injuries()
        stats['injuries_count'] = sum(len(v) for v in injuries.values())
        
        # Step 6: Armazenar dados
        await store_data(valid_games, valid_odds, injuries)
        
        # Step 7: Atualizar Feature Store
        if not skip_features:
            await update_feature_store(date)
        
        # Calcular stats finais
        stats['dlq_count'] = invalid_games + invalid_odds
        stats['success'] = True
        stats['duration'] = (datetime.now() - start_time).total_seconds()
        
        # Step 8: Enviar resumo
        await send_summary(date, stats)
        
        log.info(f"✅ Pipeline concluído em {stats['duration']:.1f}s")
        
    except Exception as e:
        log.error(f"❌ Erro no pipeline: {e}")
        stats['success'] = False
        stats['error'] = str(e)
        raise
    
    return stats


# ============= HELPER PARA RODAR LOCALMENTE =============

async def run_flow(date: str = None):
    """Helper para rodar o flow localmente"""
    return await daily_data_flow(date)


if __name__ == "__main__":
    import sys
    
    date = sys.argv[1] if len(sys.argv) > 1 else None
    result = asyncio.run(run_flow(date))
    
    print("\n" + "="*50)
    print("📊 Resultado do Pipeline:")
    for key, value in result.items():
        print(f"  {key}: {value}")
