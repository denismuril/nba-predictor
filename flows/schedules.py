"""
Prefect Schedules and Configuration
===================================
Configurações de agendamento para NBA Predictor.
"""
from datetime import timedelta
from prefect.client.schemas.schedules import CronSchedule


# =============================================================================
# SCHEDULES - Horários de execução (Brasília)
# =============================================================================

SCHEDULES = {
    # Daily Pipeline - Rodar previsões 2h antes dos jogos
    "daily_pipeline": CronSchedule(
        cron="0 17 * * *",  # 17:00 BRT
        timezone="America/Sao_Paulo"
    ),
    
    # Paper Trading - Capturar sinais durante janela de apostas
    "paper_trading": CronSchedule(
        cron="0 18 * * *",  # 18:00 BRT
        timezone="America/Sao_Paulo"
    ),
    
    # Settlement - Liquidar bets da noite anterior
    "settlement": CronSchedule(
        cron="0 9 * * *",  # 09:00 BRT
        timezone="America/Sao_Paulo"
    ),
    
    # Health Check - Verificação matinal
    "health_check": CronSchedule(
        cron="0 8 * * *",  # 08:00 BRT
        timezone="America/Sao_Paulo"
    ),
    
    # Fetch Odds - Atualizar odds periodicamente
    "fetch_odds": CronSchedule(
        cron="0 12,15,18 * * *",  # 12:00, 15:00, 18:00 BRT
        timezone="America/Sao_Paulo"
    ),
}


# =============================================================================
# CONCURRENCY LIMITS - Controle de recursos
# =============================================================================

CONCURRENCY_LIMITS = {
    "odds-api": 1,           # 1 request por vez para API de odds
    "predictions": 2,        # 2 predictions simultâneas
    "telegram": 5,           # 5 mensagens por segundo
    "database": 10,          # 10 conexões DB simultâneas
}


# =============================================================================
# RETRY CONFIG - Políticas de retry
# =============================================================================

RETRY_CONFIG = {
    "fetch_games": {
        "retries": 3,
        "delay_seconds": 60,
    },
    "fetch_odds": {
        "retries": 2,
        "delay_seconds": 120,
    },
    "predictions": {
        "retries": 1,
        "delay_seconds": 300,
    },
    "telegram": {
        "retries": 2,
        "delay_seconds": 30,
    },
}


# =============================================================================
# NOTIFICATION CONFIG
# =============================================================================

NOTIFICATION_CONFIG = {
    "on_failure": True,
    "on_success": False,  # Não notificar sucesso (muito spam)
    "channels": ["telegram"],
}
