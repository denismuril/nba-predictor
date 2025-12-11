"""
Módulo de Alertas de Lesões para Telegram

Envia notificações automáticas para o Telegram quando lesões críticas são detectadas.

Integração:
    - Usado pelo job background do nba_tigrinho_bot.py
    - Verifica novas lesões a cada 30 minutos
    - Envia alertas apenas para jogadores de alto impacto (OUT/DOUBTFUL)

Usage:
    from data.scrapers.injury_telegram_alerts import InjuryAlertService
    
    service = InjuryAlertService(bot, chat_id)
    await service.check_and_send_alerts()
"""
import logging
from typing import List, Optional, Set
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Cache de alertas já enviados (evita spam)
SENT_ALERTS_FILE = Path("data/cache/sent_injury_alerts.json")

# Jogadores de alto impacto que merecem alerta
HIGH_IMPACT_PLAYERS = {
    # MVP candidates
    'Nikola Jokic', 'Luka Doncic', 'Giannis Antetokounmpo', 
    'Joel Embiid', 'Kevin Durant', 'Stephen Curry', 'LeBron James',
    'Shai Gilgeous-Alexander', 'Jayson Tatum',
    
    # All-Stars
    'Anthony Davis', 'Damian Lillard', 'Jimmy Butler', 'Devin Booker',
    'Donovan Mitchell', 'Tyrese Haliburton', 'Ja Morant', 'Trae Young',
    'De\'Aaron Fox', 'Karl-Anthony Towns', 'Kawhi Leonard', 'Paul George',
    'Zion Williamson', 'Victor Wembanyama', 'Paolo Banchero', 'Chet Holmgren',
    
    # Key players
    'Jaylen Brown', 'Bam Adebayo', 'Jalen Brunson', 'Domantas Sabonis',
    'Lauri Markkanen', 'Brandon Ingram', 'CJ McCollum', 'Scottie Barnes',
}

# Cooldown em segundos (30 min por jogador)
ALERT_COOLDOWN_SECONDS = 1800


def load_sent_alerts() -> dict:
    """Carrega alertas já enviados do disco."""
    if not SENT_ALERTS_FILE.exists():
        return {}
    try:
        with open(SENT_ALERTS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_sent_alerts(alerts: dict):
    """Salva alertas enviados no disco."""
    try:
        SENT_ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SENT_ALERTS_FILE, 'w') as f:
            json.dump(alerts, f)
    except Exception as e:
        logger.error(f"Erro ao salvar alertas: {e}")


def is_high_impact_player(player_name: str) -> bool:
    """Verifica se jogador é de alto impacto."""
    # Normaliza nome para comparação
    normalized = player_name.strip()
    return normalized in HIGH_IMPACT_PLAYERS


class InjuryAlertService:
    """
    Serviço de alertas de lesões para Telegram.
    
    Monitora novas lesões críticas e envia notificações para jogadores importantes.
    """
    
    def __init__(self, bot, admin_chat_id: str):
        """
        Args:
            bot: Instância do Telegram Bot
            admin_chat_id: Chat ID do admin para receber alertas
        """
        self.bot = bot
        self.admin_chat_id = admin_chat_id
        self.sent_alerts = load_sent_alerts()
    
    async def check_and_send_alerts(self) -> int:
        """
        Verifica novas lesões críticas e envia alertas.
        
        Returns:
            Número de alertas enviados
        """
        try:
            from data.scrapers.injury_scraper_v2 import InjuryManager
            
            manager = InjuryManager()
            injuries = manager.get_latest_injuries()
            
            alerts_sent = 0
            import time
            current_time = time.time()
            
            for injury in injuries:
                # Apenas lesões críticas
                if not injury.is_critical():
                    continue
                
                # Apenas jogadores de alto impacto
                if not is_high_impact_player(injury.player_name):
                    continue
                
                # Verificar cooldown
                alert_key = f"{injury.player_name}_{injury.status}"
                last_sent = self.sent_alerts.get(alert_key, 0)
                
                if current_time - last_sent < ALERT_COOLDOWN_SECONDS:
                    continue
                
                # Enviar alerta
                await self._send_alert(injury)
                
                # Registrar
                self.sent_alerts[alert_key] = current_time
                alerts_sent += 1
            
            # Salvar cache
            save_sent_alerts(self.sent_alerts)
            
            if alerts_sent > 0:
                logger.info(f"🚨 {alerts_sent} alertas de lesão enviados")
            
            return alerts_sent
            
        except Exception as e:
            logger.error(f"Erro ao verificar lesões: {e}")
            return 0
    
    async def _send_alert(self, injury):
        """Envia alerta formatado para o Telegram."""
        # Emoji baseado no status
        status_emoji = {
            'OUT': '🔴',
            'DOUBTFUL': '🟠',
            'QUESTIONABLE': '🟡',
        }.get(injury.status, '⚪')
        
        # Montar mensagem
        msg = f"🚨 *ALERTA DE LESÃO*\n\n"
        msg += f"{status_emoji} *{injury.player_name}* ({injury.team})\n"
        msg += f"📋 Status: *{injury.status}*\n"
        msg += f"📝 {injury.description}\n"
        msg += f"🕐 Atualizado: {injury.updated_at[:16]}\n"
        msg += f"📡 Fonte: {injury.source}"
        
        try:
            await self.bot.send_message(
                chat_id=self.admin_chat_id,
                text=msg,
                parse_mode='Markdown'
            )
            logger.info(f"✅ Alerta enviado: {injury.player_name} ({injury.status})")
        except Exception as e:
            logger.error(f"❌ Erro ao enviar alerta: {e}")


async def check_injury_alerts_job(context):
    """
    Job para o Telegram bot que roda periodicamente.
    
    Adicionar ao bot com:
        job_queue.run_repeating(check_injury_alerts_job, interval=1800, first=60)
    """
    from telegram.ext import ContextTypes
    import os
    
    admin_chat_id = os.getenv('TELEGRAM_ADMIN_ID')
    if not admin_chat_id:
        logger.warning("TELEGRAM_ADMIN_ID não configurado")
        return
    
    service = InjuryAlertService(context.bot, admin_chat_id)
    await service.check_and_send_alerts()
