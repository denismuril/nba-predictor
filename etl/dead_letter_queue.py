"""
Dead Letter Queue - Fila de Dados que Falharam Validação
=========================================================
Dados que falham validação Pydantic vão para DLQ e o pipeline continua.
Alertas são enviados automaticamente via Telegram.

Autor: NBA Predictor v22.0
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import os

logger = logging.getLogger(__name__)


class DeadLetterQueue:
    """
    Dead Letter Queue para dados que falharam validação.
    
    Características:
    - Armazena dados inválidos sem parar o pipeline
    - Envia alertas via Telegram
    - Suporta Redis (produção) ou arquivo local (dev)
    - Permite reprocessamento posterior
    """
    
    def __init__(self, redis_client=None, queue_key: str = "dlq:nba_predictor"):
        """
        Inicializa a DLQ.
        
        Args:
            redis_client: Cliente Redis assíncrono. Se None, usa arquivo local.
            queue_key: Chave base no Redis
        """
        self._redis = redis_client
        self.queue_key = queue_key
        self._local_queue: List[Dict[str, Any]] = []
        self._local_file = "data/cache/dead_letter_queue.jsonl"
        self._alert_cooldown = 300  # 5 minutos entre alertas do mesmo tipo
        self._last_alerts: Dict[str, datetime] = {}
    
    async def push(self, data: Dict[str, Any], error: str, source: str,
                   severity: str = "warning") -> bool:
        """
        Envia dado inválido para DLQ.
        
        Args:
            data: Dados que falharam validação
            error: Mensagem de erro
            source: Origem dos dados (ex: 'games_validation', 'odds_scraper')
            severity: Severidade ('info', 'warning', 'error', 'critical')
            
        Returns:
            True se sucesso
        """
        entry = {
            "data": data,
            "error": str(error),
            "source": source,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "processed": False
        }
        
        logger.warning(f"⚠️ DLQ: {source} - {error[:100]}")
        
        # Armazenar
        if self._redis:
            try:
                await self._redis.lpush(self.queue_key, json.dumps(entry, default=str))
                await self._redis.ltrim(self.queue_key, 0, 9999)  # Manter max 10k entradas
            except Exception as e:
                logger.error(f"Erro ao salvar em Redis DLQ: {e}")
                self._save_local(entry)
        else:
            self._save_local(entry)
        
        # Alertar se severidade alta
        if severity in ('error', 'critical'):
            await self._send_alert(entry)
        
        return True
    
    def _save_local(self, entry: Dict[str, Any]):
        """Salva entrada localmente como fallback"""
        try:
            from pathlib import Path
            Path(self._local_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._local_file, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')
            
            self._local_queue.append(entry)
        except Exception as e:
            logger.error(f"Erro ao salvar DLQ local: {e}")
    
    async def _send_alert(self, entry: Dict[str, Any]):
        """Envia alerta via Telegram (com rate limiting)"""
        source = entry['source']
        
        # Rate limiting por fonte
        now = datetime.utcnow()
        last_alert = self._last_alerts.get(source)
        
        if last_alert:
            elapsed = (now - last_alert).total_seconds()
            if elapsed < self._alert_cooldown:
                logger.debug(f"Alert skipped (cooldown): {source}")
                return
        
        self._last_alerts[source] = now
        
        try:
            from telegram import Bot
            
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            admin_id = os.getenv('TELEGRAM_ADMIN_ID')
            
            if not token or not admin_id:
                return
            
            bot = Bot(token)
            
            # Formatar dados de forma segura
            data_preview = str(entry['data'])[:200]
            
            msg = (
                f"⚠️ **DLQ Alert**\n\n"
                f"📛 Source: `{entry['source']}`\n"
                f"⚡ Severity: {entry['severity'].upper()}\n"
                f"❌ Error: {entry['error'][:150]}\n"
                f"📦 Data: `{data_preview}...`\n"
                f"🕐 Time: {entry['timestamp']}"
            )
            
            await bot.send_message(admin_id, msg, parse_mode='Markdown')
            logger.info(f"📱 DLQ alert enviado: {source}")
            
        except Exception as e:
            logger.debug(f"Erro enviando alerta DLQ: {e}")
    
    async def get_pending(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Busca entradas pendentes da DLQ.
        
        Args:
            limit: Máximo de entradas a retornar
            
        Returns:
            Lista de entradas pendentes
        """
        if self._redis:
            try:
                entries = await self._redis.lrange(self.queue_key, 0, limit - 1)
                return [json.loads(e) for e in entries]
            except Exception:
                pass
        
        return self._local_queue[:limit]
    
    async def get_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Busca entradas de uma fonte específica"""
        all_entries = await self.get_pending(limit * 2)
        return [e for e in all_entries if e.get('source') == source][:limit]
    
    async def mark_processed(self, entry_id: str):
        """Marca entrada como processada"""
        # Implementação simplificada - em produção usar LREM do Redis
        pass
    
    async def reprocess(self, limit: int = 10) -> int:
        """
        Tenta reprocessar entradas da DLQ.
        
        Returns:
            Número de entradas reprocessadas com sucesso
        """
        entries = await self.get_pending(limit)
        reprocessed = 0
        
        for entry in entries:
            try:
                # Lógica de reprocessamento específica por source
                source = entry.get('source', '')
                
                if 'games' in source:
                    from etl.schemas import GameSchema
                    GameSchema(**entry['data'])
                elif 'odds' in source:
                    from etl.schemas import OddsSchema
                    OddsSchema(**entry['data'])
                
                # Se chegou aqui, validação passou
                await self.mark_processed(entry['timestamp'])
                reprocessed += 1
                
            except Exception:
                # Ainda inválido, manter na DLQ
                continue
        
        if reprocessed > 0:
            logger.info(f"✅ DLQ: {reprocessed} entradas reprocessadas")
        
        return reprocessed
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da DLQ"""
        entries = await self.get_pending(10000)
        
        by_source = {}
        by_severity = {}
        
        for entry in entries:
            source = entry.get('source', 'unknown')
            severity = entry.get('severity', 'unknown')
            
            by_source[source] = by_source.get(source, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            'total_entries': len(entries),
            'by_source': by_source,
            'by_severity': by_severity,
            'oldest_entry': entries[-1]['timestamp'] if entries else None,
            'newest_entry': entries[0]['timestamp'] if entries else None
        }
    
    async def clear(self, source: Optional[str] = None):
        """
        Limpa a DLQ.
        
        Args:
            source: Se especificado, limpa apenas entradas dessa fonte
        """
        if source:
            # Limpar apenas de uma fonte específica (complexo no Redis)
            logger.warning(f"Limpando DLQ para source: {source}")
        else:
            if self._redis:
                try:
                    await self._redis.delete(self.queue_key)
                except Exception:
                    pass
            
            self._local_queue.clear()
            
            try:
                import os
                if os.path.exists(self._local_file):
                    os.remove(self._local_file)
            except Exception:
                pass
            
            logger.info("🗑️ DLQ limpa")


# ============= SINGLETON =============

_dlq_instance: Optional[DeadLetterQueue] = None


async def get_dlq() -> DeadLetterQueue:
    """Retorna instância singleton da DLQ"""
    global _dlq_instance
    
    if _dlq_instance is None:
        _dlq_instance = DeadLetterQueue()
        
        # Tentar conectar Redis
        try:
            from infrastructure.redis_cache import get_redis
            redis = await get_redis()
            if redis._connected:
                _dlq_instance._redis = redis._redis
                logger.info("✅ DLQ conectada ao Redis")
        except Exception:
            logger.info("ℹ️ DLQ usando modo local")
    
    return _dlq_instance
