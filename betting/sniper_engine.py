"""
Sniper Engine - Monitor de Odds em Tempo Real
==============================================
Sistema de detecção de apostas de valor em tempo real.

Funcionalidades:
- Monitora Redis a cada 30 segundos buscando mudanças de odds
- Calcula Fair Price instantâneo via FeatureStore
- Dispara alertas quando detecta valor (Minha Odd > Casa + 5%)
- Integra com Kelly Criterion para stake recomendado

Autor: NBA Predictor v23.0
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# TRAP ODDS DETECTION
# =============================================================================
@dataclass
class TrapOddsCheck:
    """Result of trap odds analysis."""
    is_trap: bool
    public_pct: float
    line_movement: float
    reason: str


def detect_trap_odds(
    public_pct: float,
    line_movement: float,
    threshold_public: float = 0.70,
    threshold_movement: float = -0.03
) -> TrapOddsCheck:
    """
    Detect if odds are a 'trap' - public heavily on one side but line moving opposite.
    
    Sharp money often goes against public consensus. When 70%+ of public bets
    are on one side BUT the line moves against them, it's a trap.
    
    Args:
        public_pct: Percentage of public bets on this side (0.0-1.0)
        line_movement: Line movement as decimal (-0.05 = line got worse by 5%)
        threshold_public: Min public % to consider as heavy public action
        threshold_movement: Movement threshold (negative = line got worse)
        
    Returns:
        TrapOddsCheck with analysis results
    """
    # Trap = heavy public action BUT line moving opposite (sharps on other side)
    is_trap = public_pct >= threshold_public and line_movement <= threshold_movement
    
    reason = ""
    if is_trap:
        reason = f"TRAP: {public_pct*100:.0f}% public mas linha subiu {abs(line_movement)*100:.1f}%"
    elif public_pct >= threshold_public:
        reason = f"Heavy public: {public_pct*100:.0f}%"
    
    return TrapOddsCheck(
        is_trap=is_trap,
        public_pct=public_pct,
        line_movement=line_movement,
        reason=reason
    )


# =============================================================================
# WEBSOCKET ODDS CLIENT
# =============================================================================
class OddsWebSocketClient:
    """
    WebSocket client for real-time odds streaming.
    
    Falls back to HTTP polling if WebSocket unavailable.
    """
    
    def __init__(self, poll_interval: float = 5.0):
        self.poll_interval = poll_interval
        self._ws = None
        self._running = False
        self._callbacks = []
        
    async def connect(self, url: str) -> bool:
        """
        Connect to WebSocket endpoint.
        
        Returns True if connected, False if fallback to polling.
        """
        try:
            import websockets
            self._ws = await websockets.connect(url)
            logger.info(f"🔌 WebSocket connected to {url}")
            return True
        except ImportError:
            logger.warning("⚠️ websockets not installed, using HTTP polling")
            return False
        except Exception as e:
            logger.warning(f"⚠️ WebSocket failed: {e}, using HTTP polling")
            return False
    
    async def subscribe(self, callback):
        """Subscribe to odds updates."""
        self._callbacks.append(callback)
    
    async def start_listening(self):
        """Start listening for odds updates."""
        self._running = True
        
        if self._ws:
            # WebSocket mode
            while self._running:
                try:
                    message = await self._ws.recv()
                    for callback in self._callbacks:
                        await callback(message)
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    await asyncio.sleep(1)
        else:
            # HTTP polling fallback
            logger.info(f"📡 HTTP polling mode (interval: {self.poll_interval}s)")
    
    async def close(self):
        """Close connection."""
        self._running = False
        if self._ws:
            await self._ws.close()


@dataclass
class ValueAlert:
    """Representa um alerta de aposta de valor."""
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    bet_side: str  # 'home' ou 'away'
    market_odds: float
    fair_odds: float
    edge_pct: float
    kelly_stake_pct: float
    confidence: str
    reason: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_telegram_message(self) -> str:
        """Formata alerta para envio via Telegram."""
        emoji = "🔥" if self.edge_pct > 10 else "💰"
        side_name = self.home_team if self.bet_side == 'home' else self.away_team
        
        return (
            f"{emoji} **ALERTA DE VALOR DETECTADO!**\n\n"
            f"🏀 {self.home_team} vs {self.away_team}\n"
            f"📅 {self.game_date}\n"
            f"🎯 Apostar em: **{side_name}**\n\n"
            f"💵 Odd do Mercado: {self.market_odds:.2f}\n"
            f"📊 Fair Odd: {self.fair_odds:.2f}\n"
            f"📈 Edge: **+{self.edge_pct:.1f}%**\n"
            f"💰 Stake Kelly: {self.kelly_stake_pct:.2f}%\n"
            f"🎲 Confiança: {self.confidence}\n\n"
            f"📝 {self.reason}"
        )


class SniperEngine:
    """
    Engine de Sniper para detecção de apostas de valor em tempo real.
    
    Características:
    - Polling do Redis a cada 30 segundos
    - Detecção de Line Movement (mudanças bruscas)
    - Cálculo de Fair Price via modelo ML
    - Integração com Kelly Criterion
    - Alertas via Telegram
    """
    
    # Configurações
    POLL_INTERVAL_SECONDS = 30
    MIN_EDGE_PCT = 5.0  # Mínimo 5% de edge para alertar
    MAX_ALERTS_PER_GAME = 3  # Máximo de alertas por jogo
    LINE_MOVEMENT_THRESHOLD = 0.10  # 10% de mudança = movimento significativo
    TRAP_PUBLIC_THRESHOLD = 0.70  # 70% public = trap territory
    TRAP_MOVEMENT_THRESHOLD = -0.03  # Line moved 3% against public = trap
    
    def __init__(self, bankroll: float = 1000.0, kelly_fraction: float = 0.25):
        """
        Inicializa o Sniper Engine.
        
        Args:
            bankroll: Banca atual em reais
            kelly_fraction: Fração do Kelly a usar (0.25 = Quarter Kelly)
        """
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        
        self.redis = None
        self.db = None
        self.feature_store = None
        self.telegram_bot = None
        
        self._running = False
        self._alerts_sent: Dict[str, int] = {}  # game_id -> count
        self._last_odds: Dict[str, Dict[str, float]] = {}  # game_id -> odds
        self._public_money: Dict[str, float] = {}  # game_id -> public % on home
        self._initialized = False
        self._ws_client = OddsWebSocketClient()
        
    async def initialize(self):
        """Inicializa conexões e dependências."""
        if self._initialized:
            return
        
        logger.info("🎯 Inicializando Sniper Engine...")
        
        # Redis Cache
        try:
            from infrastructure.redis_cache import get_redis
            self.redis = await get_redis()
            health = await self.redis.health_check()
            if health['status'] != 'healthy':
                logger.warning("⚠️ Redis não disponível - Sniper limitado")
                self.redis = None
        except ImportError:
            logger.warning("⚠️ Redis não instalado")
        
        # Banco de Dados
        try:
            from infrastructure.database import get_async_db
            self.db = await get_async_db()
        except ImportError:
            logger.warning("⚠️ AsyncDataManager não disponível")
        
        # Feature Store
        try:
            from feature_store import FeatureStore
            if self.db:
                self.feature_store = FeatureStore(self.db)
        except ImportError:
            logger.warning("⚠️ FeatureStore não disponível")
        
        # Telegram Bot
        try:
            from telegram import Bot
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            admin_id = os.getenv('TELEGRAM_ADMIN_ID')
            
            if token and admin_id:
                self.telegram_bot = Bot(token)
                self.telegram_admin_id = admin_id
                logger.info("✅ Telegram configurado para alertas")
            else:
                logger.info("ℹ️ Telegram não configurado")
        except ImportError:
            logger.info("ℹ️ python-telegram-bot não instalado")
        
        self._initialized = True
        logger.info("✅ Sniper Engine inicializado")
    
    async def start(self):
        """Inicia o loop de monitoramento."""
        await self.initialize()
        
        if not self.redis:
            logger.error("❌ Redis obrigatório para Sniper Engine")
            return
        
        logger.info(f"🎯 Sniper Engine iniciado (poll: {self.POLL_INTERVAL_SECONDS}s)")
        self._running = True
        
        while self._running:
            try:
                await self._poll_and_analyze()
                await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                logger.info("🛑 Sniper Engine cancelado")
                break
            except Exception as e:
                logger.error(f"❌ Erro no Sniper: {e}")
                await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
    
    async def stop(self):
        """Para o loop de monitoramento."""
        self._running = False
        logger.info("🛑 Sniper Engine parado")
    
    async def _poll_and_analyze(self):
        """Ciclo de polling e análise."""
        # Buscar todas as odds do cache
        try:
            keys = await self.redis._redis.keys("odds:*")
            
            if not keys:
                logger.debug("Nenhuma odd em cache")
                return
            
            for key in keys:
                game_id = key.replace("odds:", "")
                current_odds = await self.redis.get_odds(game_id)
                
                if not current_odds:
                    continue
                
                # Detectar Line Movement
                line_movement = self._detect_line_movement(game_id, current_odds)
                
                # Calcular Fair Price e verificar valor
                alerts = await self._analyze_value(game_id, current_odds, line_movement)
                
                # Enviar alertas
                for alert in alerts:
                    await self._send_alert(alert)
                
                # Atualizar histórico de odds
                self._last_odds[game_id] = current_odds
                
        except Exception as e:
            logger.error(f"Erro no poll: {e}")
    
    def _detect_line_movement(self, game_id: str, current_odds: Dict[str, Any]) -> Dict[str, float]:
        """
        Detecta movimentos significativos nas linhas de odds.
        
        Returns:
            Dict com mudanças percentuais para cada lado
        """
        movement = {'home': 0.0, 'away': 0.0}
        
        if game_id not in self._last_odds:
            return movement
        
        last = self._last_odds[game_id]
        
        # Calcular mudança percentual
        if 'home_odds' in current_odds and 'home_odds' in last:
            if last['home_odds'] > 0:
                change = (current_odds['home_odds'] - last['home_odds']) / last['home_odds']
                movement['home'] = change
        
        if 'away_odds' in current_odds and 'away_odds' in last:
            if last['away_odds'] > 0:
                change = (current_odds['away_odds'] - last['away_odds']) / last['away_odds']
                movement['away'] = change
        
        # Log se movimento significativo
        if abs(movement['home']) > self.LINE_MOVEMENT_THRESHOLD:
            logger.info(f"📊 Line Movement detectado em {game_id}: HOME {movement['home']*100:+.1f}%")
        if abs(movement['away']) > self.LINE_MOVEMENT_THRESHOLD:
            logger.info(f"📊 Line Movement detectado em {game_id}: AWAY {movement['away']*100:+.1f}%")
        
        return movement
    
    async def _analyze_value(self, game_id: str, odds: Dict[str, Any], 
                              line_movement: Dict[str, float]) -> List[ValueAlert]:
        """
        Analisa se existe valor nas odds atuais.
        
        Compara:
        - Fair Odds (calculada pelo modelo)
        - Market Odds (oferecida pela casa)
        
        Se Fair Odds < Market Odds + margem, há valor.
        """
        alerts = []
        
        # Verificar limite de alertas
        alerts_count = self._alerts_sent.get(game_id, 0)
        if alerts_count >= self.MAX_ALERTS_PER_GAME:
            return alerts
        
        # Extrair odds do mercado
        home_odds = odds.get('home_odds', odds.get('odd_home', 0))
        away_odds = odds.get('away_odds', odds.get('odd_away', 0))
        home_team = odds.get('home_team', odds.get('home', 'HOME'))
        away_team = odds.get('away_team', odds.get('away', 'AWAY'))
        game_date = odds.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        if not home_odds or not away_odds:
            return alerts
        
        # Calcular Fair Price
        fair_home, fair_away = await self._calculate_fair_price(home_team, away_team, game_date)
        
        if not fair_home or not fair_away:
            # Fallback: usar implied probabilities
            fair_home, fair_away = self._implied_fair_odds(home_odds, away_odds)
        
        # Verificar valor em HOME
        home_edge = self._calculate_edge(fair_home, home_odds)
        if home_edge >= self.MIN_EDGE_PCT:
            kelly_stake = self._calculate_kelly_stake(1/fair_home, home_odds)
            confidence = self._get_confidence_level(home_edge, line_movement.get('home', 0))
            
            alerts.append(ValueAlert(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                game_date=game_date,
                bet_side='home',
                market_odds=home_odds,
                fair_odds=fair_home,
                edge_pct=home_edge,
                kelly_stake_pct=kelly_stake * 100,
                confidence=confidence,
                reason=self._get_reason(home_edge, line_movement.get('home', 0))
            ))
        
        # Verificar valor em AWAY
        away_edge = self._calculate_edge(fair_away, away_odds)
        if away_edge >= self.MIN_EDGE_PCT:
            kelly_stake = self._calculate_kelly_stake(1/fair_away, away_odds)
            confidence = self._get_confidence_level(away_edge, line_movement.get('away', 0))
            
            alerts.append(ValueAlert(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                game_date=game_date,
                bet_side='away',
                market_odds=away_odds,
                fair_odds=fair_away,
                edge_pct=away_edge,
                kelly_stake_pct=kelly_stake * 100,
                confidence=confidence,
                reason=self._get_reason(away_edge, line_movement.get('away', 0))
            ))
        
        return alerts
    
    async def _calculate_fair_price(self, home_team: str, away_team: str, 
                                     game_date: str) -> tuple:
        """
        Calcula Fair Price usando FeatureStore e modelo ML.
        
        Returns:
            (fair_home_odds, fair_away_odds)
        """
        try:
            if not self.feature_store:
                return None, None
            
            # Buscar features dos times
            home_features = await self.feature_store.get_features_for_inference(
                home_team, 
                ['pts_avg_5', 'off_rating_avg', 'def_rating_avg', 'win_rate_10']
            )
            away_features = await self.feature_store.get_features_for_inference(
                away_team,
                ['pts_avg_5', 'off_rating_avg', 'def_rating_avg', 'win_rate_10']
            )
            
            # Calcular probabilidades baseado nas features
            # Fórmula simplificada usando diferença de ratings
            home_off = home_features.get('off_rating_avg', 110) or 110
            home_def = home_features.get('def_rating_avg', 110) or 110
            away_off = away_features.get('off_rating_avg', 110) or 110
            away_def = away_features.get('def_rating_avg', 110) or 110
            
            # Net Rating esperado
            home_net = (home_off - away_def) - (away_off - home_def)
            
            # Converter para probabilidade (logística)
            import math
            home_prob = 1 / (1 + math.exp(-home_net / 10))  # /10 para suavizar
            
            # Ajuste de mando de campo (+3.5 pontos ≈ +7% prob)
            home_prob = min(0.95, home_prob + 0.07)
            away_prob = 1 - home_prob
            
            # Converter para odds decimais
            fair_home = 1 / home_prob if home_prob > 0 else 10.0
            fair_away = 1 / away_prob if away_prob > 0 else 10.0
            
            return fair_home, fair_away
            
        except Exception as e:
            logger.debug(f"Erro calculando fair price: {e}")
            return None, None
    
    def _implied_fair_odds(self, home_odds: float, away_odds: float) -> tuple:
        """
        Calcula fair odds removendo a margem da casa.
        
        Overround = (1/home + 1/away) - 1
        Fair Prob = Implied Prob / (1 + Overround)
        """
        try:
            home_implied = 1 / home_odds
            away_implied = 1 / away_odds
            overround = home_implied + away_implied - 1
            
            if overround <= 0:
                return home_odds, away_odds
            
            # Remover margem proporcionalmente
            fair_home_prob = home_implied / (1 + overround)
            fair_away_prob = away_implied / (1 + overround)
            
            fair_home = 1 / fair_home_prob
            fair_away = 1 / fair_away_prob
            
            return fair_home, fair_away
            
        except Exception:
            return home_odds, away_odds
    
    def _calculate_edge(self, fair_odds: float, market_odds: float) -> float:
        """
        Calcula edge percentual.
        
        Edge = (Market Odds / Fair Odds - 1) * 100
        
        Se Market Odds > Fair Odds, temos valor positivo.
        """
        if fair_odds <= 0:
            return 0.0
        
        edge = (market_odds / fair_odds - 1) * 100
        return max(0.0, edge)
    
    def _calculate_kelly_stake(self, prob_win: float, decimal_odds: float) -> float:
        """
        Calcula stake usando Kelly Criterion.
        
        Integra com utils/kelly.py.
        """
        try:
            from utils.kelly import kelly_criterion_advanced
            
            result = kelly_criterion_advanced(
                prob_win=prob_win,
                decimal_odds=decimal_odds,
                fractional=self.kelly_fraction
            )
            
            return result.get('kelly_fractional', 0.0)
            
        except ImportError:
            # Fallback: cálculo manual
            b = decimal_odds - 1
            q = 1 - prob_win
            kelly_full = (b * prob_win - q) / b if b > 0 else 0
            return max(0, min(0.05, kelly_full * self.kelly_fraction))
    
    def _get_confidence_level(self, edge: float, line_movement: float) -> str:
        """Determina nível de confiança baseado em edge e line movement."""
        if edge >= 15 and line_movement > 0:
            return "🔥 MUITO ALTA"
        elif edge >= 10:
            return "💪 ALTA"
        elif edge >= 7:
            return "📊 MÉDIA"
        else:
            return "📉 BAIXA"
    
    def _get_reason(self, edge: float, line_movement: float) -> str:
        """Gera razão do alerta."""
        reasons = []
        
        if edge >= 10:
            reasons.append(f"Edge excepcional de {edge:.1f}%")
        else:
            reasons.append(f"Edge de valor: {edge:.1f}%")
        
        if abs(line_movement) > self.LINE_MOVEMENT_THRESHOLD:
            direction = "subindo" if line_movement > 0 else "descendo"
            reasons.append(f"Linha {direction} ({abs(line_movement)*100:.1f}%)")
        
        return " | ".join(reasons)
    
    async def _send_alert(self, alert: ValueAlert):
        """Envia alerta via Telegram."""
        # Atualizar contador
        self._alerts_sent[alert.game_id] = self._alerts_sent.get(alert.game_id, 0) + 1
        
        # Log sempre
        logger.info(f"🎯 VALOR DETECTADO: {alert.home_team} vs {alert.away_team} "
                   f"| {alert.bet_side.upper()} @ {alert.market_odds:.2f} "
                   f"| Edge: {alert.edge_pct:.1f}%")
        
        # Enviar Telegram se disponível
        if self.telegram_bot and hasattr(self, 'telegram_admin_id'):
            try:
                await self.telegram_bot.send_message(
                    chat_id=self.telegram_admin_id,
                    text=alert.to_telegram_message(),
                    parse_mode='Markdown'
                )
                logger.info("📱 Alerta enviado via Telegram")
            except Exception as e:
                logger.warning(f"⚠️ Erro enviando alerta Telegram: {e}")
    
    # ============= MÉTODOS UTILITÁRIOS =============
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do engine."""
        return {
            'running': self._running,
            'alerts_sent': sum(self._alerts_sent.values()),
            'games_monitored': len(self._last_odds),
            'bankroll': self.bankroll,
            'kelly_fraction': self.kelly_fraction,
            'min_edge': self.MIN_EDGE_PCT
        }
    
    async def analyze_single_game(self, game_id: str, odds: Dict[str, Any]) -> List[ValueAlert]:
        """
        Analisa um único jogo manualmente.
        
        Útil para testes e análise sob demanda.
        """
        await self.initialize()
        return await self._analyze_value(game_id, odds, {'home': 0, 'away': 0})


# ============= SINGLETON =============

_sniper_instance: Optional[SniperEngine] = None


async def get_sniper_engine(bankroll: float = 1000.0) -> SniperEngine:
    """Retorna instância singleton do Sniper Engine."""
    global _sniper_instance
    if _sniper_instance is None:
        _sniper_instance = SniperEngine(bankroll=bankroll)
        await _sniper_instance.initialize()
    return _sniper_instance


# ============= CLI =============

async def main():
    """Entry point para execução standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sniper Engine - Monitor de Odds")
    parser.add_argument('--bankroll', type=float, default=1000.0, help='Banca atual')
    parser.add_argument('--kelly', type=float, default=0.25, help='Fração Kelly (0.25 = Quarter)')
    parser.add_argument('--test', action='store_true', help='Modo teste (single poll)')
    args = parser.parse_args()
    
    engine = SniperEngine(bankroll=args.bankroll, kelly_fraction=args.kelly)
    
    if args.test:
        await engine.initialize()
        print("✅ Sniper Engine Test Mode")
        print(f"   Bankroll: R$ {engine.bankroll:.2f}")
        print(f"   Kelly Fraction: {engine.kelly_fraction}")
        print(f"   Min Edge: {engine.MIN_EDGE_PCT}%")
        print(f"   Redis: {'✅' if engine.redis else '❌'}")
        print(f"   Telegram: {'✅' if engine.telegram_bot else '❌'}")
    else:
        await engine.start()


if __name__ == "__main__":
    import sys
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
