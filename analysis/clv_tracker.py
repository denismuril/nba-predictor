"""
CLV Tracker - Rastreador de Closing Line Value.

O CLV (Closing Line Value) é a métrica definitiva para medir se um modelo
de apostas está realmente "batendo o mercado".

Funcionamento:
1. Registra a odd no momento da aposta (Odd_Apostada)
2. 10 minutos antes do jogo, faz scrape da odd de fechamento (Odd_Fechamento)
3. Calcula: CLV = (Odd_Apostada / Odd_Fechamento) - 1

Interpretação:
- CLV > 0: Modelo capturou valor que o mercado precificou depois
- CLV < 0: Mercado corrigiu contra sua posição
- CLV médio positivo ao longo do tempo = edge sustentável

v27.0: Implementação inicial para arquitetura God Mode.
"""

import asyncio
import csv
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Caminhos
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
PERFORMANCE_FILE = LOGS_DIR / "betting_performance.csv"


class BetStatus(Enum):
    """Status de uma aposta rastreada."""
    PENDING = "pending"           # Aguardando fechamento
    CLOSED = "closed"             # Odds de fechamento capturadas
    SETTLED = "settled"           # Resultado final conhecido
    EXPIRED = "expired"           # Jogo passou sem capturar fechamento


@dataclass
class TrackedBet:
    """
    Representa uma aposta rastreada para CLV.
    
    Atributos:
        bet_id: Identificador único da aposta
        player_name: Nome do jogador
        prop_type: Tipo da prop (points, rebounds, etc)
        line: Linha apostada
        direction: 'over' ou 'under'
        opening_odds: Odds no momento da aposta
        game_time: Horário do jogo
        created_at: Quando a aposta foi registrada
        closing_odds: Odds de fechamento (preenchido depois)
        clv: Valor de CLV calculado
        outcome: Resultado ('win', 'loss', 'push', None)
        status: Status atual
    """
    bet_id: str
    player_name: str
    prop_type: str
    line: float
    direction: str  # 'over' ou 'under'
    opening_odds: float
    game_time: datetime
    created_at: datetime = field(default_factory=datetime.now)
    closing_odds: Optional[float] = None
    clv: Optional[float] = None
    outcome: Optional[str] = None
    status: BetStatus = BetStatus.PENDING
    source: str = "unknown"
    bookmaker: Optional[str] = None
    
    def calculate_clv(self) -> Optional[float]:
        """
        Calcula CLV se odds de fechamento disponíveis.
        
        Returns:
            CLV como decimal ou None
        """
        if self.closing_odds is None or self.opening_odds is None:
            return None
        
        if self.closing_odds <= 0:
            return None
        
        # CLV = (Odd_Apostada / Odd_Fechamento) - 1
        self.clv = (self.opening_odds / self.closing_odds) - 1
        return self.clv
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário para serialização."""
        return {
            "bet_id": self.bet_id,
            "player_name": self.player_name,
            "prop_type": self.prop_type,
            "line": self.line,
            "direction": self.direction,
            "opening_odds": self.opening_odds,
            "closing_odds": self.closing_odds,
            "clv": self.clv,
            "game_time": self.game_time.isoformat() if self.game_time else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "outcome": self.outcome,
            "status": self.status.value,
            "source": self.source,
            "bookmaker": self.bookmaker,
        }


class CLVTracker:
    """
    Rastreador de Closing Line Value.
    
    Registra apostas, captura odds de fechamento, e calcula CLV para
    avaliar a qualidade do modelo ao longo do tempo.
    
    Exemplo de uso:
        >>> tracker = CLVTracker()
        >>> tracker.register_bet(
        ...     bet_id="bet_001",
        ...     player_name="LeBron James",
        ...     prop_type="points",
        ...     line=25.5,
        ...     direction="over",
        ...     opening_odds=1.91,
        ...     game_time=datetime(2024, 12, 19, 19, 30)
        ... )
        >>> # Depois, próximo do jogo:
        >>> await tracker.update_closing_odds()
        >>> summary = tracker.get_summary()
    """
    
    def __init__(
        self,
        performance_file: Optional[Path] = None,
        auto_save: bool = True
    ):
        """
        Inicializa o tracker.
        
        Args:
            performance_file: Caminho do arquivo CSV de performance
            auto_save: Se True, salva automaticamente após cada update
        """
        self.performance_file = performance_file or PERFORMANCE_FILE
        self.auto_save = auto_save
        self._tracked_bets: Dict[str, TrackedBet] = {}
        
        # Garantir que diretório existe
        self.performance_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Carregar apostas existentes
        self._load_existing()
        
        logger.info(
            f"📊 CLVTracker inicializado. "
            f"{len(self._tracked_bets)} apostas carregadas."
        )
    
    def _load_existing(self):
        """Carrega apostas existentes do arquivo CSV."""
        if not self.performance_file.exists():
            return
        
        try:
            with open(self.performance_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        bet = TrackedBet(
                            bet_id=row["bet_id"],
                            player_name=row["player_name"],
                            prop_type=row["prop_type"],
                            line=float(row["line"]),
                            direction=row["direction"],
                            opening_odds=float(row["opening_odds"]),
                            game_time=datetime.fromisoformat(row["game_time"]) if row.get("game_time") else datetime.now(),
                            closing_odds=float(row["closing_odds"]) if row.get("closing_odds") else None,
                            clv=float(row["clv"]) if row.get("clv") else None,
                            outcome=row.get("outcome") or None,
                            status=BetStatus(row.get("status", "pending")),
                            source=row.get("source", "unknown"),
                            bookmaker=row.get("bookmaker"),
                        )
                        self._tracked_bets[bet.bet_id] = bet
                    except (KeyError, ValueError) as e:
                        logger.warning(f"⚠️ Erro ao carregar bet: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao ler arquivo de performance: {e}")
    
    def register_bet(
        self,
        bet_id: str,
        player_name: str,
        prop_type: str,
        line: float,
        direction: str,
        opening_odds: float,
        game_time: datetime,
        source: str = "unknown",
        bookmaker: Optional[str] = None
    ) -> TrackedBet:
        """
        Registra uma nova aposta para rastreamento de CLV.
        
        Args:
            bet_id: ID único da aposta
            player_name: Nome do jogador
            prop_type: Tipo da prop
            line: Valor da linha
            direction: 'over' ou 'under'
            opening_odds: Odds no momento da aposta
            game_time: Horário do jogo
            source: Fonte da aposta
            bookmaker: Casa de apostas
            
        Returns:
            TrackedBet criado
        """
        if bet_id in self._tracked_bets:
            logger.warning(f"⚠️ Bet ID {bet_id} já existe. Atualizando...")
        
        bet = TrackedBet(
            bet_id=bet_id,
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            direction=direction.lower(),
            opening_odds=opening_odds,
            game_time=game_time,
            source=source,
            bookmaker=bookmaker,
        )
        
        self._tracked_bets[bet_id] = bet
        
        logger.info(
            f"📝 Aposta registrada: {bet_id} - "
            f"{player_name} {prop_type} {direction} {line} @ {opening_odds}"
        )
        
        if self.auto_save:
            self._save()
        
        return bet
    
    async def fetch_closing_odds(
        self,
        bet_id: str,
        available_props: Optional[List[Any]] = None
    ) -> Optional[float]:
        """
        Busca odds de fechamento para uma aposta.
        
        Args:
            bet_id: ID da aposta
            available_props: Lista de props já scrapeados (otimização)
            
        Returns:
            Odds de fechamento ou None
        """
        if bet_id not in self._tracked_bets:
            return None
            
        bet = self._tracked_bets[bet_id]
        if bet.closing_odds:
            return bet.closing_odds
            
        # Se não passamos props pré-carregados, teríamos que buscar individualmente
        # Mas API do Action Network retorna tudo. Então é melhor passar a lista.
        if not available_props:
            return None
            
        # Tentar encontrar a aposta nos props disponíveis
        # Matching: Player + Type + Line (aprox) + Direction
        from data.scrapers.player_name_normalizer import normalize_player_name
        
        target_player = normalize_player_name(bet.player_name)
        
        for prop in available_props:
            # Match Player
            if normalize_player_name(prop.player_name) != target_player:
                continue
                
            # Match Type
            if prop.prop_type.lower() != bet.prop_type.lower():
                continue
                
            # Match Line (permitir pequena divergência se mercado mudou linha principal)
            # Para CLV exato, idealmente queremos a MESMA linha.
            # Se a linha mudou, o CLV é mais complexo de calcular.
            # Simplificação: Só aceita se linha for igual.
            if abs(prop.line - bet.line) > 0.1:
                continue
                
            # Match Odds
            if bet.direction == 'over':
                return prop.over_odds
            elif bet.direction == 'under':
                return prop.under_odds
                
        return None
    
    def set_closing_odds(
        self,
        bet_id: str,
        closing_odds: float
    ) -> Optional[float]:
        """
        Define manualmente as odds de fechamento e calcula CLV.
        
        Args:
            bet_id: ID da aposta
            closing_odds: Odds de fechamento
            
        Returns:
            CLV calculado ou None
        """
        if bet_id not in self._tracked_bets:
            logger.warning(f"⚠️ Bet ID {bet_id} não encontrado")
            return None
        
        bet = self._tracked_bets[bet_id]
        bet.closing_odds = closing_odds
        bet.status = BetStatus.CLOSED
        clv = bet.calculate_clv()
        
        logger.info(
            f"📊 CLV calculado para {bet_id}: "
            f"Open={bet.opening_odds} → Close={closing_odds} = CLV {clv:.2%}"
        )
        
        if self.auto_save:
            self._save()
        
        return clv
    
    def set_outcome(
        self,
        bet_id: str,
        outcome: str  # 'win', 'loss', 'push'
    ):
        """
        Define resultado da aposta.
        
        Args:
            bet_id: ID da aposta
            outcome: 'win', 'loss', ou 'push'
        """
        if bet_id not in self._tracked_bets:
            return
        
        bet = self._tracked_bets[bet_id]
        bet.outcome = outcome
        bet.status = BetStatus.SETTLED
        
        logger.info(f"✅ Resultado registrado: {bet_id} = {outcome}")
        
        if self.auto_save:
            self._save()
    
    async def update_closing_odds(
        self,
        minutes_before_game: int = 20
    ) -> int:
        """
        Atualiza odds de fechamento para apostas próximas do jogo.
        Usa ActionNetworkScraper para buscar dados reais.
        
        Args:
            minutes_before_game: Janela de tempo antes do jogo (default: 20 min)
            
        Returns:
            Número de apostas atualizadas
        """
        now = datetime.now()
        threshold = now + timedelta(minutes=minutes_before_game)
        
        # 1. Identificar apostas que precisam de update
        pending_bets = [
            bet for bet in self._tracked_bets.values()
            if bet.status == BetStatus.PENDING
            and now < bet.game_time <= threshold
        ]
        
        if not pending_bets:
            # Verificar expiradas
            self._check_expired_bets(now)
            return 0
            
        logger.info(f"🔄 Buscando closing odds para {len(pending_bets)} apostas pendentes...")
        
        # 2. Buscar TODOS os props do dia (Batch Request)
        try:
            from data.scrapers.action_network_scraper import ActionNetworkScraper
            scraper = ActionNetworkScraper()
            # Buscar dados de hoje e amanhã para garantir
            props_today = await scraper.fetch_props(date=now.strftime("%Y-%m-%d"))
            
            # Se jogos forem virando a noite, talvez precise do dia seguinte
            # props_tomorrow = await scraper.fetch_props(date=(now + timedelta(days=1)).strftime("%Y-%m-%d"))
            # all_props = props_today + props_tomorrow
            all_props = props_today
            
            if not all_props:
                logger.warning("⚠️ ActionNetwork não retornou props. CLV update pulado.")
                return 0
                
        except Exception as e:
            logger.error(f"❌ Falha ao inicializar/rodar scraper de CLV: {e}")
            return 0
            
        # 3. Match e Update
        updated = 0
        for bet in pending_bets:
            closing = await self.fetch_closing_odds(bet.bet_id, available_props=all_props)
            
            if closing:
                self.set_closing_odds(bet.bet_id, closing)
                updated += 1
            else:
                logger.debug(f"📉 Closing line não encontrada para {bet.player_name} ({bet.prop_type})")
        
        # 4. Checar expiradas
        self._check_expired_bets(now)
        
        if updated > 0:
            logger.info(f"📊 {updated} apostas atualizadas com CLV real!")
            
        return updated

    def _check_expired_bets(self, now: datetime):
        """Marca apostas como expiradas se o jogo já começou."""
        for bet in self._tracked_bets.values():
            if bet.status == BetStatus.PENDING and bet.game_time <= now:
                bet.status = BetStatus.EXPIRED
                logger.warning(f"⚠️ Bet {bet.bet_id} expirou sem closing odds (Jogo iniciou)")
                if self.auto_save:
                    self._save()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo de performance do CLV.
        
        Returns:
            Dict com estatísticas de CLV
        """
        bets_with_clv = [b for b in self._tracked_bets.values() if b.clv is not None]
        
        if not bets_with_clv:
            return {
                "total_bets": len(self._tracked_bets),
                "bets_with_clv": 0,
                "avg_clv": None,
                "positive_clv_pct": None,
                "message": "Nenhuma aposta com CLV calculado ainda",
            }
        
        clvs = [b.clv for b in bets_with_clv]
        avg_clv = sum(clvs) / len(clvs)
        positive_clv_count = sum(1 for c in clvs if c > 0)
        
        # Estatísticas de outcome
        outcomes = [b.outcome for b in bets_with_clv if b.outcome]
        wins = sum(1 for o in outcomes if o == "win")
        losses = sum(1 for o in outcomes if o == "loss")
        
        return {
            "total_bets": len(self._tracked_bets),
            "bets_with_clv": len(bets_with_clv),
            "avg_clv": avg_clv,
            "avg_clv_pct": f"{avg_clv:.2%}" if avg_clv else None,
            "positive_clv_count": positive_clv_count,
            "positive_clv_pct": positive_clv_count / len(bets_with_clv) if bets_with_clv else None,
            "min_clv": min(clvs),
            "max_clv": max(clvs),
            "settled_bets": len(outcomes),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(outcomes) if outcomes else None,
            "interpretation": self._interpret_clv(avg_clv),
        }
    
    def _interpret_clv(self, avg_clv: Optional[float]) -> str:
        """
        Interpreta o CLV médio.
        
        Args:
            avg_clv: CLV médio
            
        Returns:
            Interpretação em texto
        """
        if avg_clv is None:
            return "Sem dados suficientes"
        
        if avg_clv > 0.03:
            return "🎯 EXCELENTE: Modelo está consistentemente batendo o mercado"
        elif avg_clv > 0.01:
            return "✅ BOM: Modelo tem edge positivo sustentável"
        elif avg_clv > 0:
            return "📊 OK: Edge marginal, continue monitorando"
        elif avg_clv > -0.01:
            return "⚠️ NEUTRO: Modelo não está capturando edge"
        else:
            return "❌ RUIM: Mercado está consistentemente melhor que o modelo"
    
    def _save(self):
        """Salva apostas no arquivo CSV."""
        if not self._tracked_bets:
            return
        
        try:
            fieldnames = [
                "bet_id", "player_name", "prop_type", "line", "direction",
                "opening_odds", "closing_odds", "clv", "game_time",
                "created_at", "outcome", "status", "source", "bookmaker"
            ]
            
            with open(self.performance_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for bet in self._tracked_bets.values():
                    writer.writerow(bet.to_dict())
            
            logger.debug(f"💾 Salvo {len(self._tracked_bets)} apostas em {self.performance_file}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar performance: {e}")
    
    def get_bets_for_update(
        self,
        within_minutes: int = 15
    ) -> List[TrackedBet]:
        """
        Retorna apostas que precisam de atualização de closing odds.
        
        Args:
            within_minutes: Jogos dentro de X minutos
            
        Returns:
            Lista de apostas pendentes
        """
        now = datetime.now()
        threshold = now + timedelta(minutes=within_minutes)
        
        return [
            bet for bet in self._tracked_bets.values()
            if bet.status == BetStatus.PENDING
            and bet.game_time <= threshold
            and bet.game_time > now
        ]
    
    def export_to_dataframe(self) -> "pd.DataFrame":
        """
        Exporta apostas para DataFrame pandas.
        
        Returns:
            DataFrame com todas as apostas
        """
        import pandas as pd
        
        data = [bet.to_dict() for bet in self._tracked_bets.values()]
        return pd.DataFrame(data)


# Singleton global
_clv_tracker_instance: Optional[CLVTracker] = None


def get_clv_tracker() -> CLVTracker:
    """
    Obtém instância singleton do CLVTracker.
    
    Returns:
        Instância compartilhada
    """
    global _clv_tracker_instance
    
    if _clv_tracker_instance is None:
        _clv_tracker_instance = CLVTracker()
    
    return _clv_tracker_instance


# ============================================================================
# TESTE / DEMO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    async def demo():
        """Demo do CLV Tracker."""
        tracker = CLVTracker()
        
        # Registrar algumas apostas
        tracker.register_bet(
            bet_id="demo_001",
            player_name="LeBron James",
            prop_type="points",
            line=25.5,
            direction="over",
            opening_odds=1.91,
            game_time=datetime.now() + timedelta(hours=2),
            source="demo"
        )
        
        tracker.register_bet(
            bet_id="demo_002",
            player_name="Stephen Curry",
            prop_type="threes",
            line=4.5,
            direction="over",
            opening_odds=1.85,
            game_time=datetime.now() + timedelta(hours=2),
            source="demo"
        )
        
        # Simular closing odds
        tracker.set_closing_odds("demo_001", 1.87)  # Positivo CLV
        tracker.set_closing_odds("demo_002", 1.95)  # Negativo CLV
        
        # Simular outcomes
        tracker.set_outcome("demo_001", "win")
        tracker.set_outcome("demo_002", "loss")
        
        # Ver resumo
        print("\n📊 RESUMO DE PERFORMANCE:")
        summary = tracker.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Exportar para DataFrame
        df = tracker.export_to_dataframe()
        print("\n📋 DataFrame:")
        print(df.to_string())
    
    asyncio.run(demo())
