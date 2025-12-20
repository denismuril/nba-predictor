"""
Market Odds Shopping - V21 FIX COMPLETO
========================================

Compara probabilidades do modelo ML com odds de mercado para encontrar EV positivo.

CORREÇÕES V21:
1. Suporte a Moneyline (h2h) - comparação direta com prob do modelo
2. MAX_SPREAD_DIFF = 2.5 - ignora spreads muito distantes
3. TTL de 60 minutos para odds desatualizadas
4. Sem conversões artificiais de probabilidade

Autor: NBA Predictor Team
Data: 2025-12-06
"""

import requests
import logging
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from scipy import stats  # AUDIT FIX #1: Para cálculo estatístico de spread probability
from config.constants import ODDS_API_KEY

# Initialize logger before using it in the import fallback chain
logger = logging.getLogger(__name__)

# ROBUSTNESS FIX: Import com fallback em cadeia (3 níveis)
# Evita quebra total do sistema se módulo de staking mudar de lugar
try:
    from betting.staking_strategy import KellyCriterionStrategy
    STAKING_ENABLED = True
    logger.info("✅ Kelly Criterion Staking carregado (betting.staking_strategy)")
except ImportError as e1:
    try:
        # Fallback 1: Tentar caminho alternativo
        from betting.confidence_kelly import ConfidenceKelly as KellyCriterionStrategy
        STAKING_ENABLED = True
        logger.warning(
            "⚠️ Staking carregado de betting.confidence_kelly (fallback 1). "
            f"Original error: {e1}"
        )
    except ImportError as e2:
        # Fallback 2: Dummy Strategy para não quebrar
        logger.critical(
            "🔴 CRITICAL: betting.staking_strategy E betting.confidence_kelly "
            "NÃO encontrados!\n"
            f"Error 1: {e1}\n"
            f"  Error 2: {e2}\n"
            "Sistema rodando com estratégia DUMMY (aposta fixa 1%).\n"
            "⚠️ CORRIJA imports antes de apostar dinheiro real!"
        )
        
        # Dummy Strategy minimalista
        class KellyCriterionStrategy:
            """Estratégia dummy conservadora (1% fixo)."""
            def __init__(self, bankroll=1000.0):
                self.bankroll = bankroll
            
            def calculate_optimal_stake(self, **kwargs):
                # Aposta fixa de 1% (extremamente conservador)
                return {
                    'stake_amount': self.bankroll * 0.01,
                    'stake_pct': 1.0,
                    'kelly_full': 0.01,
                    'kelly_fraction': 0.25
                }
            
            def adjust_for_correlation(self, bets):
                return bets  # Sem ajuste de correlação
        
        STAKING_ENABLED = False  # Marcar como desabilitado

BASE_URL = "https://api.the-odds-api.com/v4/sports"

# V21 FIX: Constantes de configuração
MAX_SPREAD_DIFF = 2.5  # Ignorar spreads com mais de 2.5 pontos de diferença
MAX_PROB_DIFF = 0.20   # Ignorar Moneyline se prob difere mais de 20% do modelo
MAX_ODDS_AGE_MINUTES = 10  # TTL para odds (P1-FIX: Reduzido de 60 para 10 min)


class CircuitBreaker:
    """Circuit Breaker para evitar requests repetidas a APIs instáveis."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: int = 300):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"

    def is_open(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.cooldown_seconds:
                logger.info("🔄 Circuit Breaker: Cooldown expirado, reconectando...")
                self.state = "CLOSED"
                self.failures = 0
                return False
            return True
        return False

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            mins = self.cooldown_seconds // 60
            logger.warning(
                f"🔴 Circuit Breaker ABERTO: {self.failures} falhas. "
                f"Bloqueando por {mins} min."
            )

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"


_odds_circuit_breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=300)


# ============================================================================
# FINANCIAL PERSISTENCE - BANKROLL MANAGEMENT
# ============================================================================

def get_current_bankroll() -> float:
    """
    PERSISTENCE FIX: Lê bankroll atual de data/bankroll.json.
    Se não existir, cria com valor inicial de R$ 2000.
    
    Returns:
        float: Bankroll atual em reais
    """
    bankroll_file = Path("data/bankroll.json")
    
    try:
        if bankroll_file.exists():
            with open(bankroll_file, 'r') as f:
                data = json.load(f)
                bankroll = float(data.get('current_bankroll', 2000.0))
                logger.debug(f"💰 Bankroll carregado: R$ {bankroll:.2f}")
                return bankroll
        else:
            # Criar arquivo inicial
            logger.info("💰 Criando bankroll.json inicial...")
            bankroll_file.parent.mkdir(parents=True, exist_ok=True)
            initial_data = {
                'current_bankroll': 2000.0,
                'initial_bankroll': 2000.0,
                'last_updated': datetime.now().isoformat(),
                'note': 'Gerado automaticamente pelo sistema'
            }
            with open(bankroll_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
            logger.info(
                "✅ data/bankroll.json criado com R$ 2000 inicial"
            )
            return 2000.0
    except Exception as e:
        logger.error(f"❌ Erro ao ler bankroll: {e}")
        return 2000.0  # Fallback seguro


def update_bankroll(new_amount: float) -> bool:
    """
    Atualiza bankroll no arquivo de persistência.
    
    Args:
        new_amount: Novo valor da banca em reais
        
    Returns:
        bool: True se atualizado com sucesso
    """
    bankroll_file = Path("data/bankroll.json")
    
    try:
        # Ler dados existentes
        if bankroll_file.exists():
            with open(bankroll_file, 'r') as f:
                data = json.load(f)
        else:
            data = {'initial_bankroll': new_amount}
        
        # Atualizar
        data['current_bankroll'] = new_amount
        data['last_updated'] = datetime.now().isoformat()
        
        # Salvar
        with open(bankroll_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💰 Bankroll atualizado: R$ {new_amount:.2f}")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar bankroll: {e}")
        return False


# Mapeamento TheOddsAPI Name -> Internal ID
API_TO_INTERNAL_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "New Jersey Nets": "BRK", "Charlotte Hornets": "CHA",
    "Charlotte Bobcats": "CHA", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU",
    "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}


def fetch_multi_bookie_odds(sport='basketball_nba', regions='us,eu,uk'):
    """Busca odds de h2h E spreads, normalizando nomes dos times.
    
    V27.3 FIX: Usa scrapers locais como fallback quando API retorna erro.
    """
    if _odds_circuit_breaker.is_open():
        logger.warning("⚡ Circuit Breaker ABERTO: Pulando request de odds")
        return []

    # V27.3 FIX: Tentar scrapers locais primeiro se API key ausente ou inválida
    if not ODDS_API_KEY or ODDS_API_KEY == "SUA_CHAVE_AQUI":
        logger.info("🔄 API key ausente, usando scrapers locais...")
        return _fetch_odds_from_local_scrapers()

    try:
        logger.info(f"🛒 Buscando odds em tempo real para {sport}...")
        url = f"{BASE_URL}/{sport}/odds"
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': regions,
            'markets': 'h2h,spreads',  # Buscar AMBOS os mercados
            'oddsFormat': 'decimal'
        }

        response = requests.get(url, params=params, timeout=10)
        
        # V27.3 FIX: Se erro 401/403, usar scrapers locais como fallback
        if response.status_code in [401, 403]:
            logger.warning(f"⚠️ API retornou {response.status_code}, usando scrapers locais...")
            return _fetch_odds_from_local_scrapers()
        
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list):
            logger.error(f"❌ API retornou formato inesperado: {type(data)}")
            _odds_circuit_breaker.record_failure()
            return _fetch_odds_from_local_scrapers()

        normalized_data = []
        for game in data:
            required = ['home_team', 'away_team', 'bookmakers']
            if not all(k in game for k in required):
                continue

            home_name = game['home_team']
            away_name = game['away_team']
            game['home_team_id'] = API_TO_INTERNAL_MAP.get(home_name, home_name)
            game['away_team_id'] = API_TO_INTERNAL_MAP.get(away_name, away_name)
            normalized_data.append(game)

        logger.info(f"✅ Odds obtidas para {len(normalized_data)} jogos.")
        _odds_circuit_breaker.record_success()
        return normalized_data

    except requests.exceptions.Timeout:
        logger.warning("⚠️ Timeout ao buscar odds, usando scrapers locais...")
        _odds_circuit_breaker.record_failure()
        return _fetch_odds_from_local_scrapers()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erro de rede: {e}")
        _odds_circuit_breaker.record_failure()
        return _fetch_odds_from_local_scrapers()
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}")
        _odds_circuit_breaker.record_failure()
        return _fetch_odds_from_local_scrapers()


def _fetch_odds_from_local_scrapers():
    """
    V27.3 FIX: Busca odds de scrapers locais (OddsAgora, OddsScanner, etc).
    
    Converte formato dos scrapers para formato esperado pelo sistema.
    """
    try:
        from data.scrapers.multi_odds_scraper import MultiSourceOddsScraper
        
        scraper = MultiSourceOddsScraper(headless=True)
        raw_odds = scraper.fetch_odds()
        
        if not raw_odds:
            logger.warning("⚠️ Scrapers locais não retornaram odds")
            return []
        
        # Converter formato dos scrapers para formato esperado
        normalized_data = []
        for game_key, data in raw_odds.items():
            home_name = data.get('home_team', '')
            away_name = data.get('away_team', '')
            home_odds = data.get('home_odds')
            away_odds = data.get('away_odds')
            
            if not home_name or not away_name or not home_odds or not away_odds:
                continue
            
            game = {
                'home_team': home_name,
                'away_team': away_name,
                'home_team_id': API_TO_INTERNAL_MAP.get(home_name, home_name),
                'away_team_id': API_TO_INTERNAL_MAP.get(away_name, away_name),
                'bookmakers': [{
                    'title': data.get('source', 'LocalScraper'),
                    'markets': [{
                        'key': 'h2h',
                        'outcomes': [
                            {'name': home_name, 'price': home_odds},
                            {'name': away_name, 'price': away_odds}
                        ]
                    }]
                }]
            }
            normalized_data.append(game)
        
        logger.info(f"✅ Scrapers locais: {len(normalized_data)} jogos obtidos")
        return normalized_data
        
    except ImportError as e:
        logger.error(f"❌ Falha ao importar scrapers locais: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Erro nos scrapers locais: {e}")
        return []



def _calculate_ev(prob_model: float, decimal_odds: float) -> float:
    """
    Calcula Expected Value (EV) corretamente.

    Fórmula: EV = (Prob_Modelo * Odds_Decimal) - 1

    Args:
        prob_model: Probabilidade do modelo (0.0 a 1.0)
        decimal_odds: Odds decimais do mercado (ex: 1.95)

    Returns:
        EV como decimal (ex: 0.05 = 5%)
    """
    return (prob_model * decimal_odds) - 1


def _is_odds_fresh(game_odds: dict, max_age_minutes: int = MAX_ODDS_AGE_MINUTES) -> bool:
    """
    V21 FIX: Verifica se as odds são recentes (TTL check).

    Returns:
        True se odds são frescas, False se desatualizadas
    """
    if 'last_update' not in game_odds:
        return True  # Assume fresca se não tiver timestamp

    try:
        last_update_str = game_odds['last_update']
        last_update = datetime.fromisoformat(
            last_update_str.replace('Z', '+00:00')
        )
        now_utc = datetime.now(timezone.utc)
        age_minutes = (now_utc - last_update).total_seconds() / 60

        if age_minutes > max_age_minutes:
            return False
        return True
    except Exception:
        return True  # Em caso de erro, assume fresca


def compare_lines(
    model_prediction: dict,
    market_odds_data: list,
    bankroll: float = None,  # PERSISTENCE FIX: Padrão None para carregar do arquivo
    calculate_stakes: bool = True
) -> list:
    """
    V21 FIX COMPLETO + GESTÃO DE BANCA: Compara predição do modelo com odds de mercado.

    LÓGICA CORRETA:
    1. PRIORIZA odds h2h (Moneyline) - comparação direta com prob do modelo
    2. Para spreads: só considera se diferença <= MAX_SPREAD_DIFF (2.5 pts)
    3. Aplica TTL de 60 minutos para odds desatualizadas
    4. NOVO: Calcula stake ótimo usando Kelly Criterion
    5. NOVO: Detecta correlação entre apostas

    Args:
        model_prediction: Dict com:
            - 'Casa': ID do time mandante (ex: 'BRK')
            - 'Prob Casa %': Probabilidade de vitória (ex: 53.5)
            - 'Spread Previsto': Spread justo do modelo (ex: -2.1)
        market_odds_data: Lista de jogos com odds da API
        bankroll: Banca atual para cálculo de stakes (default: $1000)
        calculate_stakes: Se True, calcula stakes usando Kelly (default: True)

    Returns:
        Lista de oportunidades ordenadas por EV (maior primeiro)
        Cada oportunidade inclui campos de staking se calculate_stakes=True
    """
    team_id_model = model_prediction.get('Casa')
    prob_home = model_prediction.get('Prob Casa %', 50.0) / 100.0
    model_spread = model_prediction.get('Spread Previsto', 0)

    if not team_id_model:
        return []

    # Encontrar o jogo correspondente
    game_odds = None
    for game in market_odds_data:
        if game.get('home_team_id') == team_id_model:
            game_odds = game
            break
        # Fallback: match parcial
        if team_id_model in game.get('home_team', ''):
            game_odds = game
            break

    if not game_odds:
        return []

    # V21 FIX: TTL Check - ignorar odds desatualizadas
    if not _is_odds_fresh(game_odds):
        logger.warning(f"⚠️ V21 TTL: Odds para {team_id_model} ignoradas (desatualizadas)")
        return []

    opportunities = []

    for bookie in game_odds.get('bookmakers', []):
        bookie_name = bookie.get('title', 'Unknown')

        for market in bookie.get('markets', []):
            market_key = market.get('key')

            # ============================================================
            # MERCADO H2H (MONEYLINE) - PRIORIDADE MÁXIMA
            # ============================================================
            if market_key == 'h2h':
                for outcome in market.get('outcomes', []):
                    outcome_name = outcome.get('name', '')
                    outcome_id = API_TO_INTERNAL_MAP.get(outcome_name, outcome_name)

                    # Match com time da casa
                    if outcome_id == team_id_model or team_id_model in outcome_name:
                        decimal_odds = outcome.get('price', 1.0)

                        # Probabilidade implícita do mercado
                        implied_prob = 1 / decimal_odds if decimal_odds > 0 else 0

                        # V21 FIX: SANITY CHECK - Ignorar se prob difere muito
                        # Se modelo diz 51% e mercado implica 21%, há discordância
                        prob_diff = abs(prob_home - implied_prob)
                        if prob_diff > MAX_PROB_DIFF:
                            # Skip - modelo e mercado discordam demais
                            continue

                        # V21 FIX: EV correto para Moneyline
                        # EV = (Prob_Modelo * Odds) - 1
                        ev_decimal = _calculate_ev(prob_home, decimal_odds)
                        ev_pct = ev_decimal * 100

                        # V21 FIX: Cap EV máximo realista (25%)
                        # EVs > 25% são provavelmente erros ou value traps
                        if ev_pct > 25:
                            continue

                        # Classificar oportunidade
                        if ev_pct > 5:
                            recommendation = 'APOSTAR'
                        elif ev_pct > 0:
                            recommendation = 'CONSIDERAR'
                        elif ev_pct > -3:
                            recommendation = 'NEUTRO'
                        else:
                            recommendation = 'EVITAR'

                        opportunities.append({
                            'market': 'Moneyline',
                            'bookie': bookie_name,
                            'team': team_id_model,
                            'odds': round(decimal_odds, 2),
                            'model_prob': round(prob_home * 100, 1),
                            'implied_prob': round(implied_prob * 100, 1),
                            'ev': round(ev_pct, 1),
                            'recommendation': recommendation
                        })

            # ============================================================
            # MERCADO SPREADS - COM SANITY CHECK
            # ============================================================
            elif market_key == 'spreads':
                for outcome in market.get('outcomes', []):
                    outcome_name = outcome.get('name', '')
                    outcome_id = API_TO_INTERNAL_MAP.get(outcome_name, outcome_name)

                    # Match com time da casa
                    if outcome_id == team_id_model or team_id_model in outcome_name:
                        market_line = outcome.get('point', 0)
                        decimal_odds = outcome.get('price', 1.0)

                        # V21 FIX: Ignorar odds extremas para spreads
                        # Odds > 3.0 são mercados alternativos muito arriscados
                        if decimal_odds > 3.0:
                            continue

                        # V21 FIX: Calcular diferença entre spread mercado e modelo
                        spread_diff = abs(market_line - (-model_spread))

                        # V21 FIX: SANITY CHECK - Ignorar spreads muito distantes
                        if spread_diff > MAX_SPREAD_DIFF:
                            continue

                        # AUDIT FIX #1: Cálculo estatístico correto de spread probability
                        # Fórmula: P(cover) = CDF((model_spread - market_spread) / std_deviation)
                        # NBA historical std deviation of game margins ≈ 12.5 points
                        NBA_MARGIN_STD = 12.5
                        
                        # edge_points = diferença entre spread do modelo e do mercado
                        # Positivo = modelo acha que time deve ganhar por mais pontos que mercado diz
                        edge_points = market_line - (-model_spread)
                        
                        # Probabilidade de cobrir o spread usando distribuição normal
                        # Se edge_points > 0, modelo favorece mais o time -> prob_cover > 0.5
                        prob_cover = stats.norm.cdf(edge_points / NBA_MARGIN_STD)
                        
                        # Usar prob_cover ao invés de ajuste linear arbitrário
                        adjusted_prob = min(0.95, max(0.05, prob_cover))

                        # EV para spread
                        ev_decimal = _calculate_ev(adjusted_prob, decimal_odds)
                        ev_pct = ev_decimal * 100

                        # V21 FIX: Cap EV máximo realista (25%)
                        # EVs > 25% são provavelmente erros ou value traps
                        if ev_pct > 25:
                            continue

                        # Classificar
                        if ev_pct > 5:
                            recommendation = 'APOSTAR'
                        elif ev_pct > 0:
                            recommendation = 'CONSIDERAR'
                        elif ev_pct > -3:
                            recommendation = 'NEUTRO'
                        else:
                            recommendation = 'EVITAR'

                        opportunities.append({
                            'market': 'Spread',
                            'bookie': bookie_name,
                            'team': team_id_model,
                            'line': market_line,
                            'model_line': round(-model_spread, 1),
                            'odds': round(decimal_odds, 2),
                            'edge_pts': round(edge_points, 1),
                            'ev': round(ev_pct, 1),
                            'recommendation': recommendation
                        })

    # Ordenar por EV (maior primeiro)
    opportunities.sort(key=lambda x: x['ev'], reverse=True)

    # ============================================================
    # GESTÃO DE BANCA: Calcular stakes usando Kelly Criterion
    # ============================================================
    # PERSISTENCE FIX: Carregar bankroll do arquivo se não fornecido
    if bankroll is None:
        bankroll = get_current_bankroll()
        logger.debug(f"💰 Usando bankroll persistido: R$ {bankroll:.2f}")
    
    if calculate_stakes and STAKING_ENABLED and opportunities:
        try:
            # Inicializar estratégia de staking
            strategy = KellyCriterionStrategy(bankroll=bankroll)

            # Calcular stake para cada oportunidade
            for opp in opportunities:
                # Apenas calcular stake para recomendação APOSTAR ou CONSIDERAR
                if opp['recommendation'] in ['APOSTAR', 'CONSIDERAR']:
                    stake_result = strategy.calculate_optimal_stake(
                        model_prob=opp['model_prob'] / 100.0,  # Converter de % para decimal
                        market_odds=opp['odds'],
                        confidence=0.80,  # Confidence padrão
                        game_id=f"{team_id_model}_vs_OPPONENT",  # Aproximação
                        team=opp['team'],
                        market=opp['market']
                    )

                    # Adicionar campos de staking à oportunidade
                    opp['suggested_stake_amount'] = stake_result['stake_amount']
                    opp['suggested_stake_pct'] = stake_result['stake_pct']
                    opp['kelly_full'] = stake_result['kelly_full']
                    opp['kelly_fraction'] = stake_result['kelly_fraction']
                else:
                    # Não apostar
                    opp['suggested_stake_amount'] = 0
                    opp['suggested_stake_pct'] = 0
                    opp['kelly_full'] = 0
                    opp['kelly_fraction'] = 0
            
            # P0 RISK FIX: Force Stake=0 for "Estimado" odds (Double Check)
            if opp.get('bookie') == 'Estimado':
                 opp['suggested_stake_amount'] = 0
                 opp['recommendation'] = 'NEUTRO' # Downgrade recomendation

            # Detectar correlação (se múltiplas apostas)
            if len([o for o in opportunities if o['recommendation'] in ['APOSTAR', 'CONSIDERAR']]) > 1:
                # Criar lista de bets para ajuste
                bets_to_adjust = []
                for opp in opportunities:
                    if opp['recommendation'] in ['APOSTAR', 'CONSIDERAR']:
                        bets_to_adjust.append({
                            'stake_amount': opp['suggested_stake_amount'],
                            'stake_pct': opp['suggested_stake_pct'],
                            'recommendation': 'BET',
                            'game_id': f"{team_id_model}_vs_OPPONENT",
                            'team': opp['team'],
                            'market': opp['market'],
                            'correlation_alert': None
                        })

                # Ajustar para correlação
                adjusted_bets = strategy.adjust_for_correlation(bets_to_adjust)

                # Aplicar ajustes
                bet_idx = 0
                for opp in opportunities:
                    if opp['recommendation'] in ['APOSTAR', 'CONSIDERAR']:
                        opp['suggested_stake_amount'] = adjusted_bets[bet_idx]['stake_amount']
                        opp['suggested_stake_pct'] = adjusted_bets[bet_idx]['stake_pct']
                        opp['correlation_alert'] = adjusted_bets[bet_idx]['correlation_alert']
                        bet_idx += 1
                    else:
                        opp['correlation_alert'] = None

        except Exception as e:
            logger.error(f"❌ Erro ao calcular stakes: {e}")
            # Continuar sem stakes calculados
            for opp in opportunities:
                opp['suggested_stake_amount'] = None
                opp['suggested_stake_pct'] = None
                opp['correlation_alert'] = None

    return opportunities


def get_best_opportunity(opportunities: list) -> dict:
    """
    Retorna a melhor oportunidade da lista (maior EV).

    Args:
        opportunities: Lista de oportunidades de compare_lines()

    Returns:
        Melhor oportunidade ou dict vazio
    """
    if not opportunities:
        return {}

    # Priorizar Moneyline sobre Spread em caso de empate
    best = opportunities[0]

    for opp in opportunities:
        if opp['ev'] > best['ev']:
            best = opp
        elif opp['ev'] == best['ev'] and opp.get('market') == 'Moneyline':
            best = opp

    return best


# ============================================================================
# COMPATIBILIDADE (LEGACY ALIASES)
# ============================================================================
# Mantém compatibilidade com scripts que buscam 'get_live_odds'
get_live_odds = fetch_multi_bookie_odds
