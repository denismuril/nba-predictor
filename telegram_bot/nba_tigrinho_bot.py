"""
NBA Predictor Telegram Bot - Produção v20.6
=============================================
Bot profissional com:
- Predições diárias
- Alertas SNIPER (EV > 5%)
- Monitoramento de saúde do sistema
- Auto-registro de administrador
- HOTFIX: Timezone US/Eastern com late-night logic

Uso: Defina TELEGRAM_BOT_TOKEN como variável de ambiente antes de executar.
"""

import logging
import json
from pathlib import Path
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes
)
from datetime import datetime, timedelta
import pytz  # HOTFIX: Timezone handling
import pandas as pd
import os
import sys
import asyncio
import time  # HEARTBEAT FIX: Para unix timestamp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # Load .env file

from interfaces.cli import run_prediction_pipeline

# SECURITY: Token DEVE vir de variável de ambiente (nunca hardcoded)
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    raise EnvironmentError(
        "🔴 TELEGRAM_BOT_TOKEN não definido! "
        "Configure: export TELEGRAM_BOT_TOKEN='seu_token_aqui'"
    )

# SECURITY FIX: Admin ID DEVE vir do .env (Fail Fast)
ADMIN_CHAT_ID = os.getenv('TELEGRAM_ADMIN_ID')
if not ADMIN_CHAT_ID:
    raise EnvironmentError(
        "🔴 TELEGRAM_ADMIN_ID não definido no .env!\n"
        "Configure antes de iniciar o bot:\n"
        "  export TELEGRAM_ADMIN_ID='seu_chat_id_aqui'\n"
        "Obtenha seu Chat ID enviando /start para @userinfobot"
    )
try:
    ADMIN_CHAT_ID = int(ADMIN_CHAT_ID)
    logging.info(f"👑 Admin ID configurado: {ADMIN_CHAT_ID}")
except ValueError:
    raise EnvironmentError(
        f"🔴 TELEGRAM_ADMIN_ID inválido: '{ADMIN_CHAT_ID}' não é um número!"
    )

# Alerts Cache Persistence (evita spam após restart)
ALERTS_CACHE_FILE = Path("data/cache/sent_alerts.json")

# HEARTBEAT FIX: Arquivo de health check para monitoramento externo
HEALTH_CHECK_FILE = Path("data/health_check.txt")


# SECURITY FIX: Timezone-aware date for NBA games (Bot Offline Fix)
def get_nba_date() -> str:
    """
    SECURITY FIX: Returns today's date in NBA context (US/Eastern Time).
    
    Problema resolvido: Bot usava datetime.now() do servidor (UTC), 
    causando "Sem jogos hoje" para jogos à noite nos EUA.
    
    Late-night logic: Se antes das 04:00 AM ET, usa dia anterior
    para cobrir jogos da madrugada (ex: partida às 22h ET).
    
    Returns:
        str: Date in 'YYYY-MM-DD' format (NBA context)
    """
    eastern = pytz.timezone('US/Eastern')  # SECURITY FIX: Força timezone ET
    now_et = datetime.now(eastern)
    
    # SECURITY FIX: Late-night logic para jogos da madrugada
    if now_et.hour < 4:
        # Antes das 4AM ET = ainda é "ontem" no contexto NBA
        nba_date = (now_et - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        nba_date = now_et.strftime('%Y-%m-%d')
    
    return nba_date



# SECURITY FIX: Funções de admin removidas - Admin agora vem do .env apenas

def load_sent_alerts() -> dict:
    """Carrega alertas enviados do disco para evitar spam após restart."""
    if ALERTS_CACHE_FILE.exists():
        try:
            with open(ALERTS_CACHE_FILE, 'r') as f:
                data = json.load(f)
                return {k: datetime.fromisoformat(v) for k, v in data.items()}
        except Exception as e:
            logging.error(f"Erro ao carregar cache de alertas: {e}")
    return {}


def save_sent_alerts(alerts: dict):
    """Salva cache de alertas no disco."""
    try:
        ALERTS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v.isoformat() for k, v in alerts.items()}
        with open(ALERTS_CACHE_FILE, 'w') as f:
            json.dump(serializable, f)
    except Exception as e:
        logging.error(f"Erro ao salvar cache de alertas: {e}")

# Job Intervals
SNIPER_INTERVAL = 300  # 5 minutes
HEALTH_INTERVAL = 3600  # 1 hour
INJURY_ALERT_INTERVAL = 1800  # 30 minutes

# Anti-spam cache for sniper alerts (carregado do disco)
SENT_ALERTS = load_sent_alerts()
ALERT_COOLDOWN = 1800  # 30 minutes

# SECURITY: Lock para prevenir race conditions em jobs concorrentes
_sniper_lock = asyncio.Lock()

# TASK-004/005: Tracking do último sucesso de cada job para monitoramento
_job_last_success = {
    'sniper': None,
    'health': None,
    'injury_alerts': None
}

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# SECURITY FIX: Admin já validado no início do arquivo (fail-fast)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start - Boas-vindas (SECURITY FIX: Auto-registro removido)
    """
    user_id = update.effective_chat.id
    user_name = update.effective_user.first_name or "Usuário"
    
    # SECURITY FIX: Verificar se é o admin configurado
    is_admin = (user_id == ADMIN_CHAT_ID)
    
    welcome = f"""
🏀 *NBA TIGRINHO BOT* v21.6

Olá, *{user_name}*!

{'👑 Você é o Administrador' if is_admin else '📊 Bem-vindo ao sistema de previsões'}

*Comandos Disponíveis:*
/jogos - Jogos de hoje com previsões ML
/props - Estatísticas de jogadores (PTS/REB/AST)
/status - Saúde do sistema
/testes - Rodar testes de integridade

{'🎯 Alertas SNIPER ativos (EV > 5%)' if is_admin else ''}

🤖 Powered by XGBoost V6 + Kelly Criterion
    """
    
    await update.message.reply_text(welcome, parse_mode='Markdown')


async def jogos_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /jogos - Predições de Vencedor, Spread e Placar
    """
    await update.message.reply_text("⏳ Calculando predições dos jogos...")
    
    try:
        # Run pipeline para obter predições do modelo
        class Args:
            date = get_nba_date()  # HOTFIX: Timezone-aware
            ml = True
            backtest = False
        
        previsoes = run_prediction_pipeline(Args())
        
        if not previsoes:
            await update.message.reply_text("❌ Nenhum jogo hoje ou erro ao gerar predições!")
            return
        
        # Format message
        msg = f"🏀 *PREDIÇÕES DE JOGOS - {Args.date}*\n\n"
        
        # Line Shopping Integration
        try:
            from market.odds_shopping import fetch_multi_bookie_odds, compare_lines
            market_odds = fetch_multi_bookie_odds()
        except Exception as e:
            logger.error(f"Erro Line Shopping: {e}")
            market_odds = []

        for prev in previsoes:
            casa = prev['Casa']
            visitante = prev['Visitante']
            prob_casa = prev.get('Prob Casa %', 50)
            spread = prev.get('Spread Previsto', 0)
            
            msg += f"🏟️ *{casa}* vs *{visitante}*\n"
            
            # Vencedor provável
            if prob_casa > 50:
                vencedor = casa
                confianca = prob_casa
            else:
                vencedor = visitante
                confianca = 100 - prob_casa
                
            msg += f"🏆 Vencedor: *{vencedor}* ({confianca if confianca == confianca else 0.0:.1f}%)\n"
            
            # Spread Modelo
            if spread > 0:
                msg += f"📉 Spread Justo: {casa} -{spread if spread == spread else 0.0:.1f}\n"
            else:
                msg += f"📉 Spread Justo: {visitante} -{abs(spread) if spread == spread else 0.0:.1f}\n"
            
            # Placar Estimado & Totais
            total_est = prev.get('Total Previsto', 225.0)
            home_score = (total_est + spread) / 2
            away_score = (total_est - spread) / 2
            
            msg += f"🔢 *Total Previsto:* {total_est:.1f} pts\n" if total_est == total_est else "🔢 *Total Previsto:* N/A\n"
            msg += f"📊 *Placar Est:* {casa} {home_score:.0f} x {away_score:.0f} {visitante}\n" if (home_score == home_score and away_score == away_score) else f"📊 *Placar Est:* {casa} ? x ? {visitante}\n"
            
            # ML Model Details (V6)
            if 'ML Model Home %' in prev:
                msg += f"🤖 *Modelo V6:* {prev['ML Model Home %']}% vs {prev['ML Model Away %']}%\n"

            # Line Shopping - Melhor Oportunidade
            if market_odds:
                opps = compare_lines(prev, market_odds)
                if opps:
                    best = opps[0] # Melhor EV
                    icon = "✅" if best['recommendation'] == 'APOSTAR' else "⚠️"
                    msg += f"\n🛒 *Line Shopping:*\n"
                    msg += f"{icon} Melhor: *{best['bookie']}*\n"
                    line_desc = "ML" if best['market'] == 'Moneyline' else f"{best.get('line', '?'):+}"
                    msg += f"   Linha: {line_desc} @ {best.get('odds', '?')}\n"
                    msg += f"   EV: {best['ev']:.1f}% ({best['recommendation']})\n"
            
            msg += "--------------------------------\n"
        
        msg += "\n_Use /props para ver estatísticas de jogadores_"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Erro /jogos: {e}")
        await update.message.reply_text(f"❌ Erro: {str(e)}")


async def props_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /props - Estatísticas e Linhas de Jogadores (Top 3 PTS, REB, AST)
    """
    await update.message.reply_text("⏳ Buscando stats de jogadores (PTS, REB, AST)...")
    
    try:
        from data.scrapers.schedule_scraper import obter_schedule
        from data.scrapers.stats_scraper import obter_player_stats
        
        hoje = get_nba_date()  # HOTFIX: Timezone-aware
        jogos = obter_schedule(hoje)
        
        # jogos é uma lista de dicionários, não um DataFrame
        if not jogos or len(jogos) == 0:
            await update.message.reply_text("❌ Nenhum jogo hoje!")
            return
            
        # Get player stats
        stats_dfs = obter_player_stats()
        stats_df = None
        
        # Prioridade: BASIC_STATS > ALL_PLAYERS > PIE
        if isinstance(stats_dfs, dict):
            if 'BASIC_STATS' in stats_dfs and stats_dfs['BASIC_STATS'] is not None:
                stats_df = stats_dfs['BASIC_STATS']
                if hasattr(stats_df, 'empty') and stats_df.empty:
                    stats_df = None
            
            if stats_df is None and 'ALL_PLAYERS' in stats_dfs and stats_dfs['ALL_PLAYERS'] is not None:
                stats_df = stats_dfs['ALL_PLAYERS']
                if hasattr(stats_df, 'empty') and stats_df.empty:
                    stats_df = None
            
            if stats_df is None and 'pie' in stats_dfs and stats_dfs['pie'] is not None:
                stats_df = stats_dfs['pie']
                if hasattr(stats_df, 'empty') and stats_df.empty:
                    stats_df = None
        
        if stats_df is None:
            await update.message.reply_text("❌ Não foi possível carregar estatísticas de jogadores.")
            return
        
        # Verificar se stats_df tem dados
        if hasattr(stats_df, 'empty') and stats_df.empty:
            await update.message.reply_text("❌ Não foi possível carregar estatísticas de jogadores.")
            return

        # Mapa de normalização robusto (IDs canônicos após migração)
        TEAM_MAP_BOT = {
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
            "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
            "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
            "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
            "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
            "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
            "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
            "Phoenix Suns": "PHO", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
            "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
            "Washington Wizards": "WAS"
        }
        
        msg = f"👤 *PLAYER PROPS - {hoje}*\n"
        msg += "_Top 3 Médias na Temporada_\n\n"
        
        for jogo in jogos[:5]:
            casa = jogo.get('home', 'Casa')
            visitante = jogo.get('away', 'Visitante')
            
            msg += f"🏟️ *{casa}* vs *{visitante}*\n"
            
            casa_abbrev = TEAM_MAP_BOT.get(casa, casa[:3].upper())
            visitante_abbrev = TEAM_MAP_BOT.get(visitante, visitante[:3].upper())
            
            # Helper function
            def get_top_n(team_abbr, df, category='PTS', n=3):
                mask = df['TEAM'] == team_abbr
                matches = df[mask]
                if matches.empty: return []
                
                if category in matches.columns:
                    return matches.sort_values(category, ascending=False).head(n)
                return []

            # Processar cada time
            for team_name, team_abbr in [(casa, casa_abbrev), (visitante, visitante_abbrev)]:
                msg += f"\n📊 *{team_name} ({team_abbr})*\n"
                
                # Pontos
                top_pts = get_top_n(team_abbr, stats_df, 'PTS', 3)
                if isinstance(top_pts, list) and len(top_pts) == 0:
                    pass  # Lista vazia, não há dados
                elif hasattr(top_pts, 'empty') and not top_pts.empty:
                    msg += "🔥 *Pontos:*\n"
                    for _, p in top_pts.iterrows():
                        pts = p.get('PTS', 0.0)
                        val = pts if (pts is not None and pts == pts) else 0.0
                        msg += f"  • {p['PLAYER']}: {val:.1f}\n"
                
                # Rebotes
                top_reb = get_top_n(team_abbr, stats_df, 'REB', 3)
                if isinstance(top_reb, list) and len(top_reb) == 0:
                    pass  # Lista vazia, não há dados
                elif hasattr(top_reb, 'empty') and not top_reb.empty:
                    msg += "🖐️ *Rebotes:*\n"
                    for _, p in top_reb.iterrows():
                        reb = p.get('REB', 0.0)
                        val = reb if (reb is not None and reb == reb) else 0.0
                        msg += f"  • {p['PLAYER']}: {val:.1f}\n"
                        
                # Assistências
                top_ast = get_top_n(team_abbr, stats_df, 'AST', 3)
                if isinstance(top_ast, list) and len(top_ast) == 0:
                    pass  # Lista vazia, não há dados
                elif hasattr(top_ast, 'empty') and not top_ast.empty:
                    msg += "🤝 *Assistências:*\n"
                    for _, p in top_ast.iterrows():
                        ast = p.get('AST', 0.0)
                        val = ast if (ast is not None and ast == ast) else 0.0
                        msg += f"  • {p['PLAYER']}: {val:.1f}\n"

            msg += "\n--------------------------------\n"
            
        await update.message.reply_text(msg, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Erro /props: {e}")
        await update.message.reply_text(f"❌ Erro: {str(e)}")


async def testes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /testes - Rodar testes de integridade
    """
    await update.message.reply_text("🧪 Rodando testes de integridade...")
    
    try:
        import subprocess
        result = subprocess.run(
            ["pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd="/home/denis/nba-predictor"
        )
        
        # Parse resultado
        output = result.stdout + result.stderr
        
        if "passed" in output.lower():
            passed = output.count(" PASSED")
            failed = output.count(" FAILED")
            
            msg = f"✅ *TESTES DE INTEGRIDADE*\n\n"
            msg += f"Passou: {passed}\n"
            msg += f"Falhou: {failed}\n\n"
            
            if failed == 0:
                msg += "🎉 Sistema aprovado para apostas!"
            else:
                msg += "⚠️ SISTEMA COM PROBLEMAS!\n"
                msg += "NÃO APOSTAR até corrigir."
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.message.reply_text(f"```\n{output[:1000]}\n```", parse_mode='Markdown')
            
    except Exception as e:
        await update.message.reply_text(f"❌ Erro: {e}")


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /news - Últimas notícias de lesão (Woj/Shams)
    """
    await update.message.reply_text("🐦 Buscando tweets de Woj/Shams...")
    
    try:
        from data.scrapers.twitter_scraper import fetch_latest_injury_tweets
        alerts = fetch_latest_injury_tweets()
        
        if not alerts:
            await update.message.reply_text("✅ Nenhuma notícia crítica de lesão recente encontrada.")
            return
            
        msg = "🚨 *ÚLTIMAS NOTÍCIAS (LESÕES)*\n\n"
        for alert in alerts[:5]: # Top 5
            msg += f"🐦 {alert['text']}\n"
            msg += f"🕒 {alert['created_at']}\n\n"
            
        await update.message.reply_text(msg)
        
    except Exception as e:
        logger.error(f"Erro /news: {e}")
        await update.message.reply_text(f"❌ Erro: {str(e)}")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /status - Monitoramento do Sistema (Drift + Testes)
    """
    await update.message.reply_text("🔍 Verificando saúde do sistema...")
    
    try:
        # 1. Drift Check
        from ml_pipeline.drift_monitor import check_model_drift
        drift = check_model_drift()
        
        msg = "📊 *STATUS DO SISTEMA*\n\n"
        
        if drift['status'] == 'STABLE':
            msg += f"✅ Modelo: ESTÁVEL (MAE: {drift['mae']:.2f})\n"
        else:
            msg += f"⚠️ Modelo: DRIFT (MAE: {drift['mae']:.2f})\n"
            
        # 2. Testes Rápidos (Opcional, pode demorar)
        # msg += "\n🧪 Testes de Integridade: Use /testes para rodar completo."
        
        await update.message.reply_text(msg, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Erro /status: {e}")
        await update.message.reply_text(f"❌ Erro: {str(e)}")


# =============================================================================
# BACKGROUND JOBS - Sniper Alerts & Health Monitoring
# =============================================================================


async def check_live_opportunities(context: ContextTypes.DEFAULT_TYPE):
    """
    Job que roda a cada 5 minutos.
    Compara probabilidade do modelo vs odds ao vivo e envia alertas SNIPER.
    
    Gatilhos:
    - Probabilidade do modelo > 60%
    - EV (Expected Value) > 5%
    - Confiança > 55%
    
    SECURITY: Usa lock para evitar execuções paralelas (race condition fix)
    """
    global SENT_ALERTS
    
    # FIX: Adquirir lock para evitar race condition
    if _sniper_lock.locked():
        logger.debug("🔒 [SNIPER] Job anterior ainda em execução. Pulando...")
        return
    
    async with _sniper_lock:
        logger.info("🎯 [SNIPER] Verificando oportunidades ao vivo...")
    
        try:
            from betting.confidence_kelly import ConfidenceKelly
            from market.odds_shopping import fetch_multi_bookie_odds, compare_lines
            
            # Obter odds ao vivo
            try:
                market_odds = fetch_multi_bookie_odds()
            except Exception:
                logger.debug("Odds não disponíveis no momento")
                return
            
            if not market_odds:
                return
            
            # Obter predições do modelo
            class Args:
                date = get_nba_date()  # HOTFIX: Timezone-aware
                ml = True
                backtest = False
            
            try:
                previsoes = run_prediction_pipeline(Args())
            except Exception as e:
                logger.debug(f"Erro ao obter predições: {e}")
                return
            
            if not previsoes:
                return
            
            # Configurar Kelly
            kelly = ConfidenceKelly(min_edge=0.05, fraction=0.25)
            
            # Verificar cada jogo
            for prev in previsoes:
                casa = prev.get('Casa', '')
                prob_casa = prev.get('Prob Casa %', 50) / 100  # Converter para 0-1
                confidence = prev.get('Confiança', 0.5)
                
                # Proteção: não alertar se confiança baixa
                if confidence < 0.55:
                    continue
                
                # Proteção: só alertar se prob > 60%
                if prob_casa < 0.60 and (1 - prob_casa) < 0.60:
                    continue
                
                # Comparar com odds de mercado
                opps = compare_lines(prev, market_odds)
                
                for opp in opps[:1]:  # Pegar melhor oportunidade
                    if opp.get('ev', 0) > 5:  # EV > 5%
                        game_key = f"{casa}_{datetime.now().strftime('%Y%m%d')}"
                        
                        # Anti-spam: verificar cooldown
                        last_sent = SENT_ALERTS.get(game_key)
                        if last_sent:
                            # FIX: Usar total_seconds() para incluir dias completos
                            if (datetime.now() - last_sent).total_seconds() < ALERT_COOLDOWN:
                                continue  # Ainda em cooldown
                        
                        # Preparar dados financeiros (com fallback)
                        stake_amount = opp.get('suggested_stake_amount', 0) or 0
                        stake_pct = opp.get('suggested_stake_pct', 0) or 0
                        
                        # Enviar alerta com gestão de banca
                        alert_msg = f"""
🚨 *SNIPER ALERT* 🎯

🏀 *{casa}* vs {prev.get('Visitante', '?')}

✅ PICK: {opp.get('market', 'Moneyline')} @ {opp.get('odds', '?')}
💎 EV: +{opp.get('ev', 0):.1f}%

💰 STAKE: R$ {stake_amount:.2f} ({stake_pct:.1f}%)
🛡️ Confiança: {confidence*100:.0f}%
📈 Bookie: {opp.get('bookie', '?')}
"""
                        
                        # Enviar para todos os chats registrados
                        if ADMIN_CHAT_ID:
                            try:
                                await context.bot.send_message(
                                    chat_id=ADMIN_CHAT_ID,
                                    text=alert_msg,
                                    parse_mode='Markdown'
                                )
                                SENT_ALERTS[game_key] = datetime.now()
                                save_sent_alerts(SENT_ALERTS)  # FIX: Persistir imediatamente
                                logger.info(f"🚨 Alerta enviado: {casa} EV={opp.get('ev', 0):.1f}%")
                            except Exception as e:
                                logger.error(f"Erro ao enviar alerta: {e}")
        
        except Exception as e:
            logger.error(f"Erro no Sniper Job: {e}")


async def check_system_health(context: ContextTypes.DEFAULT_TYPE):
    """
    Job que roda a cada 1 hora.
    Verifica a saúde do sistema e envia alertas para o admin se houver problemas.
    """
    logger.info("🏥 [HEALTH] Verificando saúde do sistema...")
    
    try:
        from monitoring.alert_system import AlertSystem
        
        alert_system = AlertSystem()
        
        # Coletar métricas atuais
        metrics = {}
        
        # 1. Checar drift do modelo
        try:
            from ml_pipeline.drift_monitor import check_model_drift
            drift = check_model_drift()
            metrics['accuracy'] = {'test_accuracy': 1.0 if drift['status'] == 'STABLE' else 0.5}
        except Exception:
            pass
        
        # 2. Checar calibração
        try:
            from pathlib import Path
            calibrator_path = Path('data/models/calibrator.pkl')
            if calibrator_path.exists():
                from ml_pipeline.calibrator import AutoCalibrator
                calibrator = AutoCalibrator.load(str(calibrator_path))
                stats = calibrator.get_stats()
                metrics['calibration'] = {
                    'ece': stats.get('ece', 0.1),
                    'brier_calibrated': stats.get('brier_score', 0.25)
                }
        except Exception:
            pass
        
        # Verificar alertas
        alerts = alert_system.check_metrics(metrics)
        
        # Filtrar críticos
        critical = [a for a in alerts if a.get('severity') == 'CRITICAL']
        
        if critical and ADMIN_CHAT_ID:
            alert_msg = "🚨 *ALERTA CRÍTICO DO SISTEMA*\n\n"
            for alert in critical:
                alert_msg += f"⚠️ {alert.get('message', 'Erro desconhecido')}\n"
            
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=alert_msg,
                    parse_mode='Markdown'
                )
                logger.warning(f"Alertas críticos enviados ao admin: {len(critical)}")
            except Exception as e:
                logger.error(f"Erro ao enviar alerta de saúde: {e}")
        
        # Logar alertas (mesmo sem enviar)
        alert_system.send_alerts(alerts)
    
    except Exception as e:
        logger.error(f"Erro no Health Check Job: {e}")


async def check_injury_alerts(context: ContextTypes.DEFAULT_TYPE):
    """
    Job que roda a cada 30 minutos.
    Detecta novas lesões críticas e envia alertas para o admin.
    
    Foco em jogadores de alto impacto (MVP candidates, All-Stars).
    """
    logger.info("🏥 [INJURY] Verificando novas lesões críticas...")
    
    try:
        from data.scrapers.injury_telegram_alerts import InjuryAlertService
        
        service = InjuryAlertService(context.bot, str(ADMIN_CHAT_ID))
        alerts_sent = await service.check_and_send_alerts()
        
        if alerts_sent > 0:
            logger.info(f"🚨 [INJURY] {alerts_sent} alertas de lesão enviados")
        
        # Marcar sucesso
        _job_last_success['injury_alerts'] = datetime.now()
        
    except Exception as e:
        logger.error(f"Erro no Injury Alert Job: {e}")


async def update_heartbeat(context: ContextTypes.DEFAULT_TYPE):
    """
    HEARTBEAT FIX: Job que atualiza health_check.txt a cada minuto.
    
    Permite que scripts externos monitorem o status do bot.
    Se o arquivo não for atualizado por mais de 3 minutos,
    o bot provavelmente travou e precisa ser reiniciado.
    """
    try:
        HEALTH_CHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        heartbeat_data = {
            'timestamp': datetime.now().isoformat(),
            'unix_time': time.time(),
            'status': 'ALIVE',
            'version': 'v21.6',
            'jobs_status': {
                'sniper_last_run': _job_last_success.get('sniper', 'Never'),
                'health_last_run': _job_last_success.get('health', 'Never')
            }
        }
        
        with open(HEALTH_CHECK_FILE, 'w') as f:
            json.dump(heartbeat_data, f, indent=2, default=str)
        
        logger.debug(f"💓 Heartbeat: {heartbeat_data['timestamp']}")
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar heartbeat: {e}")


async def heartbeat_command(update, context):
    """
    /heartbeat - Status dos jobs background (TASK-004)
    """
    msg = "💓 *HEARTBEAT - JOBS STATUS*\n\n"

    for job_name, last_run in _job_last_success.items():
        if last_run:
            age = (datetime.now() - last_run).total_seconds()
            if age < 600:
                status = "🟢"
            elif age < 1800:
                status = "🟡"
            else:
                status = "🔴"
            time_str = last_run.strftime('%H:%M:%S')
            msg += f"{status} *{job_name.upper()}*: {time_str} ({age/60:.0f}min atrás)\n"
        else:
            msg += f"⚪ *{job_name.upper()}*: Nunca executado\n"

    await update.message.reply_text(msg, parse_mode='Markdown')


def main():
    """Run bot with background jobs"""
    app = Application.builder().token(TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("jogos", jogos_command))
    app.add_handler(CommandHandler("props", props_command))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("testes", testes_command))
    app.add_handler(CommandHandler("heartbeat", heartbeat_command))

    # Background Jobs (JobQueue)
    job_queue = app.job_queue

    if job_queue:
        # Sniper: a cada 5 minutos
        job_queue.run_repeating(
            check_live_opportunities,
            interval=SNIPER_INTERVAL,
            first=60,  # Iniciar após 1 minuto
            name='sniper_job'
        )
        logger.info(f"🎯 Job SNIPER configurado: a cada {SNIPER_INTERVAL//60} min")

        # Health Check: a cada 1 hora
        job_queue.run_repeating(
            check_system_health,
            interval=HEALTH_INTERVAL,
            first=300,  # Iniciar após 5 minutos
            name='health_job'
        )
        logger.info(f"🏥 Job HEALTH configurado: a cada {HEALTH_INTERVAL//60} min")
        
        # Injury Alerts: a cada 30 minutos
        job_queue.run_repeating(
            check_injury_alerts,
            interval=INJURY_ALERT_INTERVAL,
            first=120,  # Iniciar após 2 minutos
            name='injury_alert_job'
        )
        logger.info(f"🏥 Job INJURY ALERTS configurado: a cada {INJURY_ALERT_INTERVAL//60} min")
        
        # HEARTBEAT FIX: Atualizar health_check.txt a cada 60 segundos
        job_queue.run_repeating(
            update_heartbeat,
            interval=60,
            first=10,  # Iniciar após 10 segundos
            name='heartbeat_job'
        )
        logger.info("💓 Job HEARTBEAT configurado: a cada 60s")
    else:
        logger.warning("⚠️ JobQueue não disponível. Jobs desativados.")

    logger.info("🤖 NBA TIGRINHO BOT v20.8 iniciado!")
    logger.info("   - Comandos: /jogos, /props, /news, /status, /testes")
    logger.info("   - Jobs: Sniper (5min), Health (1h), Injury (30min), Heartbeat (60s)")
    app.run_polling()


if __name__ == "__main__":
    main()
