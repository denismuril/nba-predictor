#!/usr/bin/env python3
"""
NBA Predictor - Production Cycle Runner (O Cérebro)
====================================================
Script orquestrador que executa todo o pipeline de Player Props
de ponta a ponta, desde a ingestão até o envio de alertas.

Uso:
    python run_production_cycle.py

Fluxo:
    1. Ingestão: Busca player props via OddsDataManager
    2. Processamento: Enriquece dados via PropsProcessor
    3. Inteligência: Analisa props via PlayerPropsEngine
    4. Gestão de Banca: Aplica Kelly Criterion
    5. Entrega: Envia top 5 oportunidades via Telegram
    6. Logs: Salva resultado em logs/production_run.log

Autor: NBA Predictor Team
Versão: 1.0.0 (Dec 2025)
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Adicionar raiz do projeto ao path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / "production_run.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ProductionCycle")

# Silenciar logs verbosos
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
class ProductionConfig:
    """Configurações do ciclo de produção."""
    
    # Limites de filtragem
    MIN_EV_THRESHOLD: float = 0.05  # EV mínimo de 5%
    MIN_ODDS: float = 1.75  # Odds mínimas
    TOP_N_ALERTS: int = 5  # Número máximo de alertas
    
    # Kelly Criterion
    KELLY_FRACTION: float = 0.125  # Kelly / 8 (conservador)
    MAX_STAKE_PCT: float = 0.03  # Máximo 3% da banca por aposta
    BANKROLL: float = 1000.0  # Banca padrão para cálculos
    
    # Sniper Mode
    SNIPER_EV_THRESHOLD: float = 0.10  # EV > 10% = Sniper Bet


config = ProductionConfig()


# =============================================================================
# ETAPA 1: INGESTÃO DE DADOS
# =============================================================================
async def step_ingest_props(date: str) -> List:
    """
    Busca player props de múltiplos scrapers.
    
    Args:
        date: Data no formato YYYY-MM-DD
        
    Returns:
        Lista de PlayerProp
    """
    logger.info("=" * 60)
    logger.info("📥 ETAPA 1: INGESTÃO DE DADOS")
    logger.info("=" * 60)
    
    try:
        from data.odds_manager import OddsDataManager
        
        manager = OddsDataManager()
        logger.info(f"📅 Buscando props para: {date}")
        
        props = await manager.fetch_player_props(date)
        
        if not props:
            logger.warning("⚠️ Nenhum prop retornado pelos scrapers")
            return []
        
        logger.info(f"✅ {len(props)} props obtidos com sucesso")
        
        # Log por tipo de prop
        prop_types = {}
        for p in props:
            ptype = getattr(p, 'prop_type', 'unknown')
            prop_types[ptype] = prop_types.get(ptype, 0) + 1
        
        for ptype, count in prop_types.items():
            logger.info(f"   - {ptype}: {count}")
        
        return props
        
    except Exception as e:
        logger.error(f"❌ Erro na ingestão: {e}")
        raise


# =============================================================================
# ETAPA 2: PROCESSAMENTO (ENRIQUECIMENTO)
# =============================================================================
async def step_process_props(raw_props: List) -> 'pd.DataFrame':
    """
    Enriquece props com estatísticas de jogadores.
    
    Args:
        raw_props: Lista de PlayerProp brutos
        
    Returns:
        DataFrame processado com features
    """
    logger.info("=" * 60)
    logger.info("⚙️ ETAPA 2: PROCESSAMENTO E ENRIQUECIMENTO")
    logger.info("=" * 60)
    
    try:
        import pandas as pd
        from data.processing.props_processor import PropsProcessor
        
        processor = PropsProcessor()
        
        logger.info(f"🔄 Processando {len(raw_props)} props...")
        
        # Processar props (async)
        df = await processor.process_props(raw_props, include_odds=True)
        
        logger.info(f"✅ {len(df)} props processados com sucesso")
        logger.info(f"   Colunas: {list(df.columns)[:10]}...")
        
        return df
        
    except ValueError as e:
        logger.warning(f"⚠️ Processamento parcial: {e}")
        # Retornar DataFrame vazio em caso de erro
        import pandas as pd
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"❌ Erro no processamento: {e}")
        raise


# =============================================================================
# ETAPA 3: INTELIGÊNCIA (ANÁLISE ML)
# =============================================================================
def step_analyze_props(processed_df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Analisa props usando regras heurísticas Sniper.
    
    Args:
        processed_df: DataFrame processado
        
    Returns:
        DataFrame com análise EV+ e recomendações
    """
    logger.info("=" * 60)
    logger.info("🧠 ETAPA 3: ANÁLISE INTELIGENTE (SNIPER ENGINE)")
    logger.info("=" * 60)
    
    try:
        from ml_pipeline.player_props_engine import analyze_props
        
        logger.info(f"🔍 Analisando {len(processed_df)} props...")
        
        analyzed_df = analyze_props(
            processed_props=processed_df,
            min_ev=config.MIN_EV_THRESHOLD,
            min_odds=config.MIN_ODDS,
            edge_threshold=0.10
        )
        
        if analyzed_df.empty:
            logger.warning("⚠️ Nenhuma oportunidade EV+ encontrada")
            return analyzed_df
        
        sniper_count = 0
        if 'is_sniper' in analyzed_df.columns:
            sniper_count = analyzed_df['is_sniper'].sum()
        elif 'ev' in analyzed_df.columns:
            sniper_count = len(analyzed_df[analyzed_df['ev'] > config.SNIPER_EV_THRESHOLD])
        logger.info(f"✅ {len(analyzed_df)} oportunidades EV+ encontradas")
        logger.info(f"🎯 {sniper_count} Sniper Bets (EV > 10%)")
        
        return analyzed_df
        
    except Exception as e:
        logger.error(f"❌ Erro na análise: {e}")
        raise


# =============================================================================
# ETAPA 4: GESTÃO DE BANCA (KELLY)
# =============================================================================
def step_apply_kelly(analyzed_df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Aplica Kelly Criterion para definir tamanho das apostas.
    
    Args:
        analyzed_df: DataFrame analisado
        
    Returns:
        DataFrame com stakes calculadas
    """
    logger.info("=" * 60)
    logger.info("💰 ETAPA 4: GESTÃO DE BANCA (KELLY CRITERION)")
    logger.info("=" * 60)
    
    try:
        from utils.kelly import kelly_criterion_advanced
        
        if analyzed_df.empty:
            logger.warning("⚠️ Sem props para calcular Kelly")
            return analyzed_df
        
        stakes = []
        
        for idx, row in analyzed_df.iterrows():
            # Extrair probabilidade e odds
            prob = row.get('model_probability', row.get('sniper_prob', 0.55))
            odds = row.get('odds_over', row.get('odds', 1.90))
            
            # Calcular Kelly
            kelly_result = kelly_criterion_advanced(
                prob_win=prob,
                decimal_odds=odds,
                fractional=config.KELLY_FRACTION
            )
            
            stake_pct = kelly_result.get('kelly_fractional', 0.0)
            
            # Aplicar cap máximo
            stake_pct = min(stake_pct, config.MAX_STAKE_PCT)
            
            stake_amount = stake_pct * config.BANKROLL
            
            stakes.append({
                'stake_pct': stake_pct * 100,  # Em porcentagem
                'stake_amount': stake_amount,
                'kelly_full': kelly_result.get('kelly_full', 0),
                'should_bet': kelly_result.get('should_bet', False)
            })
        
        # Adicionar colunas ao DataFrame
        import pandas as pd
        stakes_df = pd.DataFrame(stakes)
        
        for col in stakes_df.columns:
            analyzed_df[col] = stakes_df[col].values
        
        # Filtrar apenas apostas válidas
        valid_bets = analyzed_df[analyzed_df['should_bet'] == True]
        
        logger.info(f"✅ {len(valid_bets)} apostas válidas após Kelly")
        
        if not valid_bets.empty:
            avg_stake = valid_bets['stake_pct'].mean()
            total_exposure = valid_bets['stake_pct'].sum()
            logger.info(f"   Stake médio: {avg_stake:.2f}%")
            logger.info(f"   Exposição total: {total_exposure:.2f}%")
        
        return valid_bets
        
    except Exception as e:
        logger.error(f"❌ Erro no cálculo Kelly: {e}")
        raise


# =============================================================================
# ETAPA 5: ENTREGA (TELEGRAM)
# =============================================================================
async def step_send_alerts(final_props: 'pd.DataFrame') -> int:
    """
    Envia alertas das melhores oportunidades via Telegram.
    
    Args:
        final_props: DataFrame final com top oportunidades
        
    Returns:
        Número de alertas enviados
    """
    logger.info("=" * 60)
    logger.info("📤 ETAPA 5: ENTREGA VIA TELEGRAM")
    logger.info("=" * 60)
    
    try:
        from telegram import Bot
        
        # Verificar token
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        admin_id = os.getenv('TELEGRAM_ADMIN_ID')
        
        if not token:
            logger.error("❌ TELEGRAM_BOT_TOKEN não configurado!")
            return 0
        
        if not admin_id:
            logger.error("❌ TELEGRAM_ADMIN_ID não configurado!")
            return 0
        
        bot = Bot(token=token)
        
        if final_props.empty:
            logger.info("ℹ️ Nenhuma prop para enviar")
            # Enviar mensagem de "sem oportunidades"
            await bot.send_message(
                chat_id=admin_id,
                text="📊 *CICLO DE PRODUÇÃO CONCLUÍDO*\n\n❌ Nenhuma oportunidade EV+ encontrada hoje.",
                parse_mode='Markdown'
            )
            return 0
        
        # Ordenar por EV (maior primeiro)
        ev_col = 'ev' if 'ev' in final_props.columns else 'expected_value'
        if ev_col in final_props.columns:
            sorted_props = final_props.sort_values(ev_col, ascending=False)
        else:
            sorted_props = final_props
        
        # Pegar top N
        top_props = sorted_props.head(config.TOP_N_ALERTS)
        
        logger.info(f"📨 Enviando {len(top_props)} alertas...")
        
        alerts_sent = 0
        
        for idx, (_, prop) in enumerate(top_props.iterrows(), 1):
            try:
                msg = _format_prop_alert(prop, idx)
                await bot.send_message(
                    chat_id=admin_id,
                    text=msg,
                    parse_mode='Markdown'
                )
                alerts_sent += 1
                
                # Pequeno delay para evitar rate limit
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"⚠️ Erro ao enviar alerta {idx}: {e}")
        
        # Enviar resumo final
        summary_msg = _format_summary(sorted_props)
        await bot.send_message(
            chat_id=admin_id,
            text=summary_msg,
            parse_mode='Markdown'
        )
        
        logger.info(f"✅ {alerts_sent} alertas enviados com sucesso")
        return alerts_sent
        
    except Exception as e:
        logger.error(f"❌ Erro no envio Telegram: {e}")
        raise


def _format_prop_alert(prop: 'pd.Series', rank: int) -> str:
    """
    Formata mensagem de alerta de prop.
    
    Args:
        prop: Série com dados do prop
        rank: Posição no ranking
        
    Returns:
        Mensagem formatada
    """
    # Extrair dados
    player = prop.get('player_name', prop.get('player', 'Unknown'))
    team = prop.get('team', prop.get('team_abbr', '???'))
    prop_type = prop.get('prop_type', 'unknown')
    line = prop.get('line', 0)
    odds = prop.get('odds_over', prop.get('odds', 1.90))
    
    # Projeção do modelo
    projection = prop.get('model_projection', prop.get('season_avg', line))
    
    # EV
    ev = prop.get('ev', prop.get('expected_value', 0)) * 100
    
    # Stake
    stake_pct = prop.get('stake_pct', 0)
    
    # Determinar direção
    direction = prop.get('direction', 'over').upper()
    direction_emoji = "📈" if direction == "OVER" else "📉"
    
    # Determinar se é Sniper Bet
    is_sniper = ev > 10 or prop.get('is_sniper', False)
    header = "🎯 *SNIPER ALERT*" if is_sniper else f"💡 *OPORTUNIDADE #{rank}*"
    
    msg = f"""{header}

🏀 *{player}* ({team})
{direction_emoji} *{direction} {line} {prop_type.upper()}* @ {odds:.2f}

🤖 Modelo: {projection:.1f} | ✅ EV: +{ev:.1f}%
💰 Stake Sugerida: {stake_pct:.1f}% (Kelly)
"""
    
    # Adicionar razão se disponível
    reason = prop.get('sniper_reason', prop.get('recommendation', ''))
    if reason:
        msg += f"\n📋 _{reason}_"
    
    return msg


def _format_summary(props: 'pd.DataFrame') -> str:
    """
    Formata resumo do ciclo de produção.
    
    Args:
        props: DataFrame de props analisados
        
    Returns:
        Mensagem de resumo
    """
    total = len(props)
    
    # Contar snipers
    sniper_count = 0
    if 'is_sniper' in props.columns:
        sniper_count = props['is_sniper'].sum()
    elif 'ev' in props.columns:
        sniper_count = len(props[props['ev'] > 0.10])
    
    # Calcular exposição total
    total_stake = props['stake_pct'].sum() if 'stake_pct' in props.columns else 0
    
    # EV médio
    ev_col = 'ev' if 'ev' in props.columns else 'expected_value'
    avg_ev = props[ev_col].mean() * 100 if ev_col in props.columns else 0
    
    now = datetime.now().strftime('%d/%m/%Y %H:%M')
    
    return f"""
📊 *RESUMO DO CICLO* | {now}

📋 Total de Oportunidades: {total}
🎯 Sniper Bets (EV>10%): {sniper_count}
📈 EV Médio: +{avg_ev:.1f}%
💰 Exposição Total: {total_stake:.1f}%

🤖 _Powered by NBA Predictor v27.5_
"""


# =============================================================================
# ORQUESTRADOR PRINCIPAL
# =============================================================================
async def run_production_cycle():
    """
    Executa o ciclo completo de produção.
    
    Fluxo:
        Ingestão → Processamento → Análise → Kelly → Entrega
    """
    start_time = datetime.now()
    
    logger.info("🚀" + "=" * 58)
    logger.info("🚀 INICIANDO CICLO DE PRODUÇÃO - NBA PREDICTOR")
    logger.info("🚀" + "=" * 58)
    logger.info(f"📅 Data/Hora: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determinar data dos jogos
    import pytz
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
    
    # Late-night logic: antes das 4AM ainda é "ontem"
    if now_et.hour < 4:
        from datetime import timedelta
        game_date = (now_et - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        game_date = now_et.strftime('%Y-%m-%d')
    
    logger.info(f"🏀 Data dos jogos (ET): {game_date}")
    
    results = {
        'date': game_date,
        'start_time': start_time.isoformat(),
        'steps': {},
        'success': False,
        'alerts_sent': 0
    }
    
    try:
        # ETAPA 1: Ingestão
        raw_props = await step_ingest_props(game_date)
        results['steps']['ingest'] = {'status': 'ok', 'count': len(raw_props)}
        
        if not raw_props:
            logger.warning("⚠️ Pipeline encerrado: sem props disponíveis")
            results['steps']['ingest']['status'] = 'empty'
            results['end_time'] = datetime.now().isoformat()
            _save_run_log(results)
            return
        
        # ETAPA 2: Processamento
        processed_df = await step_process_props(raw_props)
        results['steps']['process'] = {'status': 'ok', 'count': len(processed_df)}
        
        if processed_df.empty:
            logger.warning("⚠️ Pipeline encerrado: processamento retornou vazio")
            results['steps']['process']['status'] = 'empty'
            results['end_time'] = datetime.now().isoformat()
            _save_run_log(results)
            return
        
        # ETAPA 3: Análise
        analyzed_df = step_analyze_props(processed_df)
        results['steps']['analyze'] = {'status': 'ok', 'count': len(analyzed_df)}
        
        if analyzed_df.empty:
            logger.info("ℹ️ Nenhuma oportunidade EV+ hoje")
            results['steps']['analyze']['status'] = 'no_ev'
            # Ainda envia notificação de "sem oportunidades"
            await step_send_alerts(analyzed_df)
            results['end_time'] = datetime.now().isoformat()
            _save_run_log(results)
            return
        
        # ETAPA 4: Kelly
        final_df = step_apply_kelly(analyzed_df)
        results['steps']['kelly'] = {'status': 'ok', 'count': len(final_df)}
        
        # ETAPA 5: Envio
        alerts_sent = await step_send_alerts(final_df)
        results['steps']['send'] = {'status': 'ok', 'count': alerts_sent}
        results['alerts_sent'] = alerts_sent
        
        results['success'] = True
        
    except Exception as e:
        logger.error(f"💥 ERRO FATAL NO PIPELINE: {e}")
        logger.exception(e)
        results['error'] = str(e)
        
        # Tentar notificar erro via Telegram
        try:
            from telegram import Bot
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            admin_id = os.getenv('TELEGRAM_ADMIN_ID')
            if token and admin_id:
                bot = Bot(token=token)
                await bot.send_message(
                    chat_id=admin_id,
                    text=f"❌ *ERRO NO PIPELINE*\n\n`{str(e)[:500]}`",
                    parse_mode='Markdown'
                )
        except:
            pass
    
    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = duration
        
        logger.info("=" * 60)
        logger.info(f"🏁 CICLO FINALIZADO | Duração: {duration:.1f}s")
        logger.info(f"   Sucesso: {'✅' if results['success'] else '❌'}")
        logger.info(f"   Alertas enviados: {results['alerts_sent']}")
        logger.info("=" * 60)
        
        _save_run_log(results)


def _save_run_log(results: Dict):
    """Salva log estruturado da execução."""
    import json
    
    log_file = LOG_DIR / "production_runs.jsonl"
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.warning(f"Erro ao salvar log estruturado: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    """Entry point do script."""
    try:
        # Verificar dependências
        required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_ADMIN_ID']
        missing = [v for v in required_vars if not os.getenv(v)]
        
        if missing:
            logger.error(f"❌ Variáveis de ambiente faltando: {missing}")
            logger.error("Configure no .env antes de executar!")
            sys.exit(1)
        
        # Executar ciclo
        asyncio.run(run_production_cycle())
        
    except KeyboardInterrupt:
        logger.info("⚠️ Execução interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
