import logging
import argparse
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from scipy.stats import norm
import pandas as pd

# Importar logger configurado
try:
    from utils.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback se logger_config não estiver disponível
    logger = logging.getLogger(__name__)

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.repositories.db_manager import get_db_manager
from data.scrapers.schedule_scraper import obter_schedule
from data.scrapers.injury_scraper import obter_injury_report
from data.scrapers.standings_scraper import obter_standings
from data.scrapers.stats_scraper import obter_player_stats, get_shot_quality_data
from data.scrapers.referee_scraper import scrape_referees
from data.scrapers.odds_scraper import obter_odds  # FIX: Usar scraper robusto com fallback
from core.algorithms import calcular_power_rating_v11
from core.simulation import simular_monte_carlo
from ml_pipeline.train_ensemble import train_ensemble_model
from ml_pipeline.predict import predict_next_games
from utils.confidence import get_prediction_confidence
from utils.kelly import get_bet_recommendation
from utils.export import exportar_para_csv, exportar_para_excel
from utils.validation import (
    validate_game_schedule,
    validate_team_name,
    validate_date,
    validate_prediction,
    validate_odds,
    ValidationError,
    safe_get
)

# Logger já configurado no início do arquivo

def run_prediction_pipeline(args: argparse.Namespace) -> Optional[List[Dict[str, Any]]]:
    """
    Executa o pipeline completo de previsão.

    Args:
        args: Argumentos da linha de comando.

    Returns:
        Lista de previsões ou None em caso de erro.
    """
    logger.info("🚀 Iniciando NBA Predictor v21.4 - Data Leakage Audit Complete...")

    # 0. Atualizar resultados pendentes
    logger.info("\n--- ETAPA 0: ATUALIZAÇÃO DE RESULTADOS ---")
    try:
        db = get_db_manager()
        db.update_pending_results()
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar resultados pendentes: {e}")
        logger.warning("⚠️  Continuando sem atualizar resultados...")

    # 1. Definir e validar data
    if args.date:
        try:
            data_analise = validate_date(args.date)
        except ValidationError as e:
            logger.error(f"❌ Data inválida: {e}")
            return None
    else:
        data_analise = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"📅 Data de Análise: {data_analise}")

    # 2. Coleta de dados
    logger.info("\n--- ETAPA 1: COLETA DE DADOS ---")
    try:
        schedule = obter_schedule(data_analise)
    except Exception as e:
        logger.error(f"❌ Erro ao obter schedule: {e}")
        return None

    if not schedule:
        logger.error("❌ Nenhum jogo encontrado para a data. Encerrando.")
        return None

    # Validar schedule
    valid_schedule = []
    for jogo in schedule:
        try:
            validate_game_schedule(jogo)
            valid_schedule.append(jogo)
        except ValidationError as e:
            logger.warning(f"⚠️  Jogo inválido ignorado: {e}")
            continue

    if not valid_schedule:
        logger.error("❌ Nenhum jogo válido encontrado após validação.")
        return None

    schedule = valid_schedule
    try:
        injuries = obter_injury_report()
    except Exception as e:
        logger.warning(f"⚠️  Erro ao obter injury report: {e}. Continuando sem dados de lesões.")
        injuries = {}

    try:
        standings = obter_standings()
    except Exception as e:
        logger.warning(f"⚠️  Erro ao obter standings: {e}. Continuando sem standings.")
        standings = {}

    try:
        dfs_stats = obter_player_stats()
    except Exception as e:
        logger.error(f"❌ Erro crítico ao obter player stats: {e}")
        return None

    try:
        shot_quality_data = get_shot_quality_data()
    except Exception as e:
        logger.warning(f"⚠️  Erro ao obter shot quality data: {e}. Continuando sem ajuste de shot quality.")
        shot_quality_data = None

    try:
        referee_assignments = scrape_referees(data_analise)
    except Exception as e:
        logger.warning(f"⚠️  Erro ao obter árbitros: {e}. Continuando sem dados de árbitros.")
        referee_assignments = {}

    try:
        odds_map = obter_odds()
    except Exception as e:
        logger.warning(f"⚠️  Erro ao obter odds: {e}. Usando odds padrão (1.90).")
        odds_map = {}

    # 3. Processamento e previsão
    logger.info("\n--- ETAPA 2: PROCESSAMENTO E PREVISÃO ---")

    # ML Pre-calculations
    ml_lookup = {}

    if args.ml:
        logger.info("🤖 Executando Modelos de Machine Learning (V18/V6)...")

        try:
            from ml_pipeline.predict import predict_next_games

            ml_df = predict_next_games()

            if not ml_df.empty:
                for _, row in ml_df.iterrows():
                    # Armazenar tudo no lookup
                    ml_lookup[(row['home_team'], row['away_team'])] = row.to_dict()
                logger.info(f"✅ Previsões ML geradas para {len(ml_df)} jogos.")
            else:
                logger.warning("⚠️  Nenhuma previsão ML gerada.")

        except Exception as e:
            logger.error(f"❌ Falha no pipeline ML: {e}")

    previsoes = []
    for jogo in schedule:
        try:
            home_team = validate_team_name(jogo['home'])
            away_team = validate_team_name(jogo['away'])
        except ValidationError as e:
            logger.warning(f"⚠️  Jogo ignorado devido a erro de validação: {e}")
            continue

        logger.info(f"\n🏀 Analisando: {away_team} @ {home_team}")

        # Árbitros
        refs = safe_get(referee_assignments, home_team, [])
        if refs:
            logger.info(f"   ⚖️ Árbitros: {', '.join(refs)}")

        # Power Rating & Monte Carlo
        try:
            resultado = calcular_power_rating_v11(
                home_team, away_team,
                injuries, standings, dfs_stats,
                referees=refs,
                shot_quality_data=shot_quality_data
            )
        except Exception as e:
            logger.error(f"❌ Erro ao calcular Power Rating para {home_team} vs {away_team}: {e}")
            continue

        try:
            # Pegar HCA usado no Power Rating (ou default 3.0 se não existir)
            hca_usado = resultado.get('hca_usado', 3.0)

            prob_mc = simular_monte_carlo(
                resultado['prob_casa'],
                resultado['nr_ajustado_casa'],
                resultado['nr_ajustado_visitante'],
                hca_value=hca_usado
            )
        except Exception as e:
            logger.warning(f"⚠️  Erro na simulação Monte Carlo: {e}. Usando probabilidade base.")
            prob_mc = resultado['prob_casa']

        # Odds - Hierarquia: API Real → Fair Odds Calculadas
        odds_source = "api"
        odd_home = None
        odd_away = None

        if home_team in odds_map:
            try:
                odd_home = validate_odds(safe_get(odds_map[home_team], 'home', None), "Odd Casa")
                odd_away = validate_odds(safe_get(odds_map[home_team], 'away', None), "Odd Visitante")

                # Se retornou None ou inválido, usar Fair Odds
                if odd_home is None or odd_home == 0:
                    raise ValidationError("Odd inválida")

                logger.info(f"✅ Odds da API para {home_team}: {odd_home:.2f} / {odd_away:.2f}")
            except (ValidationError, Exception) as e:
                logger.warning(f"⚠️ Odds inválidas da API: {e}. Calculando Fair Odds...")
                odds_source = "calculated"
                odd_home = None  # Forçar cálculo abaixo
        else:
            logger.info(f"ℹ️ Odds não encontradas na API para {home_team}. Usando Fair Odds.")
            odds_source = "calculated"
            odd_home = None

        # FALLBACK: Calcular Fair Odds baseadas nas probabilidades do modelo
        if odd_home is None or odd_home == 0:
            # Fair Odd = 100 / Probabilidade%
            # Ex: 55.8% → 100/55.8 = 1.79
            prob_home_pct = resultado['prob_casa']
            prob_away_pct = resultado['prob_visitante']

            if prob_home_pct > 0:
                odd_home = 100 / prob_home_pct
            else:
                odd_home = 2.00  # Último recurso (50/50)

            if prob_away_pct > 0:
                odd_away = 100 / prob_away_pct
            else:
                odd_away = 2.00

            logger.info(f"📊 Fair Odds calculadas: {home_team} {odd_home:.2f} | {away_team} {odd_away:.2f}")
            odds_source = "calculated"

        # Kelly Criterion (Moneyline)
        kelly_rec = get_bet_recommendation(
            resultado['prob_casa'],
            resultado['prob_visitante'],
            odd_home,
            odd_away
        )

        # Get ML probabilities if available, otherwise use Monte Carlo
        ml_data = ml_lookup.get((home_team, away_team))

        if ml_data and ml_data.get('prob_home', 0) > 0:
            # USE ML PREDICTIONS (V6 Calibrated Model)
            prob_final_home = ml_data['prob_home']
            prob_final_away = ml_data['prob_away']
            prediction_source = "ML"
        else:
            # FALLBACK: Use Power Rating + Monte Carlo
            prob_final_home = resultado['prob_casa']
            prob_final_away = resultado['prob_visitante']
            prediction_source = "MC"

        previsao = {
            "Data": data_analise,
            "Casa": home_team,
            "Visitante": away_team,
            "Prob Casa %": round(prob_final_home, 2),
            "Prob Visitante %": round(prob_final_away, 2),
            "Prob MC Casa %": round(prob_mc, 2),
            "Prob MC Visitante %": round(100 - prob_mc, 2),
            "NR Casa": round(resultado['nr_ajustado_casa'], 2),
            "NR Visitante": round(resultado['nr_ajustado_visitante'], 2),
            "Fator Lesão Casa": round(resultado['fator_lesao_casa'], 2),
            "Fator Lesão Visitante": round(resultado['fator_lesao_visitante'], 2),
            "Odd Casa": odd_home,
            "Odd Visitante": odd_away,
            "Confiança": get_prediction_confidence(prob_final_home),
            "Aposta Recomendada": kelly_rec['recommendation'],
            "Stake %": round(kelly_rec['stake_pct'], 2),
            "EV %": round(kelly_rec['ev'], 2)
        }

        # Add extra ML columns and Totals
        if ml_data:
            # Extra ML columns for display
            previsao['ML Model Home %'] = round(ml_data.get('prob_home', 0), 1)
            previsao['ML Model Away %'] = round(ml_data.get('prob_away', 0), 1)

            # Totals
            if 'predicted_total' in ml_data:
                previsao['Total Previsto'] = round(ml_data['predicted_total'], 1)
                logger.info(f"   🔢 Totals Model (V18): {previsao['Total Previsto']}")

            logger.info(f"   🤖 ML Moneyline (V6): {home_team} {prob_final_home:.1f}% [Source: {prediction_source}]")

        # SEMPRE adicionar colunas de lesões (seja de ML ou vazias)
        if ml_data:
            previsao['home_injuries_list'] = ml_data.get('home_injuries_list', '')
            previsao['away_injuries_list'] = ml_data.get('away_injuries_list', '')
        else:
            previsao['home_injuries_list'] = ''
            previsao['away_injuries_list'] = ''

        # Spread Calculation using final probabilities
        prob_diff = prob_final_home - prob_final_away
        pred_margin = prob_diff * 0.3
        previsao['Spread Previsto'] = round(pred_margin, 1)

        if pred_margin > 0:
            spread_str = f"{home_team} por {pred_margin:.1f}"
        else:
            spread_str = f"{away_team} por {abs(pred_margin):.1f}"
        logger.info(f"   📉 Spread (probabilístico): {spread_str}")

        # Totals Fallback (se não veio do ML)
        if 'Total Previsto' not in previsao:
            try:
                pr_casa = safe_get(resultado, 'pr_casa', 10.0)
                pr_visitante = safe_get(resultado, 'pr_visitante', 10.0)

                # Fórmula melhorada: usar média de pontos da liga + ajuste por Power Rating
                league_avg_points = 110.0
                home_offensive = league_avg_points + (pr_casa * 0.5)
                away_offensive = league_avg_points + (pr_visitante * 0.5)
                total_estimate = max(200, min(240, home_offensive + away_offensive))
                previsao['Total Previsto'] = round(total_estimate, 1)
                logger.info(f"   🔢 Totals (Fallback): {previsao['Total Previsto']}")
            except Exception as e:
                logger.warning(f"⚠️  Erro ao calcular total (fallback): {e}")
                previsao['Total Previsto'] = 218.0

        # Validar previsão antes de adicionar
        try:
            validate_prediction(previsao)
            previsoes.append(previsao)
        except ValidationError as e:
            logger.error(f"❌ Previsão inválida ignorada: {e}")
            continue

        # Logs finais do jogo
        logger.info(f"   📊 Previsão Final: {home_team} {previsao['Prob Casa %']}% vs {away_team} {previsao['Prob Visitante %']}%")
        if kelly_rec['recommendation'] != 'NO BET':
            logger.info(f"   💰 {kelly_rec['reasoning']} | Apostar {kelly_rec['stake_pct']:.2f}% da banca")

    # 4. Exportação e persistência
    logger.info("\n--- ETAPA 3: EXPORTAÇÃO ---")
    if previsoes:
        try:
            import pandas as pd
            df_previsoes = pd.DataFrame(previsoes)

            try:
                db = get_db_manager()
                db.save_predictions(previsoes)
            except Exception as e:
                logger.warning(f"⚠️  Erro ao salvar previsões no banco: {e}")

            filename_base = f"nba_predictions_{data_analise}"
            try:
                exportar_para_csv(df_previsoes, f"{filename_base}.csv")
                exportar_para_excel(df_previsoes, f"{filename_base}.xlsx")
            except Exception as e:
                logger.error(f"❌ Erro ao exportar arquivos: {e}")

            logger.info("✅ Processo concluído com sucesso!")
        except Exception as e:
            logger.error(f"❌ Erro crítico na exportação: {e}")
            return previsoes

        # --- CLI OUTPUT ---

        # Line Shopping
        try:
            from market.odds_shopping import fetch_multi_bookie_odds, compare_lines
            market_odds = fetch_multi_bookie_odds()
            print(f"\n🛒 Line Shopping: {len(market_odds)} jogos encontrados no mercado.")
        except Exception as e:
            market_odds = []
            print(f"⚠️  Erro Line Shopping: {e}")

        print("\n" + "="*80)
        print(f"🏀 PREDIÇÕES NBA - {data_analise}")
        print("="*80)

        for p in previsoes:
            casa = p['Casa']
            visitante = p['Visitante']
            prob_casa = p['Prob Casa %']
            spread = p.get('Spread Previsto', 0)

            # Determinar vencedor e confiança
            if prob_casa > 50:
                vencedor = casa
                confianca = prob_casa
            else:
                vencedor = visitante
                confianca = 100 - prob_casa

            # Determinar spread display
            if spread > 0:
                spread_str = f"{casa} -{spread:.1f}"
            else:
                spread_str = f"{visitante} -{abs(spread):.1f}"

            print(f"\n🏟️  {casa} vs {visitante}")
            print(f"🏆 Vencedor: {vencedor} ({confianca:.1f}%)")
            print(f"📉 Spread Justo: {spread_str}")

            # Mostrar Line Shopping
            if market_odds:
                opps = compare_lines(p, market_odds)
                if opps:
                    best = opps[0]
                    ev = best.get('ev', 0)
                    rec = best.get('recommendation', 'NEUTRO')

                    # Cores por recomendação
                    if rec == 'APOSTAR':
                        color = "\033[92m"  # Verde
                    elif rec == 'CONSIDERAR':
                        color = "\033[93m"  # Amarelo
                    else:
                        color = "\033[0m"  # Reset
                    reset = "\033[0m"

                    # Formato diferente para Moneyline vs Spread
                    market_type = best.get('market', 'Unknown')
                    bookie = best.get('bookie', 'Unknown')

                    if market_type == 'Moneyline':
                        odds = best.get('odds', 0)
                        model_prob = best.get('model_prob', 0)
                        print(f"🛒 Melhor ({market_type}): {color}{bookie} @ {odds:.2f} | Modelo: {model_prob}% | EV: {ev:.1f}%{reset}")
                    else:
                        line = best.get('line', 0)
                        odds = best.get('odds', 0)
                        print(f"🛒 Melhor ({market_type}): {color}{bookie} {line:+.1f} @ {odds:.2f} | EV: {ev:.1f}%{reset}")
                else:
                    print("🛒 Sem oportunidades válidas (spreads muito distantes).")

            print("-" * 40)

        return df_previsoes.to_dict(orient='records')
    else:
        logger.warning("⚠️  Nenhuma previsão gerada.")
        return []

def main():
    parser = argparse.ArgumentParser(description="NBA Predictor CLI")
    parser.add_argument("--date", type=str, help="Data para previsão (YYYY-MM-DD)")
    parser.add_argument("--backtest", action="store_true", help="Executar Backtest do modelo ML")
    parser.add_argument("--ml", action="store_true", default=True, help="Usar modelos ML (padrão: ativado)")
    parser.add_argument("--no-ml", action="store_true", help="Desativar modelos ML (usar só Power Rating)")
    args = parser.parse_args()

    # Suporte a --no-ml
    if args.no_ml:
        args.ml = False

    # Fix encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

    if args.backtest:
        logger.warning("Backtest not implemented in CLI yet.")
        return

    run_prediction_pipeline(args)

if __name__ == "__main__":
    main()
