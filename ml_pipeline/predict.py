"""
Predict Next Games - Pipeline Unificado V21
============================================

Gera previsões para jogos futuros usando os modelos V6 (Moneyline) e V18 (Totals).
Implementa FAIL FAST para evitar predições com dados inventados.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from data.scrapers.schedule_scraper import obter_schedule
from ml_pipeline.calibrator import get_calibrator
from ml_pipeline.data_preparation import (
    load_historical_data,
    calculate_four_factors
)
from ml_pipeline.opponent_adjusted_stats import calcular_stats_ajustados_oponente
from ml_pipeline.elo_system import calcular_elo_ratings_historico
from core.contextual_features import add_all_contextual_features
from utils.team_normalization import normalize_team

try:
    from utils.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


def propagate_features_from_history(df_history, df_today, feature_names):
    """
    CRITICAL FIX: Propaga features rolling do ultimo jogo de cada time para jogos futuros.
    
    Problema: Jogos futuros tem features zeradas porque rolling e calculado do historico.
    Solucao: Para cada time no jogo futuro, copiar features do seu ultimo jogo historico.
    
    Args:
        df_history: DataFrame com jogos historicos (ja com features)
        df_today: DataFrame com jogos de hoje (sem features rolling corretas)
        feature_names: Lista de features do modelo
    
    Returns:
        df_today com features propagadas
    """
    rolling_feats = [f for f in feature_names if 'rolling' in f.lower()]
    
    if not rolling_feats:
        return df_today
    
    logger.info(f"   Propagando {len(rolling_feats)} rolling features do historico...")
    
    # Cache de features por time
    team_home_features = {}
    team_away_features = {}
    
    teams = set(df_today['home_team'].unique()) | set(df_today['away_team'].unique())
    
    for team in teams:
        # Ultimo jogo em casa
        home_games = df_history[df_history['home_team'] == team].tail(1)
        if not home_games.empty:
            team_home_features[team] = {}
            for feat in rolling_feats:
                if feat.startswith('home_') and feat in home_games.columns:
                    val = home_games[feat].values[0]
                    if pd.notna(val):
                        team_home_features[team][feat] = val
        
        # Ultimo jogo fora
        away_games = df_history[df_history['away_team'] == team].tail(1)
        if not away_games.empty:
            team_away_features[team] = {}
            for feat in rolling_feats:
                if feat.startswith('away_') and feat in away_games.columns:
                    val = away_games[feat].values[0]
                    if pd.notna(val):
                        team_away_features[team][feat] = val
    
    # Aplicar features nos jogos de hoje
    propagated_count = 0
    for idx in df_today.index:
        home = df_today.loc[idx, 'home_team']
        away = df_today.loc[idx, 'away_team']
        
        # Features HOME do time da casa
        if home in team_home_features:
            for feat, val in team_home_features[home].items():
                # Criar coluna se nao existir
                if feat not in df_today.columns:
                    df_today[feat] = 0.0
                # Preencher se zerado/NaN
                current = df_today.loc[idx, feat]
                if pd.isna(current) or current == 0:
                    df_today.loc[idx, feat] = val
                    propagated_count += 1
        
        # Features AWAY do time visitante
        if away in team_away_features:
            for feat, val in team_away_features[away].items():
                # Criar coluna se nao existir
                if feat not in df_today.columns:
                    df_today[feat] = 0.0
                # Preencher se zerado/NaN
                current = df_today.loc[idx, feat]
                if pd.isna(current) or current == 0:
                    df_today.loc[idx, feat] = val
                    propagated_count += 1
    
    logger.info(f"   OK: {propagated_count} valores propagados do historico")
    return df_today


# PERFORMANCE FIX: Global cache para modelos (Singleton Pattern)
# Evita carregar modelos do disco a cada execução (288x/dia = 14GB I/O desperdiçado)
_MODEL_CACHE = {
    'totals_model': None,
    'ensemble_model': None,
    'calibrator': None,
    'last_load_time': None
}



def load_models(force_reload: bool = False):
    """
    Carrega os modelos V18 (Totals) e V6 (Moneyline) e o calibrador.
    
    PERFORMANCE FIX: Usa cache em memória (Singleton Pattern).
    Carrega do disco apenas na primeira chamada ou se force_reload=True.
    
    Args:
        force_reload: Se True, ignora cache e recarrega do disco
        
    Returns:
        tuple: (totals_model, ensemble_model, calibrator)
    """
    global _MODEL_CACHE
    
    # PERFORMANCE FIX: Retornar do cache se disponível
    if not force_reload and _MODEL_CACHE['totals_model'] is not None:
        logger.debug("✅ Modelos carregados DO CACHE (Singleton)")
        return (
            _MODEL_CACHE['totals_model'],
            _MODEL_CACHE['ensemble_model'],
            _MODEL_CACHE['calibrator']
        )
    
    # Carregar do disco
    logger.info("📀 Carregando modelos DO DISCO...")
    models_dir = Path('data/models')

    try:
        totals_model = joblib.load(models_dir / 'totals_model_v18.joblib')
        logger.info("✅ Modelo Totals V18 carregado DO DISCO")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar Totals V18: {e}")
        totals_model = None

    try:
        ensemble_model = joblib.load(models_dir / 'ensemble_model_v6.joblib')
        logger.info("✅ Modelo Ensemble V6 carregado DO DISCO")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar Ensemble V6: {e}")
        ensemble_model = None

    try:
        calibrator = get_calibrator('data/models/calibrator.pkl')
        logger.info("✅ Calibrador carregado DO DISCO")
    except Exception as e:
        calibrator = None
        logger.warning(f"⚠️ Calibrador não encontrado: {e}")

    # PERFORMANCE FIX: Armazenar no cache
    _MODEL_CACHE['totals_model'] = totals_model
    _MODEL_CACHE['ensemble_model'] = ensemble_model
    _MODEL_CACHE['calibrator'] = calibrator
    _MODEL_CACHE['last_load_time'] = datetime.now()
    
    logger.info(f"💾 Modelos cacheados em memória (timestamp: {_MODEL_CACHE['last_load_time']})")

    return totals_model, ensemble_model, calibrator


def predict_next_games(date=None):
    """
    Gera previsões para os próximos jogos usando Pipeline Unificado.

    V21 FIX: Implementa FAIL FAST - não gera predições com dados inventados.
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"🚀 Iniciando previsões para {date} (Pipeline Unificado)")

    # 1. Obter Schedule
    try:
        schedule_data = obter_schedule(date)
        if not schedule_data:
            logger.warning(f"⚠️ Nenhum jogo encontrado para {date}")
            return pd.DataFrame()

        if isinstance(schedule_data, list):
            df_schedule = pd.DataFrame(schedule_data)
            if 'home' in df_schedule.columns:
                df_schedule = df_schedule.rename(
                    columns={'home': 'home_team', 'away': 'away_team'}
                )
            if 'date' not in df_schedule.columns:
                df_schedule['date'] = date
        else:
            df_schedule = schedule_data

        if df_schedule.empty:
            logger.warning(f"⚠️ DataFrame de schedule vazio para {date}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"❌ Erro ao obter schedule: {e}")
        return pd.DataFrame()

    # 2. Carregar Histórico (RAW)
    logger.info("📊 Carregando histórico para feature engineering...")
    df_history = load_historical_data(raw=True)

    if df_history is None or df_history.empty:
        logger.error("❌ Falha ao carregar histórico")
        return pd.DataFrame()

    # 3. Combinar Schedule com Histórico
    df_upcoming = df_schedule.copy()
    df_upcoming['date'] = pd.to_datetime(df_upcoming['date'])

    df_upcoming['home_team'] = df_upcoming['home_team'].apply(normalize_team)
    df_upcoming['away_team'] = df_upcoming['away_team'].apply(normalize_team)
    df_upcoming = df_upcoming.dropna(subset=['home_team', 'away_team'])

    for col in ['home_score', 'away_score', 'winner']:
        df_upcoming[col] = np.nan

    df_full = pd.concat([df_history, df_upcoming], ignore_index=True)
    df_full['date'] = pd.to_datetime(df_full['date'])
    df_full = df_full.sort_values('date').reset_index(drop=True)

    logger.info(f"📚 Dataset combinado: {len(df_full)} jogos")

    # 4. Executar Pipeline de Features
    logger.info("🔧 Executando Pipeline de Features...")
    try:
        # Keep this - rolling features need the BASE features from here
        df_full = calculate_four_factors(df_full)
        
        # CRITICAL FIX: DISABLED - ortg_adj calculation uses current game scores
        # This was causing 100% probabilities in predictions (data leakage)
        # Model V6 doesn't need ortg_adj BASE (only rolling_ortg_adj which are safe)
        # df_full = calcular_stats_ajustados_oponente(df_full)
        
        df_full = calcular_elo_ratings_historico(df_full)
        df_full = add_all_contextual_features(df_full)
        
        # CRITICAL: Use add_rolling_features for model compatibility
        # Model V6 expects: home_rolling_5_points, home_rolling_5_efg, etc.
        from ml_pipeline.data_preparation import add_rolling_features
        df_full = add_rolling_features(df_full, windows=[5, 10])
        
        # Add Four Factors rolling for V4 advanced features (ts_pct, off_rating)
        # BUG FIXED: Now preserves existing columns and only adds new ones
        from ml_pipeline.feature_engineering_v2 import add_rolling_four_factors
        df_full = add_rolling_four_factors(df_full, windows=[5, 10, 30])

        # V18 FIX: Add Pace Volatility Features (required for Totals model)
        # Features: home_pace_trend_10, away_pace_trend_10, home_pace_trend_5, away_pace_trend_5
        try:
            from ml_pipeline.pace_volatility import add_pace_volatility_features
            df_full = add_pace_volatility_features(df_full, windows=[5, 10])
            logger.info("   ✅ Pace Volatility features added (pace_trend_5, pace_trend_10)")
        except Exception as e:
            logger.warning(f"   ⚠️ Pace Volatility failed: {e}")

        # Features opcionais
        try:
            from ml_pipeline.feature_engineering_v2 import (
                add_contextual_rolling_features
            )
            df_full = add_contextual_rolling_features(df_full, window=10)
            logger.info("   ✅ Context-aware rolling features added")
        except Exception as e:
            logger.warning(f"   ⚠️ Context-aware rolling failed: {e}")


        try:
            from ml_pipeline.feature_engineering_v2 import add_referee_features
            df_full = add_referee_features(df_full, referee_names_col=None)
            logger.info("   ✅ Referee features added")
        except Exception as e:
            logger.warning(f"   ⚠️ Referee features failed: {e}")

        try:
            from ml_pipeline.feature_engineering_v2 import add_smart_money_features
            # AUDIT FIX #1: Skip Smart Money if we don't have real opening/closing odds
            # Using odds_home for both creates constant zero features (useless)
            if 'opening_odds' in df_full.columns and 'closing_odds' in df_full.columns:
                df_full = add_smart_money_features(df_full, opening_col='opening_odds', closing_col='closing_odds')
                logger.info("   ✅ Smart Money features added (real movement data)")
            else:
                logger.info("   ⏭️ Smart Money features SKIPPED (no real movement data)")
        except Exception as e:
            logger.warning(f"   ⚠️ Smart Money failed: {e}")

        try:
            from ml_pipeline.player_aggregation import (
                aggregate_player_stats_by_team,
                merge_player_features_to_games,
                get_cached_player_stats
            )
            df_players = get_cached_player_stats()
            if df_players is not None and not df_players.empty:
                df_agg = aggregate_player_stats_by_team(df_players, top_n=5)
                df_full = merge_player_features_to_games(df_full, df_agg)
                logger.info("   ✅ Player RAPM/BPM features added")
        except Exception as e:
            logger.warning(f"   ⚠️ Player features failed: {e}")

        # V4: Advanced Pace, Matchup, Volatility & Shooting Luck (MIGRADO para V2)
        try:
            from ml_pipeline.feature_engineering_v2 import prepare_advanced_features_only
            df_full = prepare_advanced_features_only(df_full)
            logger.info("   ✅ Advanced Features V2 (Pace, Matchup, Volatility, Shooting Luck) added")
        except Exception as e:
            logger.warning(f"   ⚠️ Feature Pipeline V4 Modular failed: {e}")


        # V2: Player Impact with RAPM (lesões)
        # CRITICAL: Initialize injury columns BEFORE try/except
        df_full['home_injuries_list'] = ''
        df_full['away_injuries_list'] = ''
        df_full['home_rapm_penalty'] = 0.0
        df_full['away_rapm_penalty'] = 0.0
        df_full['rapm_impact_diff'] = 0.0
        
        try:
            from ml_pipeline.player_impact import PlayerImpactCalculator
            from data.scrapers.injury_scraper import get_injuries_with_cache
            
            calculator = PlayerImpactCalculator()
            injuries_raw = get_injuries_with_cache()  # Retorna dict
            
            if injuries_raw is not None and len(injuries_raw) > 0:
                # BUG FIX: Converter dict para DataFrame
                # injuries_raw format: {'Minnesota Timberwolves': {'Player': 'STATUS'}}
                # PlayerImpactCalculator espera DataFrame com ['team', 'player', 'status']
                from config.constants import TEAM_ABBREV_MAP
                
                injuries_rows = []
                for team_full_name, players in injuries_raw.items():
                    # Converter nome completo -> código de 3 letras
                    team_code = TEAM_ABBREV_MAP.get(team_full_name, team_full_name)
                    
                    for player_name, status in players.items():
                        injuries_rows.append({
                            'team': team_code,
                            'player': player_name,
                            'status': status
                        })
                
                injuries_df = pd.DataFrame(injuries_rows)
                logger.info(f"   📋 {len(injuries_df)} lesões convertidas para DataFrame")
                
                # Calcular impacto para jogos futuros
                target_date = pd.to_datetime(date)
                future_mask = df_full['date'] == target_date
                
                # Helper function to format injury list
                def format_injuries_for_team(team_name, injuries_dict):
                    """Extract injured players for a team and format as string."""
                    if not isinstance(injuries_dict, dict):
                        return ""
                    
                    # Create reverse map: abbreviation -> full name
                    abbrev_to_full = {v: k for k, v in TEAM_ABBREV_MAP.items()}
                    
                    # Convert abbreviation to full name
                    team_full_name = abbrev_to_full.get(team_name, team_name)
                    
                    # injuries_dict format: {'Minnesota Timberwolves': {'Player': 'STATUS'}}
                    injuries_list = []
                    for team_key, players in injuries_dict.items():
                        # Exact match or fuzzy match
                        if team_full_name == team_key or team_name == team_key:
                            for player, status in players.items():
                                injuries_list.append(f"{player} ({status})")
                    
                    return ", ".join(injuries_list) if injuries_list else ""
                
                for idx in df_full[future_mask].index:
                    home = df_full.loc[idx, 'home_team']
                    away = df_full.loc[idx, 'away_team']
                    
                    # BUG FIX: Passar DataFrame em vez de dict
                    home_impact = calculator.get_team_impact_penalty(home, injuries_df)
                    away_impact = calculator.get_team_impact_penalty(away, injuries_df)
                    
                    df_full.loc[idx, 'home_rapm_penalty'] = home_impact
                    df_full.loc[idx, 'away_rapm_penalty'] = away_impact
                    df_full.loc[idx, 'rapm_impact_diff'] = away_impact - home_impact
                    
                    # NEW: Add injured players list
                    df_full.loc[idx, 'home_injuries_list'] = format_injuries_for_team(home, injuries_raw)
                    df_full.loc[idx, 'away_injuries_list'] = format_injuries_for_team(away, injuries_raw)
                
                logger.info("   ✅ Player Impact RAPM features added (with injury details)")
        except Exception as e:
            logger.warning(f"   ⚠️ Player Impact RAPM failed: {e}")

        logger.info(f"✅ Features geradas: {df_full.shape[1]} colunas")

    except Exception as e:
        logger.error(f"❌ Falha crítica no Pipeline de Features: {e}")
        raise

    # 5. Filtrar jogos do dia
    target_date = pd.to_datetime(date)
    df_today = df_full[df_full['date'] == target_date].copy()

    if df_today.empty:
        logger.warning(f"⚠️ Nenhuma feature gerada para {date}")
        return pd.DataFrame()

    logger.info(f"🎯 Preparando previsões para {len(df_today)} jogos")

    # 6. Carregar Modelos
    totals_model, ensemble_model, calibrator = load_models()

    # AUDIT FIX: Definir explicitamente todas as colunas calculadas a manter
    # Problema: A seleção anterior descartava métricas críticas calculadas no pipeline
    # Solução: Expandir lista e usar filter defensivo para evitar KeyError
    cols_to_keep = [
        'date', 'home_team', 'away_team', 
        'home_injuries_list', 'away_injuries_list',
        # Shooting Luck (V4)
        'home_shooting_luck', 'away_shooting_luck',
        'home_shooting_luck_ts', 'away_shooting_luck_ts',
        'home_shooting_luck_efg', 'away_shooting_luck_efg',
        # RAPM Penalties (Player Impact)
        'home_rapm_penalty', 'away_rapm_penalty', 'rapm_impact_diff',
        # Fatigue (se disponível)
        'home_fatigue_score', 'away_fatigue_score',
        # Monte Carlo (se disponível)
        'prob_mc_home', 'prob_mc_away',
        # Elo
        'home_elo', 'away_elo',
        # Pace
        'home_rolling_10_pace', 'away_rolling_10_pace', 'projected_pace_vegas'
    ]
    
    # Filtrar apenas colunas que existem no df_today (evita KeyError)
    available_cols = [c for c in cols_to_keep if c in df_today.columns]
    results = df_today[available_cols].copy()
    
    # Garantir que colunas críticas existem com valor default se falharam no cálculo
    # Isto previne errors downstream no web app / telegram bot
    critical_defaults = {
        'home_shooting_luck': 0.0,
        'away_shooting_luck': 0.0,
        'home_rapm_penalty': 0.0,
        'away_rapm_penalty': 0.0,
        'rapm_impact_diff': 0.0
    }
    for col, default_val in critical_defaults.items():
        if col not in results.columns:
            results[col] = default_val
            logger.warning(f"⚠️ Coluna {col} ausente, usando default={default_val}")
    
    logger.info(f"📊 Results DataFrame: {len(results.columns)} colunas preservadas")


    # --- PREVISÃO MONEYLINE (V6) - V21 FIX: FAIL FAST ---
    if ensemble_model:
        try:
            try:
                feature_names_v6 = joblib.load(
                    'data/models/feature_names_v6.joblib'
                )
            except FileNotFoundError:
                feature_names_v6 = joblib.load(
                    'data/models/ensemble_feature_names_v6.joblib'
                )

            # CRITICAL FIX: Propagar features rolling do historico para jogos de hoje
            # Problema: jogos futuros tem features zeradas porque nao tem score ainda
            # Solucao: copiar features do ultimo jogo de cada time
            # IMPORTANTE: Usa CACHE para evitar recálculo pesado (8h -> 10s)
            try:
                from ml_pipeline.data_cache import load_historical_data_cached
                df_hist_processed = load_historical_data_cached()  # COM CACHE
            except ImportError:
                df_hist_processed = load_historical_data()  # Fallback sem cache
            
            df_today = propagate_features_from_history(
                df_hist_processed,  # Historico COM features
                df_today,
                feature_names_v6
            )

            # V21 FIX: FAIL FAST - Detectar features críticas faltantes
            missing = [f for f in feature_names_v6 if f not in df_today.columns]
            critical_missing = [
                f for f in missing
                if 'rolling' in f.lower() or 'elo' in f.lower() or 'ortg' in f.lower()
            ]

            if critical_missing:
                logger.error(
                    f"❌ FAIL FAST: {len(critical_missing)} features críticas "
                    f"faltando. Exemplos: {critical_missing[:5]}"
                )
                results['prob_home'] = np.nan
                results['prob_away'] = np.nan
                results['confidence'] = 'NO_DATA'
                return results

            # CRITICAL FIX: Preencher TODAS features faltantes com 0
            # Problema original: features faltantes causavam probabilidades extremas (100%)
            # Solução: garantir que TODAS as features existam, preenchendo com 0
            if missing:
                logger.warning(f"⚠️ {len(missing)} features faltando, preenchendo com 0")
                for f in missing:
                    df_today[f] = 0.0

            # CRITICAL FIX: Zerar features RAPM/BPM/Referee para consistencia com treino
            # Problema: Modelo foi treinado com essas features = 0 (dados indisponiveis)
            # Mas predict.py gera valores reais, causando predicoes extremas (100%)
            # Solucao: Forcar essas features para 0 como estavam no treino
            features_to_zero = [
                'home_rapm_avg', 'home_rapm_top', 'home_rapm_std',
                'home_bpm_avg', 'home_bpm_top', 'home_bpm_std',
                'away_rapm_avg', 'away_rapm_top', 'away_rapm_std',
                'away_bpm_avg', 'away_bpm_top', 'away_bpm_std',
                'referee_home_win_pct', 'referee_foul_avg'
            ]
            for feat in features_to_zero:
                if feat in df_today.columns:
                    df_today[feat] = 0.0
            logger.info(f"   FIX: {len(features_to_zero)} features RAPM/BPM/Referee zeradas")

            # CRITICAL FIX: Reordenar colunas para match exato com feature_names_v6
            # Problema: ordem errada de colunas causa modelo ler features erradas
            X_v6 = df_today[feature_names_v6].copy()
            
            # Replace inf/nan
            X_v6 = X_v6.replace([np.inf, -np.inf], np.nan)

            # V21 FIX: FAIL FAST por jogo
            critical_feats = [f for f in feature_names_v6 if 'rolling' in f.lower()]
            games_missing = []
            valid_indices = []

            for idx, row in X_v6.iterrows():
                nan_count = row[critical_feats].isna().sum() if critical_feats else 0
                threshold = len(critical_feats) * 0.3

                if critical_feats and nan_count > threshold:
                    home = df_today.loc[idx, 'home_team']
                    away = df_today.loc[idx, 'away_team']
                    logger.error(
                        f"❌ Dados insuficientes: [{home}] vs [{away}]. "
                        f"{nan_count}/{len(critical_feats)} rolling NaN. PULADO."
                    )
                    games_missing.append(idx)
                else:
                    valid_indices.append(idx)

            if games_missing:
                logger.warning(f"⚠️ {len(games_missing)} jogos pulados")

            if not valid_indices:
                logger.error("❌ FAIL FAST: Nenhum jogo tem dados suficientes")
                results['prob_home'] = np.nan
                results['prob_away'] = np.nan
                results['confidence'] = 'SKIPPED'
            else:
                X_valid = X_v6.loc[valid_indices].fillna(0)
                probs = ensemble_model.predict_proba(X_valid)[:, 1]

                if calibrator:
                    probs_cal = calibrator.predict(probs)
                    if all(p <= 0.01 or p >= 0.99 for p in probs_cal):
                        logger.warning("⚠️ Calibrador extremo, usando raw")
                        probs_cal = probs
                else:
                    probs_cal = probs

                results['prob_home'] = np.nan
                results['prob_away'] = np.nan
                results['confidence'] = 'SKIPPED'

                for i, idx in enumerate(valid_indices):
                    p = probs_cal[i]
                    results.loc[idx, 'prob_home'] = p * 100
                    results.loc[idx, 'prob_away'] = (1 - p) * 100

                    p_max = max(p, 1 - p)
                    if p_max > 0.65:
                        conf = 'HIGH'
                    elif p_max > 0.55:
                        conf = 'MEDIUM'
                    else:
                        conf = 'LOW'
                    results.loc[idx, 'confidence'] = conf

                logger.info(
                    f"✅ Moneyline V6: {len(valid_indices)} ok, "
                    f"{len(games_missing)} puladas"
                )

        except Exception as e:
            logger.error(f"❌ Erro Moneyline: {e}")
            results['prob_home'] = np.nan
            results['prob_away'] = np.nan
            results['confidence'] = 'ERROR'

    # --- PREVISÃO TOTALS (V18) - V21 FIX: FAIL FAST ---
    if totals_model:
        try:
            feature_names_v18 = joblib.load(
                'data/models/totals_feature_names_v18.joblib'
            )

            LEAGUE_AVG_PACE = 99.5
            has_pace = (
                'home_rolling_10_pace' in df_today.columns and
                'away_rolling_10_pace' in df_today.columns
            )
            if has_pace:
                df_today['projected_pace_vegas'] = (
                    df_today['home_rolling_10_pace'] *
                    df_today['away_rolling_10_pace']
                ) / LEAGUE_AVG_PACE
            else:
                df_today['projected_pace_vegas'] = LEAGUE_AVG_PACE

            # V21 FIX: FAIL FAST
            missing = [f for f in feature_names_v18 if f not in df_today.columns]
            critical = [
                f for f in missing
                if 'rolling' in f.lower() or 'pace' in f.lower() or 'points' in f.lower()
            ]

            if critical:
                logger.error(
                    f"❌ V21 FAIL FAST Totals: {len(critical)} features "
                    f"críticas faltando. Exemplos: {critical[:5]}"
                )
                results['predicted_total'] = np.nan
            else:
                non_critical = [f for f in missing if f not in critical]
                if non_critical:
                    logger.warning(
                        f"⚠️ V21: {len(non_critical)} features não-críticas = 0"
                    )
                    for f in non_critical:
                        df_today[f] = 0.0

                X_v18 = df_today[feature_names_v18].fillna(0)
                preds_total = totals_model.predict(X_v18)
                results['predicted_total'] = preds_total
                logger.info("✅ Previsões Totals V18 geradas")

        except Exception as e:
            logger.error(f"❌ Erro Totals: {e}")
            results['predicted_total'] = np.nan

    # Adicionar Odds (se disponíveis)
    if 'odds_home' in df_schedule.columns:
        schedule_odds = df_schedule[['home_team', 'odds_home', 'odds_away']].copy()
        schedule_odds['home_team'] = schedule_odds['home_team'].apply(normalize_team)
        results = results.merge(schedule_odds, on='home_team', how='left')

    return results


if __name__ == "__main__":
    preds = predict_next_games()
    print(preds)
