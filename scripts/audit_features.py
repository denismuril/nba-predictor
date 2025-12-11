#!/usr/bin/env python3
"""
Auditoria de Features - Data Quality & Leakage Detection
=========================================================

V2: Auditoria expandida com verificações de:
- Porcentagem de zeros em colunas críticas (injury_factor, etc)
- Correlação com score_diff (além de winner)
- Presença de features de Pace e Rest
- Relatório consolidado

Usage:
    python scripts/audit_features.py
"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import pandas as pd
import numpy as np
from ml_pipeline.data_preparation import load_historical_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def audit_zeros_in_critical_columns(df: pd.DataFrame) -> dict:
    """
    Verifica porcentagem de zeros em colunas críticas.
    
    Returns:
        Dict com estatísticas de zeros por coluna
    """
    critical_patterns = [
        'injury', 'impact', 'rapm', 'bpm', 'lebron',
        'rest', 'fatigue', 'travel'
    ]
    
    results = {}
    
    for col in df.columns:
        if any(pattern in col.lower() for pattern in critical_patterns):
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                total = len(df)
                zeros = (df[col] == 0).sum()
                nans = df[col].isna().sum()
                pct_zeros = (zeros / total) * 100 if total > 0 else 0
                pct_nans = (nans / total) * 100 if total > 0 else 0
                
                results[col] = {
                    'zeros': int(zeros),
                    'nans': int(nans),
                    'pct_zeros': round(pct_zeros, 2),
                    'pct_nans': round(pct_nans, 2),
                    'total': total
                }
    
    return results


def audit_correlations(df: pd.DataFrame) -> dict:
    """
    Calcula correlação das features com targets (winner e score_diff).
    
    Returns:
        Dict com correlações por feature
    """
    results = {'with_winner': [], 'with_score_diff': []}
    
    # Preparar targets
    if 'winner' in df.columns:
        y_winner = (df['winner'] == 'HOME').astype(int)
    else:
        y_winner = None
    
    # Calcular score_diff se possível
    if 'home_score' in df.columns and 'away_score' in df.columns:
        score_diff = df['home_score'] - df['away_score']
    else:
        score_diff = None
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    exclude_cols = ['winner', 'correct', 'home_score', 'away_score', 'total_points']
    
    for col in numeric_cols:
        if col in exclude_cols:
            continue
        
        try:
            # Correlação com winner
            if y_winner is not None:
                corr_winner = df[col].corr(y_winner)
                if abs(corr_winner) > 0.3:  # Considerar relevante
                    results['with_winner'].append((col, round(corr_winner, 4)))
            
            # Correlação com score_diff
            if score_diff is not None:
                corr_diff = df[col].corr(score_diff)
                if abs(corr_diff) > 0.3:
                    results['with_score_diff'].append((col, round(corr_diff, 4)))
        except Exception:
            continue
    
    # Ordenar por magnitude
    results['with_winner'].sort(key=lambda x: abs(x[1]), reverse=True)
    results['with_score_diff'].sort(key=lambda x: abs(x[1]), reverse=True)
    
    return results


def audit_pace_features(df: pd.DataFrame) -> dict:
    """Verifica presença e qualidade de features de Pace (Ritmo)."""
    pace_patterns = ['pace', 'possessions', 'tempo']
    
    found = []
    for col in df.columns:
        if any(p in col.lower() for p in pace_patterns):
            stats = {
                'column': col,
                'non_null': int(df[col].notna().sum()),
                'mean': round(df[col].mean(), 2) if df[col].dtype in ['float64', 'int64'] else None,
                'std': round(df[col].std(), 2) if df[col].dtype in ['float64', 'int64'] else None
            }
            found.append(stats)
    
    return {
        'found': len(found) > 0,
        'count': len(found),
        'features': found
    }


def audit_rest_features(df: pd.DataFrame) -> dict:
    """Verifica presença e qualidade de features de Rest (Descanso)."""
    rest_patterns = ['rest', 'back_to_back', 'b2b', 'days_off', 'schedule']
    
    found = []
    for col in df.columns:
        if any(p in col.lower() for p in rest_patterns):
            stats = {
                'column': col,
                'non_null': int(df[col].notna().sum()),
                'unique_values': int(df[col].nunique()) if df[col].dtype in ['int64', 'float64', 'object'] else None
            }
            found.append(stats)
    
    return {
        'found': len(found) > 0,
        'count': len(found),
        'features': found
    }


def audit_data_leakage(df: pd.DataFrame) -> dict:
    """Detecta possível data leakage (features que vazam informação do futuro)."""
    
    # Padrões suspeitos
    critical_patterns = ['winner', 'correct', 'prediction', 'prob_home', 'prob_away']
    high_risk_patterns = ['home_score', 'away_score', 'pt_diff', 'total_points', 'opp_pts']
    
    critical = [col for col in df.columns if any(p in col.lower() for p in critical_patterns)]
    high_risk = [col for col in df.columns if any(p in col.lower() for p in high_risk_patterns)]
    
    # Verificar correlações muito altas com target
    y = (df['winner'] == 'HOME').astype(int) if 'winner' in df.columns else None
    suspicious_corr = []
    
    if y is not None:
        for col in df.select_dtypes(include=['number']).columns:
            try:
                corr = abs(df[col].corr(y))
                if corr > 0.8 and col not in critical and col not in high_risk:
                    suspicious_corr.append((col, round(corr, 4)))
            except Exception:
                continue
    
    return {
        'critical_leaks': critical,
        'high_risk': high_risk,
        'suspicious_correlations': sorted(suspicious_corr, key=lambda x: x[1], reverse=True)
    }


def run_full_audit():
    """Executa auditoria completa de features."""
    
    logger.info("=" * 80)
    logger.info("🔍 AUDITORIA COMPLETA DE FEATURES - NBA Predictor")
    logger.info("=" * 80)
    
    # Carregar dados
    logger.info("\n📊 Carregando dados históricos...")
    try:
        df, _ = load_historical_data(
            seasons=['2024-25', '2023-24'],
            apply_weights=True
        )
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados: {e}")
        return
    
    logger.info(f"   Total: {len(df)} jogos, {len(df.columns)} colunas")
    
    # 1. ZEROS EM COLUNAS CRÍTICAS
    logger.info("\n" + "=" * 80)
    logger.info("📋 1. ZEROS EM COLUNAS CRÍTICAS (Injury, Impact, RAPM)")
    logger.info("=" * 80)
    
    zeros_audit = audit_zeros_in_critical_columns(df)
    
    if zeros_audit:
        for col, stats in sorted(zeros_audit.items(), key=lambda x: x[1]['pct_zeros'], reverse=True):
            status = "🔴" if stats['pct_zeros'] > 90 else "🟡" if stats['pct_zeros'] > 50 else "🟢"
            logger.info(f"   {status} {col}: {stats['pct_zeros']:.1f}% zeros, {stats['pct_nans']:.1f}% NaN")
    else:
        logger.info("   ℹ️ Nenhuma coluna crítica encontrada")
    
    # 2. CORRELAÇÕES COM TARGET
    logger.info("\n" + "=" * 80)
    logger.info("🎯 2. CORRELAÇÕES COM TARGETS (winner e score_diff)")
    logger.info("=" * 80)
    
    corr_audit = audit_correlations(df)
    
    logger.info("\n   📌 Correlação com WINNER (|r| > 0.3):")
    for col, corr in corr_audit['with_winner'][:15]:
        status = "🔴" if abs(corr) > 0.8 else "🟠" if abs(corr) > 0.5 else "🟡"
        logger.info(f"   {status} {col}: {corr:+.4f}")
    
    logger.info(f"\n   📌 Correlação com SCORE_DIFF (|r| > 0.3):")
    for col, corr in corr_audit['with_score_diff'][:15]:
        status = "🔴" if abs(corr) > 0.8 else "🟠" if abs(corr) > 0.5 else "🟡"
        logger.info(f"   {status} {col}: {corr:+.4f}")
    
    # 3. FEATURES DE PACE
    logger.info("\n" + "=" * 80)
    logger.info("⚡ 3. FEATURES DE PACE (Ritmo)")
    logger.info("=" * 80)
    
    pace_audit = audit_pace_features(df)
    
    if pace_audit['found']:
        logger.info(f"   ✅ {pace_audit['count']} features de Pace encontradas:")
        for feat in pace_audit['features'][:10]:
            logger.info(f"      • {feat['column']}: mean={feat['mean']}, std={feat['std']}")
    else:
        logger.info("   ❌ NENHUMA feature de Pace encontrada!")
        logger.info("   ℹ️ Recomendação: Adicionar rolling_*_pace, projected_pace_vegas")
    
    # 4. FEATURES DE REST
    logger.info("\n" + "=" * 80)
    logger.info("😴 4. FEATURES DE REST (Descanso)")
    logger.info("=" * 80)
    
    rest_audit = audit_rest_features(df)
    
    if rest_audit['found']:
        logger.info(f"   ✅ {rest_audit['count']} features de Rest encontradas:")
        for feat in rest_audit['features'][:10]:
            logger.info(f"      • {feat['column']}: {feat['non_null']} non-null, {feat['unique_values']} unique")
    else:
        logger.info("   ❌ NENHUMA feature de Rest encontrada!")
        logger.info("   ℹ️ Recomendação: Adicionar home_rest_days, is_back_to_back")
    
    # 5. DATA LEAKAGE
    logger.info("\n" + "=" * 80)
    logger.info("🚨 5. DETECÇÃO DE DATA LEAKAGE")
    logger.info("=" * 80)
    
    leakage_audit = audit_data_leakage(df)
    
    if leakage_audit['critical_leaks']:
        logger.info(f"\n   🔴 CRÍTICO - Features derivadas do resultado:")
        for col in leakage_audit['critical_leaks']:
            logger.info(f"      ❌ {col}")
    
    if leakage_audit['high_risk']:
        logger.info(f"\n   🟠 ALTO RISCO - Stats do próprio jogo:")
        for col in leakage_audit['high_risk'][:10]:
            logger.info(f"      ⚠️ {col}")
    
    if leakage_audit['suspicious_correlations']:
        logger.info(f"\n   🟡 SUSPEITO - Correlação > 0.8 com target:")
        for col, corr in leakage_audit['suspicious_correlations'][:10]:
            logger.info(f"      ⚠️ {col}: {corr:.4f}")
    
    # SUMÁRIO
    logger.info("\n" + "=" * 80)
    logger.info("📊 SUMÁRIO DA AUDITORIA")
    logger.info("=" * 80)
    
    high_zero_cols = [col for col, stats in zeros_audit.items() if stats['pct_zeros'] > 90]
    
    logger.info(f"""
    📈 Total de jogos: {len(df)}
    📊 Total de features: {len(df.columns)}
    
    🏥 Colunas críticas com >90% zeros: {len(high_zero_cols)}
    ⚡ Features de Pace: {'✅ ' + str(pace_audit['count']) if pace_audit['found'] else '❌ 0'}
    😴 Features de Rest: {'✅ ' + str(rest_audit['count']) if rest_audit['found'] else '❌ 0'}
    🚨 Data Leakage crítico: {len(leakage_audit['critical_leaks'])}
    """)
    
    if high_zero_cols:
        logger.info("   ⚠️ AÇÕES RECOMENDADAS:")
        logger.info(f"      1. Verificar por que {len(high_zero_cols)} colunas têm >90% zeros")
        if not pace_audit['found']:
            logger.info("      2. Adicionar features de Pace (ritmo de jogo)")
        if not rest_audit['found']:
            logger.info("      3. Adicionar features de Rest (dias de descanso)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Auditoria concluída!")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_full_audit()
