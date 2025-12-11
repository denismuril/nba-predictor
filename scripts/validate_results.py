#!/usr/bin/env python3
"""
Script de Validação de Resultados
==================================
Compara previsões salvas com resultados reais para calcular accuracy.

Usage:
    python scripts/validate_results.py           # Últimos 30 dias
    python scripts/validate_results.py --days 7  # Últimos 7 dias
    python scripts/validate_results.py --all     # Todos os dados
"""
import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from data.repositories.db_manager import get_db_manager


def get_predictions_with_results(days: int = 30) -> pd.DataFrame:
    """Busca previsões que já têm resultados."""
    db = get_db_manager()
    
    if days > 0:
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        date_filter = f"AND p.date >= '{cutoff}'"
    else:
        date_filter = ""
    
    # Match by date and teams (more reliable than game_id)
    query = f"""
        SELECT 
            p.date,
            p.home_team,
            p.away_team,
            p.prob_home,
            p.prob_away,
            p.prediction,
            p.predicted_spread,
            p.predicted_total,
            g.home_score,
            g.away_score,
            CASE 
                WHEN g.home_score > g.away_score THEN g.home_team
                ELSE g.away_team
            END as actual_winner
        FROM predictions p
        JOIN games g ON p.date::date = g.date::date
            AND p.home_team = g.home_team 
            AND p.away_team = g.away_team
        WHERE g.home_score > 0 
        AND p.prob_home IS NOT NULL
        {date_filter}
        ORDER BY p.date DESC
    """
    
    with db.get_connection() as conn:
        df = pd.read_sql_query(query, conn)
    
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calcula métricas de accuracy."""
    if len(df) == 0:
        return None
    
    # Determinar vencedor previsto (prob_home > 50% = home wins)
    df['predicted_winner'] = df.apply(
        lambda x: x['home_team'] if x['prob_home'] > 50 else x['away_team'],
        axis=1
    )
    
    # Calcular acertos
    df['correct'] = df['predicted_winner'] == df['actual_winner']
    
    # Totals: calcular erro
    if 'predicted_total' in df.columns:
        df['actual_total'] = df['home_score'] + df['away_score']
        df['total_error'] = abs(df['predicted_total'] - df['actual_total'])
    
    # Métricas
    accuracy = df['correct'].mean() * 100
    total_games = len(df)
    correct_games = df['correct'].sum()
    
    # Por confiança
    high_conf = df[df['prob_home'].apply(lambda x: abs(x - 50) > 10)]
    high_conf_acc = high_conf['correct'].mean() * 100 if len(high_conf) > 0 else 0
    
    # Totals MAE
    mae_total = df['total_error'].mean() if 'total_error' in df.columns else None
    
    return {
        'total_games': total_games,
        'correct_games': correct_games,
        'accuracy': accuracy,
        'high_conf_games': len(high_conf),
        'high_conf_accuracy': high_conf_acc,
        'totals_mae': mae_total,
        'df': df
    }


def print_report(metrics: dict, days: int):
    """Imprime relatório de validação."""
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO DE VALIDAÇÃO DE RESULTADOS")
    print("=" * 60)
    
    if metrics is None:
        print("❌ Nenhuma previsão com resultado encontrada.")
        return
    
    period = f"últimos {days} dias" if days > 0 else "todos os dados"
    print(f"\n📅 Período: {period}")
    print(f"📈 Total de jogos: {metrics['total_games']}")
    print(f"✅ Acertos: {metrics['correct_games']}")
    print(f"❌ Erros: {metrics['total_games'] - metrics['correct_games']}")
    
    print("\n" + "-" * 40)
    print("🎯 ACCURACY MONEYLINE")
    print("-" * 40)
    print(f"   Geral: {metrics['accuracy']:.1f}%")
    print(f"   Alta Confiança (>60%): {metrics['high_conf_accuracy']:.1f}% ({metrics['high_conf_games']} jogos)")
    
    if metrics['totals_mae']:
        print("\n" + "-" * 40)
        print("🔢 TOTALS")
        print("-" * 40)
        print(f"   MAE: {metrics['totals_mae']:.1f} pontos")
    
    # Últimos 10 jogos
    df = metrics['df']
    print("\n" + "-" * 40)
    print("📋 ÚLTIMOS 10 JOGOS")
    print("-" * 40)
    
    for _, row in df.head(10).iterrows():
        status = "✅" if row['correct'] else "❌"
        prob = max(row['prob_home'], row['prob_away'])
        pred = row['predicted_winner']
        actual = row['actual_winner']
        score = f"{row['home_score']}-{row['away_score']}"
        print(f"   {status} {row['home_team']} vs {row['away_team']}: "
              f"Pred={pred} ({prob:.0f}%) | Real={actual} ({score})")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Validar previsões vs resultados')
    parser.add_argument('--days', type=int, default=30, help='Dias para análise (default: 30)')
    parser.add_argument('--all', action='store_true', help='Analisar todos os dados')
    parser.add_argument('--csv', type=str, help='Exportar para CSV')
    args = parser.parse_args()
    
    days = 0 if args.all else args.days
    
    print("🔍 Buscando previsões com resultados...")
    df = get_predictions_with_results(days)
    
    print(f"   Encontrados: {len(df)} jogos")
    
    metrics = calculate_metrics(df)
    print_report(metrics, days)
    
    if args.csv and metrics:
        output_path = Path(args.csv)
        metrics['df'].to_csv(output_path, index=False)
        print(f"\n💾 Exportado para: {output_path}")


if __name__ == "__main__":
    main()
