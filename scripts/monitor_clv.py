"""
Monitor de CLV (Closing Line Value)
====================================

Valida a qualidade das apostas comparando odds apostadas vs odds de fechamento.

CLV (Closing Line Value):
    Métrica fundamental para apostadores profissionais.
    CLV = (Odd Apostada / Odd Fechamento) - 1
    
    CLV Positivo: Conseguimos odds melhores que o mercado final (bom!)
    CLV Negativo: Linha andou contra nós (mal sinal)

Uso:
    # Rodar análise dos últimos 30 dias
    python scripts/monitor_clv.py --days 30
    
    # Rodar análise de todas as apostas
    python scripts/monitor_clv.py --all
    
    # Exportar relatório para CSV
    python scripts/monitor_clv.py --days 30 --export reports/clv_report.csv

Autor: NBA Predictor Team
Data: 2025-12-06
"""

import sys
from pathlib import Path
import pandas as pd
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from market.odds_shopping import fetch_multi_bookie_odds, API_TO_INTERNAL_MAP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BET_TRACKING_FILE = Path(__file__).parent.parent / 'data' / 'bet_tracking.csv'


def load_bet_history(days: int = 30) -> pd.DataFrame:
    """
    Carrega histórico de apostas do CSV.
    
    Args:
        days: Número de dias para analisar (0 = todas as apostas)
    
    Returns:
        DataFrame com apostas
    """
    if not BET_TRACKING_FILE.exists():
        logger.warning(f"❌ Arquivo não encontrado: {BET_TRACKING_FILE}")
        logger.info("💡 Criando estrutura CSV vazia...")
        
        # Criar diretório se não existir
        BET_TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Criar CSV vazio com headers
        df_empty = pd.DataFrame(columns=[
            'date', 'game_id', 'home_team', 'away_team', 'bet_team',
            'market', 'odds_taken', 'odds_closing', 'stake_pct', 'stake_amount',
            'model_prob', 'ev', 'result', 'profit', 'clv'
        ])
        df_empty.to_csv(BET_TRACKING_FILE, index=False)
        return df_empty
    
    df = pd.read_csv(BET_TRACKING_FILE)
    
    # Filtrar por data se necessário
    if days > 0 and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['date'] >= cutoff_date]
    
    logger.info(f"📊 Carregadas {len(df)} apostas do histórico")
    return df


def fetch_closing_odds(game_id: str, team: str, market: str) -> Optional[float]:
    """
    Busca odds de fechamento para uma aposta.
    
    NOTA: Esta é uma implementação mock. Na produção, você precisaria:
        1. Armazenar odds de fechamento em banco de dados
        2. Ou usar API histórica (ex: Pinnacle closing lines)
    
    Args:
        game_id: ID do jogo (ex: "LAL_vs_BRK")
        team: Time apostado (ex: "LAL")
        market: Mercado (ex: "Moneyline")
    
    Returns:
        Odd de fechamento ou None se não disponível
    """
    # MOCK: Para demonstração, retornar None
    # Na produção, substituir por consulta real
    logger.debug(f"   Buscando odd de fechamento para {team} {market} em {game_id}")
    return None


def calculate_clv(odds_taken: float, odds_closing: float) -> float:
    """
    Calcula Closing Line Value (CLV).
    
    Args:
        odds_taken: Odd na qual apostamos
        odds_closing: Odd de fechamento
    
    Returns:
        CLV em decimal (ex: 0.05 = +5% CLV)
    """
    if odds_closing <= 0:
        return 0.0
    
    clv = (odds_taken / odds_closing) - 1
    return clv


def generate_clv_report(df_bets: pd.DataFrame) -> Dict:
    """
    Gera relatório de CLV.
    
    Args:
        df_bets: DataFrame com apostas (deve ter coluna 'clv')
    
    Returns:
        Dict com métricas do relatório
    """
    if df_bets.empty:
        return {
            'total_bets': 0,
            'avg_clv': 0.0,
            'positive_clv_count': 0,
            'negative_clv_count': 0,
            'positive_clv_pct': 0.0,
            'worst_bets': []
        }
    
    # Filtrar apenas apostas com CLV calculado
    df_with_clv = df_bets[df_bets['clv'].notna()].copy()
    
    if df_with_clv.empty:
        logger.warning("⚠️ Nenhuma aposta possui CLV calculado (odds de fechamento ausentes)")
        return {
            'total_bets': len(df_bets),
            'avg_clv': 0.0,
            'positive_clv_count': 0,
            'negative_clv_count': 0,
            'positive_clv_pct': 0.0,
            'worst_bets': []
        }
    
    total_bets = len(df_with_clv)
    avg_clv = df_with_clv['clv'].mean()
    positive_clv = df_with_clv[df_with_clv['clv'] > 0]
    negative_clv = df_with_clv[df_with_clv['clv'] < 0]
    
    positive_clv_count = len(positive_clv)
    negative_clv_count = len(negative_clv)
    positive_clv_pct = (positive_clv_count / total_bets * 100) if total_bets > 0 else 0
    
    # Top 5 piores apostas (maior CLV negativo)
    worst_bets = df_with_clv.nsmallest(5, 'clv')[
        ['date', 'bet_team', 'market', 'odds_taken', 'odds_closing', 'clv']
    ].to_dict('records')
    
    return {
        'total_bets': total_bets,
        'avg_clv': avg_clv,
        'positive_clv_count': positive_clv_count,
        'negative_clv_count': negative_clv_count,
        'positive_clv_pct': positive_clv_pct,
        'worst_bets': worst_bets
    }


def print_clv_report(report: Dict, days: int = 30):
    """
    Imprime relatório de CLV no console.
    
    Args:
        report: Dict retornado por generate_clv_report()
        days: Número de dias analisados
    """
    print("\n" + "=" * 70)
    print(f"📊 RELATÓRIO CLV - Últimos {days if days > 0 else 'Todos os'} Dias")
    print("=" * 70)
    print(f"Total de Apostas: {report['total_bets']}")
    
    if report['total_bets'] == 0:
        print("⚠️ Nenhuma aposta encontrada no período")
        print("=" * 70)
        return
    
    print(f"CLV Médio: {report['avg_clv']:+.2%}")
    print(f"CLV Positivo: {report['positive_clv_count']} ({report['positive_clv_pct']:.1f}%)")
    print(f"CLV Negativo: {report['negative_clv_count']} ({100 - report['positive_clv_pct']:.1f}%)")
    
    # Interpretação
    if report['avg_clv'] > 0.02:
        print("\n✅ EXCELENTE: CLV médio positivo indica que estamos batendo o mercado!")
    elif report['avg_clv'] > 0:
        print("\n✅ BOM: CLV médio levemente positivo. Continue assim!")
    elif report['avg_clv'] > -0.02:
        print("\n⚠️ NEUTRO: CLV próximo de zero. Modelo está na média do mercado.")
    else:
        print("\n🚨 ATENÇÃO: CLV negativo indica que o mercado fecha contra nós.")
        print("   Considere revisar seu modelo ou estratégia de entrada.")
    
    # Piores apostas
    if report['worst_bets']:
        print("\n🚨 Top 5 Piores Apostas (Linha Contra Nós):")
        print("-" * 70)
        for bet in report['worst_bets']:
            print(f"   {bet['date']} | {bet['bet_team']} {bet['market']}")
            print(f"      Odd Apostada: {bet['odds_taken']:.2f} → Fechou: {bet['odds_closing']:.2f} | CLV: {bet['clv']:+.2%}")
    
    print("=" * 70)


def update_clv_for_bets(df_bets: pd.DataFrame) -> pd.DataFrame:
    """
    Atualiza coluna CLV para apostas que não têm odds de fechamento.
    
    NOTA: Esta implementação é MOCK. Na produção, substituir por:
        - Consulta a API histórica de odds
        - Banco de dados com odds de fechamento armazenadas
    
    Args:
        df_bets: DataFrame com apostas
    
    Returns:
        DataFrame atualizado com CLVs
    """
    df = df_bets.copy()
    
    # Garantir que coluna CLV existe
    if 'clv' not in df.columns:
        df['clv'] = None
    
    if 'odds_closing' not in df.columns:
        df['odds_closing'] = None
    
    for idx, row in df.iterrows():
        # Se já tem CLV, pular
        if pd.notna(row.get('clv')):
            continue
        
        # Se já tem odd de fechamento, calcular
        if pd.notna(row.get('odds_closing')) and row['odds_closing'] > 0:
            df.at[idx, 'clv'] = calculate_clv(row['odds_taken'], row['odds_closing'])
            continue
        
        # Tentar buscar odd de fechamento (MOCK - sempre retorna None)
        closing_odd = fetch_closing_odds(
            game_id=row.get('game_id', ''),
            team=row.get('bet_team', ''),
            market=row.get('market', '')
        )
        
        if closing_odd:
            df.at[idx, 'odds_closing'] = closing_odd
            df.at[idx, 'clv'] = calculate_clv(row['odds_taken'], closing_odd)
    
    return df


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Monitor de CLV (Closing Line Value)')
    parser.add_argument('--days', type=int, default=30, help='Dias para analisar (default: 30)')
    parser.add_argument('--all', action='store_true', help='Analisar todas as apostas')
    parser.add_argument('--export', type=str, help='Exportar relatório para CSV')
    args = parser.parse_args()
    
    days = 0 if args.all else args.days
    
    logger.info("🔍 Carregando histórico de apostas...")
    df_bets = load_bet_history(days=days)
    
    if df_bets.empty:
        logger.warning("⚠️ Nenhuma aposta encontrada. Use o sistema para fazer apostas primeiro.")
        return
    
    logger.info("📡 Atualizando CLVs (consultando odds de fechamento)...")
    df_bets = update_clv_for_bets(df_bets)
    
    # Salvar atualizações
    df_bets.to_csv(BET_TRACKING_FILE, index=False)
    logger.info(f"💾 Apostas atualizadas salvas em: {BET_TRACKING_FILE}")
    
    # Gerar relatório
    logger.info("📊 Gerando relatório CLV...")
    report = generate_clv_report(df_bets)
    
    # Imprimir relatório
    print_clv_report(report, days=days)
    
    # Exportar se solicitado
    if args.export:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        df_bets.to_csv(export_path, index=False)
        logger.info(f"\n💾 Relatório exportado para: {export_path}")


if __name__ == '__main__':
    main()
