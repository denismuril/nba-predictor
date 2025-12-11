"""
Script utilitário para identificar e corrigir jogos com estatísticas zeradas no banco de dados.

Este script:
1. Identifica jogos passados com placar zerado (pts=0 ou fga=0)
2. Força atualização desses jogos baixando os dados novamente
3. Permite execução idempotente (seguro rodar múltiplas vezes)

Uso:
    python scripts/fix_zeros.py
"""

import pandas as pd
from datetime import datetime
import logging
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.repositories.db_manager import get_db_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Definir temporada atual
SEASON_START_DATE = '2025-10-01'


def find_zero_stats_games():
    """
    Identifica jogos passados com estatísticas zeradas ou inválidas.
    
    Returns:
        pd.DataFrame: DataFrame com jogos problemáticos
    """
    db = get_db_manager()
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Query para encontrar jogos problemáticos
    # Jogos do passado que têm pts=0 ou fg a=0 (indicando falta de stats)
    query = f"""
    SELECT DISTINCT
        g.date,
        g.home_team,
        g.away_team,
        sh.pts as home_pts,
        sh.fga as home_fga,
        sa.pts as away_pts,
        sa.fga as away_fga
    FROM games g
    LEFT JOIN game_stats sh ON g.game_id = sh.game_id AND sh.is_home = TRUE
    LEFT JOIN game_stats sa ON g.game_id = sa.game_id AND sa.is_home = FALSE
    WHERE g.date < '{today}'
      AND g.date >= '{SEASON_START_DATE}'
      AND (sh.pts = 0 OR sh.fga = 0 OR sa.pts = 0 OR sa.fga = 0
           OR sh.pts IS NULL OR sa.pts IS NULL)
    ORDER BY g.date DESC
    """
    
    try:
        with db.get_connection() as conn:
            df_broken = pd.read_sql_query(query, conn)
        
        return df_broken
    except Exception as e:
        logger.error(f"❌ Erro ao consultar banco de dados: {e}")
        return pd.DataFrame()


def display_problematic_games(df_broken):
    """
    Exibe jogos problemáticos para o usuário.
    
    Args:
        df_broken: DataFrame com jogos com stats zeradas
    """
    if df_broken.empty:
        logger.info("✅ Nenhum jogo com stats zerados encontrado na temporada atual.")
        return
    
    logger.info(f"\n⚠️  Encontrados {len(df_broken)} jogos com stats zeradas/inválidas:")
    logger.info("=" * 80)
    
    # Mostrar primeiros 10 jogos
    display_df = df_broken.head(10)
    for idx, row in display_df.iterrows():
        logger.info(
            f"  {row['date']} | {row['home_team']} vs {row['away_team']} | "
            f"PTS: {row['home_pts']}-{row['away_pts']} | "
            f"FGA: {row['home_fga']}-{row['away_fga']}"
        )
    
    if len(df_broken) > 10:
        logger.info(f"  ... e mais {len(df_broken) - 10} jogos")
    
    logger.info("=" * 80)


def force_rescrape_dates(dates):
    """
    Força re-scraping de datas específicas usando results_scraper.
    
    Args:
        dates: Lista de datas para re-scrapear
    
    Returns:
        dict: Estatísticas de sucesso/falha
    """
    from data.scrapers.results_scraper import get_game_results
    from data.repositories.db_manager import get_db_manager
    
    logger.info(f"\n🔄 Iniciando re-scraping de {len(dates)} datas...")
    logger.info("=" * 80)
    
    db = get_db_manager()
    stats = {
        'total_dates': len(dates),
        'successful_dates': 0,
        'failed_dates': 0,
        'games_updated': 0,
        'errors': []
    }
    
    # Agrupar datas para processar em batch
    # Calcular days_back a partir da data mais antiga
    today = datetime.now()
    dates_sorted = sorted([datetime.strptime(str(d), '%Y-%m-%d') for d in dates])
    
    if not dates_sorted:
        logger.error("❌ Nenhuma data válida para processar")
        return stats
    
    oldest_date = dates_sorted[0]
    days_back = (today - oldest_date).days + 1
    
    logger.info(f"📅 Período de busca: últimos {days_back} dias")
    logger.info(f"   Data mais antiga: {oldest_date.date()}")
    logger.info(f"   Até hoje: {today.date()}")
    
    try:
        # Buscar todos os resultados do período
        logger.info(f"\n🔍 Buscando resultados via ESPN API...")
        all_results = get_game_results(days_back=days_back)
        
        if not all_results:
            logger.warning("⚠️  Nenhum resultado encontrado na API")
            stats['failed_dates'] = len(dates)
            return stats
        
        logger.info(f"✅ API retornou {len(all_results)} jogos")
        
        # Filtrar apenas jogos das datas problemáticas
        dates_str = set([str(d) for d in dates])
        relevant_games = [g for g in all_results if g['date'] in dates_str]
        
        logger.info(f"🎯 {len(relevant_games)} jogos correspondem às datas problemáticas")
        
        # Atualizar cada jogo no banco
        logger.info(f"\n💾 Atualizando banco de dados...")
        for game in relevant_games:
            try:
                db.update_game_score(
                    game_id=game['id'],
                    home_score=game['home_score'],
                    away_score=game['away_score']
                )
                stats['games_updated'] += 1
                logger.info(
                    f"   ✅ {game['date']}: {game['away_team']} @ {game['home_team']} "
                    f"= {game['away_score']}-{game['home_score']}"
                )
            except Exception as e:
                error_msg = f"Erro ao atualizar {game['id']}: {e}"
                logger.error(f"   ❌ {error_msg}")
                stats['errors'].append(error_msg)
        
        stats['successful_dates'] = len(set([g['date'] for g in relevant_games]))
        stats['failed_dates'] = len(dates) - stats['successful_dates']
        
    except Exception as e:
        logger.error(f"❌ Erro durante re-scraping: {e}")
        import traceback
        traceback.print_exc()
        stats['failed_dates'] = len(dates)
        stats['errors'].append(f"Erro geral: {e}")
    
    # Relatório final
    logger.info("\n" + "=" * 80)
    logger.info("📊 RELATÓRIO FINAL:")
    logger.info(f"   Datas processadas: {stats['successful_dates']}/{stats['total_dates']}")
    logger.info(f"   Jogos atualizados: {stats['games_updated']}")
    logger.info(f"   Erros: {len(stats['errors'])}")
    
    if stats['errors']:
        logger.warning(f"\n⚠️  {len(stats['errors'])} erros durante o processo:")
        for error in stats['errors'][:5]:
            logger.warning(f"   - {error}")
        if len(stats['errors']) > 5:
            logger.warning(f"   ... e mais {len(stats['errors']) - 5} erros")
    
    logger.info("=" * 80)
    
    return stats


def main():
    """
    Função principal do script.
    """
    logger.info("🛠️  Iniciando Script de Reparo de Dados Zerados")
    logger.info(f"📅 Temporada Atual: {SEASON_START_DATE} em diante")
    logger.info("=" * 80)
    
    # 1. Identificar jogos problemáticos
    df_broken = find_zero_stats_games()
    
    # 2. Exibir para o usuário
    display_problematic_games(df_broken)
    
    if df_broken.empty:
        logger.info("\n✅ Nenhuma ação necessária. Banco de dados está OK!")
        return
    
    # 3. Preparar lista de datas únicas
    dates_to_scrape = sorted(df_broken['date'].unique())
    logger.info(f"\n📋 {len(dates_to_scrape)} datas únicas precisam ser re-scrapeadas:")
    for date in dates_to_scrape[:10]:
        logger.info(f"  - {date}")
    if len(dates_to_scrape) > 10:
        logger.info(f"  ... e mais {len(dates_to_scrape) - 10} datas")
    
    # 4. Confirmar com usuário
    response = input("\n🤔 Deseja prosseguir com o re-scraping? (s/n): ").lower().strip()
    
    if response != 's':
        logger.info("❌ Operação cancelada pelo usuário.")
        return
    
    # 5. Executar re-scraping
    force_rescrape_dates(dates_to_scrape)
    
    logger.info("\n✅ Script concluído!")


if __name__ == "__main__":
    main()
