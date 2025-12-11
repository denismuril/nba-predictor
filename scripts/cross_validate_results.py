"""
Script para validar resultados de jogos usando múltiplas fontes de dados.

FONTES DE VALIDAÇÃO:
1. results_scraper (ESPN/Basketball Reference) - fonte atual
2. SportsBlaze API - fonte adicional para cross-validation
3. NBA Official API - fallback

OBJETIVO:
- Garantir que resultados estejam corretos
- Cross-validation entre múltiplas fontes
- Identificar discrepâncias nos dados
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime, timedelta
from data.repositories.db_manager import DatabaseManager
from data.scrapers.results_scraper import get_game_results
from scripts.sportsblaze_integration import SportsBlazeClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_team_name(name):
    """Normaliza nome de time para comparação"""
    return name.upper().replace(" ", "").replace("-", "").replace(".", "")

def validate_with_sportsblaze(date_str, days_back=7):
    """
    Valida resultados usando SportsBlaze API.
    
    Args:
        date_str: Data no formato 'YYYY-MM-DD'
        days_back: Quantos dias para trás buscar
    
    Returns:
        Dict com resultados do SportsBlaze
    """
    logger.info(f"🔍 Validando com SportsBlaze API (últimos {days_back} dias)...")
    
    client = SportsBlazeClient()
    
    try:
        # Buscar jogos da data
        data = client.get_nba_boxscores(date_str)
        
        if not data or 'games' not in data:
            logger.warning(f"⚠️  SportsBlaze: Nenhum jogo encontrado para {date_str}")
            return {}
        
        games = data['games']
        logger.info(f"✅ SportsBlaze: {len(games)} jogos encontrados")
        
        # Normalizar formato
        results = {}
        for game in games:
            home_team = game.get('home_team', {}).get('name', '')
            away_team = game.get('away_team', {}).get('name', '')
            
            if home_team and away_team:
                key = f"{normalize_team_name(home_team)}_vs_{normalize_team_name(away_team)}"
                results[key] = {
                    'date': date_str,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': game.get('home_team', {}).get('score', 0),
                    'away_score': game.get('away_team', {}).get('score', 0),
                    'source': 'sportsblaze'
                }
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro ao buscar do SportsBlaze: {e}")
        return {}

def cross_validate_results(days_back=3):
    """
    Cross-valida resultados entre múltiplas fontes.
    
    Args:
        days_back: Quantos dias para trás validar
    """
    logger.info("=" * 80)
    logger.info("🔍 VALIDAÇÃO CRUZADA DE RESULTADOS")
    logger.info("=" * 80)
    print()
    
    # Fonte 1: Results Scraper (atual)
    logger.info("📊 FONTE 1: Results Scraper (ESPN/Basketball Reference)")
    logger.info("-" * 80)
    
    results_scraper = get_game_results(days_back=days_back)
    logger.info(f"✅ {len(results_scraper)} jogos encontrados")
    print()
    
    # Fonte 2: SportsBlaze API
    logger.info("📊 FONTE 2: SportsBlaze API")
    logger.info("-" * 80)
    
    # Buscar últimos dias
    sportsblaze_results = {}
    for i in range(days_back):
        date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
        sb_data = validate_with_sportsblaze(date)
        sportsblaze_results.update(sb_data)
    
    logger.info(f"✅ {len(sportsblaze_results)} jogos encontrados")
    print()
    
    # Cross-validation
    logger.info("=" * 80)
    logger.info("🔄 CROSS-VALIDATION")
    logger.info("=" * 80)
    print()
    
    matches = 0
    discrepancies = 0
    only_in_scraper = 0
    only_in_sportsblaze = 0
    
    # Normalizar results_scraper para comparação
    scraper_dict = {}
    for game in results_scraper:
        home_norm = normalize_team_name(game['home_team'])
        away_norm = normalize_team_name(game['away_team'])
        key = f"{home_norm}_vs_{away_norm}"
        scraper_dict[key] = game
    
    # Comparar
    all_keys = set(scraper_dict.keys()) | set(sportsblaze_results.keys())
    
    for key in all_keys:
        in_scraper = key in scraper_dict
        in_sportsblaze = key in sportsblaze_results
        
        if in_scraper and in_sportsblaze:
            # Comparar placares
            scraper_game = scraper_dict[key]
            sb_game = sportsblaze_results[key]
            
            if (scraper_game['home_score'] == sb_game['home_score'] and 
                scraper_game['away_score'] == sb_game['away_score']):
                matches += 1
                logger.debug(f"✅ Match: {key}")
            else:
                discrepancies += 1
                logger.warning(
                    f"⚠️  DISCREPÂNCIA: {key}\\n"
                    f"   Scraper: {scraper_game['home_score']}-{scraper_game['away_score']}\\n"
                    f"   SportsBlaze: {sb_game['home_score']}-{sb_game['away_score']}"
                )
        
        elif in_scraper:
            only_in_scraper += 1
            logger.debug(f"📌 Apenas em Scraper: {key}")
        
        else:  # only in sportsblaze
            only_in_sportsblaze += 1
            logger.debug(f"📌 Apenas em SportsBlaze: {key}")
    
    # Resumo
    print()
    logger.info("=" * 80)
    logger.info("📊 RESUMO DA VALIDAÇÃO")
    logger.info("=" * 80)
    print()
    
    total_compared = matches + discrepancies
    match_rate = (matches / total_compared * 100) if total_compared > 0 else 0
    
    logger.info(f"✅ Matches (placares idênticos): {matches}")
    logger.info(f"⚠️  Discrepâncias (placares diferentes): {discrepancies}")
    logger.info(f"📌 Apenas em Results Scraper: {only_in_scraper}")
    logger.info(f"📌 Apenas em SportsBlaze: {only_in_sportsblaze}")
    print()
    logger.info(f"🎯 Taxa de Concordância: {match_rate:.1f}%")
    print()
    
    if match_rate >= 95:
        logger.info("✅ VALIDAÇÃO APROVADA - Dados consistentes entre fontes!")
    elif match_rate >= 80:
        logger.warning("⚠️  VALIDAÇÃO PARCIAL - Algumas discrepâncias encontradas")
    else:
        logger.error("❌ VALIDAÇÃO FALHADA - Muitas discrepâncias detectadas!")
    
    print()
    logger.info("=" * 80)
    logger.info("💡 RECOMENDAÇÕES")
    logger.info("=" * 80)
    print()
    
    if discrepancies > 0:
        logger.info("1. Revisar discrepâncias manualmente")
        logger.info("2. Verificar se jogos foram adiados/reprogramados")
        logger.info("3. Confirmar placares finais em fonte oficial (NBA.com)")
    
    if only_in_scraper > only_in_sportsblaze:
        logger.info("4. Results Scraper pode estar mais atualizado")
    elif only_in_sportsblaze > only_in_scraper:
        logger.info("4. SportsBlaze API pode estar mais completo")
    
    logger.info("5. Usar TeamID matching previne falsos positivos (Lakers ≠ Clippers)")
    print()
    
    return {
        'matches': matches,
        'discrepancies': discrepancies,
        'match_rate': match_rate,
        'only_scraper': only_in_scraper,
        'only_sportsblaze': only_in_sportsblaze
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validação cruzada de resultados')
    parser.add_argument('--days', type=int, default=3, help='Dias para validar (default: 3)')
    
    args = parser.parse_args()
    
    stats = cross_validate_results(days_back=args.days)
    
    # Exit code baseado em taxa de match
    if stats['match_rate'] >= 95:
        return 0
    elif stats['match_rate'] >= 80:
        return 1
    else:
        return 2

if __name__ == "__main__":
    sys.exit(main())
