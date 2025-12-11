"""
Dynamic Home Court Advantage Calculator

Calcula HCA dinâmico baseado em:
1. Arena-specific multipliers (altitude, crowd, etc)
2. Recent home performance do time
3. Season adjustments

Referências:
- Historical NBA HCA data (2019-2025)
- Altitude studies (Denver, Utah)
- Crowd impact research (Portland, Miami, Boston)
"""
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from config.arena_constants import (
        ARENA_HCA_MULTIPLIERS,
        BASE_HCA,
        CURRENT_SEASON
    )
except ImportError:
    logger.warning("arena_constants not found, using defaults")
    ARENA_HCA_MULTIPLIERS = {}
    BASE_HCA = 2.8


def calculate_dynamic_hca(
    home_team: str,
    date: Optional[pd.Timestamp] = None,
    recent_home_games: Optional[pd.DataFrame] = None,
    base_hca: float = BASE_HCA
) -> float:
    """
    Calcula Home Court Advantage dinâmico para um time.
    
    Formula:
        HCA = Base_League_HCA × Arena_Multiplier × Recency_Weight
    
    Args:
        home_team: Código do time (3 letras, ex: 'LAL', 'BOS')
        date: Data do jogo (opcional, para ajustes sazonais)
        recent_home_games: DataFrame com últimos jogos em casa (opcional)
        base_hca: HCA base da liga (default: 2.8 pts)
    
    Returns:
        HCA ajustado em pontos
    
    Examples:
        >>> # Denver com altitude
        >>> hca_den = calculate_dynamic_hca('DEN')
        >>> print(f"{hca_den:.2f} pts")  # ~3.78 pts (2.8 × 1.35)
        
        >>> # Lakers (muitos torcedores visitantes)
        >>> hca_lal = calculate_dynamic_hca('LAL')
        >>> print(f"{hca_lal:.2f} pts")  # ~2.58 pts (2.8 × 0.92)
    """
    # 1. Arena Multiplier
    arena_multiplier = ARENA_HCA_MULTIPLIERS.get(home_team, 1.0)
    
    # 2. Recency Adjustment (baseado em performance recente em casa)
    recency_weight = 1.0  # default (neutro)
    
    if recent_home_games is not None and not recent_home_games.empty:
        try:
            # Calcular win% em casa nos últimos N jogos
            home_win_pct = recent_home_games['won'].mean()
            
            # Ajuste: 0.8x a 1.2x baseado em performance
            # 0% wins → 0.8x
            # 50% wins → 1.0x
            # 100% wins → 1.2x
            recency_weight = 0.8 + (0.4 * home_win_pct)
            
            logger.debug(
                f"   {home_team} recent home win%: {home_win_pct:.1%} "
                f"→ recency weight: {recency_weight:.2f}x"
            )
        except Exception as e:
            logger.debug(f"   Could not calculate recency for {home_team}: {e}")
            recency_weight = 1.0
    
    # 3. Final HCA
    hca_final = base_hca * arena_multiplier * recency_weight
    
    # AUDIT FIX #5: Cap de 5.0 pts para permitir Denver+altitude+crowd+hot streak
    hca_final = max(0.5, min(5.0, hca_final))
    
    logger.debug(
        f"   🏠 HCA {home_team}: {base_hca:.2f} × {arena_multiplier:.2f} (arena) "
        f"× {recency_weight:.2f} (recency) = {hca_final:.2f} pts"
    )
    
    return hca_final


def get_recent_home_performance(
    team: str,
    df_games: pd.DataFrame,
    n_games: int = 10
) -> Optional[pd.DataFrame]:
    """
    Retorna últimos N jogos em casa de um time.
    
    Args:
        team: Código do time
        df_games: DataFrame com histórico de jogos
        n_games: Número de jogos recentes (default: 10)
    
    Returns:
        DataFrame com jogos em casa ou None
    """
    try:
        # Filtrar jogos em casa
        home_games = df_games[df_games['home_team'] == team].copy()
        
        # Ordenar por data (mais recente primeiro)
        home_games = home_games.sort_values('date', ascending=False)
        
        # Pegar últimos N
        recent = home_games.head(n_games).copy()
        
        # Adicionar coluna 'won' se não existir
        if 'won' not in recent.columns:
            if 'winner' in recent.columns:
                recent['won'] = (recent['winner'] == 'HOME').astype(int)
            elif 'home_score' in recent.columns and 'away_score' in recent.columns:
                recent['won'] = (recent['home_score'] > recent['away_score']).astype(int)
            else:
                logger.warning(f"Cannot determine wins for {team}")
                return None
        
        return recent
        
    except Exception as e:
        logger.error(f"Error getting recent home performance for {team}: {e}")
        return None


def hca_sanity_check():
    """
    Valida que todos os multiplicadores estão em range razoável.
    """
    issues = []
    
    for team, multiplier in ARENA_HCA_MULTIPLIERS.items():
        if multiplier < 0.7 or multiplier > 1.5:
            issues.append(f"{team}: {multiplier:.2f}x (fora do range 0.7-1.5)")
    
    if issues:
        logger.warning(f"⚠️ Arena multipliers com valores suspeitos:\n" + "\n".join(issues))
    else:
        logger.info(f"✅ {len(ARENA_HCA_MULTIPLIERS)} arena multipliers validados (range: 0.7-1.5x)")
    
    return len(issues) == 0


if __name__ == '__main__':
    # Demo
    logging.basicConfig(level=logging.DEBUG)
    
    print("🏀 Dynamic HCA Calculator Demo\n")
    
    # Teste básico
    teams_to_test = ['DEN', 'UTA', 'POR', 'LAL', 'BOS', 'BRK']
    
    print("HCA sem recency adjustment:")
    for team in teams_to_test:
        hca = calculate_dynamic_hca(team)
        multiplier = ARENA_HCA_MULTIPLIERS.get(team, 1.0)
        print(f"  {team}: {hca:.2f} pts (multiplier: {multiplier:.2f}x)")
    
    print("\n✅ Sanity check:")
    hca_sanity_check()
    
    # Simulate recent performance
    print("\n🔄 Com recency adjustment (simulado):")
    
    # Simular jogos recentes
    sample_games = pd.DataFrame({
        'date': pd.date_range('2025-11-01', periods=15),
        'home_team': ['DEN'] * 15,
        'away_team': ['Various'],
        'won': [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1]  # 11-4 record
    })
    
    recent = get_recent_home_performance('DEN', sample_games, n_games=10)
    hca_den_adjusted = calculate_dynamic_hca('DEN', recent_home_games=recent)
    
    print(f"  DEN with 70% home win rate: {hca_den_adjusted:.2f} pts")
    print(f"  (Base: {BASE_HCA:.2f} × Arena: {ARENA_HCA_MULTIPLIERS.get('DEN', 1.0):.2f} × Recency: ~1.08)")
    
    print("\n✅ Demo completo!")
