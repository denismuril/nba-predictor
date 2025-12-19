import numpy as np
import logging
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from config.constants import (
    TEAM_ABBREV_MAP,  # Legacy - será removido gradualmente
    ALL_STARS_2025, 
    RATING_WEIGHTS, 
    NORMALIZATION_LIMITS, 
    REFEREE_ADJUSTMENTS, 
    HCA_VALUE  # Legacy - ainda usado como fallback
)
from utils.team_normalization import normalize_team  # NEW: Centralized normalization
from core.dynamic_hca import calculate_dynamic_hca  # NEW: Dynamic HCA calculator

# Importar logger configurado
try:
    from utils.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback se logger_config não estiver disponível
    logger = logging.getLogger(__name__)

from core.referee_cache import get_referee_stats

def normalizar_metrica(valor: float, min_val: float, max_val: float) -> float:
    """
    Normaliza valor para 0-100.
    
    Args:
        valor: Valor a normalizar.
        min_val: Valor mínimo da escala original.
        max_val: Valor máximo da escala original.
        
    Returns:
        Valor normalizado entre 0 e 100.
    """
    if max_val == min_val:
        return 50.0 # Evitar divisão por zero, retorna valor neutro
        
    norm = (valor - min_val) / (max_val - min_val) * 100
    return max(0.0, min(100.0, norm))

def calcular_net_rating_v11(team_name: str, dfs: Dict[str, pd.DataFrame]) -> float:
    """
    Calcula Net Rating usando média ponderada (v10.1).
    
    Args:
        team_name: Nome do time.
        dfs: Dicionário com DataFrames de estatísticas (RAPM, LEBRON, BPM, PIE).
        
    Returns:
        Net Rating calculado (escala -10 a +10). Retorna 0.0 se falhar.
    """
    
    metricas_coletadas = {}
    
    team_abbrev = TEAM_ABBREV_MAP.get(team_name, "")
    logger.debug(f"      🔍 Calculando Net Rating: {team_name} (abbrev: {team_abbrev})")
    
    norm_min = NORMALIZATION_LIMITS['min']
    norm_max = NORMALIZATION_LIMITS['max']
    
    # Identificar chave correta para RAPM (case-insensitive)
    rapm_key = next((k for k in dfs.keys() if k.upper() == 'RAPM'), None)
    
    # RAPM
    if rapm_key and not dfs[rapm_key].empty:
        try:
            # Usar normalização centralizada para buscar time no DataFrame
            # O DataFrame de stats deve ter nomes normalizados ou usar IDs
            
            df_rapm = pd.DataFrame()
            
            if 'Team' in dfs[rapm_key].columns:
                # Normalizar o nome do time alvo
                target_team_norm = normalize_team(team_name)
                
                # Criar coluna temporária normalizada no DF para busca
                dfs[rapm_key]['_team_norm'] = dfs[rapm_key]['Team'].apply(normalize_team)
                
                df_rapm = dfs[rapm_key][dfs[rapm_key]['_team_norm'] == target_team_norm]
                
                # Limpar coluna temporária
                dfs[rapm_key].drop(columns=['_team_norm'], inplace=True)
            
            cols = [c.lower() for c in dfs[rapm_key].columns]
            cols_map = {c.lower(): c for c in dfs[rapm_key].columns}
            
            # Definir colunas alvo (suportar formatos novos e antigos)
            col_orapm = next((c for c in ['orapm', 'time decay orapm'] if c in cols), None)
            col_drapm = next((c for c in ['drapm', 'time decay drapm'] if c in cols), None)
            
            if col_orapm and col_drapm:
                if not df_rapm.empty:
                    rapm_vals = []
                    for _, row in df_rapm.iterrows():
                        try:
                            o_rapm = float(row.get(cols_map[col_orapm], 0))
                            d_rapm = float(row.get(cols_map[col_drapm], 0))
                            combined = (o_rapm + d_rapm) / 2
                            rapm_vals.append(combined)
                        except (ValueError, TypeError):
                            pass
                    
                    if rapm_vals:
                        rapm_media = float(np.mean(rapm_vals))
                        rapm_norm = normalizar_metrica(rapm_media, norm_min, norm_max)
                        metricas_coletadas['rapm'] = {"valor": rapm_norm, "peso": RATING_WEIGHTS['rapm']}
                        logger.debug(f"         ✅ RAPM: {rapm_media:.2f} → norm: {rapm_norm:.2f}")
            else:
                logger.warning(f"         ⚠️ RAPM: Colunas obrigatórias ausentes. Encontradas: {cols}")
                
        except Exception as e:
            logger.debug(f"         ❌ RAPM: Erro - {e}")
    
    # LEBRON
    if 'lebron' in dfs and not dfs['lebron'].empty:
        try:
            team_abbrev_upper = team_abbrev.upper()
            col_team = 'Team'
            for col in dfs['lebron'].columns:
                if 'team' in col.lower():
                    col_team = col
                    break
            
            # Validar colunas
            if 'O-LEBRON' in dfs['lebron'].columns and 'D-LEBRON' in dfs['lebron'].columns:
                df_lebron = dfs['lebron'][dfs['lebron'][col_team].str.upper() == team_abbrev_upper]
                
                if not df_lebron.empty:
                    lebron_vals = []
                    for _, row in df_lebron.iterrows():
                        try:
                            o_lbr = float(row.get('O-LEBRON', 0))
                            d_lbr = float(row.get('D-LEBRON', 0))
                            combined = (o_lbr + d_lbr) / 2
                            lebron_vals.append(combined)
                        except (ValueError, TypeError):
                            pass
                    
                    if lebron_vals:
                        lebron_media = float(np.mean(lebron_vals))
                        lebron_norm = normalizar_metrica(lebron_media, norm_min, norm_max)
                        metricas_coletadas['lebron'] = {"valor": lebron_norm, "peso": RATING_WEIGHTS['lebron']}
                        logger.debug(f"         ✅ LEBRON: {lebron_media:.2f} → norm: {lebron_norm:.2f}")
            else:
                 logger.warning(f"         ⚠️ LEBRON: Colunas O-LEBRON/D-LEBRON ausentes.")

        except Exception as e:
            logger.debug(f"         ❌ LEBRON: Erro - {e}")
    
    # BPM
    if 'bpm' in dfs and not dfs['bpm'].empty:
        try:
            team_abbrev_upper = team_abbrev.upper()
            if 'Team' in dfs['bpm'].columns and 'OBPM' in dfs['bpm'].columns and 'DBPM' in dfs['bpm'].columns:
                df_bpm = dfs['bpm'][dfs['bpm']['Team'].str.upper() == team_abbrev_upper]
                
                if not df_bpm.empty:
                    bpm_vals = []
                    mp_total = 0.0
                    for _, row in df_bpm.iterrows():
                        try:
                            obpm = float(row.get('OBPM', 0))
                            dbpm = float(row.get('DBPM', 0))
                            mp = float(row.get('MP', 0))
                            combined = (obpm + dbpm) / 2
                            bpm_vals.append(combined)
                            mp_total += mp
                        except (ValueError, TypeError):
                            pass
                    
                    if bpm_vals:
                        bpm_media = float(np.mean(bpm_vals))
                        bpm_norm = normalizar_metrica(bpm_media, norm_min, norm_max)
                        
                        # Peso dinâmico baseado em minutos jogados
                        peso_bpm = max(
                            RATING_WEIGHTS['bpm_min'], 
                            min(RATING_WEIGHTS['bpm_max'], mp_total / RATING_WEIGHTS['bpm_divisor'])
                        )
                        
                        metricas_coletadas['bpm'] = {"valor": bpm_norm, "peso": peso_bpm}
                        logger.debug(f"         ✅ BPM: {bpm_media:.2f} → norm: {bpm_norm:.2f} (peso: {peso_bpm:.2f})")
            else:
                logger.warning("         ⚠️ BPM: Colunas Team/OBPM/DBPM ausentes.")
                
        except Exception as e:
            logger.debug(f"         ❌ BPM: Erro - {e}")
    
    # PIE
    if 'pie' in dfs and not dfs['pie'].empty:
        try:
            team_abbrev_upper = team_abbrev.upper()
            col_team = 'Team'
            # Tentar encontrar coluna de time
            for c in dfs['pie'].columns:
                if 'team' in c.lower():
                    col_team = c
                    break
            
            if 'PIE' in dfs['pie'].columns:
                df_pie = dfs['pie'][dfs['pie'][col_team].str.upper().str.contains(team_abbrev_upper, na=False)]
                
                if not df_pie.empty:
                    pie_val = float(df_pie.iloc[0]['PIE'])
                    # PIE já é 0-100 geralmente, ou 0-1. Assumindo 0-100 ou ajustando
                    if pie_val < 1.0: pie_val *= 100
                    
                    metricas_coletadas['pie'] = {"valor": pie_val, "peso": 1.5} # Peso fixo por enquanto
                    logger.debug(f"         ✅ PIE: {pie_val:.2f}")
        except Exception as e:
            logger.debug(f"         ❌ PIE: Erro - {e}")

    # Calcular média ponderada
    if not metricas_coletadas:
        logger.warning(f"      ⚠️ Nenhuma métrica coletada para {team_name}. Retornando 0.0")
        return 0.0
        
    soma_valores = 0.0
    soma_pesos = 0.0
    
    for m, dados in metricas_coletadas.items():
        soma_valores += dados['valor'] * dados['peso']
        soma_pesos += dados['peso']
        
    if soma_pesos == 0:
        return 0.0
        
    net_rating_norm = soma_valores / soma_pesos
    
    # Desnormalizar para escala -10 a +10 (aproximada)
    # 0 -> -10, 50 -> 0, 100 -> +10
    net_rating_final = (net_rating_norm - 50) / 5
    
    if np.isnan(net_rating_final):
        return 0.0
        
    return net_rating_final

def calculate_expected_efg(team_abbr: str, shot_quality_data: Dict[str, Any]) -> Optional[float]:
    """
    Calcula xeFG% (Expected eFG%).
    
    Args:
        team_abbr: Abreviação do time.
        shot_quality_data: Dados de qualidade de arremesso.
        
    Returns:
        Expected eFG% ou None se não puder ser calculado.
    """
    if not shot_quality_data:
        return None
    
    team_data = shot_quality_data.get(team_abbr, {})
    val = team_data.get('xEFG')
    return float(val) if val is not None else None

def apply_shot_quality_adjustment(
    team_name: str, 
    net_rating: float, 
    shot_quality_data: Dict[str, Any]
) -> float:
    """
    Aplica ajuste de Shot Quality Luck ao Net Rating.
    
    Se um time tem xEFG > Actual EFG, ele está 'azarado' e merece um bump no rating.
    Se xEFG < Actual EFG, ele está 'sortudo' (ou elite shooting) e pode regredir.
    """
    if not shot_quality_data:
        return net_rating
        
    team_abbr = TEAM_ABBREV_MAP.get(team_name)
    if not team_abbr:
        return net_rating
        
    team_data = shot_quality_data.get(team_abbr)
    if not team_data:
        return net_rating
        
    x_efg = team_data.get('xEFG')
    actual_efg = team_data.get('actual_EFG')
    
    if x_efg is None or actual_efg is None:
        return net_rating
        
    # Diferença: Positivo = Azarado (Merece mais), Negativo = Sortudo (Merece menos)
    diff = x_efg - actual_efg
    
    # Ajuste conservador: 50% da diferença impacta o Net Rating
    # Ex: Diff +0.02 (2%) -> Ajuste +1.0 no Net Rating (aprox)
    adjustment = diff * 50.0 
    
    # Limitar ajuste a +/- 2.0 pontos de Net Rating para evitar distorções
    adjustment = max(-2.0, min(2.0, adjustment))
    
    logger.debug(f"      🎯 SQ Adj {team_name}: Diff {diff:+.3f} -> Adj {adjustment:+.2f}")
    
    return float(net_rating + adjustment)

def calcular_net_rating_v11_with_shot_quality(team_name, dfs, shot_quality_data=None):
    """Wrapper que aplica Shot Quality adjustment ao Net Rating"""
    nr = calcular_net_rating_v11(team_name, dfs)
    if shot_quality_data:
        return apply_shot_quality_adjustment(team_name, nr, shot_quality_data)
    return nr

def calcular_fator_lesao(team_name: str, injuries: Dict[str, Dict[str, str]]) -> float:
    """
    Calcula fator de lesão do time.
    
    Args:
        team_name: Nome do time.
        injuries: Dicionário com relatório de lesões por time.
        
    Returns:
        Fator de lesão (negativo = impacto negativo, positivo = impacto positivo).
    """
    if not injuries:
        return 0.0
        
    team_injuries = injuries.get(team_name, {})
    if not team_injuries:
        return 0.0
        
    impacto_total = 0.0
    
    # Lógica simplificada de impacto baseada em estrelas
    for player, status in team_injuries.items():
        if player in ALL_STARS_2025:
            if status.lower() in ['out', 'doubtful']:
                impacto_total -= 0.15 # 15% de penalidade no rating
            elif status.lower() in ['questionable']:
                impacto_total -= 0.05
                
    return impacto_total

def calcular_power_rating_v11(
    home_team: str,
    away_team: str,
    injuries: Dict[str, Dict[str, str]],
    standings: Dict[str, Any],
    dfs: Dict[str, pd.DataFrame],
    referees: Optional[List[str]] = None,
    shot_quality_data: Optional[Dict[str, Any]] = None,
    recent_games_df: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Cálculo completo do Power Rating v11.1 (Dynamic HCA).
    
    UPGRADE: HCA agora é dinâmico baseado em:
    - Arena multipliers (altitude, crowd, etc)
    - Recent home performance
    
    Args:
        home_team: Nome do time da casa.
        away_team: Nome do time visitante.
        injuries: Relatório de lesões.
        standings: Classificação dos times.
        dfs: DataFrames com estatísticas.
        referees: Lista de árbitros (opcional).
        shot_quality_data: Dados de qualidade de arremesso (opcional).
        recent_games_df: DataFrame com jogos recentes para HCA dinâmico (opcional).
        
    Returns:
        Dicionário com Power Rating, probabilidades e ajustes.
    """
    if referees is None:
        referees = []
    
    # Calcular Net Rating (com ajuste de Shot Quality se disponível)
    if shot_quality_data:
        nr_home = calcular_net_rating_v11_with_shot_quality(home_team, dfs, shot_quality_data)
        nr_away = calcular_net_rating_v11_with_shot_quality(away_team, dfs, shot_quality_data)
    else:
        nr_home = calcular_net_rating_v11(home_team, dfs)
        nr_away = calcular_net_rating_v11(away_team, dfs)
    
    # Garantir que nr_home e nr_away sejam floats
    if nr_home is None: nr_home = 0.0
    if nr_away is None: nr_away = 0.0
    
    fator_lesao_home = calcular_fator_lesao(home_team, injuries)
    fator_lesao_away = calcular_fator_lesao(away_team, injuries)
    
    # Garantir que fatores de lesão sejam floats
    if fator_lesao_home is None: fator_lesao_home = 0.0
    if fator_lesao_away is None: fator_lesao_away = 0.0
    
    nr_ajustado_home = nr_home + (nr_home * fator_lesao_home)
    nr_ajustado_away = nr_away + (nr_away * fator_lesao_away)
    
    ajuste_referee = 0.0
    if referees:
        total_home_win_pct = 0.0
        count = 0
        for ref in referees:
            stats = get_referee_stats(ref)
            if stats:
                total_home_win_pct += stats.get('home_win_pct', REFEREE_ADJUSTMENTS['default_home_win_pct'])
                count += 1
        
        if count > 0:
            avg_hw_pct = total_home_win_pct / count
            if avg_hw_pct >= REFEREE_ADJUSTMENTS['high_win_pct_threshold']:
                ajuste_referee = REFEREE_ADJUSTMENTS['high_adjustment']
            elif avg_hw_pct <= REFEREE_ADJUSTMENTS['low_win_pct_threshold']:
                ajuste_referee = REFEREE_ADJUSTMENTS['low_adjustment']
    
    # Fórmula base do Power Rating
    # Base 10 + Net Rating Ajustado * 1.3 (fator de escala)
    base_home = (10 + nr_ajustado_home) * 1.3 + ajuste_referee
    base_away = (10 + nr_ajustado_away) * 1.3
    
    # 🚀 UPGRADE V11.1: Dynamic HCA
    # Normalizar home_team para código de 3 letras
    home_team_code = normalize_team(home_team) if home_team else None
    
    if home_team_code:
        try:
            # Calcular HCA dinâmico
            hca = calculate_dynamic_hca(
                home_team_code,
                recent_home_games=None  # TODO: passar recent_games_df filtrado
            )
            logger.debug(f"🏠 Dynamic HCA for {home_team_code}: {hca:.2f} pts")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao calcular HCA dinâmico: {e}. Usando estático.")
            hca = HCA_VALUE
    else:
        # Fallback para HCA estático se normalização falhar
        hca = HCA_VALUE
    
    pr_home = max(base_home + hca, 1.0)
    pr_away = max(base_away, 1.0)
    
    prob_home = (pr_home / (pr_home + pr_away)) * 100
    prob_away = 100 - prob_home
    
    return {
        "pr_casa": float(pr_home),
        "pr_visitante": float(pr_away),
        "prob_casa": float(prob_home),
        "prob_visitante": float(prob_away),
        "nr_bruto_casa": float(nr_home),
        "nr_bruto_visitante": float(nr_away),
        "nr_ajustado_casa": float(nr_ajustado_home),
        "nr_ajustado_visitante": float(nr_ajustado_away),
        "fator_lesao_casa": float(fator_lesao_home),
        "fator_lesao_visitante": float(fator_lesao_away),
        "ajuste_referee": float(ajuste_referee),
        "hca_usado": float(hca)  # NEW: Track dynamic HCA used
    }
