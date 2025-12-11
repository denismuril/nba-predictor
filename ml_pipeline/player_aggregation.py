"""
Módulo para Agregar Estatísticas de Jogadores por Time.

Este módulo processa dados de jogadores (RAPM, BPM, etc.) e agrega por time,
criando features para o modelo de ML.

Features criadas:
- team_rapm_avg: Média RAPM dos top 5 jogadores
- team_rapm_top: RAPM do melhor jogador
- team_rapm_std: Desvio padrão (indica profundidade do elenco)
- Similar para BPM

Autor: Sistema NBA Predictor
Data: 2025-12-01
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def aggregate_player_stats_by_team(
    df_players: pd.DataFrame,
    top_n: int = 5,
    min_minutes: float = 10.0
) -> pd.DataFrame:
    """
    Agrega estatísticas de jogadores por time.
    
    Args:
        df_players: DataFrame com colunas ['Player', 'Team', 'RAPM', 'ORAPM', 'DRAPM', 'BPM', 'MP']
        top_n: Número de top jogadores a considerar
        min_minutes: Minutos mínimos para considerar o jogador
        
    Returns:
        DataFrame com features agregadas por time:
        - Team
        - team_rapm_avg: Média RAPM dos top N
        - team_rapm_top: RAPM do melhor jogador
        - team_rapm_std: Desvio padrão
        - team_bpm_avg, team_bpm_top, team_bpm_std (mesmo para BPM)
        - team_depth_score: Métrica de profundidade do elenco
    """
    if df_players is None or df_players.empty:
        logger.warning("⚠️ DataFrame de jogadores vazio. Retornando features vazias.")
        return pd.DataFrame()
    
    # Filtrar jogadores com minutos mínimos (se coluna MP existir)
    if 'MP' in df_players.columns:
        df_filtered = df_players[df_players['MP'] >= min_minutes].copy()
        logger.info(f"📊 Filtrados {len(df_filtered)}/{len(df_players)} jogadores com MP >= {min_minutes}")
    else:
        df_filtered = df_players.copy()
        logger.info(f"📊 Processando {len(df_filtered)} jogadores (sem filtro de MP)")
    
    # Garantir que RAPM e BPM são numéricos
    for col in ['RAPM', 'ORAPM', 'DRAPM']:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce').fillna(0)
    
    # Se BPM não existir, criar columns vazias
    if 'BPM' not in df_filtered.columns:
        df_filtered['BPM'] = 0.0
        logger.info("ℹ️ Coluna BPM não encontrada. Usando 0.0 como fallback.")
    
    # Agregar por time
    team_features = []
    
    for team, group in df_filtered.groupby('Team'):
        # Ordenar por RAPM (descendente)
        group_sorted = group.sort_values('RAPM', ascending=False)
        
        # Top N jogadores
        top_players = group_sorted.head(top_n)
        
        if len(top_players) == 0:
            continue
        
        # Features RAPM
        rapm_avg = top_players['RAPM'].mean()
        rapm_top = top_players['RAPM'].iloc[0] if len(top_players) > 0 else 0.0
        rapm_std = top_players['RAPM'].std() if len(top_players) > 1 else 0.0
        
        orapm_avg = top_players['ORAPM'].mean() if 'ORAPM' in top_players else 0.0
        drapm_avg = top_players['DRAPM'].mean() if 'DRAPM' in top_players else 0.0
        
        # Features BPM
        bpm_avg = top_players['BPM'].mean()
        bpm_top = top_players['BPM'].max()
        bpm_std = top_players['BPM'].std() if len(top_players) > 1 else 0.0
        
        # Depth Score: quanto mais jogadores bons, maior o score
        # Métrica: soma dos RAPMs positivos / número de jogadores
        positive_rapm = top_players[top_players['RAPM'] > 0]['RAPM']
        depth_score = positive_rapm.sum() / max(len(positive_rapm), 1) if len(positive_rapm) > 0 else 0.0
        
        team_features.append({
            'Team': team,
            'rapm_avg': rapm_avg,
            'rapm_top': rapm_top,
            'rapm_std': rapm_std,
            'orapm_avg': orapm_avg,
            'drapm_avg': drapm_avg,
            'bpm_avg': bpm_avg,
            'bpm_top': bpm_top,
            'bpm_std': bpm_std,
            'depth_score': depth_score,
            'player_count': len(group)  # Total de jogadores no roster
        })
    
    result_df = pd.DataFrame(team_features)
    
    if not result_df.empty:
        logger.info(f"✅ Agregadas features de jogadores para {len(result_df)} times")
        logger.info(f"   RAPM médio range: [{result_df['rapm_avg'].min():.2f}, {result_df['rapm_avg'].max():.2f}]")
    else:
        logger.warning("⚠️ Nenhuma feature de jogador foi agregada")
    
    return result_df


def merge_player_features_to_games(
    df_games: pd.DataFrame,
    df_player_features: pd.DataFrame,
    fillna_strategy: str = 'median'
) -> pd.DataFrame:
    """
    Merge features de jogadores com DataFrame de jogos.
    
    Args:
        df_games: DataFrame com jogos (deve ter 'home_team' e 'away_team')
        df_player_features: DataFrame retornado por aggregate_player_stats_by_team()
        fillna_strategy: Estratégia para preencher valores faltantes
                        - 'median': usar mediana dos times
                        - 'zero': preencher com 0
                        
    Returns:
        df_games com features de jogadores adicionadas para home e away
    """
    if df_player_features is None or df_player_features.empty:
        logger.warning("⚠️ Features de jogadores vazias. Usando fallback (zeros).")
        # Criar features vazias
        for prefix in ['home', 'away']:
            for col in ['rapm_avg', 'rapm_top', 'rapm_std', 'bpm_avg', 'bpm_top', 
                       'depth_score', 'orapm_avg', 'drapm_avg']:
                df_games[f'{prefix}_{col}'] = 0.0
        return df_games
    
    # Merge para home team
    df_home = df_player_features.copy()
    df_home.columns = ['Team'] + [f'home_{c}' if c != 'Team' else c 
                                   for c in df_home.columns if c != 'Team']
    
    df_games = df_games.merge(df_home, left_on='home_team', right_on='Team', how='left')
    df_games = df_games.drop(columns=['Team'], errors='ignore')
    
    # Merge para away team
    df_away = df_player_features.copy()
    df_away.columns = ['Team'] + [f'away_{c}' if c != 'Team' else c 
                                   for c in df_away.columns if c != 'Team']
    
    df_games = df_games.merge(df_away, left_on='away_team', right_on='Team', how='left')
    df_games = df_games.drop(columns=['Team'], errors='ignore')
    
    # Preencher valores faltantes
    player_cols = [c for c in df_games.columns if c.startswith(('home_rapm', 'away_rapm', 
                                                                 'home_bpm', 'away_bpm',
                                                                 'home_depth', 'away_depth',
                                                                 'home_orapm', 'away_orapm',
                                                                 'home_drapm', 'away_drapm'))]
    
    if fillna_strategy == 'median':
        for col in player_cols:
            if col in df_games.columns:
                median_val = df_games[col].median()
                df_games[col] = df_games[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                logger.debug(f"Preenchido {col} com mediana: {median_val:.2f}")
    else:
        df_games[player_cols] = df_games[player_cols].fillna(0.0)
    
    # Criar features de diferença
    df_games['rapm_diff'] = df_games['home_rapm_avg'] - df_games['away_rapm_avg']
    df_games['bpm_diff'] = df_games['home_bpm_avg'] - df_games['away_bpm_avg']
    df_games['depth_diff'] = df_games['home_depth_score'] - df_games['away_depth_score']
    
    logger.info(f"✅ Merged player features. Criadas {len(player_cols) + 3} colunas")
    
    return df_games


def get_cached_player_stats(cache_dir: Path = None) -> Optional[pd.DataFrame]:
    """
    Tenta carregar estatísticas de jogadores do cache.
    
    Args:
        cache_dir: Diretório do cache (padrão: data/)
        
    Returns:
        DataFrame com stats de jogadores ou None se não encontrado
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data"
    
    # Tentar CSV de RAPM
    rapm_csv = cache_dir / "nba_rapm.csv"
    logger.info(f"🔍 Verificando caminho: {rapm_csv.absolute()}")
    
    if rapm_csv.exists():
        logger.info(f"📂 Carregando stats de jogadores do cache: {rapm_csv}")
        try:
            df = pd.read_csv(rapm_csv)
            
            # Mapeamento de colunas do scraper para o esperado pelo pipeline
            column_map = {
                'player_name': 'Player',
                'team': 'Team',
                'rapm_timedecay': 'RAPM',
                'orapm_timedecay': 'ORAPM',
                'drapm_timedecay': 'DRAPM',
                'rapm_darko': 'RAPM_Darko' 
            }
            
            # Renomear colunas existentes
            df = df.rename(columns=column_map)
            
            # Verificar se as colunas essenciais existem após renomear
            required = ['Player', 'Team', 'RAPM']
            if not all(col in df.columns for col in required):
                logger.warning(f"⚠️ Colunas obrigatórias faltando após renomear: {required}")
                logger.warning(f"   Colunas disponíveis: {list(df.columns)}")
                return None
            
            # 🔄 Tentar carregar BPM do Excel (se existir)
            bpm_xlsx = cache_dir / "nba_bpm.xlsx"
            if bpm_xlsx.exists():
                logger.info(f"📂 Carregando BPM do Excel: {bpm_xlsx}")
                try:
                    df_bpm = pd.read_excel(bpm_xlsx)
                    
                    # Normalizar colunas do BPM
                    # Geralmente vem como 'Player', 'BPM', 'OBPM', 'DBPM' do Bball-Ref
                    # Mas pode precisar de limpeza (remover asteriscos de nomes, etc)
                    
                    if 'Player' in df_bpm.columns and 'BPM' in df_bpm.columns:
                        # Limpar nomes de jogadores (ex: "LeBron James*" -> "LeBron James")
                        df_bpm['Player'] = df_bpm['Player'].astype(str).str.replace('*', '', regex=False).str.strip()
                        
                        # Selecionar colunas úteis
                        bpm_cols = ['Player', 'BPM', 'OBPM', 'DBPM', 'VORP']
                        cols_to_merge = [c for c in bpm_cols if c in df_bpm.columns]
                        
                        if cols_to_merge:
                            # Merge com RAPM
                            logger.info(f"🔄 Merging BPM data ({len(df_bpm)} players)...")
                            # Merge left on Player (assumindo que RAPM é a base principal)
                            # Team pode variar (TOT vs time atual), então usamos Player como chave principal
                            # Idealmente usaria Player + Team, mas Bball-Ref usa TOT para trocas
                            
                            # Remover duplicatas em BPM (pegar a última linha, geralmente TOT ou time atual)
                            df_bpm = df_bpm.drop_duplicates(subset=['Player'], keep='first')
                            
                            df = df.merge(df_bpm[cols_to_merge], on='Player', how='left')
                            
                            # Preencher BPM nulo com 0
                            for c in cols_to_merge:
                                if c != 'Player':
                                    df[c] = df[c].fillna(0.0)
                                    
                            logger.info("✅ BPM merged com sucesso!")
                    else:
                        logger.warning("⚠️ Excel de BPM não tem colunas esperadas (Player, BPM)")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao carregar/mergear BPM: {e}")
            else:
                logger.warning(f"⚠️ Arquivo de BPM não encontrado: {bpm_xlsx}")

            return df
        except Exception as e:
            logger.warning(f"⚠️ Erro lendo cache: {e}")
    else:
        logger.warning(f"⚠️ Arquivo não encontrado: {rapm_csv}")
        try:
            if cache_dir.exists():
                logger.info(f"📂 Conteúdo de {cache_dir.absolute()}: {list(cache_dir.glob('*'))}")
            else:
                logger.warning(f"⚠️ Diretório de cache não existe: {cache_dir.absolute()}")
        except Exception as e:
            logger.error(f"Erro listando diretório: {e}")
    
    return None
