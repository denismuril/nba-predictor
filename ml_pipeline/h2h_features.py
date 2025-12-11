"""
Módulo para cálculo de features Head-to-Head (H2H).

Este módulo calcula estatísticas baseadas no histórico de confrontos diretos
entre duas equipes, como taxa de vitória e diferença de pontos média.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_h2h_stats(df_games: pd.DataFrame, lookback_window: int = 10) -> pd.DataFrame:
    """
    Calcula estatísticas de confronto direto (H2H) para cada jogo.
    
    Para cada jogo (Home vs Away), olha para os últimos N jogos entre
    esses dois times ANTES da data do jogo atual.
    
    Args:
        df_games: DataFrame contendo histórico de jogos.
                  Deve ter colunas: 'date', 'home_team', 'away_team', 'home_score', 'away_score'
        lookback_window: Número máximo de jogos passados a considerar (default: 10)
        
    Returns:
        DataFrame original com novas colunas H2H:
        - h2h_home_win_rate: % de vitórias do time da casa contra este oponente
        - h2h_avg_point_diff: Diferença média de pontos (Home - Away)
        - h2h_games_played: Número de jogos no histórico considerado
    """
    logger.info(f"⚔️ Calculando features Head-to-Head (H2H) (Janela: {lookback_window} jogos)...")
    
    # Garantir ordenação por data
    df = df_games.sort_values('date').copy()
    
    # Dicionário para armazenar histórico de confrontos
    # Chave: tuple(sorted(team_a, team_b)) -> Lista de resultados
    # Valor: Lista de dicts {'date': date, 'winner': team, 'point_diff': score_a - score_b}
    matchup_history = {}
    
    # Listas para armazenar as novas features
    h2h_win_rates = []
    h2h_point_diffs = []
    h2h_counts = []
    
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        date = row['date']
        
        # Chave única para o matchup (independente de quem é casa/fora)
        matchup_key = tuple(sorted([home, away]))
        
        if matchup_key not in matchup_history:
            matchup_history[matchup_key] = []
            
        history = matchup_history[matchup_key]
        
        # Filtrar apenas jogos ANTERIORES a este (embora o loop já garanta ordem, é bom ser explícito se houver datas iguais)
        # Como estamos iterando em ordem, o histórico atual já contém apenas jogos passados processados
        
        # Pegar os últimos N jogos
        recent_history = history[-lookback_window:]
        
        if not recent_history:
            # Sem histórico: valores neutros
            h2h_win_rates.append(0.5)
            h2h_point_diffs.append(0.0)
            h2h_counts.append(0)
        else:
            # Calcular stats
            # Win rate do HOME team atual
            home_wins = sum(1 for game in recent_history if game['winner'] == home)
            win_rate = home_wins / len(recent_history)
            
            # Point diff (Home - Away)
            # No histórico, point_diff precisa ser ajustado para a perspectiva do HOME atual
            total_diff = 0
            for game in recent_history:
                # Se no jogo histórico o time 'home' atual foi 'team_a' (da chave sorted), usamos o diff direto
                # Se não, invertemos
                # Mas simplificando: armazenar diff sempre da perspectiva do time alfabeticamente primeiro
                # E ajustar na leitura
                
                # Vamos simplificar o armazenamento:
                # Armazenar: {'home_team': h, 'away_team': a, 'home_score': hs, 'away_score': as}
                pass
            
            # Re-implementando lógica de diff com armazenamento mais simples
            diffs = []
            for game in recent_history:
                if game['home_team'] == home:
                    diff = game['home_score'] - game['away_score']
                else:
                    # O time 'home' atual estava como 'away' no jogo histórico
                    diff = game['away_score'] - game['home_score']
                diffs.append(diff)
            
            avg_diff = sum(diffs) / len(diffs)
            
            h2h_win_rates.append(win_rate)
            h2h_point_diffs.append(avg_diff)
            h2h_counts.append(len(recent_history))
        
        # Adicionar jogo atual ao histórico (para os próximos loops)
        # Só se tiver placar (treino vs inferência futura)
        if pd.notna(row.get('home_score')) and pd.notna(row.get('away_score')):
            winner = home if row['home_score'] > row['away_score'] else away
            matchup_history[matchup_key].append({
                'date': date,
                'home_team': home,
                'away_team': away,
                'home_score': row['home_score'],
                'away_score': row['away_score'],
                'winner': winner
            })
            
    # Adicionar colunas ao DF
    df['h2h_home_win_rate'] = h2h_win_rates
    df['h2h_avg_point_diff'] = h2h_point_diffs
    df['h2h_games_played'] = h2h_counts
    
    logger.info(f"✅ Features H2H calculadas. Média de jogos prévios: {np.mean(h2h_counts):.1f}")
    
    return df
