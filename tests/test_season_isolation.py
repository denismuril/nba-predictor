"""
Teste de integração para validar isolamento de temporada nas features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.feature_engineering_v2 import add_rolling_four_factors


def test_season_isolation():
    """
    Testa que rolling features não cruzam limites de temporada.
    """
    print("🧪 Teste: Isolamento de Temporada")
    print("=" * 60)
    
    # Criar dados de teste com duas temporadas
    dates_old_season = pd.date_range('2024-10-01', '2024-12-31', freq='3D')
    dates_new_season = pd.date_range('2025-10-01', '2025-12-15', freq='3D')
    
    # Time fictício: GS
    games_old = []
    for date in dates_old_season:
        games_old.append({
            'date': date,
            'home_team': 'GSW',
            'away_team': 'LAL',
            'home_score': 110,
            'away_score': 105,
            'fgm': 40, 'fga': 85, 'fg3m': 15,
            'ftm': 15, 'fta': 18,
            'oreb': 10, 'dreb': 35, 'ast': 25, 'stl': 8, 'blk': 5, 'tov': 12, 'pf': 20, 'pts': 110,
            'opp_fgm': 38, 'opp_fga': 83, 'opp_fg3m': 12,
            'opp_ftm': 17, 'opp_fta': 20,
            'opp_oreb': 8, 'opp_dreb': 33, 'opp_ast': 22, 'opp_stl': 6, 'opp_blk': 4, 'opp_tov': 14, 'opp_pf': 22, 'opp_pts': 105
        })
    
    games_new = []
    for date in dates_new_season:
        games_new.append({
            'date': date,
            'home_team': 'GSW',
            'away_team': 'LAL',
            'home_score': 115,
            'away_score': 108,
            'fgm': 42, 'fga': 87, 'fg3m': 16,
            'ftm': 15, 'fta': 18,
            'oreb': 12, 'dreb': 36, 'ast': 27, 'stl': 9, 'blk': 6, 'tov': 11, 'pf': 19, 'pts': 115,
            'opp_fgm': 40, 'opp_fga': 85, 'opp_fg3m': 13,
            'opp_ftm': 15, 'opp_fta': 19,
            'opp_oreb': 9, 'opp_dreb': 34, 'opp_ast': 23, 'opp_stl': 7, 'opp_blk': 5, 'opp_tov': 13, 'opp_pf': 21, 'opp_pts': 108
        })
    
    df = pd.DataFrame(games_old + games_new)
    
    # Aplicar rolling features
    df_with_features = add_rolling_four_factors(df, windows=[5, 10])
    
    # Verificar jogos de  2025: rolling features NÃO devem incluir dados de 2024
    df_new = df_with_features[df_with_features['date'] >= '2025-10-01']
    
    print(f"\n✅ Total de jogos na nova temporada: {len(df_new)}")
    print(f"✅ Primeiras rolling features (deveriam usar apenas dados de 2025):")
    
    first_game = df_new.iloc[0]
    print(f"   Data: {first_game['date'].date()}")
    print(f"   home_rolling_5_pts: {first_game.get('home_rolling_5_pts', 'AUSENTE')}")
    print(f"   home_rolling_10_pts: {first_game.get('home_rolling_10_pts', 'AUSENTE')}")
    
    # Se funcionou corretamente, os rolling values devem ser NaN ou média da liga (não 110 da season antiga)
    rolling_5 = first_game.get('home_rolling_5_pts', None)
    if rolling_5 is not None and not pd.isna(rolling_5):
        if abs(rolling_5 - 110) < 1:  # Se estiver próximo de 110, usou dados antigos (BUG)
            print(f"   ❌ FALHA: Rolling feature usou dados da temporada antiga (valor={rolling_5})")
            return False
        else:
            print(f"   ✅ SUCESSO: Rolling feature não contaminou com temporada antiga")
    else:
        print(f"   ℹ️  Rolling feature é NaN (esperado para primeiro jogo da temporada)")
    
    return True


def test_zero_stats_filtering():
    """
    Testa que jogos com stats zeradas são filtrados antes do rolling.
    """
    print("\n🧪 Teste: Filtragem de Stats Zeradas")
    print("=" * 60)
    
    # Criar dados com alguns jogos zerados
    games = []
    for i in range(15):
        is_zero = (i % 5 == 0)  # Cada 5º jogo tem stats zeradas
        games.append({
            'date': pd.Timestamp('2025-10-01') + timedelta(days=i*2),
            'home_team': 'BOS',
            'away_team': 'MIA',
            'home_score': 0 if is_zero else 112,
            'away_score': 0 if is_zero else 108,
            'fgm': 0 if is_zero else 41, 'fga': 0 if is_zero else 86, 'fg3m': 0 if is_zero else 14,
            'ftm': 0 if is_zero else 16, 'fta': 0 if is_zero else 19,
            'oreb': 0 if is_zero else 11, 'dreb': 0 if is_zero else 35, 
            'ast': 0 if is_zero else 26, 'stl': 0 if is_zero else 8, 
            'blk': 0 if is_zero else 5, 'tov': 0 if is_zero else 12, 
            'pf': 0 if is_zero else 20, 'pts': 0 if is_zero else 112,
            'opp_fgm': 0 if is_zero else 39, 'opp_fga': 0 if is_zero else 84, 'opp_fg3m': 0 if is_zero else 13,
            'opp_ftm': 0 if is_zero else 17, 'opp_fta': 0 if is_zero else 20,
            'opp_oreb': 0 if is_zero else 9, 'opp_dreb': 0 if is_zero else 34, 
            'opp_ast': 0 if is_zero else 24, 'opp_stl': 0 if is_zero else 7, 
            'opp_blk': 0 if is_zero else 4, 'opp_tov': 0 if is_zero else 13, 
            'opp_pf': 0 if is_zero else 21, 'opp_pts': 0 if is_zero else 108
        })
    
    df = pd.DataFrame(games)
    
    print(f"\n📊 Dataset original: {len(df)} jogos")
    print(f"   Jogos com stats zeradas: {(df['pts'] == 0).sum()}")
    
    # Aplicar rolling features
    df_with_features = add_rolling_four_factors(df, windows=[5])
    
    # Verificar rolling averages são realistas (não contaminadas por zeros)
    avg_pts = df_with_features['home_rolling_5_pts'].mean()
    
    print(f"\n✅ Média de home_rolling_5_pts: {avg_pts:.1f}")
    
    if avg_pts < 90:  # Se média for muito baixa, zeros contaminaram
        print(f"   ❌ FALHA: Rolling averages parecem contaminadas por zeros (média={avg_pts:.1f})")
        return False
    else:
        print(f"   ✅ SUCESSO: Rolling averages realistas (zeros foram filtrados)")
    
    return True


if __name__ == "__main__":
    print("🚀 Executando Testes de Isolamento de Temporada\n")
    
    test1 = test_season_isolation()
    test2 = test_zero_stats_filtering()
    
    print("\n" + "=" * 60)
    print("📊 Resultados Finais:")
    print(f"   Teste de Isolamento de Temporada: {'✅ PASSOU' if test1 else '❌ FALHOU'}")
    print(f"   Teste de Filtragem de Zeros: {'✅ PASSOU' if test2 else '❌ FALHOU'}")
    print("=" * 60)
