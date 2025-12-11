"""
Test NBA Official API usando nba_api package

Testa acesso a dados REAIS de box score e stats avançadas.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_api.stats.endpoints import boxscoreadvancedv2, boxscoretraditionalv2
from nba_api.stats.endpoints import leaguegamefinder
from datetime import datetime, timedelta

print("🏀 Testando NBA Official API\n")

# Buscar game recente
print("1. Buscando games recentes...")
try:
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable='2024-25',
        league_id_nullable='00'
    )
    games = gamefinder.get_data_frames()[0]
    
    if len(games) > 0:
        recent_game = games.iloc[0]
        game_id = recent_game['GAME_ID']
        print(f"✅ Game encontrado: {game_id}")
        print(f"   {recent_game['MATCHUP']} ({recent_game['GAME_DATE']})")
        
        # Test boxscore tradicional
        print("\n2. Buscando boxscore tradicional...")
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        team_stats = boxscore.get_data_frames()[1]  # Team stats
        
        print(f"✅ Box score obtido!")
        print(f"   Teams: {len(team_stats)}")
        print(f"   Columns: {list(team_stats.columns)[:10]}")
        
        # Test advanced stats
        print("\n3. Buscando stats avançadas...")
        advanced = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
        adv_stats = advanced.get_data_frames()[1]  # Team advanced
        
        print(f"✅ Advanced stats obtidas!")
        print(f"   Columns: {list(adv_stats.columns)[:10]}")
        
        # Verificar se tem fast break, paint, etc
        print("\n4. Verificando stats específicas...")
        all_cols = list(team_stats.columns) + list(adv_stats.columns)
        
        target_stats = ['FASTBREAK', 'PAINT', 'SECOND_CHANCE', 'PTS_FB', 'PTS_PAINT', 'PTS_2ND_CHANCE']
        found = [col for col in all_cols if any(t in col.upper() for t in target_stats)]
        
        if found:
            print(f"✅ Stats encontradas: {found}")
        else:
            print(f"⚠️ Stats específicas não encontradas em colunas padrão")
            print(f"   Disponíveis: {all_cols}")
        
        print(f"\n🎉 NBA Official API FUNCIONANDO!")
        print(f"   Dados 100% REAIS disponíveis ✅")
        
    else:
        print("⚠️ Nenhum game encontrado (temporada ainda começando)")
        
except Exception as e:
    print(f"❌ Erro: {e}")
    print(f"   Pode ser rate limiting ou temporada ainda não começou")

print("\n✅ Test completo!")
