#!/usr/bin/env python3
"""
Debug script para testar o fluxo completo de injuries
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.scrapers.injury_scraper import get_injuries_with_cache
from config.constants import TEAM_ABBREV_MAP
import pprint

print("=" * 80)
print("TESTE 1: get_injuries_with_cache() - Formato de Saída")
print("=" * 80)

injuries = get_injuries_with_cache()

print(f"\nTipo: {type(injuries)}")
print(f"Num teams com lesões: {len(injuries) if injuries else 0}")
print("\nConteúdo:")
pprint.pprint(injuries)

print("\n" + "=" * 80)
print("TESTE 2: Conversão de Abreviação para Nome Completo")
print("=" * 80)

abbrev_to_full = {v: k for k, v in TEAM_ABBREV_MAP.items()}

test_teams = ['MIN', 'PHO', 'IND', 'SAC', 'NOP', 'SAS']

for abbr in test_teams:
    full_name = abbrev_to_full.get(abbr, abbr)
    injuries_for_team = injuries.get(full_name, {})
    print(f"\n{abbr} → {full_name}")
    print(f"  Lesões encontradas: {len(injuries_for_team)}")
    if injuries_for_team:
        for player, status in injuries_for_team.items():
            print(f"    - {player}: {status}")

print("\n" + "=" * 80)
print("TESTE 3: Formato que format_injuries_for_team deve retornar")
print("=" * 80)

def format_injuries_for_team(team_name, injuries_dict):
    """Extract injured players for a team and format as string."""
    if not isinstance(injuries_dict, dict):
        return ""
    
    # Create reverse map
    abbrev_to_full = {v: k for k, v in TEAM_ABBREV_MAP.items()}
    team_full_name = abbrev_to_full.get(team_name, team_name)
    
    injuries_list = []
    for team_key, players in injuries_dict.items():
        if team_full_name == team_key or team_name == team_key:
            for player, status in players.items():
                injuries_list.append(f"{player} ({status})")
    
    return ", ".join(injuries_list) if injuries_list else ""

for abbr in test_teams:
    result = format_injuries_for_team(abbr, injuries)
    print(f"\n{abbr}: '{result}'")

print("\n✅ Teste concluído!")
