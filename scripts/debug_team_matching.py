from ml_pipeline.predict import predict_next_games
from data.scrapers.injury_scraper import get_injuries_with_cache
from config.constants import TEAM_ABBREV_MAP

print("=== CHECKING TEAM NAME MATCHING ===\n")

# Get raw injuries
injuries_raw = get_injuries_with_cache()
print(f"Injuries keys (team names): {list(injuries_raw.keys())}\n")

# Get predictions
df = predict_next_games()
print(f"Teams in predictions:")
for _, row in df.iterrows():
    print(f"  {row['home_team']} vs {row['away_team']}")

print(f"\n=== TEAM_ABBREV_MAP ===")
print(f"Sample entries:")
for i, (full, abbr) in enumerate(TEAM_ABBREV_MAP.items()):
    if i < 3:
        print(f"  '{full}' -> '{abbr}'")

print(f"\n=== REVERSE MAP (abbrev_to_full) ===")
abbrev_to_full = {v: k for k, v in TEAM_ABBREV_MAP.items()}
print(f"MIN -> '{abbrev_to_full.get('MIN', 'NOT FOUND')}'")
print(f"PHO -> '{abbrev_to_full.get('PHO', 'NOT FOUND')}'")
print(f"IND -> '{abbrev_to_full.get('IND', 'NOT FOUND')}'")

print(f"\n=== MATCHING TEST ===")
for team_abbr in ['MIN', 'PHO', 'IND', 'SAC', 'NOP', 'SAS']:
    team_full = abbrev_to_full.get(team_abbr, team_abbr)
    print(f"{team_abbr} -> '{team_full}'")
    
    # Check if in injuries
    if team_full in injuries_raw:
        print(f"  ✅ MATCH! {injuries_raw[team_full]}")
    elif team_abbr in injuries_raw:
        print(f"  ✅ ABBR MATCH! {injuries_raw[team_abbr]}")
    else:
        print(f"  ❌ NO MATCH")
