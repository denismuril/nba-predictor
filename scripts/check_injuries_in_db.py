from data.repositories.db_manager import get_db_manager

db = get_db_manager()
df = db.get_latest_predictions('2025-12-08')

print('\n=== INJURY DATA ===')
print(df[['home_team', 'away_team', 'home_injuries_list', 'away_injuries_list']].to_string())

if not df.empty:
    print('\n=== SAMPLE VALUE ===')
    row = df.iloc[0]
    print(f'{row.home_team}: {row.home_injuries_list}')
    print(f'{row.away_team}: {row.away_injuries_list}')
