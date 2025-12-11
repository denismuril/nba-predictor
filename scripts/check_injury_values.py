from ml_pipeline.predict import predict_next_games

df = predict_next_games()

print("=== INJURY DATA FROM PREDICTIONS ===\n")
for _, row in df.iterrows():
    print(f"{row['home_team']} vs {row['away_team']}")
    print(f"   Home injuries: '{row.get('home_injuries_list', 'COLUMN MISSING')}'")
    print(f"   Away injuries: '{row.get('away_injuries_list', 'COLUMN MISSING')}'")
    print()
