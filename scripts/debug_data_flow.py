import sys
sys.path.insert(0, '/mnt/wsl/Ubuntu/home/denis/nba-predictor')

from data.repositories.db_manager import get_db_manager
from ml_pipeline.predict import predict_next_games

# Simular o que cli.py faz
print("=== STEP 1: Predict ML ===")
ml_df = predict_next_games()
print(f"Columns: {ml_df.columns.tolist()}")
print(f"\nInjury columns exist? home_injuries_list: {'home_injuries_list' in ml_df.columns}")
print(f"Injury columns exist? away_injuries_list: {'away_injuries_list' in ml_df.columns}")

print("\n=== STEP 2: Create ml_lookup (como cli.py) ===")
ml_lookup = {}
for _, row in ml_df.iterrows():
    ml_lookup[(row['home_team'], row['away_team'])] = row.to_dict()

print(f"ml_lookup keys: {list(ml_lookup.keys())}")
first_game = list(ml_lookup.values())[0]
print(f"\nFirst game dict keys: {list(first_game.keys())}")
print(f"home_injuries_list in dict? {'home_injuries_list' in first_game}")
print(f"home_injuries_list value: '{first_game.get('home_injuries_list', 'KEY MISSING')}'")

print("\n=== STEP 3: Create previsao dict (como cli.py) ===")
ml_data = first_game
previsao = {
    "Data": "2025-12-08",
    "Casa": first_game['home_team'],
    "Visitante": first_game['away_team'],
}

# SEMPRE adicionar colunas de lesões
if ml_data:
    previsao['home_injuries_list'] = ml_data.get('home_injuries_list', '')
    previsao['away_injuries_list'] = ml_data.get('away_injuries_list', '')
else:
    previsao['home_injuries_list'] = ''
    previsao['away_injuries_list'] = ''

print(f"previsao keys: {list(previsao.keys())}")
print(f"previsao['home_injuries_list']: '{previsao['home_injuries_list']}'")
print(f"previsao['away_injuries_list']: '{previsao['away_injuries_list']}'")

print("\n=== STEP 4: Simulate save_predictions ===")
print(f"Sending to DB: home_injuries_list = '{previsao.get('home_injuries_list', '')}'")
print(f"Sending to DB: away_injuries_list = '{previsao.get('away_injuries_list', '')}'")
