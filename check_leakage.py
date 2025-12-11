import joblib

features = joblib.load('data/models/feature_names_v6.joblib')

print('='*60)
print('ANÁLISE DETALHADA DE FEATURES - ENSEMBLE V6')
print('='*60)
print(f'Total: {len(features)} features\n')

# TODAS as features
print('LISTA COMPLETA DE FEATURES:')
for i, f in enumerate(sorted(features), 1):
    print(f'{i:2}. {f}')

print()
print('='*60)
print('ANÁLISE DE LEAKAGE')
print('='*60)

# Features que podem ser leakage
suspect_keywords = ['pts', 'score', 'win', 'efg_pct', 'ortg', 'drtg']
print('\n🔍 Features com keywords suspeitas:')
for f in sorted(features):
    for kw in suspect_keywords:
        if kw in f.lower() and 'rolling' not in f.lower():
            print(f'   ⚠️ {f} (contém "{kw}")')
            break
