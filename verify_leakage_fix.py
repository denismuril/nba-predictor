#!/usr/bin/env python3
"""Script de verificação das correções de Data Leakage."""

from ml_pipeline.data_preparation import load_historical_data, prepare_data_for_training

print("=" * 60)
print("🔍 VERIFICANDO CORREÇÕES DE DATA LEAKAGE")
print("=" * 60)

# Carregar dados
print("\n📦 Carregando dados históricos...")
df = load_historical_data()
print(f"   Total colunas no DataFrame: {len(df.columns)}")

# Preparar para treino (usa whitelist)
print("\n🔒 Aplicando whitelist de features...")
X, y = prepare_data_for_training(df)
print(f"   Features após whitelist: {len(X.columns)}")

# Verificar se colunas perigosas foram removidas
dangerous_cols = ['home_efg', 'away_efg', 'home_score', 'away_score', 
                  'home_off_rating', 'away_off_rating', 'fgm', 'fga', 'pts',
                  'home_tov_pct', 'away_tov_pct']
found_dangerous = [c for c in dangerous_cols if c in X.columns]

print("\n" + "=" * 60)
if found_dangerous:
    print(f"❌ LEAK DETECTADO: {found_dangerous}")
else:
    print("✅ WHITELIST FUNCIONANDO!")
    print("   Nenhuma coluna de estatísticas do jogo atual encontrada!")
print("=" * 60)

# Mostrar algumas features selecionadas
print("\n📋 Exemplo de features permitidas:")
for col in sorted(X.columns)[:15]:
    print(f"   ✓ {col}")
print(f"   ... e mais {len(X.columns) - 15} features")
