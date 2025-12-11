"""
Demo e teste do Confidence-Adjusted Kelly Criterion

Simula apostas com dados sintéticos para validar implementação.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from betting.confidence_kelly import ConfidenceKelly, backtest_kelly

# Gerar dados sintéticos realistas
np.random.seed(42)
n_games = 200

print("🎲 Gerando dados sintéticos...")
print("="*60)

# Simular um modelo com performance realista
true_win_prob = 0.58  # 58% accuracy real
predicted_probs = np.clip(np.random.normal(true_win_prob, 0.08, n_games), 0.1, 0.9)

# Simular odds de mercado (inverse da prob + margin)
market_margin = 1.05  # 5% margin
odds = (1 / predicted_probs) * market_margin

# Simular resultados reais
results = np.random.binomial(1, predicted_probs)

# Simular confidence scores (varying)
confidence_scores = np.random.uniform(0.6, 0.9, n_games)

# Criar DataFrame
df = pd.DataFrame({
    'prob': predicted_probs,
    'odds': odds,
    'result': results,
    'confidence': confidence_scores
})

print(f"✅ {n_games} jogos gerados")
print(f"   Win Rate Real: {results.mean():.1%}")
print(f"   Odds Médio: {odds.mean():.2f}")
print("")

# Test 1: Baseline (sem Kelly, flat bet)
print("📊 Test 1: Baseline (Flat Betting)")
print("-"*60)

initial_bankroll = 1000
flat_bet_pct = 0.02  # 2% flat
bankroll_flat = initial_bankroll

for idx, row in df.iterrows():
    bet_size = bankroll_flat * flat_bet_pct
    
    if row['result'] == 1:
        bankroll_flat += bet_size * (row['odds'] - 1)
    else:
        bankroll_flat -= bet_size

profit_flat = bankroll_flat - initial_bankroll
roi_flat = (profit_flat / initial_bankroll) * 100

print(f"Final Bankroll: ${bankroll_flat:.2f}")
print(f"Profit: ${profit_flat:.2f} ({roi_flat:+.1f}%)")
print("")

# Test 2: Confidence-Adjusted Kelly
print("📊 Test 2: Confidence-Adjusted Kelly")
print("-"*60)

kelly = ConfidenceKelly(
    fraction=0.25,      # Quarter Kelly
    min_edge=0.02,      # 2% edge mínimo
    max_bet_pct=0.05,   # 5% max
    min_confidence=0.6  # 60% confidence mínimo
)

results_kelly = backtest_kelly(df, kelly, initial_bankroll=1000)

print("")

# Comparação
print("📊 COMPARAÇÃO")
print("="*60)
print(f"{'Método':<20} {'ROI':>10} {'Final':>15} {'Sharpe':>10}")
print("-"*60)
print(f"{'Flat Betting':<20} {roi_flat:>9.1f}% ${bankroll_flat:>13.2f} {'N/A':>10}")
print(f"{'Kelly (Adjusted)':<20} {results_kelly['roi_pct']:>9.1f}% ${results_kelly['final_bankroll']:>13.2f} {results_kelly['sharpe_ratio']:>9.2f}")
print("="*60)

improvement = results_kelly['roi_pct'] - roi_flat
print(f"\n✨ Melhoria: {improvement:+.1f}% ROI")
print(f"✨ Max Drawdown Kelly: {results_kelly['max_drawdown_pct']:.1f}%")
print(f"✨ Win Rate: {results_kelly['win_rate']:.1f}%")
print(f"✨ Apostas: {results_kelly['total_bets']} placed, {results_kelly['skipped']} skipped")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bankroll evolution
ax1.plot(results_kelly['bankroll_history'], label='Kelly Adjusted', linewidth=2)
ax1.axhline(y=initial_bankroll, color='red', linestyle='--', alpha=0.5, label='Initial')
ax1.set_xlabel('Bets')
ax1.set_ylabel('Bankroll ($)')
ax1.set_title('Bankroll Evolution - Kelly Criterion')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Bet size distribution
if len(results_kelly['bets']) > 0:
    bet_sizes = [b['bet_pct'] * 100 for b in results_kelly['bets']]
    ax2.hist(bet_sizes, bins=20, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Bet Size (% of Bankroll)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Bet Size Distribution')
    ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reports/kelly_demo.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Gráficos salvos em: reports/kelly_demo.png")

print("\n✅ Demo completo!")
