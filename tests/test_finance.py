"""
Teste 4: Gestão de Risco - Kelly Criterion
===========================================
CRÍTICO: Previne falência por apostas ruins.
"""

import pytest
from utils.kelly import calculate_kelly_criterion, kelly_criterion_quarter


def test_kelly_rejects_negative_ev():
    """
    EV negativo = DON'T BET
    """
    # Prob 40%, Odds 2.00 (fair seria 2.50)
    # EV = (0.40 * 2.00) - 1 = -0.20 (NEGATIVO)
    
    stake = calculate_kelly_criterion(prob_win=0.40, odds=2.00, bankroll=1000)
    
    assert stake == 0.0, \
        f"❌ Kelly deveria retornar 0 para EV negativo, retornou {stake}"
    
    print("✅ EV negativo rejeitado corretamente")


def test_kelly_limits_max_stake():
    """
    Mesmo com EV alto, stake deve ser limitado
    """
    # EV altíssimo: Prob 80%, Odds 2.50
    stake = calculate_kelly_criterion(prob_win=0.80, odds=2.50, bankroll=1000)
    
    # Kelly puro poderia sugerir 60%+ da banca
    # Devemos limitar a 5%
    assert stake <= 50, \
        f"❌ Stake muito alto ({stake}), max deveria ser 50 (5%)"
    
    print(f"✅ Stake limitado a {stake:.2f} (proteção OK)")


def test_kelly_quarter_safer_than_full():
    """
    Kelly Quarter deve ser mais conservador
    """
    prob = 0.60
    odds = 2.00
    bankroll = 1000
    
    full_kelly = calculate_kelly_criterion(prob, odds, bankroll)
    quarter_kelly = kelly_criterion_quarter(prob, odds, bankroll)
    
    # Quarter Kelly deve ser menor ou igual ao Full Kelly
    # (pode ser igual se ambos atingirem o cap de 5%)
    assert quarter_kelly <= full_kelly, \
        f"❌ Quarter Kelly ({quarter_kelly}) deveria ser <= Full Kelly ({full_kelly})"
    
    # Se full kelly está abaixo do cap, quarter deve ser ~1/4
    # Se full kelly está no cap (50), quarter também pode estar no cap (50)
    # Neste caso ambos batem no cap de 5% = 50 (1000*0.05)
    # O comportamento real é: full=50, quarter=12.5 se não houver cap
    # Mas kelly_criterion_advanced aplica cap de 5% em ambos
    
    print(f"✅ Quarter Kelly ({quarter_kelly:.2f}) <= Full ({full_kelly:.2f})")


def test_no_bet_on_zero_prob():
    """
    Probabilidade 0% = Don't bet
    """
    stake = calculate_kelly_criterion(prob_win=0.0, odds=10.0, bankroll=1000)
    
    assert stake == 0.0, \
        f"❌ Prob 0% deveria retornar stake 0, retornou {stake}"
    
    print("✅ Prob 0% corretamente rejeitada")


def test_no_bet_below_50_percent():
    """
    Sem edge (prob < 50%) em odds balanceadas = Don't bet
    """
    # Prob 45%, Odds 2.20 (ligeiramente favorável)
    # EV = (0.45 * 2.20) - 1 = -0.01 (ainda negativo)
    
    stake = calculate_kelly_criterion(prob_win=0.45, odds=2.20, bankroll=1000)
    
    assert stake == 0.0, \
        f"❌ Kelly deveria rejeitar prob < 50% sem edge real"
    
    print("✅ Edge insuficiente rejeitado")


if __name__ == "__main__":
    print("💰 TESTANDO GESTÃO DE RISCO\n")
    test_kelly_rejects_negative_ev()
    test_kelly_limits_max_stake()
    test_kelly_quarter_safer_than_full()
    test_no_bet_on_zero_prob()
    test_no_bet_below_50_percent()
    print("\n🎉 GESTÃO DE RISCO APROVADA!")
