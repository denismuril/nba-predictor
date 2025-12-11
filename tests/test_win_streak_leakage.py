"""
Teste unitário para validar que win_streak NÃO vaza dados do jogo atual.

Objetivo: Garantir que a streak do Jogo N use APENAS resultados até o Jogo N-1.
"""
import pandas as pd
import numpy as np


def test_win_streak_no_leakage():
    """
    Teste crítico: Win streak do jogo N deve usar apenas dados até N-1.
    
    Cenário de teste:
    - Time A: W, W, W, L, W (jogos 1-5)
    - No jogo 5 (vitória), a streak deveria ser 0 (pois jogo 4 foi derrota)
    - Se a streak for 1, significa que está vendo o resultado do jogo 5 (LEAKAGE!)
    """
    # Simular histórico de um time
    test_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'team': ['TeamA'] * 5,
        'win': [1, 1, 1, 0, 1],  # W, W, W, L, W
        'is_home': [1, 1, 0, 1, 0]
    })
    
    # Aplicar a MESMA lógica de data_preparation.py:249-253
    long_df = test_data.copy()
    long_df['win_shifted'] = long_df.groupby('team')['win'].shift(1)
    long_df['win_streak'] = long_df.groupby('team')['win_shifted'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    ) * long_df['win_shifted'].fillna(0)
    
    print("\n" + "="*70)
    print("TEST: Win Streak Data Leakage")
    print("="*70)
    print("\nInput Data:")
    print(test_data[['date', 'team', 'win']].to_string(index=False))
    
    print("\nCalculated Win Streak:")
    print(long_df[['date', 'team', 'win', 'win_shifted', 'win_streak']].to_string(index=False))
    
    # VALIDAÇÕES CRÍTICAS
    print("\n" + "="*70)
    print("VALIDATIONS:")
    print("="*70)
    
    # Jogo 1: Primeira partida, sem histórico
    # Streak esperada: 0 (sem dados anteriores)
    assert long_df.iloc[0]['win_streak'] == 0, \
        f"❌ Jogo 1 deveria ter streak=0 (sem histórico), mas tem {long_df.iloc[0]['win_streak']}"
    print("✅ Jogo 1: streak=0 (correto, sem histórico)")
    
    # Jogo 2: Após 1ª vitória
    # Streak esperada: 1 (1 vitória anterior)
    assert long_df.iloc[1]['win_streak'] == 1, \
        f"❌ Jogo 2 deveria ter streak=1, mas tem {long_df.iloc[1]['win_streak']}"
    print("✅ Jogo 2: streak=1 (correto, 1 vitória anterior)")
    
    # Jogo 3: Após 2 vitórias
    # Streak esperada: 2
    assert long_df.iloc[2]['win_streak'] == 2, \
        f"❌ Jogo 3 deveria ter streak=2, mas tem {long_df.iloc[2]['win_streak']}"
    print("✅ Jogo 3: streak=2 (correto, 2 vitórias anteriores)")
    
    # Jogo 4: Após 3 vitórias (mas este jogo é derrota)
    # Streak esperada: 3 (olhando apenas para o passado, não sabe que vai perder)
    assert long_df.iloc[3]['win_streak'] == 3, \
        f"❌ Jogo 4 deveria ter streak=3, mas tem {long_df.iloc[3]['win_streak']}"
    print("✅ Jogo 4: streak=3 (correto, 3 vitórias anteriores)")
    
    # Jogo 5: Após derrota no jogo 4 (mas este jogo é vitória)
    # Streak esperada: 0 (a derrota anterior quebrou a streak)
    # Se streak=1, significa que está vendo a vitória atual (LEAKAGE!)
    expected_streak = 0
    actual_streak = long_df.iloc[4]['win_streak']
    
    print(f"\n🔍 TESTE CRÍTICO - Jogo 5:")
    print(f"   Jogo anterior: Derrota (win=0)")
    print(f"   Jogo atual: Vitória (win=1)")
    print(f"   Streak esperada (sem leakage): {expected_streak}")
    print(f"   Streak calculada: {actual_streak}")
    
    if actual_streak == 0:
        print("   ✅ PASSOU - Win streak NÃO está vazando!")
    elif actual_streak == 1:
        print("   ❌ FALHOU - Win streak está vazando o resultado do jogo atual!")
        raise AssertionError("DATA LEAKAGE DETECTADO: win_streak vê o resultado do jogo atual")
    else:
        print(f"   ⚠️ INESPERADO - Streak={actual_streak} (esperado 0 ou 1)")
    
    assert actual_streak == expected_streak, \
        f"❌ LEAKAGE: Jogo 5 deveria ter streak={expected_streak}, mas tem {actual_streak}"
    
    print("\n" + "="*70)
    print("✅ TODOS OS TESTES PASSARAM - Win Streak está LIVRE de vazamento!")
    print("="*70)
    

def test_win_streak_loss_after_wins():
    """
    Teste adicional: Sequência W-W-L-W
    """
    test_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=4),
        'team': ['TeamB'] * 4,
        'win': [1, 1, 0, 1],  # W, W, L, W
        'is_home': [1, 0, 1, 0]
    })
    
    long_df = test_data.copy()
    long_df['win_shifted'] = long_df.groupby('team')['win'].shift(1)
    long_df['win_streak'] = long_df.groupby('team')['win_shifted'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    ) * long_df['win_shifted'].fillna(0)
    
    print("\n" + "="*70)
    print("TEST 2: W-W-L-W Pattern")
    print("="*70)
    print(long_df[['date', 'team', 'win', 'win_shifted', 'win_streak']].to_string(index=False))
    
    # Jogo 4 (vitória após derrota): streak deveria ser 0
    assert long_df.iloc[3]['win_streak'] == 0, \
        f"Jogo 4 após derrota deveria ter streak=0, mas tem {long_df.iloc[3]['win_streak']}"
    print("✅ Jogo 4 após derrota: streak=0 (correto)")


if __name__ == "__main__":
    try:
        test_win_streak_no_leakage()
        test_win_streak_loss_after_wins()
        print("\n🎉 CONCLUSÃO: A lógica de win_streak está CORRETA (sem vazamento)")
    except AssertionError as e:
        print(f"\n💥 ERRO DETECTADO: {e}")
        print("\n❌ A lógica de win_streak precisa ser CORRIGIDA")
