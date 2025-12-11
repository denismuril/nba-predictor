import sys
import os
import time
import statistics

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation import simular_monte_carlo

def test_simulation(name, nr_home, nr_away, expected_range=None, hca=3.0):
    print(f"\n--- Teste: {name} ---")
    print(f"Net Rating Casa: {nr_home}")
    print(f"Net Rating Visitante: {nr_away}")
    print(f"HCA Aplicado: {hca}")
    
    start_time = time.time()
    prob_home = simular_monte_carlo(0.5, nr_home, nr_away, iterations=300000, hca_value=hca)
    end_time = time.time()
    
    elapsed = end_time - start_time
    
    print(f"Probabilidade Casa (Monte Carlo): {prob_home:.2f}%")
    print(f"Tempo de execução (300k iterações): {elapsed:.4f} segundos")
    
    if expected_range:
        min_p, max_p = expected_range
        if min_p <= prob_home <= max_p:
            print("✅ Resultado dentro do esperado.")
        else:
            print(f"❌ Resultado FORA do esperado ({min_p}-{max_p}%).")
    return prob_home

def main():
    print("Iniciando verificação da Simulação Monte Carlo (300k iterações)...")
    
    # Caso 1: Times Iguais (Net Ratings iguais)
    # HCA = +3.0. Spread = +3.0. 
    # Normal(3, 13). P(X > 0).
    # Z = (0 - 3) / 13 = -0.2307
    # P(Z < 0.2307) approx 0.591 -> 59.1%
    test_simulation("Times Iguais (HCA favorável)", 0.0, 0.0, expected_range=(58.5, 59.7))

    # Caso 2: Casa Muito Forte vs Visitante Fraco
    # Casa +10, Vis -10. Diff = 20. Spread = 20 + 3 = 23.
    # Normal(23, 13). P(X > 0). Z = -23/13 = -1.769
    # P(Z < 1.769) approx 0.961 -> 96.1%
    test_simulation("Favorito Casa Forte", 10.0, -10.0, expected_range=(95.5, 96.8))

    # Caso 3: Zebra Casa Fraca
    # Casa -10, Vis +10. Diff = -20. Spread = -20 + 3 = -17.
    # Normal(-17, 13). P(X > 0). Z = -(-17)/13 = 1.307
    # P(X > 0) = P(Z > 1.307) = 1 - 0.9044 = 0.0956 -> 9.56%
    test_simulation("Zebra Casa Fraca", -10.0, 10.0, expected_range=(9.0, 10.5))

    # Caso 4: HCA Alto (Ex: Denver)
    # Net Ratings Iguais, HCA 5.0
    # Spread = 5.0. Normal(5, 13). Z = -5/13 = -0.3846
    # P(Z < 0.3846) = 0.65 -> 65%
    test_simulation("Denver Altitude (HCA 5.0)", 0.0, 0.0, expected_range=(64.5, 65.5), hca=5.0)

    # Caso 5: HCA Baixo (Ex: Lakers)
    # Net Ratings Iguais, HCA 1.0
    # Spread = 1.0. Normal(1, 13). Z = -1/13 = -0.0769
    # P(Z < 0.0769) = 0.53 -> 53%
    test_simulation("Lakers Crowd (HCA 1.0)", 0.0, 0.0, expected_range=(52.5, 53.5), hca=1.0)

    print("\nVerificação concluída.")

if __name__ == "__main__":
    main()
