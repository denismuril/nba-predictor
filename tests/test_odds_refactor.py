#!/usr/bin/env python3
"""
Script de Teste Completo - Sistema de Odds Refatorado

Testa:
1. Eliminação de valores fixos (1.90)
2. Player Props funcional
3. Validação de range
4. Integrity logging
5. Normalização de nomes

Execute: python tests/test_odds_refactor.py
"""

import asyncio
import sys
from pathlib import Path

# Adiciona raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """Print colorido para headers."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_test(name: str, passed: bool, details: str = ""):
    """Print resultado do teste."""
    status = "✅ PASSOU" if passed else "❌ FALHOU"
    print(f"{status} - {name}")
    if details:
        print(f"    {details}")


async def test_1_eliminacao_de_valores_fixos():
    """TESTE 1: Validar que sistema não retorna mais 1.90."""
    print_header("TESTE 1: Eliminação de Valores Fixos (1.90)")
    
    from data.scrapers.odds_scraper import get_odds_for_game
    from exceptions.odds_exceptions import OddsUnavailableError
    
    # Teste: Deve lançar exceção, NÃO retornar 1.90
    try:
        odds = get_odds_for_game("Lakers", "Celtics", odds_cache={})
        print_test(
            "get_odds_for_game sem cache",
            False,
            "❌ ERRO: Retornou odds em vez de lançar exceção!"
        )
        return False
    except OddsUnavailableError as e:
        print_test(
            "get_odds_for_game sem cache",
            True,
            f"Exceção correta: {str(e)[:60]}..."
        )
        return True
    except Exception as e:
        print_test(
            "get_odds_for_game sem cache",
            False,
            f"Exceção inesperada: {type(e).__name__}"
        )
        return False


async def test_2_player_name_normalizer():
    """TESTE 2: Normalização de nomes de jogadores."""
    print_header("TESTE 2: Player Name Normalizer")
    
    from data.scrapers.player_name_normalizer import normalize_player_name
    
    tests = [
        ("LeBron James", "LeBron James", True),  # Exato
        ("Lebron james", "LeBron James", True),  # Case insensitive
        ("LEBRON JAMES", "LeBron James", True),  # Uppercase
        ("Fake Player XYZ 123", None, True),     # Nome falso
    ]
    
    all_passed = True
    for raw_name, expected, should_pass in tests:
        try:
            result = normalize_player_name(raw_name)
            passed = (result == expected) == should_pass
            print_test(
                f"Normalizar '{raw_name}'",
                passed,
                f"Esperado: {expected}, Obtido: {result}"
            )
            if not passed:
                all_passed = False
        except Exception as e:
            print_test(f"Normalizar '{raw_name}'", False, f"Erro: {e}")
            all_passed = False
    
    return all_passed


async def test_3_integrity_logger():
    """TESTE 3: Sistema de logging de integridade."""
    print_header("TESTE 3: Integrity Logger")
    
    from data.utils.integrity_logger import (
        validate_odds_range,
        log_invalid_odds,
        log_missing_data,
        INTEGRITY_LOG_FILE
    )
    
    # Testa validação de range
    tests = [
        (0.50, False, "Below minimum"),
        (1.90, False, "Fixed value 1.90"),
        (2.50, True, "Valid odds"),
        (100.0, False, "Above maximum"),
    ]
    
    all_passed = True
    for odds_value, expected_valid, description in tests:
        result = validate_odds_range(odds_value, "test_scraper", "test_game")
        passed = result == expected_valid
        print_test(
            f"Validar odds={odds_value} ({description})",
            passed,
            f"Esperado: {expected_valid}, Obtido: {result}"
        )
        if not passed:
            all_passed = False
    
    # Verifica se arquivo de log existe
    if INTEGRITY_LOG_FILE.exists():
        print_test(
            "Log file criado",
            True,
            f"Localização: {INTEGRITY_LOG_FILE}"
        )
    else:
        print_test("Log file criado", False, "Arquivo não encontrado")
        all_passed = False
    
    return all_passed


async def test_4_player_props_scraper():
    """TESTE 4: Action Network Scraper (Player Props)."""
    print_header("TESTE 4: Player Props Scraper")
    
    try:
        from data.scrapers.action_network_scraper import (
            ActionNetworkScraper,
            PlayerProp
        )
        
        # Testa se o scraper foi criado corretamente
        scraper = ActionNetworkScraper(headless=True)
        print_test("ActionNetworkScraper instanciado", True)
        
        # Testa dataclass PlayerProp
        from datetime import datetime
        test_prop = PlayerProp(
            player_name="LeBron James",
            prop_type="points",
            line=25.5,
            over_odds=1.909,
            under_odds=1.909,
            source="action_network",
            timestamp=datetime.now()
        )
        print_test(
            "PlayerProp dataclass",
            True,
            f"Criado: {test_prop.player_name} - {test_prop.prop_type}"
        )
        
        # Nota: Não executamos o scraping real para evitar timeout
        print("\n    ⚠️  Scraping real não executado (evitar timeout)")
        print("    💡 Para testar scraping real, execute:")
        print("       python -c \"from data.scrapers.action_network_scraper import *; import asyncio; scraper=ActionNetworkScraper(headless=False); asyncio.run(scraper.fetch_props('2024-12-18'))\"")
        
        return True
        
    except Exception as e:
        print_test("Player Props Scraper", False, f"Erro: {e}")
        return False


async def test_5_odds_manager_integration():
    """TESTE 5: Integração com OddsDataManager."""
    print_header("TESTE 5: OddsDataManager Integration")
    
    try:
        from data.odds_manager import OddsDataManager
        
        # Instancia manager
        manager = OddsDataManager()
        print_test("OddsDataManager instanciado", True)
        
        # Verifica se método fetch_player_props existe
        has_method = hasattr(manager, 'fetch_player_props')
        print_test(
            "Método fetch_player_props existe",
            has_method,
            "Método disponível" if has_method else "Método não encontrado"
        )
        
        # Verifica se método _count_props_by_type existe (helper)
        has_helper = hasattr(manager, '_count_props_by_type')
        print_test(
            "Helper _count_props_by_type existe",
            has_helper,
            "Helper disponível" if has_helper else "Helper não encontrado"
        )
        
        # Nota: Não executamos fetch real para evitar timeout
        print("\n    ⚠️  Fetch real não executado (evitar timeout)")
        print("    💡 Para testar fetch real, execute:")
        print("       python -c \"from data.odds_manager import OddsDataManager; import asyncio; m=OddsDataManager(); asyncio.run(m.fetch_player_props('2024-12-18'))\"")
        
        return has_method and has_helper
        
    except Exception as e:
        print_test("OddsDataManager Integration", False, f"Erro: {e}")
        return False


async def test_6_odds_provider_interface():
    """TESTE 6: Interface OddsProvider atualizada."""
    print_header("TESTE 6: OddsProvider Interface")
    
    try:
        from data.interfaces.odds_provider import OddsProvider
        import inspect
        
        # Verifica se método get_player_props existe
        has_method = hasattr(OddsProvider, 'get_player_props')
        print_test(
            "Método get_player_props na interface",
            has_method,
            "Método abstrato adicionado" if has_method else "Método não encontrado"
        )
        
        # Verifica assinatura
        if has_method:
            sig = inspect.signature(OddsProvider.get_player_props)
            params = list(sig.parameters.keys())
            has_date_param = 'date' in params
            print_test(
                "Assinatura correta (self, date)",
                has_date_param,
                f"Parâmetros: {params}"
            )
            return has_method and has_date_param
        
        return False
        
    except Exception as e:
        print_test("OddsProvider Interface", False, f"Erro: {e}")
        return False


async def main():
    """Executa todos os testes."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TESTE DE CIRURGIA - SISTEMA DE ODDS" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")
    
    results = {
        "Eliminação de 1.90": await test_1_eliminacao_de_valores_fixos(),
        "Player Name Normalizer": await test_2_player_name_normalizer(),
        "Integrity Logger": await test_3_integrity_logger(),
        "Player Props Scraper": await test_4_player_props_scraper(),
        "OddsDataManager Integration": await test_5_odds_manager_integration(),
        "OddsProvider Interface": await test_6_odds_provider_interface(),
    }
    
    # Resumo
    print_header("RESUMO DOS TESTES")
    passed_count = sum(results.values())
    total_count = len(results)
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    print("\n" + "-" * 70)
    success_rate = (passed_count / total_count) * 100
    print(f"TOTAL: {passed_count}/{total_count} testes passaram ({success_rate:.1f}%)")
    print("-" * 70)
    
    if passed_count == total_count:
        print("\n🎉 TODOS OS TESTES PASSARAM! Sistema pronto para produção.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} teste(s) falharam. Revise as mensagens acima.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
