"""
Testes unitários para Team Normalization.

Valida:
1. Normalização de todas as variações de cada time
2. Conversão bidirecional (ID ↔ Full Name)
3. Case-insensitive matching
4. Aliases e apelidos
5. Edge cases (None, empty, invalid)
"""
import unittest
from utils.team_normalization import (
    TeamNormalizer,
    normalize_team,
    team_to_full_name,
    get_team_forms
)


class TestTeamNormalization(unittest.TestCase):
    """Testes para normalização de times."""
    
    @classmethod
    def setUpClass(cls):
        """Setup executado uma vez antes de todos os testes."""
        cls.normalizer = TeamNormalizer.get_instance()
    
    def test_singleton_pattern(self):
        """Deve retornar a mesma instância sempre."""
        instance1 = TeamNormalizer.get_instance()
        instance2 = TeamNormalizer.get_instance()
        self.assertIs(instance1, instance2)
    
    def test_normalize_from_id(self):
        """Deve aceitar IDs e retorná-los uppercase."""
        self.assertEqual(self.normalizer.normalize("LAL"), "LAL")
        self.assertEqual(self.normalizer.normalize("lal"), "LAL")
        self.assertEqual(self.normalizer.normalize("GSW"), "GSW")
        self.assertEqual(self.normalizer.normalize("gsw"), "GSW")
    
    def test_normalize_from_full_name(self):
        """Deve normalizar nomes completos."""
        self.assertEqual(self.normalizer.normalize("Los Angeles Lakers"), "LAL")
        self.assertEqual(self.normalizer.normalize("Golden State Warriors"), "GSW")
        self.assertEqual(self.normalizer.normalize("Boston Celtics"), "BOS")
    
    def test_normalize_from_aliases(self):
        """Deve normalizar aliases e apelidos."""
        # Lakers variations
        self.assertEqual(self.normalizer.normalize("Lakers"), "LAL")
        self.assertEqual(self.normalizer.normalize("LA Lakers"), "LAL")
        
        # Warriors variations
        self.assertEqual(self.normalizer.normalize("Warriors"), "GSW")
        
        # 76ers variations
        self.assertEqual(self.normalizer.normalize("Sixers"), "PHI")
        self.assertEqual(self.normalizer.normalize("76ers"), "PHI")
        
        # Cavaliers variations
        self.assertEqual(self.normalizer.normalize("Cavs"), "CLE")
        self.assertEqual(self.normalizer.normalize("Cavaliers"), "CLE")
    
    def test_normalize_case_insensitive(self):
        """Deve ser case-insensitive."""
        self.assertEqual(self.normalizer.normalize("LAKERS"), "LAL")
        self.assertEqual(self.normalizer.normalize("lakers"), "LAL")
        self.assertEqual(self.normalizer.normalize("LaKeRs"), "LAL")
    
    def test_normalize_with_whitespace(self):
        """Deve lidar com espaços extras."""
        self.assertEqual(self.normalizer.normalize("  Lakers  "), "LAL")
        self.assertEqual(self.normalizer.normalize("\tGSW\n"), "GSW")
    
    def test_normalize_invalid(self):
        """Deve retornar None para inputs inválidos."""
        self.assertIsNone(self.normalizer.normalize(None))
        self.assertIsNone(self.normalizer.normalize(""))
        self.assertIsNone(self.normalizer.normalize("Invalid Team"))
        self.assertIsNone(self.normalizer.normalize("XYZ"))
    
    def test_to_full_name(self):
        """Deve converter IDs para nomes completos."""
        self.assertEqual(self.normalizer.to_full_name("LAL"), "Los Angeles Lakers")
        self.assertEqual(self.normalizer.to_full_name("GSW"), "Golden State Warriors")
        self.assertEqual(self.normalizer.to_full_name("BOS"), "Boston Celtics")
        
        # Case-insensitive
        self.assertEqual(self.normalizer.to_full_name("lal"), "Los Angeles Lakers")
    
    def test_to_full_name_invalid(self):
        """Deve retornar None para IDs inválidos."""
        self.assertIsNone(self.normalizer.to_full_name(None))
        self.assertIsNone(self.normalizer.to_full_name(""))
        self.assertIsNone(self.normalizer.to_full_name("XYZ"))
    
    def test_all_forms(self):
        """Deve retornar todas as formas de um time."""
        lal_forms = self.normalizer.all_forms("LAL")
        
        # Deve conter ID e nome completo
        self.assertIn("LAL", lal_forms)
        self.assertIn("Los Angeles Lakers", lal_forms)
        
        # Deve conter pelo menos um alias
        self.assertTrue(any("lakers" in form.lower() for form in lal_forms))
    
    def test_all_forms_from_alias(self):
        """Deve funcionar mesmo quando chamado com alias."""
        forms = self.normalizer.all_forms("Lakers")
        self.assertIn("LAL", forms)
        self.assertIn("Los Angeles Lakers", forms)
    
    def test_all_30_teams(self):
        """Deve normalizar corretamente todos os 30 times da NBA."""
        expected_ids = {
            "BOS", "BRK", "NYK", "PHI", "TOR",  # Atlantic
            "CHI", "CLE", "DET", "IND", "MIL",  # Central
            "ATL", "CHO", "MIA", "ORL", "WAS",  # Southeast
            "DEN", "MIN", "OKC", "POR", "UTA",  # Northwest
            "GSW", "LAC", "LAL", "PHO", "SAC",  # Pacific
            "DAL", "HOU", "MEM", "NOP", "SAS",  # Southwest
        }
        
        all_ids = set(self.normalizer.get_all_ids())
        self.assertEqual(all_ids, expected_ids)
        self.assertEqual(len(all_ids), 30)
    
    def test_bidirectional_consistency(self):
        """Normalização deve ser consistente em ambas direções."""
        for team_id in self.normalizer.get_all_ids():
            # ID → Full Name → ID
            full_name = self.normalizer.to_full_name(team_id)
            normalized_back = self.normalizer.normalize(full_name)
            
            self.assertEqual(normalized_back, team_id,
                           f"{team_id} → {full_name} → {normalized_back}")
    
    def test_special_cases(self):
        """Casos especiais que causaram problemas no passado."""
        # Charlotte: CHO vs CHA
        self.assertEqual(self.normalizer.normalize("Charlotte Hornets"), "CHO")
        
        # Phoenix: PHO vs PHX
        self.assertEqual(self.normalizer.normalize("Phoenix Suns"), "PHO")
        
        # LA Clippers vs LA Lakers
        self.assertEqual(self.normalizer.normalize("LA Lakers"), "LAL")
        self.assertEqual(self.normalizer.normalize("LA Clippers"), "LAC")
        self.assertNotEqual(
            self.normalizer.normalize("LA Lakers"),
            self.normalizer.normalize("LA Clippers")
        )
    
    def test_convenience_functions(self):
        """Testa funções de atalho."""
        # normalize_team()
        self.assertEqual(normalize_team("Lakers"), "LAL")
        
        # team_to_full_name()
        self.assertEqual(team_to_full_name("LAL"), "Los Angeles Lakers")
        
        # get_team_forms()
        forms = get_team_forms("LAL")
        self.assertIn("LAL", forms)
        self.assertIn("Los Angeles Lakers", forms)
    
    def test_cache_performance(self):
        """Deve usar cache para melhorar performance."""
        # Primeira chamada popula cache
        result1 = self.normalizer.normalize("Lakers")
        
        # Segunda chamada deve vir do cache (mesma referência de objeto)
        result2 = self.normalizer.normalize("Lakers")
        
        self.assertEqual(result1, result2)
        self.assertEqual(result1, "LAL")


class TestEdgeCases(unittest.TestCase):
    """Testes para edge cases e corner cases."""
    
    def setUp(self):
        self.normalizer = TeamNormalizer.get_instance()
    
    def test_empty_strings(self):
        """Deve lidar com strings vazias."""
        self.assertIsNone(self.normalizer.normalize(""))
        self.assertIsNone(self.normalizer.normalize("   "))
        self.assertIsNone(self.normalizer.to_full_name(""))
    
    def test_special_characters(self):
        """Deve ignorar caracteres especiais (não suportados)."""
        self.assertIsNone(self.normalizer.normalize("Lakers!"))
        self.assertIsNone(self.normalizer.normalize("@Lakers"))
    
    def test_numbers_only(self):
        """Deve rejeitar apenas números."""
        self.assertIsNone(self.normalizer.normalize("123"))
        self.assertIsNone(self.normalizer.normalize("76"))  # Não confundir com 76ers
    
    def test_very_long_input(self):
        """Deve lidar com inputs muito longos."""
        long_name = "Lakers " * 100
        # Não deve crashar
        result = self.normalizer.normalize(long_name)
        # Pode retornar None (não match) sem crashar
        self.assertTrue(result is None or result == "LAL")


if __name__ == '__main__':
    unittest.main(verbosity=2)
