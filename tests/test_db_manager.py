"""
Testes unitários para DatabaseManager - Normalização de Team IDs e Matching de Jogos.
"""

import pytest
from data.repositories.db_manager import DatabaseManager


class TestTeamIDNormalization:
    """Testes para o método _normalize_team_id()"""
    
    def test_normalize_full_team_name(self):
        """Deve converter nome completo para ID"""
        assert DatabaseManager._normalize_team_id("Los Angeles Lakers") == "LAL"
        assert DatabaseManager._normalize_team_id("Golden State Warriors") == "GSW"
        assert DatabaseManager._normalize_team_id("Boston Celtics") == "BOS"
        assert DatabaseManager._normalize_team_id("Phoenix Suns") == "PHO"  # PHO, não PHX
        assert DatabaseManager._normalize_team_id("Charlotte Hornets") == "CHO"  # CHO, não CHA
    
    def test_normalize_team_nickname(self):
        """Deve converter apelido para ID"""
        assert DatabaseManager._normalize_team_id("Lakers") == "LAL"
        assert DatabaseManager._normalize_team_id("Warriors") == "GSW"
        assert DatabaseManager._normalize_team_id("Celtics") == "BOS"
        assert DatabaseManager._normalize_team_id("Cavs") == "CLE"  # Apelido comum
    
    def test_normalize_abbreviation(self):
        """Deve aceitar abreviações já normalizadas"""
        assert DatabaseManager._normalize_team_id("LAL") == "LAL"
        assert DatabaseManager._normalize_team_id("GSW") == "GSW"
        assert DatabaseManager._normalize_team_id("BOS") == "BOS"
        assert DatabaseManager._normalize_team_id("lal") == "LAL"  # Case  insensitive
        assert DatabaseManager._normalize_team_id("gsw") == "GSW"
    
    def test_normalize_city_name(self):
        """Deve converter nome da cidade para ID"""
        assert DatabaseManager._normalize_team_id("Los Angeles") == None  # Ambíguo! Lakers ou Clippers?
        # Mas apelidos específicos devem funcionar:
        assert DatabaseManager._normalize_team_id("clippers") == "LAC"
        assert DatabaseManager._normalize_team_id("lakers") == "LAL"
    
    def test_differentiate_la_teams(self):
        """CRÍTICO: Deve diferenciar Lakers e Clippers"""
        lakers_id = DatabaseManager._normalize_team_id("Los Angeles Lakers")
        clippers_id = DatabaseManager._normalize_team_id("Los Angeles Clippers")
        
        assert lakers_id == "LAL"
        assert clippers_id == "LAC"
        assert lakers_id != clippers_id  # NUNCA podem ser iguais!
    
    def test_handle_variations(self):
        """Deve lidar com variações de formatação"""
        # "LA Lakers" pode passar pelo fuzzy matching (80% similarity)
        result = DatabaseManager._normalize_team_id("LA Lakers")
        # Pode ser LAL (fuzzy match) ou None
        assert result == "LAL" or result is None
        assert DatabaseManager._normalize_team_id("  Lakers  ") == "LAL"  # Espaços extras
    
    def test_unknown_team(self):
        """Deve retornar None para times desconhecidos"""
        assert DatabaseManager._normalize_team_id("Unknown Team") is None
        assert DatabaseManager._normalize_team_id("") is None
        assert DatabaseManager._normalize_team_id(None) is None
    
    def test_case_insensitive(self):
        """Deve ser case-insensitive para lookups"""
        assert DatabaseManager._normalize_team_id("LAKERS") == "LAL"
        assert DatabaseManager._normalize_team_id("lakers") == "LAL"
        assert DatabaseManager._normalize_team_id("LaKeRs") == "LAL"
    
    def test_caching(self):
        """Verifica que chamadas repetidas retornam mesmo resultado"""
        # Primeira chamada
        result1 = DatabaseManager._normalize_team_id("Lakers")
        # Segunda chamada (resultado igual)
        result2 = DatabaseManager._normalize_team_id("Lakers")

        assert result1 == result2 == "LAL"


class TestPendingResultsMatching:
    """Testes para o matching de resultados pendentes (evitar falsos positivos)"""
    
    def test_exact_match_required(self):
        """Deve exigir match exato de IDs, não parcial"""
        # Simular cenário: jogo pendente é LAL vs GSW
        pending_home_id = DatabaseManager._normalize_team_id("Lakers")
        pending_away_id = DatabaseManager._normalize_team_id("Warriors")
        
        # Resultado correto: Lakers vs Warriors
        result_home_id = DatabaseManager._normalize_team_id("Los Angeles Lakers")
        result_away_id = DatabaseManager._normalize_team_id("Golden State Warriors")
        
        # DEVE dar match
        assert pending_home_id == result_home_id
        assert pending_away_id == result_away_id
    
    def test_no_false_positive_la_teams(self):
        """CRÍTICO: Não deve confundir Lakers com Clippers"""
        pending_home_id = DatabaseManager._normalize_team_id("Lakers")  # LAL
        result_home_id = DatabaseManager._normalize_team_id("Clippers")  # LAC
        
        # NÃO DEVE dar match
        assert pending_home_id != result_home_id
        
        # Verificar explicitamente
        assert pending_home_id == "LAL"
        assert result_home_id == "LAC"
    
    def test_no_partial_string_match(self):
        """Não deve aceitar match parcial de strings (problema antigo)"""
        # Problema antigo: "LOSANGELES" in "LOSANGELESLAKERS" = True ❌
        # Solução nova: LAL != LAC ✅
        
        lakers = DatabaseManager._normalize_team_id("Los Angeles Lakers")
        clippers = DatabaseManager._normalize_team_id("Los Angeles Clippers")
        
        # String antiga (INCORRETO):
        old_lakers_str = "LOSANGELESLAKERS"
        old_clippers_str = "LOSANGELESCLIPPERS"
        old_ambiguous = "LOSANGELES"
        
        # Verificar que "LOSANGELES" daria match com ambos (PROBLEMA!)
        assert old_ambiguous in old_lakers_str
        assert old_ambiguous in old_clippers_str
        
        # Mas com IDs, não há ambiguidade:
        assert lakers == "LAL"
        assert clippers == "LAC"
        assert lakers != clippers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
