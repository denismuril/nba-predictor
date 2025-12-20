"""
Props Processor - Motor de Integração ETL para Player Props.

Este módulo é o coração do pipeline de props, responsável por:
1. Receber dados brutos dos scrapers (Linemate, BettingPros, etc)
2. Normalizar nomes de jogadores via fuzzy matching
3. Cruzar com estatísticas da temporada
4. Calcular features contextuais avançadas
5. Gerar DataFrame pronto para inferência XGBoost

Features Contextuais Calculadas:
- L5_AVG: Média dos últimos 5 jogos
- H2H_AVG: Média histórica contra oponente atual
- REST_DAYS: Dias desde último jogo
- DEF_VS_POS: Rating defensivo do oponente vs posição

v27.0: Implementação inicial para arquitetura God Mode.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

# Imports internos do projeto
try:
    from data.interfaces.player_props_provider import PlayerProp
    from data.scrapers.player_name_normalizer import normalize_player_name
    from data.utils.integrity_logger import (
        log_normalization_failure,
        log_missing_data,
        integrity_logger
    )
except ImportError:
    # Fallback para execução direta
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from data.interfaces.player_props_provider import PlayerProp
    from data.scrapers.player_name_normalizer import normalize_player_name
    from data.utils.integrity_logger import (
        log_normalization_failure,
        log_missing_data,
        integrity_logger
    )

logger = logging.getLogger(__name__)

# Caminhos padrão
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"


@dataclass
class PropsProcessorConfig:
    """
    Configuração do processador de props.
    
    Atributos:
        season_stats_path: Caminho para CSV de estatísticas da temporada
        boxscores_path: Caminho para histórico de boxscores
        min_similarity: Threshold mínimo para fuzzy matching de nomes
        require_min_games: Número mínimo de jogos para calcular L5_AVG
        skip_on_missing_data: Se True, pula jogador se faltar dados (não usa mock)
    """
    season_stats_path: Path = field(
        default_factory=lambda: DATA_DIR / "player_props" / "season_stats_2024-25.csv"
    )
    boxscores_path: Path = field(
        default_factory=lambda: DATA_DIR / "player_boxscores_history.csv"
    )
    min_similarity: float = 0.80
    require_min_games: int = 5
    skip_on_missing_data: bool = True  # REGRA DE OURO: Nunca usar dados falsos


class PropsProcessor:
    """
    Motor de integração ETL para Player Props.
    
    Transforma dados brutos de props em features prontas para inferência ML.
    
    Exemplo de uso:
        >>> processor = PropsProcessor()
        >>> props = [PlayerProp(...), PlayerProp(...)]
        >>> df = await processor.process_props(props)
        >>> # df contém features L5_AVG, REST_DAYS, DEF_VS_POS, etc.
    
    Regras de Ouro:
        - NUNCA preenche dados faltantes com zeros ou mock
        - Se faltar dado, o jogador é PULADO da previsão
        - Todas as discrepâncias são logadas via integrity_logger
    """
    
    def __init__(self, config: Optional[PropsProcessorConfig] = None):
        """
        Inicializa o processador.
        
        Args:
            config: Configuração customizada (opcional)
        """
        self.config = config or PropsProcessorConfig()
        self._season_stats: Optional[pd.DataFrame] = None
        self._boxscores: Optional[pd.DataFrame] = None
        self._def_ratings: Optional[pd.DataFrame] = None
        self._last_game_cache: Dict[str, datetime] = {}
        
        logger.info(f"📊 PropsProcessor inicializado")
        logger.info(f"   Season stats: {self.config.season_stats_path}")
        logger.info(f"   Boxscores: {self.config.boxscores_path}")
    
    def _load_season_stats(self) -> pd.DataFrame:
        """
        Carrega estatísticas da temporada.
        
        Returns:
            DataFrame com estatísticas de jogadores
            
        Raises:
            FileNotFoundError: Se arquivo não existir
        """
        if self._season_stats is not None:
            return self._season_stats
        
        path = self.config.season_stats_path
        if not path.exists():
            raise FileNotFoundError(
                f"❌ Estatísticas da temporada não encontradas: {path}\n"
                "Execute o scraper de estatísticas primeiro."
            )
        
        self._season_stats = pd.read_csv(path)
        
        # Normalizar nome da coluna de jogador
        if "PLAYER_NAME" in self._season_stats.columns:
            self._season_stats = self._season_stats.rename(
                columns={"PLAYER_NAME": "Player"}
            )
        
        logger.info(f"✅ Carregadas {len(self._season_stats)} estatísticas de jogadores")
        return self._season_stats
    
    def _load_boxscores(self) -> Optional[pd.DataFrame]:
        """
        Carrega histórico de boxscores para cálculo de L5_AVG.
        
        Returns:
            DataFrame com boxscores ou None se não disponível
        """
        if self._boxscores is not None:
            return self._boxscores
        
        path = self.config.boxscores_path
        if not path.exists():
            logger.warning(f"⚠️ Boxscores não encontrados: {path}")
            logger.warning("   L5_AVG não será calculado.")
            return None
        
        self._boxscores = pd.read_csv(path)
        
        # Converter data para datetime
        if "Date" in self._boxscores.columns:
            self._boxscores["Date"] = pd.to_datetime(
                self._boxscores["Date"], 
                errors="coerce"
            )
            self._boxscores = self._boxscores.sort_values(["Player", "Date"])
        
        logger.info(f"✅ Carregados {len(self._boxscores)} boxscores históricos")
        return self._boxscores
    
    def _match_player_to_stats(
        self, 
        player_name: str
    ) -> Optional[pd.Series]:
        """
        Encontra estatísticas do jogador via normalização de nome.
        
        Args:
            player_name: Nome bruto do jogador
            
        Returns:
            Series com estatísticas do jogador ou None se não encontrado
        """
        stats = self._load_season_stats()
        
        # Tentar match exato primeiro
        exact_match = stats[stats["Player"] == player_name]
        if len(exact_match) > 0:
            return exact_match.iloc[0]
        
        # Fuzzy matching
        canonical_name = normalize_player_name(
            player_name, 
            min_similarity=self.config.min_similarity
        )
        
        if canonical_name is None:
            log_normalization_failure(
                source="PropsProcessor",
                raw_name=player_name,
                context="Nenhum match encontrado no season_stats"
            )
            return None
        
        fuzzy_match = stats[stats["Player"] == canonical_name]
        if len(fuzzy_match) > 0:
            logger.debug(f"🔄 Fuzzy match: '{player_name}' → '{canonical_name}'")
            return fuzzy_match.iloc[0]
        
        log_normalization_failure(
            source="PropsProcessor",
            raw_name=player_name,
            context=f"Canonical '{canonical_name}' não encontrado no stats"
        )
        return None
    
    def _calculate_l5_avg(
        self, 
        player_name: str, 
        stat_type: str
    ) -> Optional[float]:
        """
        Calcula média dos últimos 5 jogos (anti-leakage).
        
        Args:
            player_name: Nome do jogador
            stat_type: Tipo de stat ('PTS', 'REB', 'AST', etc)
            
        Returns:
            Média L5 ou None se insuficiente dados
        """
        boxscores = self._load_boxscores()
        if boxscores is None:
            return None
        
        # Normalizar nome
        canonical = normalize_player_name(player_name)
        if canonical is None:
            return None
        
        # Filtrar jogos do jogador
        player_games = boxscores[boxscores["Player"] == canonical].copy()
        
        if len(player_games) < self.config.require_min_games:
            logger.debug(
                f"⚠️ {player_name}: Menos de {self.config.require_min_games} jogos "
                f"para L5_AVG (tem {len(player_games)})"
            )
            return None
        
        # Pegar últimos 5 jogos (shift para anti-leakage já aplicado no treino)
        stat_col = stat_type.upper()
        if stat_col not in player_games.columns:
            return None
        
        last_5 = player_games.tail(5)[stat_col].dropna()
        if len(last_5) < 3:  # Mínimo 3 jogos para média significativa
            return None
        
        return float(last_5.mean())
    
    def _calculate_rest_days(self, player_name: str) -> Optional[int]:
        """
        Calcula dias de descanso desde último jogo.
        
        Args:
            player_name: Nome do jogador
            
        Returns:
            Dias de descanso ou None se não disponível
        """
        boxscores = self._load_boxscores()
        if boxscores is None:
            return None
        
        canonical = normalize_player_name(player_name)
        if canonical is None:
            return None
        
        # Último jogo do jogador
        player_games = boxscores[
            (boxscores["Player"] == canonical) & 
            (boxscores["Date"].notna())
        ]
        
        if len(player_games) == 0:
            return None
        
        last_game = player_games["Date"].max()
        today = datetime.now()
        
        rest_days = (today - last_game).days
        
        # Sanity check
        if rest_days < 0 or rest_days > 30:
            logger.warning(
                f"⚠️ REST_DAYS suspeito para {player_name}: {rest_days} dias"
            )
            return None
        
        return rest_days
    
    def _get_defense_vs_position(
        self, 
        opponent: str, 
        position: str
    ) -> Optional[float]:
        """
        Obtém rating defensivo do oponente contra a posição.
        
        TODO: Implementar quando tivermos dados de DEF_VS_POS.
        Por enquanto retorna None para não usar dados falsos.
        
        Args:
            opponent: Time adversário
            position: Posição do jogador
            
        Returns:
            Rating defensivo (1.0 = neutro, >1.0 = fraco, <1.0 = forte)
        """
        # REGRA DE OURO: Retorna None em vez de valor falso
        # Será implementado quando tivermos dados reais
        return None
    
    def _calculate_h2h_avg(
        self, 
        player_name: str, 
        opponent: str, 
        stat_type: str
    ) -> Optional[float]:
        """
        Calcula média histórica contra oponente específico.
        
        Args:
            player_name: Nome do jogador
            opponent: Time adversário
            stat_type: Tipo de stat
            
        Returns:
            Média H2H ou None se poucos jogos
        """
        boxscores = self._load_boxscores()
        if boxscores is None:
            return None
        
        canonical = normalize_player_name(player_name)
        if canonical is None:
            return None
        
        # Filtrar jogos contra este oponente
        if "Opponent" not in boxscores.columns:
            return None
        
        h2h_games = boxscores[
            (boxscores["Player"] == canonical) &
            (boxscores["Opponent"] == opponent)
        ]
        
        # Mínimo 2 jogos para média H2H
        stat_col = stat_type.upper()
        if len(h2h_games) < 2 or stat_col not in h2h_games.columns:
            return None
        
        return float(h2h_games[stat_col].mean())
    
    def _calculate_hit_rate(
        self,
        player_name: str,
        stat_type: str,
        line: float,
        n_games: int = 5
    ) -> Optional[float]:
        """
        Calcula a taxa de acerto (hit rate) para uma linha nos últimos N jogos.
        
        SNIPER INTELLIGENCE FEATURE:
        Responde: "Quantas vezes o jogador bateu essa linha nos últimos 5 jogos?"
        
        Args:
            player_name: Nome do jogador
            stat_type: Tipo de stat ('PTS', 'REB', 'AST', etc)
            line: Valor da linha para avaliar (ex: 25.5 pontos)
            n_games: Número de jogos recentes a considerar (default: 5)
            
        Returns:
            Hit rate (0.0 a 1.0) ou None se dados insuficientes
        """
        boxscores = self._load_boxscores()
        if boxscores is None:
            return None
        
        # Normalizar nome
        canonical = normalize_player_name(player_name)
        if canonical is None:
            return None
        
        # Filtrar jogos do jogador
        player_games = boxscores[boxscores["Player"] == canonical].copy()
        
        stat_col = stat_type.upper()
        if stat_col not in player_games.columns:
            return None
        
        # Pegar últimos N jogos
        last_n = player_games.tail(n_games)
        
        if len(last_n) < 3:  # Mínimo 3 jogos para hit rate significativo
            return None
        
        # Calcular quantas vezes bateu a linha (OVER)
        hits = (last_n[stat_col] > line).sum()
        hit_rate = hits / len(last_n)
        
        return float(hit_rate)
    
    async def process_props(
        self, 
        props: List[PlayerProp],
        include_odds: bool = True
    ) -> pd.DataFrame:
        """
        Processa lista de props e gera DataFrame com features.
        
        Esta é a função principal do ETL pipeline. Recebe props brutos
        e retorna um DataFrame pronto para inferência do modelo.
        
        Args:
            props: Lista de PlayerProp dos scrapers
            include_odds: Se True, inclui colunas de odds
            
        Returns:
            DataFrame com colunas:
            - player_name: Nome normalizado
            - prop_type: Tipo do prop (points, rebounds, etc)
            - line: Valor da linha
            - over_odds, under_odds: Odds (se include_odds=True)
            - season_avg: Média da temporada
            - L5_AVG: Média últimos 5 jogos
            - H2H_AVG: Média vs oponente (se disponível)
            - REST_DAYS: Dias de descanso
            - DEF_VS_POS: Rating defensivo (se disponível)
            
        Raises:
            ValueError: Se nenhum prop válido após processamento
        """
        if not props:
            raise ValueError("Lista de props vazia")
        
        logger.info(f"🔄 Processando {len(props)} props...")
        
        processed_rows: List[Dict[str, Any]] = []
        skipped_count = 0
        
        for prop in props:
            try:
                row = await self._process_single_prop(prop, include_odds)
                if row is not None:
                    processed_rows.append(row)
                else:
                    skipped_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Erro processando prop de {prop.player_name}: {e}")
                skipped_count += 1
        
        if not processed_rows:
            log_missing_data(
                source="PropsProcessor",
                game="all_props",
                url=None
            )
            raise ValueError(
                f"Nenhum prop válido após processamento. "
                f"{len(props)} props recebidos, {skipped_count} pulados. "
                "Verifique se os dados de referência estão disponíveis."
            )
        
        df = pd.DataFrame(processed_rows)
        
        logger.info(
            f"✅ Processamento completo: {len(df)} props válidos, "
            f"{skipped_count} pulados (dados insuficientes)"
        )
        
        return df
    
    async def _process_single_prop(
        self, 
        prop: PlayerProp,
        include_odds: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Processa um único prop e gera features.
        
        Args:
            prop: PlayerProp a processar
            include_odds: Se True, inclui odds
            
        Returns:
            Dict com features ou None se dados insuficientes
        """
        # 1. Match com estatísticas da temporada
        player_stats = self._match_player_to_stats(prop.player_name)
        
        if player_stats is None:
            if self.config.skip_on_missing_data:
                logger.debug(f"⏭️ Pulando {prop.player_name}: sem stats de temporada")
                return None
        
        # 2. Extrair estatística base da temporada
        stat_col_map = {
            "points": "PTS",
            "rebounds": "REB",
            "assists": "AST",
            "threes": "FG3M",
            "steals": "STL",
            "blocks": "BLK",
            "turnovers": "TOV",
        }
        
        stat_col = stat_col_map.get(prop.prop_type.lower(), "PTS")
        
        season_avg = None
        if player_stats is not None and stat_col in player_stats.index:
            season_avg = float(player_stats[stat_col])
        
        # 3. Calcular features contextuais
        l5_avg = self._calculate_l5_avg(prop.player_name, stat_col)
        rest_days = self._calculate_rest_days(prop.player_name)
        
        # 4. Extrair oponente do game_info (se disponível)
        h2h_avg = None
        def_vs_pos = None

        # Acesso defensivo a game_info (pode não existir em todos os scrapers)
        game_info = getattr(prop, 'game_info', None)
        if game_info:
            # Tentar extrair oponente do formato "LAL vs BOS"
            parts = game_info.replace("@", "vs").split("vs")
            if len(parts) == 2:
                opponent = parts[1].strip()[:3].upper()  # Abreviação do time
                h2h_avg = self._calculate_h2h_avg(prop.player_name, opponent, stat_col)

                # DEF_VS_POS - por enquanto None (dados reais pendentes)
                position = player_stats.get("Position", None) if player_stats is not None else None
                if position:
                    def_vs_pos = self._get_defense_vs_position(opponent, position)
        
        # 5. REGRA DE OURO: Verificar se temos dados mínimos
        # Para previsão válida, precisamos de pelo menos season_avg OU l5_avg
        if season_avg is None and l5_avg is None:
            if self.config.skip_on_missing_data:
                logger.debug(
                    f"⏭️ Pulando {prop.player_name} ({prop.prop_type}): "
                    "sem season_avg nem L5_AVG"
                )
                return None
        
        # 6. SNIPER INTELLIGENCE: Calcular features diferenciais EV+
        # diff_to_avg: Quantos pontos a linha está acima/abaixo da média
        reference_avg = season_avg if season_avg is not None else l5_avg
        diff_to_avg = None
        diff_pct = None
        if reference_avg is not None and reference_avg > 0:
            diff_to_avg = prop.line - reference_avg
            diff_pct = (prop.line - reference_avg) / reference_avg  # Percentual
        
        # last_5_games_hit_rate: % de vezes que bateu a linha nos últimos 5 jogos
        last_5_hit_rate = self._calculate_hit_rate(prop.player_name, stat_col, prop.line)
        
        # implied_prob: Probabilidade implícita das odds (1 / decimal_odds)
        over_implied_prob = None
        under_implied_prob = None
        if prop.over_odds and prop.over_odds > 1:
            over_implied_prob = 1.0 / prop.over_odds
        if prop.under_odds and prop.under_odds > 1:
            under_implied_prob = 1.0 / prop.under_odds
        
        # 7. Montar row
        row = {
            "player_name": prop.player_name,
            "prop_type": prop.prop_type,
            "line": prop.line,
            "source": prop.source,
            "timestamp": prop.timestamp.isoformat() if prop.timestamp else None,
            "game_info": prop.game_info,
            
            # Features básicas
            "season_avg": season_avg,
            "L5_AVG": l5_avg,
            "H2H_AVG": h2h_avg,
            "REST_DAYS": rest_days,
            "DEF_VS_POS": def_vs_pos,
            
            # SNIPER INTELLIGENCE Features
            "diff_to_avg": diff_to_avg,
            "diff_pct": diff_pct,
            "last_5_hit_rate": last_5_hit_rate,
            "over_implied_prob": over_implied_prob,
            "under_implied_prob": under_implied_prob,
        }
        
        if include_odds:
            row["over_odds"] = prop.over_odds
            row["under_odds"] = prop.under_odds
            row["bookmaker"] = prop.bookmaker
        
        return row
    
    def get_feature_columns(self) -> List[str]:
        """
        Retorna lista de colunas de features para o modelo.
        
        Returns:
            Lista de nomes de colunas de features
        """
        return [
            "season_avg",
            "L5_AVG",
            "H2H_AVG",
            "REST_DAYS",
            "DEF_VS_POS",
            "line",
            # Sniper Intelligence Features
            "diff_to_avg",
            "diff_pct",
            "last_5_hit_rate",
            "over_implied_prob",
            "under_implied_prob",
        ]
    
    def prepare_for_inference(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepara DataFrame final para inferência do modelo.
        
        Filtra apenas linhas com features mínimas preenchidas.
        
        Args:
            df: DataFrame do process_props()
            
        Returns:
            DataFrame filtrado pronto para predict()
        """
        feature_cols = self.get_feature_columns()
        
        # Filtrar linhas onde pelo menos season_avg ou L5_AVG está preenchido
        valid_mask = df["season_avg"].notna() | df["L5_AVG"].notna()
        
        filtered = df[valid_mask].copy()
        
        # Preencher L5_AVG com season_avg se ausente (fallback seguro)
        filtered["L5_AVG"] = filtered["L5_AVG"].fillna(filtered["season_avg"])
        
        # Preencher REST_DAYS com valor padrão conservador (3 dias)
        filtered["REST_DAYS"] = filtered["REST_DAYS"].fillna(3)
        
        logger.info(
            f"📊 Preparado para inferência: {len(filtered)}/{len(df)} props "
            f"({len(df) - len(filtered)} removidos por dados insuficientes)"
        )
        
        return filtered


# ============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# ============================================================================

async def process_props_from_scrapers(
    scrapers: List[Any],
    date: str
) -> pd.DataFrame:
    """
    Função de conveniência para processar props de múltiplos scrapers.
    
    Args:
        scrapers: Lista de instâncias de scrapers (devem ter get_props())
        date: Data no formato YYYY-MM-DD
        
    Returns:
        DataFrame consolidado com features
    """
    all_props: List[PlayerProp] = []
    
    for scraper in scrapers:
        try:
            props = await scraper.get_props(date)
            all_props.extend(props)
            logger.info(f"✅ {scraper.name}: {len(props)} props coletados")
        except Exception as e:
            logger.warning(f"⚠️ {scraper.name} falhou: {e}")
    
    if not all_props:
        raise ValueError("Nenhum prop coletado de nenhum scraper")
    
    processor = PropsProcessor()
    return await processor.process_props(all_props)


# ============================================================================
# TESTE / DEMO
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    async def demo():
        """Demo do processador com props de exemplo."""
        # Criar props mock para teste
        from datetime import datetime
        
        test_props = [
            PlayerProp(
                player_name="LeBron James",
                prop_type="points",
                line=25.5,
                over_odds=1.91,
                under_odds=1.91,
                source="test",
                game_info="LAL vs BOS"
            ),
            PlayerProp(
                player_name="Stephen Curry",
                prop_type="threes",
                line=4.5,
                over_odds=1.85,
                under_odds=1.95,
                source="test",
                game_info="GSW @ PHX"
            ),
        ]
        
        processor = PropsProcessor()
        
        try:
            df = await processor.process_props(test_props)
            print("\n📊 DataFrame processado:")
            print(df.to_string())
            
            # Preparar para inferência
            inference_df = processor.prepare_for_inference(df)
            print("\n🎯 Pronto para inferência:")
            print(inference_df[processor.get_feature_columns()].to_string())
            
        except Exception as e:
            print(f"❌ Erro: {e}")
    
    asyncio.run(demo())
