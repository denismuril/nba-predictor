"""
Sistema de Elo Ratings para NBA Predictor
==========================================

Implementa power ratings estilo Elo (usado por FiveThirtyEight, Vegas)
para capturar força relativa dos times em um único número.

Feature Adicionada: elo_diff (diferença Elo home - away)
Esta é a feature mais preditiva isoladamente em modelagem NBA.

Autor: NBA Predictor Team
Data: 2025-12-03
Padrão: Vegas/Pinnacle
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Constantes Elo
ELO_INICIAL = 1500  # Rating médio da liga
K_FACTOR_BASE = 20  # Taxa de aprendizado base
HCA_ELO = 70        # Vantagem de casa em pontos Elo (~2.1 pontos no spread) - NBA Moderna
B2B_PENALTY = 50    # Penalidade para times em Back-to-Back (~1.5 pontos)


class NBAEloSystem:
    """
    Sistema Elo para NBA com ajustes por margin of victory e home court advantage.
    
    Baseado em:
    - FiveThirtyEight NBA Elo
    - Nate Silver's methodology
    - Vegas power ratings
    """
    
    def __init__(self, k_factor=20, hca=HCA_ELO, elo_floor=1000, elo_ceiling=2000):
        """
        Args:
            k_factor: Taxa de aprendizado (maior = mais volátil)
            hca: Home Court Advantage em pontos Elo
            elo_floor: Rating mínimo permitido
            elo_ceiling: Rating máximo permitido
        """
        self.k_factor = k_factor
        self.hca = hca
        self.elo_floor = elo_floor
        self.elo_ceiling = elo_ceiling
        self.ratings = {}  # {team: elo_rating}
        
    def inicializar_times(self, teams):
        """Inicializa todos os times com rating 1500."""
        for team in teams:
            self.ratings[team] = ELO_INICIAL
        logger.info(f"✅ {len(teams)} times inicializados com Elo {ELO_INICIAL}")
    
    def calcular_vitoria_esperada(self, elo_time, elo_oponente, is_home=False, is_b2b=False):
        """
        Calcula probabilidade de vitória baseada em diferença Elo.
        
        Formula: P(win) = 1 / (1 + 10^((Elo_Opp - Elo_Team) / 400))
        
        Args:
            elo_time: Elo do time
            elo_oponente: Elo do oponente
            is_home: Se o time está jogando em casa
            is_b2b: Se o time está em Back-to-Back (jogou ontem)
            
        Returns:
            Probabilidade de vitória (0-1)
        """
        # Ajuste HCA (Home Court Advantage)
        elo_ajustado = elo_time + (self.hca if is_home else 0)
        
        # Ajuste B2B (Penalidade por fadiga) - NBA Moderna: ~1.5 pts de desvantagem
        if is_b2b:
            elo_ajustado -= B2B_PENALTY
        
        diff = elo_oponente - elo_ajustado
        
        win_prob = 1 / (1 + 10 ** (diff / 400))
        
        return win_prob
    
    def multiplicador_margin(self, margin, elo_vencedor, elo_perdedor):
        """
        Ajusta K-factor baseado em margin of victory.
        
        Goleadas (margin > 15) = maior mudança de rating
        Jogos apertados (margin < 5) = menor mudança
        
        Formula: MOV_mult = ln(abs(margin) + 1) / ln(10) * autocorrelação
        
        Autocorrelação penaliza goleadas quando Elo favorito já era muito maior.
        """
        # Componente de margin logarítmico (diminui retornos marginais)
        mov_component = np.log(abs(margin) + 1) / np.log(10)
        
        # Autocorrelação: Se favorito ganha por muito, reduz impacto
        # (evita inflar Elo de times que já são dominantes)
        elo_diff = abs(elo_vencedor - elo_perdedor)
        
        if elo_diff > 200:  # Favorito muito forte
            autocorr = 0.8  # Reduz impacto em 20%
        else:
            autocorr = 1.0
        
        return mov_component * autocorr
    
    def atualizar_elo(
        self, 
        vencedor: str, 
        perdedor: str, 
        margin: int,
        vencedor_era_home: bool
    ) -> Tuple[float, float]:
        """
        Atualiza ratings Elo após um jogo.
        
        Args:
            vencedor: Time vencedor
            perdedor: Time perdedor
            margin: Margem de vitória (pontos)
            vencedor_era_home: Se o vencedor jogou em casa
            
        Returns:
            (novo_elo_vencedor, novo_elo_perdedor)
        """
        elo_venc = self.ratings.get(vencedor, ELO_INICIAL)
        elo_perd = self.ratings.get(perdedor, ELO_INICIAL)
        
        # Calcular vitória esperada (do ponto de vista do vencedor)
        vitoria_esperada = self.calcular_vitoria_esperada(
            elo_venc, 
            elo_perd, 
            is_home=vencedor_era_home
        )
        
        # Multiplicador de margin
        mov_mult = self.multiplicador_margin(margin, elo_venc, elo_perd)
        
        # Calcular ajuste Elo
        # Resultado real = 1 (vitória), Esperado = vitoria_esperada
        ajuste = self.k_factor * mov_mult * (1 - vitoria_esperada)
        
        # Atualizar ratings
        novo_elo_venc = elo_venc + ajuste
        novo_elo_perd = elo_perd - ajuste
        
        # Aplicar floor/ceiling
        novo_elo_venc = np.clip(novo_elo_venc, self.elo_floor, self.elo_ceiling)
        novo_elo_perd = np.clip(novo_elo_perd, self.elo_floor, self.elo_ceiling)
        
        # Salvar
        self.ratings[vencedor] = novo_elo_venc
        self.ratings[perdedor] = novo_elo_perd
        
        logger.debug(
            f"   {vencedor} {elo_venc:.0f} → {novo_elo_venc:.0f} (+{ajuste:.1f}) | "
            f"{perdedor} {elo_perd:.0f} → {novo_elo_perd:.0f} (-{ajuste:.1f})"
        )
        
        return novo_elo_venc, novo_elo_perd
    
    def regressao_temporada(self, regressao_pct=0.25):
        """
        Aplica regressão à média entre temporadas.
        
        No início de cada temporada, times regridem 25% em direção à média (1500).
        Isso evita que ratings antigos dominem previsões da nova temporada.
        
        Args:
            regressao_pct: Quanto regredir (0.25 = 25% em direção à média)
        """
        for team in self.ratings:
            elo_atual = self.ratings[team]
            elo_novo = elo_atual * (1 - regressao_pct) + ELO_INICIAL * regressao_pct
            self.ratings[team] = elo_novo
        
        logger.info(f"✅ Regressão de temporada aplicada ({regressao_pct*100:.0f}%)")
    
    def get_rating(self, team):
        """Retorna rating Elo de um time."""
        return self.ratings.get(team, ELO_INICIAL)
    
    def prever_jogo(self, home_team, away_team, home_is_b2b=False, away_is_b2b=False):
        """
        Prevê resultado de um jogo baseado em Elo.
        
        Args:
            home_team: Time da casa
            away_team: Time visitante
            home_is_b2b: Se o time da casa está em Back-to-Back
            away_is_b2b: Se o time visitante está em Back-to-Back
        
        Returns:
            Dict com prob_home, prob_away, spread_implícito, ajustes B2B
        """
        elo_home = self.get_rating(home_team)
        elo_away = self.get_rating(away_team)
        
        prob_home = self.calcular_vitoria_esperada(
            elo_home, elo_away, is_home=True, is_b2b=home_is_b2b
        )
        prob_away = self.calcular_vitoria_esperada(
            elo_away, elo_home, is_home=False, is_b2b=away_is_b2b
        )
        
        # Normalizar probabilidades (podem não somar 1.0 devido aos ajustes B2B)
        total_prob = prob_home + prob_away
        prob_home = prob_home / total_prob
        prob_away = prob_away / total_prob
        
        # Spread implícito: ~0.03 pontos por ponto Elo
        # Inclui ajustes de HCA e B2B no cálculo
        elo_home_adjusted = elo_home + self.hca - (B2B_PENALTY if home_is_b2b else 0)
        elo_away_adjusted = elo_away - (B2B_PENALTY if away_is_b2b else 0)
        elo_diff_adjusted = elo_home_adjusted - elo_away_adjusted
        spread_implicito = elo_diff_adjusted * 0.03
        
        return {
            'prob_home': prob_home,
            'prob_away': prob_away,
            'spread_implicito': spread_implicito,
            'elo_home': elo_home,
            'elo_away': elo_away,
            'elo_diff': elo_home - elo_away,
            'home_is_b2b': home_is_b2b,
            'away_is_b2b': away_is_b2b
        }
    
    def salvar(self, filepath='data/models/elo_ratings.pkl'):
        """Salva ratings Elo em arquivo pickle."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.ratings, filepath)
        logger.info(f"💾 Elo ratings salvos em {filepath}")
    
    @classmethod
    def carregar(cls, filepath='data/models/elo_ratings.pkl', **kwargs):
        """Carrega ratings Elo de arquivo."""
        sistema = cls(**kwargs)
        
        filepath = Path(filepath)
        if filepath.exists():
            sistema.ratings = joblib.load(filepath)
            logger.info(f"📂 Elo ratings carregados de {filepath} ({len(sistema.ratings)} times)")
        else:
            logger.warning(f"⚠️ Arquivo {filepath} não encontrado. Ratings vazios.")
        
        return sistema


# ==============================================================================
# INTEGRAÇÃO COM PIPELINE DE FEATURES
# ==============================================================================

def calcular_elo_ratings_historico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Elo ratings para todo o histórico de jogos.
    
    IMPORTANTE: Chama esta função ANTES de criar rolling features.
    
    Args:
        df: DataFrame com colunas:
            - date, home_team, away_team, home_score, away_score
            
    Returns:
        DataFrame com colunas adicionadas:
            - home_elo, away_elo, elo_diff
    """
    logger.info("🏀 Calculando Elo Ratings para histórico completo...")
    
    # Ordenar por data (CRÍTICO!)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Inicializar sistema Elo
    sistema = NBAEloSystem(k_factor=20, hca=HCA_ELO)
    
    # Obter todos os times
    todos_times = set(df['home_team'].unique()) | set(df['away_team'].unique())
    sistema.inicializar_times(todos_times)
    
    # Arrays para armazenar Elo de cada jogo
    home_elos = []
    away_elos = []
    elo_diffs = []
    
    # Iterar por cada jogo
    temporada_anterior = None
    
    for idx, row in df.iterrows():
        # Detectar mudança de temporada (aplicar regressão)
        if 'season' in df.columns:
            temporada_atual = row['season']
            if temporada_anterior and temporada_atual != temporada_anterior:
                logger.info(f"🔄 Nova temporada: {temporada_atual}. Aplicando regressão...")
                sistema.regressao_temporada(regressao_pct=0.25)
            temporada_anterior = temporada_atual
        
        home_team = row['home_team']
        away_team = row['away_team']
        
        # ANTES do jogo: salvar Elo atual
        elo_home_pre = sistema.get_rating(home_team)
        elo_away_pre = sistema.get_rating(away_team)
        
        home_elos.append(elo_home_pre)
        away_elos.append(elo_away_pre)
        elo_diffs.append(elo_home_pre - elo_away_pre)
        
        # DEPOIS do jogo: atualizar Elo
        if pd.notna(row['home_score']) and pd.notna(row['away_score']):
            home_score = int(row['home_score'])
            away_score = int(row['away_score'])
            
            if home_score > away_score:
                vencedor = home_team
                perdedor = away_team
                margin = home_score - away_score
                vencedor_home = True
            else:
                vencedor = away_team
                perdedor = home_team
                margin = away_score - home_score
                vencedor_home = False
            
            sistema.atualizar_elo(vencedor, perdedor, margin, vencedor_home)
    
    # Adicionar ao DataFrame
    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    df['elo_diff'] = elo_diffs
    
    logger.info(f"✅ Elo ratings calculados para {len(df)} jogos")
    logger.info(f"   Range Elo Home: {min(home_elos):.0f} - {max(home_elos):.0f}")
    logger.info(f"   Range Elo Away: {min(away_elos):.0f} - {max(away_elos):.0f}")
    
    # Salvar ratings finais
    sistema.salvar()
    
    return df


if __name__ == '__main__':
    # Demo rápido
    logging.basicConfig(level=logging.INFO)
    
    print("🏀 Sistema Elo NBA - Demo\n")
    
    # Simular alguns jogos
    jogos_demo = pd.DataFrame({
        'date': pd.date_range('2024-10-22', periods=10),
        'home_team': ['LAL', 'BOS', 'DEN', 'GSW', 'MIA'] * 2,
        'away_team': ['BOS', 'LAL', 'GSW', 'DEN', 'BOS'] * 2,
        'home_score': [110, 105, 120, 115, 108, 112, 98, 125, 105, 110],
        'away_score': [108, 110, 115, 112, 110, 108, 105, 118, 110, 108]
    })
    
    df_com_elo = calcular_elo_ratings_historico(jogos_demo)
    
    print("\n📊 Jogos com Elo:")
    print(df_com_elo[['home_team', 'away_team', 'home_elo', 'away_elo', 'elo_diff']].head(5))
    
    print("\n✅ Sistema Elo implementado com sucesso!")
