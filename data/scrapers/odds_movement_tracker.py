"""
Odds Movement Tracker - Captura Odds de Abertura e Fechamento

Este módulo rastreia a movimentação de odds ao longo do dia para detectar
"Smart Money" - quando profissionais movem as linhas.

Estratégia:
1. Captura odds de ABERTURA logo que disponíveis (manhã)
2. Captura odds de FECHAMENTO próximo ao início do jogo
3. Calcula line_movement e implied_prob_diff

Autor: NBA Predictor System
Data: 2025-12-05
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import requests

logger = logging.getLogger(__name__)


class OddsMovementTracker:
    """
    Rastreia movimentação de odds ao longo do dia.
    
    Math-Context:
    - Odds de ABERTURA: Primeira linha publicada pelas casas
    - Odds de FECHAMENTO: Última linha antes do jogo começar
    - line_movement: closing - opening
    - implied_prob_diff: (1/closing) - (1/opening)
    
    Quando implied_prob_diff > 0.03 (3%), há forte sinal de smart money.
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path('data/odds_tracking')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivo de tracking diário
        self.today_file = self.data_dir / f"odds_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Carregar dados do dia se existir
        self.today_data = self._load_today_data()
    
    def _load_today_data(self) -> Dict:
        """Carrega dados de odds do dia atual."""
        if self.today_file.exists():
            with open(self.today_file, 'r') as f:
                return json.load(f)
        return {'games': {}, 'snapshots': []}
    
    def _save_today_data(self):
        """Salva dados de odds do dia atual."""
        with open(self.today_file, 'w') as f:
            json.dump(self.today_data, f, indent=2)
    
    def record_odds_snapshot(self, odds_data: Dict[str, Dict]):
        """
        Registra um snapshot de odds atual.
        
        Args:
            odds_data: Dict no formato {
                'BOS_vs_LAL': {'home_odds': 1.85, 'away_odds': 2.00, ...},
                ...
            }
        """
        timestamp = datetime.now().isoformat()
        
        # Adicionar ao histórico de snapshots
        snapshot = {
            'timestamp': timestamp,
            'odds': odds_data
        }
        self.today_data['snapshots'].append(snapshot)
        
        # Atualizar dados por jogo
        for game_key, odds in odds_data.items():
            if game_key not in self.today_data['games']:
                # Primeiro registro = opening odds
                self.today_data['games'][game_key] = {
                    'opening_odds_home': odds.get('home_odds'),
                    'opening_odds_away': odds.get('away_odds'),
                    'opening_timestamp': timestamp,
                    'closing_odds_home': odds.get('home_odds'),
                    'closing_odds_away': odds.get('away_odds'),
                    'closing_timestamp': timestamp,
                    'snapshots_count': 1
                }
            else:
                # Atualizar closing (sempre a última)
                self.today_data['games'][game_key]['closing_odds_home'] = odds.get('home_odds')
                self.today_data['games'][game_key]['closing_odds_away'] = odds.get('away_odds')
                self.today_data['games'][game_key]['closing_timestamp'] = timestamp
                self.today_data['games'][game_key]['snapshots_count'] += 1
        
        self._save_today_data()
        logger.info(f"📊 Odds snapshot registrado: {len(odds_data)} jogos às {timestamp}")
    
    def get_odds_movement(self, game_key: str = None) -> Dict:
        """
        Calcula movimentação de odds.
        
        Args:
            game_key: Chave do jogo (ex: 'BOS_vs_LAL'). Se None, retorna todos.
            
        Returns:
            Dict com opening, closing, line_movement, implied_prob_diff
        """
        if game_key:
            game_data = self.today_data['games'].get(game_key)
            if not game_data:
                return {}
            return self._calculate_movement(game_data)
        
        # Retornar todos
        movements = {}
        for key, data in self.today_data['games'].items():
            movements[key] = self._calculate_movement(data)
        return movements
    
    def _calculate_movement(self, game_data: Dict) -> Dict:
        """Calcula métricas de movimentação para um jogo."""
        opening_home = game_data.get('opening_odds_home', 0)
        closing_home = game_data.get('closing_odds_home', 0)
        
        # Evitar divisão por zero
        if not opening_home or not closing_home:
            return {
                'line_movement': 0,
                'implied_prob_diff': 0,
                'smart_money_signal': 0,
                **game_data
            }
        
        line_movement = closing_home - opening_home
        
        # Probabilidade implícita
        opening_prob = 1 / opening_home
        closing_prob = 1 / closing_home
        implied_prob_diff = closing_prob - opening_prob
        
        # Categorizar sinal
        if implied_prob_diff > 0.03:
            signal = 1  # Forte sinal HOME
        elif implied_prob_diff < -0.03:
            signal = -1  # Forte sinal AWAY
        else:
            signal = 0  # Neutro
        
        return {
            'line_movement': round(line_movement, 4),
            'implied_prob_diff': round(implied_prob_diff, 4),
            'smart_money_signal': signal,
            **game_data
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Exporta dados de movimentação para DataFrame."""
        movements = self.get_odds_movement()
        
        if not movements:
            return pd.DataFrame()
        
        records = []
        for game_key, data in movements.items():
            records.append({
                'game_key': game_key,
                **data
            })
        
        return pd.DataFrame(records)
    
    def load_historical_movements(self, days_back: int = 30) -> pd.DataFrame:
        """
        Carrega movimentações históricas dos últimos N dias.
        
        Args:
            days_back: Número de dias para carregar
            
        Returns:
            DataFrame consolidado com todas as movimentações
        """
        all_dfs = []
        
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            file_path = self.data_dir / f"odds_{date.strftime('%Y%m%d')}.json"
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                for game_key, game_data in data.get('games', {}).items():
                    movement = self._calculate_movement(game_data)
                    movement['game_key'] = game_key
                    movement['date'] = date.strftime('%Y-%m-%d')
                    all_dfs.append(movement)
        
        if not all_dfs:
            return pd.DataFrame()
        
        return pd.DataFrame(all_dfs)


def integrate_with_odds_scraper():
    """
    Função auxiliar para integrar com o odds_scraper existente.
    
    Uso:
        from data.scrapers.odds_scraper import get_nba_odds
        from data.scrapers.odds_movement_tracker import OddsMovementTracker
        
        tracker = OddsMovementTracker()
        odds = get_nba_odds()
        tracker.record_odds_snapshot(odds)
    """
    try:
        from data.scrapers.odds_scraper import get_nba_odds
        
        tracker = OddsMovementTracker()
        odds = get_nba_odds()
        
        if odds:
            tracker.record_odds_snapshot(odds)
            logger.info("✅ Odds snapshot registrado com sucesso")
            return tracker.get_odds_movement()
        else:
            logger.warning("⚠️ Nenhuma odd obtida do scraper")
            return {}
            
    except Exception as e:
        logger.error(f"❌ Erro ao integrar com odds_scraper: {e}")
        return {}


def run_tracking_job():
    """
    Job para ser executado periodicamente (ex: a cada 2 horas).
    
    Sugestão de cron:
        0 8,10,12,14,16,18 * * * cd /home/denis/nba-predictor && python -c "from data.scrapers.odds_movement_tracker import run_tracking_job; run_tracking_job()"
    """
    logger.info("🔄 Iniciando job de tracking de odds...")
    
    result = integrate_with_odds_scraper()
    
    if result:
        logger.info(f"✅ {len(result)} jogos rastreados")
        
        # Mostrar movimentações significativas
        for game_key, data in result.items():
            if data.get('smart_money_signal', 0) != 0:
                signal = "↑ HOME" if data['smart_money_signal'] > 0 else "↓ AWAY"
                logger.info(f"   🎯 {game_key}: {signal} (Δprob: {data['implied_prob_diff']:.2%})")
    else:
        logger.warning("⚠️ Nenhum resultado obtido")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🔄 Testando OddsMovementTracker...")
    
    tracker = OddsMovementTracker()
    
    # Simular alguns snapshots
    test_odds_1 = {
        'BOS_vs_LAL': {'home_odds': 1.85, 'away_odds': 2.00},
        'MIA_vs_NYK': {'home_odds': 2.10, 'away_odds': 1.75},
    }
    
    tracker.record_odds_snapshot(test_odds_1)
    
    # Simular segundo snapshot (odds mudaram)
    import time
    time.sleep(1)
    
    test_odds_2 = {
        'BOS_vs_LAL': {'home_odds': 1.75, 'away_odds': 2.15},  # Boston encurtou
        'MIA_vs_NYK': {'home_odds': 2.05, 'away_odds': 1.80},  # Pouca mudança
    }
    
    tracker.record_odds_snapshot(test_odds_2)
    
    # Ver resultados
    print("\n📊 Movimentação de Odds:")
    movements = tracker.get_odds_movement()
    
    for game, data in movements.items():
        print(f"\n{game}:")
        print(f"   Opening: {data.get('opening_odds_home')}")
        print(f"   Closing: {data.get('closing_odds_home')}")
        print(f"   Line Movement: {data.get('line_movement')}")
        print(f"   Implied Prob Diff: {data.get('implied_prob_diff'):.2%}")
        print(f"   Smart Money Signal: {data.get('smart_money_signal')}")
    
    print("\n✅ Teste concluído!")
