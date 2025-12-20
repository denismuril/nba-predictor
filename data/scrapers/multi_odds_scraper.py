"""
Multi-Source Odds Scraper - Orquestrador de múltiplos scrapers.

Gerencia múltiplos scrapers de odds com fallback inteligente.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class MultiSourceOddsScraper:
    """
    Gerenciador de múltiplos scrapers de odds com fallback.
    
    Hierarquia de Prioridade:
    1. OddsPedia (Brasil) - dados Nuxt.js 
    2. OddsAgora (Brasil) - tabelas HTML
    3. OddsScanner (Brasil) - cards dinâmicos
    4. SportyTrader (PT-BR) - tabelas HTML
    5. OddsShark (US) - backup internacional
    
    Features:
    - Fallback automático se fonte falhar
    - Execução paralela para rapidez
    - Consenso de odds de múltiplas fontes
    - Métricas de sucesso por scraper
    """
    
    def __init__(self, headless: bool = True, max_workers: int = 3):
        """
        Inicializa o multi-scraper.
        
        Args:
            headless: Se True, navegadores rodam sem GUI
            max_workers: Número máximo de scrapers paralelos
        """
        self.headless = headless
        self.max_workers = max_workers
        self._metrics = {}
        self._scraper_order = [
            'oddspedia',
            'oddsagora', 
            'oddsscanner',
            'sportytrader',
            'oddsshark',
        ]
    
    def _get_scraper(self, name: str):
        """
        Factory para obter instância de scraper pelo nome.
        
        Args:
            name: Nome do scraper
            
        Returns:
            Instância do scraper ou None
        """
        try:
            if name == 'oddspedia':
                from data.scrapers.odds_web_scraper import OddsPediaScraper
                return OddsPediaScraper(headless=self.headless)
            elif name == 'oddsagora':
                from data.scrapers.odds_sites.odds_agora import OddsAgoraScraper
                return OddsAgoraScraper(headless=self.headless)
            elif name == 'oddsscanner':
                from data.scrapers.odds_sites.odds_scanner import OddsScannerScraper
                return OddsScannerScraper(headless=self.headless)
            elif name == 'sportytrader':
                from data.scrapers.odds_sites.sporty_trader import SportyTraderScraper
                return SportyTraderScraper(headless=self.headless)
            elif name == 'oddsshark':
                from data.scrapers.odds_sites.odds_shark import OddsSharkScraper
                return OddsSharkScraper(headless=self.headless)
            else:
                logger.warning(f"Scraper desconhecido: {name}")
                return None
        except ImportError as e:
            logger.error(f"Erro ao importar scraper {name}: {e}")
            return None
    
    def _run_scraper(self, name: str) -> Tuple[str, Dict, float]:
        """
        Executa um scraper individual.
        
        Args:
            name: Nome do scraper
            
        Returns:
            Tuple (nome, resultados, tempo_execução)
        """
        start = time.time()
        results = {}
        
        try:
            scraper = self._get_scraper(name)
            if scraper:
                results = scraper.fetch_odds()
                elapsed = time.time() - start
                
                # Atualiza métricas
                self._metrics[name] = {
                    'games_found': len(results),
                    'execution_time': elapsed,
                    'success': len(results) > 0,
                    'last_run': datetime.now().isoformat()
                }
                
                if results:
                    logger.info(f"✅ [{name}] {len(results)} jogos em {elapsed:.1f}s")
                else:
                    logger.warning(f"⚠️ [{name}] Nenhum jogo encontrado ({elapsed:.1f}s)")
                    
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"❌ [{name}] Falhou: {e} ({elapsed:.1f}s)")
            self._metrics[name] = {
                'games_found': 0,
                'execution_time': elapsed,
                'success': False,
                'error': str(e),
                'last_run': datetime.now().isoformat()
            }
        
        return name, results, time.time() - start
    
    def fetch_odds(self, 
                   parallel: bool = False,
                   sources: Optional[List[str]] = None,
                   consensus: bool = False) -> Dict:
        """
        Busca odds de múltiplas fontes.
        
        Args:
            parallel: Se True, executa scrapers em paralelo
            sources: Lista de fontes a usar (default: todas em ordem)
            consensus: Se True, retorna consenso de múltiplas fontes
            
        Returns:
            Dict com odds no formato padronizado
        """
        sources = sources or self._scraper_order
        all_results = {}
        
        if parallel:
            all_results = self._fetch_parallel(sources)
        else:
            all_results = self._fetch_sequential(sources)
        
        if consensus and len(all_results) > 1:
            return self._build_consensus(all_results)
        
        # Retorna primeiro resultado bem-sucedido
        for source in sources:
            if source in all_results and all_results[source]:
                return all_results[source]
        
        return {}
    
    def _fetch_sequential(self, sources: List[str]) -> Dict[str, Dict]:
        """
        Busca odds sequencialmente com fallback.
        
        Para no primeiro scraper que retornar dados.
        """
        all_results = {}
        
        for name in sources:
            logger.info(f"🔄 Tentando {name}...")
            _, results, _ = self._run_scraper(name)
            
            if results:
                all_results[name] = results
                break  # Para no primeiro sucesso
            
        return all_results
    
    def _fetch_parallel(self, sources: List[str]) -> Dict[str, Dict]:
        """
        Busca odds de múltiplas fontes em paralelo.
        
        Útil para consenso ou quando velocidade é crítica.
        """
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_scraper, name): name 
                for name in sources
            }
            
            for future in as_completed(futures):
                try:
                    name, results, _ = future.result()
                    if results:
                        all_results[name] = results
                except Exception as e:
                    logger.error(f"Erro em thread: {e}")
        
        return all_results
    
    def _build_consensus(self, all_results: Dict[str, Dict]) -> Dict:
        """
        Constrói consenso de odds de múltiplas fontes.
        
        Usa a média das odds quando disponível em múltiplas fontes.
        """
        consensus = {}
        game_odds = {}  # {game_key: [(source, home_odds, away_odds), ...]}
        
        # Agrupa odds por jogo
        for source, games in all_results.items():
            for game_key, data in games.items():
                if game_key not in game_odds:
                    game_odds[game_key] = []
                game_odds[game_key].append({
                    'source': source,
                    'home_odds': data.get('home_odds'),
                    'away_odds': data.get('away_odds'),
                    'home_team': data.get('home_team'),
                    'away_team': data.get('away_team'),
                })
        
        # Calcula média
        for game_key, odds_list in game_odds.items():
            valid_home = [o['home_odds'] for o in odds_list if o['home_odds']]
            valid_away = [o['away_odds'] for o in odds_list if o['away_odds']]
            
            if valid_home and valid_away:
                consensus[game_key] = {
                    'home_team': odds_list[0]['home_team'],
                    'away_team': odds_list[0]['away_team'],
                    'home_odds': round(sum(valid_home) / len(valid_home), 2),
                    'away_odds': round(sum(valid_away) / len(valid_away), 2),
                    'source': 'consensus',
                    'sources_count': len(odds_list),
                    'sources': [o['source'] for o in odds_list],
                    'timestamp': datetime.now().isoformat()
                }
        
        logger.info(f"📊 Consenso: {len(consensus)} jogos de {len(all_results)} fontes")
        return consensus
    
    def get_metrics(self) -> Dict:
        """Retorna métricas de execução dos scrapers."""
        return self._metrics
    
    def test_scrapers(self) -> Dict[str, bool]:
        """
        Testa cada scraper e retorna status.
        
        Returns:
            Dict com status de cada scraper
        """
        status = {}
        
        for name in self._scraper_order:
            logger.info(f"🧪 Testando {name}...")
            _, results, elapsed = self._run_scraper(name)
            status[name] = {
                'working': len(results) > 0,
                'games': len(results),
                'time': round(elapsed, 1)
            }
        
        return status
