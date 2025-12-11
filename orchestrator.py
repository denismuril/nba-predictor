"""
NBA Predictor Orchestrator
Gerencia o fluxo diário de execução:
1. Coleta de Dados (Async)
2. Validação
3. Treinamento (Opcional)
4. Previsão
5. Atualização de Dashboard
"""
import sys
import logging
import asyncio
import time
from pathlib import Path
from datetime import datetime

# Setup paths
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

from data.repositories.db_manager import get_db_manager
from data.scrapers.stats_scraper import obter_player_stats
from data.scrapers.async_scraper import AsyncScraper

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Orchestrator")

class PipelineOrchestrator:
    def __init__(self):
        self.db = get_db_manager()
        self.start_time = time.time()

    def run_step(self, step_name, func, *args, **kwargs):
        """Executa uma etapa com tratamento de erro e medição de tempo"""
        logger.info(f"\n{'='*50}")
        logger.info(f"🚀 Iniciando Etapa: {step_name}")
        logger.info(f"{'='*50}")

        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"✅ Etapa '{step_name}' concluída em {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"❌ Falha na etapa '{step_name}': {e}", exc_info=True)
            # Dependendo da criticidade, podemos levantar erro ou continuar
            if step_name in ['Data Collection']: # Crítico
                raise e
            return None

    def step_twitter_collection(self):
        """Coleta tweets de insiders para sentiment analysis (Tarefa 4)"""
        logger.info("🐦 Coletando tweets de lesão (Woj/Shams)...")

        try:
            from data.scrapers.twitter_scraper import fetch_latest_injury_tweets
            from ml_pipeline.sentiment import NewsSentimentAnalyzer

            tweets = fetch_latest_injury_tweets()

            if tweets:
                analyzer = NewsSentimentAnalyzer()
                sentiments = analyzer.analyze_tweets(tweets)
                logger.info(f"📰 {len(tweets)} tweets coletados, {len(sentiments)} times afetados")

                # Salvar em cache para uso posterior
                from pathlib import Path
                import json
                cache_file = Path('data/cache/sentiment_cache.json')
                cache_file.parent.mkdir(exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump({'tweets': len(tweets), 'sentiments': sentiments}, f)
            else:
                logger.info("✅ Nenhum tweet crítico encontrado")

            return True
        except Exception as e:
            logger.warning(f"⚠️ Twitter collection failed (não crítico): {e}")
            return True  # Não bloqueia pipeline

    def step_data_collection(self):
        """Coleta dados históricos e stats avançados (Grão-Mestre)"""
        logger.info("📡 Coletando Dados Históricos e Avançados...")

        # 1. Ingestão Robusta (fetch_historical_data)
        try:
            from scripts.fetch_historical_data import process_historical_data
            # Processar apenas temporada atual para o fluxo diário ser rápido
            process_historical_data(seasons=['2025-26'])
        except Exception as e:
            logger.error(f"❌ Erro na ingestão de dados: {e}")
            return False

        # 2. Player Stats (Async) - Mantido para props
        logger.info("📡 Coletando Player Stats (Async)...")
        stats = obter_player_stats()

        # 3. Atualizar Resultados Pendentes (agora feito pelo fetch_historical_data, mas mantemos log)
        logger.info("🔄 Atualizando resultados de jogos anteriores...")
        updated = self.db.update_pending_results()

        return True


    def step_model_training(self, force=False):
        """Treina modelos V7 (Grão-Mestre) se necessário"""
        last_train_file = Path("logs/last_train_date.txt")
        should_train = force

        if not should_train and last_train_file.exists():
            last_date = last_train_file.read_text().strip()
            if last_date != datetime.now().strftime("%Y-%m-%d"):
                should_train = True
        else:
            should_train = True

        if should_train:
            logger.info("🧠 Iniciando retreino dos modelos V18/V6...")

            # 1. Treinar Totals V18 e Ensemble V6
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, "ml_pipeline/train_all.py"],
                    check=True
                )
            except Exception as e:
                logger.error(f"❌ Erro no treino dos modelos: {e}")

            # 2. Player Props
            try:
                from ml_pipeline.train_player_props import load_and_prepare_data, train_prop_model
                data = load_and_prepare_data()
                if data:
                    df, features, targets = data
                    for target in targets:
                        train_prop_model(target, df, features)
            except Exception as e:
                logger.error(f"❌ Erro no treino de Props: {e}")

            # Atualizar data de treino
            last_train_file.parent.mkdir(exist_ok=True)
            last_train_file.write_text(datetime.now().strftime("%Y-%m-%d"))
        else:
            logger.info("⏭️  Pulando treinamento (já treinado hoje).")

    def step_prediction(self):
        """Gera previsões para os jogos de hoje usando modelos ML"""
        logger.info("🔮 Gerando previsões com ML (V18/V6)...")

        import subprocess

        # 1. Executar main.py com flag --ml (SEMPRE usar ML)
        try:
            subprocess.run(
                [sys.executable, "main.py", "--ml"],
                check=True
            )
        except Exception as e:
            logger.error(f"❌ Erro na predição principal: {e}")

        # 2. Player Props Prediction
        try:
            logger.info("🔮 Gerando previsões de Player Props...")
            subprocess.run(
                [sys.executable, "ml_pipeline/predict_player_props.py"],
                check=True
            )
        except Exception as e:
            logger.warning(f"⚠️ Erro ao gerar Player Props: {e}")

    def step_fetch_odds(self):
        """Busca odds reais antes de gerar previsões"""
        logger.info("💰 Buscando odds reais da APIs...")

        try:
            from data.scrapers.odds_scraper import obter_odds

            odds_data = obter_odds()  # Retorna dict com odds por jogo

            if odds_data:
                logger.info(f"✅ {len(odds_data)} jogos com odds encontrados")
                # Opcional: Salvar em cache ou passar para o próximo passo via self
                # Como o obter_odds usa cache interno, o próximo passo (prediction)
                # vai pegar do cache quando chamar obter_odds() novamente.
            else:
                logger.warning("⚠️ Nenhuma odd encontrada - sistema usará Fair Odds calculadas")

        except Exception as e:
            logger.error(f"❌ Erro ao buscar odds: {e}")

    def step_generate_player_props(self):
        """Gera player props baseado nos jogos de hoje"""
        logger.info("⛹️ Gerando Player Props para combos...")

        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "scripts/generate_player_props_quick.py"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✅ Player props gerados com sucesso")
            if result.stdout:
                logger.info(result.stdout.strip())
        except Exception as e:
            logger.warning(f"⚠️ Erro ao gerar player props (não crítico): {e}")

    def run_daily_flow(self):
        """Executa o fluxo completo v21.4"""
        logger.info("🎬 Iniciando Orchestrator Diário NBA v21.4 (Production)")

        try:
            # NOVO: Coletar tweets ANTES do pipeline
            self.run_step("Twitter Sentiment", self.step_twitter_collection)
            self.run_step("Data Collection", self.step_data_collection)
            self.run_step("Model Training", self.step_model_training)
            self.run_step("Fetch Real Odds", self.step_fetch_odds)
            self.run_step("Prediction Generation", self.step_prediction)
            self.run_step("Player Props Generation", self.step_generate_player_props)

            total_time = time.time() - self.start_time
            logger.info(f"\n🎉 Fluxo v21.4 concluído com sucesso em {total_time:.1f}s")


        except Exception as e:
            logger.critical(f"🔥 Falha crítica no pipeline: {e}")
            sys.exit(1)

if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    orchestrator.run_daily_flow()
