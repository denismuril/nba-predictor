"""
NBA Predictor API - FastAPI Backend

API REST para o sistema de previsões NBA.
Integra com o orchestrator existente e expõe endpoints para:
- Atualização de dados (nba_api + scrapers de odds)
- Previsões do dia
- Health check do sistema

v27.0: Implementação inicial com:
- POST /update-data: Atualiza dados históricos e odds
- GET /predict-today: Retorna previsões cruzadas com odds
- GET /health: Health check completo

Uso:
    # Desenvolvimento
    uvicorn api.main:app --reload --port 8000
    
    # Produção
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configurar path para imports do projeto
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS PYDANTIC
# =============================================================================

class HealthResponse(BaseModel):
    """Resposta do health check."""
    status: str
    version: str
    timestamp: str
    components: Dict[str, bool]


class UpdateDataRequest(BaseModel):
    """Request para atualização de dados."""
    force_stats: bool = False  # Forçar atualização de stats mesmo se cache existe
    force_odds: bool = True    # Forçar atualização de odds (sempre fresco)


class UpdateDataResponse(BaseModel):
    """Resposta da atualização de dados."""
    success: bool
    message: str
    stats_updated: bool
    odds_updated: bool
    games_found: int
    timestamp: str


class GamePrediction(BaseModel):
    """Previsão individual de jogo."""
    game_id: str
    home_team: str
    away_team: str
    game_time: Optional[str] = None
    
    # Previsões do modelo
    home_win_prob: float
    predicted_winner: str
    confidence: float
    
    # Previsão de totais
    predicted_total: Optional[float] = None
    over_prob: Optional[float] = None
    
    # Odds de mercado
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    
    # Valor esperado
    home_ev: Optional[float] = None
    away_ev: Optional[float] = None


class PredictTodayResponse(BaseModel):
    """Resposta com previsões do dia."""
    success: bool
    date: str
    predictions: List[GamePrediction]
    total_games: int
    value_bets_count: int  # Jogos com EV > 0
    timestamp: str


# =============================================================================
# APLICAÇÃO FASTAPI
# =============================================================================

def get_app() -> FastAPI:
    """Cria e configura a aplicação FastAPI."""
    app = FastAPI(
        title="NBA Predictor API",
        description="API para previsões de jogos NBA usando ML",
        version="27.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS - permitir frontend local
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8501"],  # Streamlit
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = get_app()


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz com informações básicas."""
    return {
        "app": "NBA Predictor API",
        "version": "27.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check completo do sistema.
    
    Verifica:
    - Disponibilidade de modelos
    - Conexão com cache/DB
    - nba_api disponível
    - Scrapers de odds funcionando
    """
    components = {}
    
    # Verificar modelo
    try:
        model_path = PROJECT_ROOT / "data" / "models" / "ensemble_model_v6.joblib"
        components["model_v6"] = model_path.exists()
    except Exception:
        components["model_v6"] = False
    
    # Verificar nba_api
    try:
        from data.ingestion.stats_client import NBAStatsClient
        client = NBAStatsClient()
        health = client.health_check()
        components["nba_api"] = health.get("nba_api_available", False)
        components["stats_cache"] = health.get("cache_db_exists", False)
    except Exception:
        components["nba_api"] = False
        components["stats_cache"] = False
    
    # Verificar scrapers de odds
    try:
        from data.scrapers.odds_scraper import OddsCollector
        components["odds_scraper"] = True
    except Exception:
        components["odds_scraper"] = False
    
    # Verificar feature engineering
    try:
        from ml_pipeline.features import create_rolling_features
        components["feature_engineering"] = True
    except Exception:
        components["feature_engineering"] = False
    
    # Status geral
    all_healthy = all(components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="27.0",
        timestamp=datetime.now().isoformat(),
        components=components
    )


@app.post("/update-data", response_model=UpdateDataResponse, tags=["Data"])
async def update_data(
    request: UpdateDataRequest = None,
    background_tasks: BackgroundTasks = None
):
    """
    Atualiza dados do sistema.
    
    1. Busca jogos de ontem via nba_api (stats históricas)
    2. Busca odds de hoje via scrapers de mercado
    
    Os scrapers de odds (linemate, bettingpros) são PRESERVADOS e usados.
    """
    if request is None:
        request = UpdateDataRequest()
    
    stats_updated = False
    odds_updated = False
    games_found = 0
    
    try:
        # 1. Atualizar stats via nba_api
        if request.force_stats:
            try:
                from data.ingestion.stats_client import NBAStatsClient
                
                client = NBAStatsClient()
                # Buscar jogos de ontem para atualizar histórico
                yesterday_games = await client.get_yesterday_games()
                
                if yesterday_games is not None:
                    stats_updated = True
                    logger.info("✅ Stats atualizadas via nba_api")
            except Exception as e:
                logger.warning(f"⚠️ Erro atualizando stats: {e}")
        
        # 2. Atualizar odds via scrapers existentes
        if request.force_odds:
            try:
                # Usar orchestrator para buscar odds
                from orchestrator import EnterpriseOrchestrator
                
                orch = EnterpriseOrchestrator()
                await orch.step_fetch_odds()
                odds_updated = True
                logger.info("✅ Odds atualizadas via scrapers")
            except Exception as e:
                logger.warning(f"⚠️ Erro atualizando odds: {e}")
                
                # Fallback: chamar scrapers diretamente
                try:
                    from data.scrapers.odds_scraper import obter_odds
                    
                    odds = obter_odds()
                    
                    if odds:
                        odds_updated = True
                        games_found = len(odds)
                except Exception as e2:
                    logger.error(f"❌ Fallback de odds falhou: {e2}")
        
        return UpdateDataResponse(
            success=stats_updated or odds_updated,
            message="Dados atualizados com sucesso" if (stats_updated or odds_updated) else "Nenhuma atualização realizada",
            stats_updated=stats_updated,
            odds_updated=odds_updated,
            games_found=games_found,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Erro em update_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict-today", response_model=PredictTodayResponse, tags=["Predictions"])
async def predict_today():
    """
    Retorna previsões do dia cruzadas com odds.
    
    Usa o modelo ensemble calibrado para gerar probabilidades,
    cruza com odds de mercado e calcula EV (Expected Value).
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        predictions = []
        
        # 1. Obter previsões do modelo
        try:
            from ml_pipeline.predict import predict_next_games
            
            raw_predictions = predict_next_games(date=today)
            
            if raw_predictions is not None and len(raw_predictions) > 0:
                for _, row in raw_predictions.iterrows():
                    pred = _row_to_prediction(row)
                    predictions.append(pred)
                logger.info(f"✅ {len(predictions)} previsões geradas")
        except Exception as e:
            logger.warning(f"⚠️ Erro obtendo previsões: {e}")
        
        # 2. Cruzar com odds de mercado
        try:
            from data.scrapers.odds_scraper import obter_odds
            
            odds_data = obter_odds()
            
            if odds_data:
                predictions = _enrich_with_odds(predictions, odds_data)
                logger.info(f"✅ Odds obtidas para {len(odds_data)} jogos")
        except Exception as e:
            logger.warning(f"⚠️ Erro buscando odds: {e}")
        
        # 3. Calcular value bets
        value_bets = sum(1 for p in predictions 
                       if p.home_ev is not None and p.home_ev > 0)
        
        return PredictTodayResponse(
            success=True,
            date=today,
            predictions=predictions,
            total_games=len(predictions),
            value_bets_count=value_bets,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Erro em predict_today: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/{date}", response_model=PredictTodayResponse, tags=["Predictions"])
async def get_predictions_by_date(date: str):
    """
    Retorna previsões para uma data específica.
    
    Args:
        date: Data no formato YYYY-MM-DD
    """
    try:
        # Validar formato da data
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail="Data inválida. Use formato YYYY-MM-DD"
        )
    
    # Reutilizar lógica de predict_today com data específica
    # TODO: Implementar busca por data específica
    
    return PredictTodayResponse(
        success=True,
        date=date,
        predictions=[],
        total_games=0,
        value_bets_count=0,
        timestamp=datetime.now().isoformat()
    )


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def _row_to_prediction(row) -> GamePrediction:
    """Converte uma linha do DataFrame para GamePrediction."""
    return GamePrediction(
        game_id=str(row.get('game_id', '')),
        home_team=str(row.get('home_team', row.get('home', ''))),
        away_team=str(row.get('away_team', row.get('away', ''))),
        game_time=str(row.get('game_time', '')) if row.get('game_time') else None,
        home_win_prob=float(row.get('home_win_prob', row.get('prob_home', 0.5))),
        predicted_winner=str(row.get('predicted_winner', row.get('prediction', 'HOME'))),
        confidence=float(row.get('confidence', abs(row.get('prob_home', 0.5) - 0.5) * 2)),
        predicted_total=float(row.get('predicted_total', 0)) if row.get('predicted_total') else None,
        over_prob=float(row.get('over_prob', 0)) if row.get('over_prob') else None,
        home_odds=float(row.get('home_odds', 0)) if row.get('home_odds') else None,
        away_odds=float(row.get('away_odds', 0)) if row.get('away_odds') else None,
        over_odds=float(row.get('over_odds', 0)) if row.get('over_odds') else None,
        under_odds=float(row.get('under_odds', 0)) if row.get('under_odds') else None,
        home_ev=_calculate_ev(
            row.get('home_win_prob', row.get('prob_home', 0.5)),
            row.get('home_odds')
        ),
        away_ev=_calculate_ev(
            1 - row.get('home_win_prob', row.get('prob_home', 0.5)),
            row.get('away_odds')
        )
    )


def _calculate_ev(prob: float, odds: Optional[float]) -> Optional[float]:
    """
    Calcula Expected Value.
    
    EV = (prob * (odds - 1)) - (1 - prob)
    
    Positivo = aposta com valor
    """
    if odds is None or odds <= 1:
        return None
    
    try:
        ev = (prob * (odds - 1)) - (1 - prob)
        return round(ev, 4)
    except Exception:
        return None


def _enrich_with_odds(
    predictions: List[GamePrediction], 
    odds_data: Dict[str, Any]
) -> List[GamePrediction]:
    """Enriquece previsões com odds de mercado."""
    # Match by team names
    for pred in predictions:
        home_key = pred.home_team.upper()
        away_key = pred.away_team.upper()
        
        # Buscar odds correspondentes
        for game_odds in odds_data.get('games', []):
            if (home_key in game_odds.get('home', '').upper() or
                away_key in game_odds.get('away', '').upper()):
                
                pred.home_odds = game_odds.get('home_ml')
                pred.away_odds = game_odds.get('away_ml')
                pred.over_odds = game_odds.get('over_odds')
                pred.under_odds = game_odds.get('under_odds')
                
                # Recalcular EVs com odds reais
                if pred.home_odds:
                    pred.home_ev = _calculate_ev(pred.home_win_prob, pred.home_odds)
                if pred.away_odds:
                    pred.away_ev = _calculate_ev(1 - pred.home_win_prob, pred.away_odds)
                
                break
    
    return predictions


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("🏀 NBA Predictor API")
    print("="*60)
    print("\nIniciando servidor...")
    print("Documentação: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("="*60 + "\n")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
