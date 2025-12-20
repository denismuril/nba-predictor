import asyncio
import pandas as pd
import logging
from datetime import datetime
from data.processing.props_processor import PropsProcessor
from data.interfaces.player_props_provider import PlayerProp
from ml_pipeline.player_props_engine import analyze_props

# Configurar logging para ver o output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PropsIntelligence")

async def run_verification():
    print("\n🚀 INICIANDO VERIFICAÇÃO DO SISTEMA PROPS INTELLIGENCE\n")

    # 1. Criar MOCK PROPS (Simulando Scraper)
    # Vamos criar props que devem ativar as regras Sniper (OVER e UNDER)
    # Precisamos de jogadores que existam no season_stats_2024-25.csv
    
    # Exemplo: Jalen Brunson (NYK) - Vamos assumir que ele tem média alta de pontos
    # Se a média dele for ~28, e a linha for 24.5, deve dar OVER.
    
    mock_props = [
        # CENÁRIO 1: SNIPER OVER
        # Jalen Brunson: Média esperada ~27-28. Linha 24.5. Odd 1.90.
        PlayerProp(
            player_name="Jalen Brunson",
            prop_type="points",
            line=24.5,
            over_odds=1.90,
            under_odds=1.90,
            source="MockBookie",
            game_info="NYK vs BOS"
        ),
        # CENÁRIO 2: SNIPER UNDER
        # Trae Young: Média ~26. Linha 30.5. Odd 1.95.
        PlayerProp(
            player_name="Trae Young",
            prop_type="points",
            line=32.5,  # Linha inflacionada
            over_odds=1.85,
            under_odds=1.95, # Odd boa pro Under
            source="MockBookie",
            game_info="ATL vs MIA"
        ),
         # CENÁRIO 3: NO VALUE (Linha justa)
        PlayerProp(
            player_name="LeBron James",
            prop_type="points",
            line=25.5, 
            over_odds=1.91,
            under_odds=1.91,
            source="MockBookie",
            game_info="LAL vs PHX"
        ),
         # CENÁRIO 4: Player Inexistente (Teste de Robustez)
        PlayerProp(
            player_name="Jogador Inexistente 123",
            prop_type="points",
            line=10.5,
            over_odds=1.90,
            under_odds=1.90,
            source="MockBookie",
            game_info="Unknown"
        )
    ]
    
    print(f"📥 1. Recebidos {len(mock_props)} props simulados.\n")

    # 2. PROCESSAMENTO (ETL)
    print("🔄 2. Executando PropsProcessor (ETL)...")
    processor = PropsProcessor()
    
    # Verificar se temos o arquivo de stats
    try:
        df_processed = await processor.process_props(mock_props)
        print(f"✅ Processamento concluído. {len(df_processed)} props enriquecidos.\n")
        
        # Mostrar algumas colunas debug
        cols = ['player_name', 'prop_type', 'line', 'season_avg', 'L5_AVG', 'diff_to_avg', 'last_5_hit_rate']
        print(df_processed[cols].to_string(index=False))
        print("\n")
        
    except Exception as e:
        print(f"❌ Erro no processador: {e}")
        return

    # 3. ANÁLISE (SNIPER ENGINE)
    print("🎯 3. Executando Sniper Engine (Heurísticas)...")
    recommendations = analyze_props(df_processed, min_ev=0.05)
    
    print("\n📊 RESULTADOS FINAIS:\n")
    
    if recommendations.empty:
        print("⚠️ Nenhuma aposta recomendada.")
    else:
        for _, row in recommendations.iterrows():
            player = row['player_name']
            market = row['prop_type']
            line = row['line']
            prediction = row['prediction'] # OVER/UNDER
            avg = row['season_avg']
            if pd.isna(avg): avg = row['L5_AVG']
            
            ev_pct = row['ev_pct']
            prob = row['estimated_prob'] * 100
            odds = row['odds']
            reason = row['sniper_reason']
            
            # Formato solicitado pelo usuário
            print(f"🔥 RECOMENDAÇÃO: Aposte no {prediction} {player} {market}")
            print(f"   Porque a média dele é {avg:.1f} e a linha é {line} ({row['edge_pct']:.1f}% de vantagem)")
            print(f"   Odds: {odds} | EV: {ev_pct}% | Prob Est: {prob:.1f}%")
            print(f"   Rationale: {reason}")
            print("-" * 60)

if __name__ == "__main__":
    asyncio.run(run_verification())
