#!/usr/bin/env python3
"""
Force Update All Results
========================
Este script força a atualização de TODOS os jogos da temporada atual no banco de dados,
usando a NBA API (LeagueGameFinder) para obter todos os resultados de uma vez.

Isso corrige problemas onde jogos antigos ficaram como 'None' ou 'NULL' devido a falhas
de scraping ou inconsistências de IDs.
"""

import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

# Adicionar raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.repositories.db_manager import get_db_manager
from utils.team_normalization import normalize_team

def force_update_all():
    print("🚀 Iniciando atualização forçada de TODOS os resultados...")
    
    # 1. Buscar todos os jogos da temporada na NBA API
    print("📡 Consultando NBA API (LeagueGameFinder)...")
    try:
        # Season 2024-25 (verificar formato, geralmente '2024-25')
        # Se estamos em nov/2025 (como sugere a imagem do usuario), a season é 2025-26?
        # A imagem mostra datas de 2025-11-21. Isso seria temporada 2025-26.
        # Vou tentar buscar as duas últimas seasons para garantir.
        
        # Collect both Regular Season and Pre Season
        all_games = []
        season_types = ['Regular Season', 'Pre Season']
        
        for st in season_types:
            print(f"   - Buscando: {st}...")
            try:
                lgf = leaguegamefinder.LeagueGameFinder(
                    league_id_nullable='00',
                    season_type_nullable=st
                )
                df = lgf.get_data_frames()[0]
                if not df.empty:
                    all_games.append(df)
            except Exception as e:
                print(f"⚠️ Falha ao buscar {st}: {e}")
        
        if not all_games:
            raise ValueError("Nenhum jogo retornado da API.")
            
        games_df = pd.concat(all_games, ignore_index=True)
        
        # Filtrar jogos recentes (ex: últimos 12 meses)
        # O formato da data no DF é 'YYYY-MM-DD'
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        cutoff_date = datetime.now() - pd.Timedelta(days=365)
        recent_games = games_df[games_df['GAME_DATE'] > cutoff_date].copy()
        
        print(f"✅ Encontrados {len(recent_games)} registros de jogos nos últimos 365 dias.")
        
    except Exception as e:
        print(f"❌ Erro ao consultar NBA API: {e}")
        return

    # 2. Processar e Agrupar (LeagueGameFinder retorna 2 linhas por jogo)
    # Precisamos juntar Home e Away
    
    # Mapear ID -> Abbr
    nba_teams = teams.get_teams()
    id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
    
    # Agrupar por GAME_ID
    updates = {}
    
    print("🔄 Processando dados...")
    for game_id, group in recent_games.groupby('GAME_ID'):
        if len(group) < 2:
            continue # Dados incompletos
            
        # Identificar Home e Away
        # MATCHUP geralmente é "ATL vs. BOS" (Home) ou "ATL @ BOS" (Away)
        # Mas podemos usar o campo WL ou PTS para saber quem ganhou, mas precisamos saber quem é quem.
        # Geralmente a API tem uma coluna MATCHUP.
        # "GSW @ PHX" -> GSW é visitante, PHX é casa.
        
        row1 = group.iloc[0]
        row2 = group.iloc[1]
        
        # Tentar deduzir pelo MATCHUP string
        matchup1 = row1['MATCHUP']
        
        if ' @ ' in matchup1:
            # row1 é visitante
            away_row = row1
            home_row = row2
        else:
            # row1 é casa (bulls vs heat)
            home_row = row1
            away_row = row2
            
        # Extrair dados
        try:
            h_team_abbr = row1['TEAM_ABBREVIATION'] if 'vs.' in row1['MATCHUP'] else row2['TEAM_ABBREVIATION']
            a_team_abbr = row2['TEAM_ABBREVIATION'] if 'vs.' in row1['MATCHUP'] else row1['TEAM_ABBREVIATION']
            
            # Verificar se a lógica acima está certa.
            # Se row1 Matchup tem "vs.", row1 é CASA.
            if 'vs.' in row1['MATCHUP']:
                h_row = row1
                a_row = row2
            else:
                a_row = row1
                h_row = row2
                
            h_score = int(h_row['PTS'])
            a_score = int(a_row['PTS'])
            
            h_team = normalize_team(h_row['TEAM_ABBREVIATION'])
            a_team = normalize_team(a_row['TEAM_ABBREVIATION'])
            
            game_date_str = h_row['GAME_DATE'].strftime('%Y-%m-%d')
            
            # Construir ID do DB
            # Importante: Sem espaços!
            db_id = f"{game_date_str}_{h_team}_{a_team}".replace(" ", "")
            
            updates[db_id] = (h_score, a_score, h_team, a_team, game_date_str)
            
        except Exception as e:
            print(f"Erro processando jogo {game_id}: {e}")
            continue

    print(f"📦 Preparado para atualizar {len(updates)} jogos únicos.")
    
    # 3. Atualizar Banco de Dados
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    
    # Detectar tipo de placeholder
    ph = "%s" if db_manager.db_type == 'postgres' else "?"
    
    print(f"💾 Escrevendo no banco de dados ({db_manager.db_type.upper()})...")
    
    updates_games = 0
    inserts_games = 0
    for db_id, (h_s, a_s, h_tm, a_tm, g_date) in updates.items():
        try:
            # Calcular vencedor
            winner = 'HOME' if h_s > a_s else 'AWAY'
            
            # Update GAMES table
            # Este é o local correto onde os resultados ficam.
            # O front-end faz o merge entre Predictions e Games via game_id
            query_games = f"""
                UPDATE games 
                SET home_score = {ph}, away_score = {ph}, winner = {ph}, status = 'Final',
                    home_team = {ph}, away_team = {ph}
                WHERE game_id = {ph}
            """
            cursor.execute(query_games, (h_s, a_s, winner, h_tm, a_tm, db_id))
            if cursor.rowcount > 0:
                updates_games += 1
            else:
                # INSERT se não existe
                season = '2025-26' 
                query_insert = f"""
                    INSERT INTO games (game_id, date, season, home_team, away_team, home_score, away_score, winner, status)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, 'Final')
                """
                cursor.execute(query_insert, (db_id, g_date, season, h_tm, a_tm, h_s, a_s, winner))
                inserts_games += 1
                
        except Exception as e:
            # Important for Postgres: unexpected error aborts transaction
            conn.rollback() 
            print(f"Erro SQL {db_id}: {e}")
            
    conn.commit()
    conn.close()
    
    print("\n✅ ATUALIZAÇÃO CONCLUÍDA!")
    print(f"   - Jogos processados da API: {len(updates)}")
    print(f"   - Games atualizados no DB: {updates_games}")

if __name__ == "__main__":
    force_update_all()
