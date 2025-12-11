    @retry_on_lock()
    def bulk_insert_games(self, games_list):
        """
        Insere múltiplos jogos e suas estatísticas de uma vez (Bulk Insert).
        
        Args:
            games_list: Lista de tuplas (game_data, home_stats, away_stats)
        """
        if not games_list:
            return

        self.init_db()
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if self.db_type == 'sqlite':
                cursor.execute('BEGIN TRANSACTION')
            
            # Preparar dados
            games_values = []
            stats_values = []
            
            for game_data, home_stats, away_stats in games_list:
                # Game
                games_values.append((
                    game_data['id'], game_data['date'], game_data.get('season', '2024-25'),
                    game_data['home_team'], game_data['away_team'],
                    game_data['home_score'], game_data['away_score'],
                    game_data['winner'], 'Final'
                ))
                
                # Home Stats
                if home_stats:
                    stats_values.append(self._prepare_stats_record(game_data['id'], game_data['home_team'], True, home_stats))
                
                # Away Stats
                if away_stats:
                    stats_values.append(self._prepare_stats_record(game_data['id'], game_data['away_team'], False, away_stats))

            # Bulk Insert Games
            if self.db_type == 'postgres':
                # Postgres: execute_values para performance máxima
                from psycopg2.extras import execute_values
                
                game_query = '''
                INSERT INTO games (
                    game_id, date, season, home_team, away_team, 
                    home_score, away_score, winner, status
                ) VALUES %s
                ON CONFLICT (game_id) DO UPDATE SET
                    date = EXCLUDED.date,
                    season = EXCLUDED.season,
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    winner = EXCLUDED.winner
                '''
                execute_values(cursor, game_query, games_values)
                
                # Bulk Insert Stats
                if stats_values:
                    # Obter colunas dinamicamente do primeiro registro
                    columns = list(stats_values[0].keys())
                    cols_str = ', '.join(columns)
                    vals_str = %s
                    
                    # Converter dicts para tuplas na ordem das colunas
                    stats_tuples = [[s[c] for c in columns] for s in stats_values]
                    
                    stats_query = f'''
                    INSERT INTO game_stats ({cols_str}) VALUES %s
                    ON CONFLICT (game_id, team_id) DO UPDATE SET
                        pts = EXCLUDED.pts,
                        fg_pct = EXCLUDED.fg_pct
                    '''
                    execute_values(cursor, stats_query, stats_tuples)
                    
            else:
                # SQLite: executemany
                game_query = '''
                INSERT OR REPLACE INTO games (
                    game_id, date, season, home_team, away_team, 
                    home_score, away_score, winner, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                cursor.executemany(game_query, games_values)
                
                if stats_values:
                    columns = list(stats_values[0].keys())
                    cols_str = ', '.join(columns)
                    placeholders = ', '.join(['?'] * len(columns))
                    
                    stats_tuples = [[s[c] for c in columns] for s in stats_values]
                    
                    stats_query = f'''
                    INSERT OR REPLACE INTO game_stats ({cols_str}) VALUES ({placeholders})
                    '''
                    cursor.executemany(stats_query, stats_tuples)

            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erro no Bulk Insert: {e}")
            raise
        finally:
            self.return_connection(conn)

    def _prepare_stats_record(self, game_id, team_id, is_home, stats):
        """Helper para preparar dicionário de stats para bulk insert."""
        record = {
            'game_id': game_id,
            'team_id': self._normalize_team_id(team_id),
            'is_home': is_home,
            'pts': stats.get('PTS', 0),
            'fgm': stats.get('FGM', 0),
            'fga': stats.get('FGA', 0),
            'fg_pct': stats.get('FG_PCT', 0.0),
            'fg3m': stats.get('FG3M', 0),
            'fg3a': stats.get('FG3A', 0),
            'fg3_pct': stats.get('FG3_PCT', 0.0),
            'ftm': stats.get('FTM', 0),
            'fta': stats.get('FTA', 0),
            'ft_pct': stats.get('FT_PCT', 0.0),
            'oreb': stats.get('OREB', 0),
            'dreb': stats.get('DREB', 0),
            'reb': stats.get('REB', 0),
            'ast': stats.get('AST', 0),
            'stl': stats.get('STL', 0),
            'blk': stats.get('BLK', 0),
            'tov': stats.get('TOV', 0),
            'pf': stats.get('PF', 0),
            'plus_minus': stats.get('PLUS_MINUS', 0),
            'off_rating': stats.get('OFF_RATING', 0.0),
            'def_rating': stats.get('DEF_RATING', 0.0),
            'pace': stats.get('PACE', 0.0)
        }
        return record
