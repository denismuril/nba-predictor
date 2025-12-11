#!/usr/bin/env python3
"""
Sistema de Monitoramento Contínuo de Performance

Rastreia performance dos modelos ao longo do tempo, detecta drift,
e gera alertas quando accuracy cai abaixo de thresholds.

Métricas monitoradas:
- Accuracy diária/semanal/mensal
- Distribuição de probabilidades
- Performance por segmento (favoritos/underdogs, spread ranges)
- Drift de features

Usage:
    python scripts/monitoring_system.py --update-daily
    python scripts/monitoring_system.py --generate-report
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import joblib

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thresholds de alerta
ALERT_THRESHOLDS = {
    'accuracy_daily_min': 0.60,      # Mínimo 60% em um dia
    'accuracy_weekly_min': 0.70,     # Mínimo 70% na semana
    'accuracy_monthly_min': 0.74,    # Mínimo 74% no mês (baseline)
    'brier_max': 0.20,               # Máximo Brier score
    'drift_max': 0.15                # Máximo drift permitido
}

class PerformanceMonitor:
    """Monitor de performance dos modelos."""
    
    def __init__(self, metrics_file='data/monitoring/metrics_history.json'):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.weekly_file = self.metrics_file.parent / 'weekly_metrics.json'
        
        # Carregar histórico (JSON Lines)
        self.history = []
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                self.history.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                logger.error(f"Erro ao carregar histórico: {e}")

    def update_daily_metrics(self, date=None):
        """Atualiza métricas do dia."""
        if date is None:
            date = datetime.now().date()
        else:
            date = pd.to_datetime(date).date()
        
        logger.info(f"📊 Atualizando métricas para {date}...")
        
        # Carregar modelo
        model_file = Path('data/models/ensemble_model_calibrated_isotonic.joblib')
        if not model_file.exists():
            model_file = Path('data/models/ensemble_model_final.joblib')
        
        if not model_file.exists():
            logger.error("❌ Modelo não encontrado")
            return None
        
        # Buscar jogos do dia
        from data.repositories.db_manager import get_db_manager
        db = get_db_manager()
        
        try:
            with db.get_connection() as conn:
                games_df = pd.read_sql('''
                    SELECT * FROM predictions 
                    WHERE date = ?
                    AND home_score > 0 AND away_score > 0
                ''', conn, params=(str(date),))
        except Exception as e:
            logger.error(f"Erro ao buscar jogos: {e}")
            return None
        
        if games_df.empty:
            logger.warning(f"⚠️  Nenhum jogo finalizado encontrado para {date}")
            return None
        
        logger.info(f"   Jogos encontrados: {len(games_df)}")
        
        # Calcular métricas reais
        # Accuracy
        games_df['actual_winner'] = np.where(games_df['home_score'] > games_df['away_score'], 
                                           games_df['home_team'], games_df['away_team'])
        games_df['correct'] = (games_df['prediction'] == games_df['actual_winner']).astype(int)
        accuracy = games_df['correct'].mean()
        
        # Brier Score (MSE das probabilidades)
        # Se home ganhou (1), prob deve ser 1. Se perdeu (0), prob deve ser 0.
        games_df['home_win'] = (games_df['home_score'] > games_df['away_score']).astype(int)
        brier_score = ((games_df['prob_home'] - games_df['home_win']) ** 2).mean()
        
        metrics = {
            'date': str(date),
            'games_count': len(games_df),
            'accuracy': float(accuracy),
            'brier_score': float(brier_score),
            'avg_confidence': float(games_df['confidence'].apply(lambda x: float(str(x).strip('%'))/100 if isinstance(x, str) else 0).mean()),
            'timestamp': datetime.now().isoformat(),
            'drift_score': 0.0, # Placeholder, atualizado pelo detect_drift
            'drift_detected': False,
            'features_drifted': 0
        }
        
        # Salvar (Append JSON Line)
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        self.history.append(metrics)
        logger.info(f"✅ Métricas salvas: {accuracy*100:.1f}% Acc, {brier_score:.4f} Brier")
        
        return metrics
    
    def calculate_weekly_metrics(self):
        """Calcula métricas da última semana e salva em arquivo separado."""
        logger.info("📊 Calculando métricas semanais...")
        
        if not self.history:
            logger.warning("⚠️  Sem dados diários para calcular")
            return None
            
        # Converter para DF para facilitar
        df = pd.DataFrame(self.history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Pegar últimos 7 dias com dados
        recent = df.tail(7)
        
        if recent.empty:
            return None
            
        metrics = {
            'week_end': str(recent['date'].max().date()),
            'week_start': str(recent['date'].min().date()),
            'games_count': int(recent['games_count'].sum()),
            'accuracy': float(recent['accuracy'].mean()),
            'days_with_data': len(recent),
            'timestamp': datetime.now().isoformat()
        }
        
        # Salvar em arquivo separado (JSON Lines)
        with open(self.weekly_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
            
        logger.info(f"✅ Métricas semanais salvas em {self.weekly_file}")
        return metrics
    
    def check_alerts(self):
        """Verifica se há alertas a serem disparados."""
        logger.info("\n🚨 VERIFICANDO ALERTAS")
        logger.info("="*80)
        
        alerts = []
        
        # Verificar accuracy diária
        if self.history:
            recent_daily = self.history[-1]
            if recent_daily.get('accuracy', 0) < ALERT_THRESHOLDS['accuracy_daily_min']:
                alert = {
                    'type': 'LOW_DAILY_ACCURACY',
                    'severity': 'WARNING',
                    'message': f"Accuracy diária baixa: {recent_daily.get('accuracy', 0)*100:.2f}% (mín: {ALERT_THRESHOLDS['accuracy_daily_min']*100:.0f}%)",
                    'date': recent_daily.get('date')
                }
                alerts.append(alert)
                logger.warning(f"⚠️  {alert['message']}")
        
        # Verificar accuracy semanal (Carregar do arquivo)
        weekly_history = []
        if self.weekly_file.exists():
            try:
                with open(self.weekly_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            weekly_history.append(json.loads(line))
            except Exception:
                pass

        if weekly_history:
            recent_weekly = weekly_history[-1]
            if recent_weekly.get('accuracy', 0) < ALERT_THRESHOLDS['accuracy_weekly_min']:
                alert = {
                    'type': 'LOW_WEEKLY_ACCURACY',
                    'severity': 'WARNING',
                    'message': f"Accuracy semanal baixa: {recent_weekly.get('accuracy', 0)*100:.2f}% (mín: {ALERT_THRESHOLDS['accuracy_weekly_min']*100:.0f}%)",
                    'week': f"{recent_weekly.get('week_start')} a {recent_weekly.get('week_end')}"
                }
                alerts.append(alert)
                logger.warning(f"⚠️  {alert['message']}")
        
        if not alerts:
            logger.info("✅ Nenhum alerta - Sistema operando normalmente")
        else:
            logger.info(f"\n📋 Total de alertas: {len(alerts)}")
            
            # Salvar alertas
            alerts_file = Path('data/monitoring/alerts.json')
            with open(alerts_file, 'w') as f:
                json.dump(alerts, f, indent=2)
            
            logger.info(f"💾 Alertas salvos em: {alerts_file}")
        
        return alerts
    
    def generate_report(self):
        """Gera relatório de monitoramento."""
        logger.info("="*80)
        logger.info("📊 RELATÓRIO DE MONITORAMENTO")
        logger.info("="*80)
        
        # Métricas diárias
        if self.history:
            daily_df = pd.DataFrame(self.history)
            
            logger.info(f"\n📅 MÉTRICAS DIÁRIAS (últimos 30 dias):")
            logger.info(f"   Total de dias: {len(daily_df)}")
            logger.info(f"   Jogos monitorados: {daily_df['games_count'].sum()}")
            logger.info(f"   Accuracy média: {daily_df['accuracy'].mean()*100:.2f}%")
            logger.info(f"   Accuracy mín: {daily_df['accuracy'].min()*100:.2f}%")
            logger.info(f"   Accuracy máx: {daily_df['accuracy'].max()*100:.2f}%")
        
        # Métricas semanais
        weekly_history = []
        if self.weekly_file.exists():
            try:
                with open(self.weekly_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            weekly_history.append(json.loads(line))
            except Exception:
                pass

        if weekly_history:
            weekly_df = pd.DataFrame(weekly_history)
            
            logger.info(f"\n📆 MÉTRICAS SEMANAIS (últimas 12 semanas):")
            logger.info(f"   Total de semanas: {len(weekly_df)}")
            logger.info(f"   Accuracy média: {weekly_df['accuracy'].mean()*100:.2f}%")
            
            # Tendência
            if len(weekly_df) >= 2:
                recent_avg = weekly_df.tail(4)['accuracy'].mean()
                old_avg = weekly_df.head(4)['accuracy'].mean()
                trend = recent_avg - old_avg
                
                trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                logger.info(f"   Tendência: {trend_emoji} {trend*100:+.2f}%")
        
        logger.info("\n" + "="*80)
        
        # Salvar report
        report = {
            'generated_at': datetime.now().isoformat(),
            'daily': self.history[-30:] if self.history else [],
            'weekly': weekly_history[-12:] if weekly_history else [],
            'summary': {
                'total_days_monitored': len(self.history),
                'total_weeks_monitored': len(weekly_history),
                'current_accuracy': self.history[-1]['accuracy'] if self.history else None
            }
        }
        
        report_file = Path('data/monitoring/performance_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Relatório salvo em: {report_file}")
        
        return report
    
    def detect_drift(self, reference_data_path='data/training/reference_data.csv'):
        """
        Detecta drift de features usando Kolmogorov-Smirnov test.
        
        Compara distribuição atual das features com dados de referência (treino).
        KS test detecta mudanças na distribuição das features.
        
        Args:
            reference_data_path: Path para dados de referência (usado no treino)
            
        Returns:
            Dict com drift score por feature e drift geral
        """
        logger.info("\n🔬 DETECÇÃO DE DRIFT")
        logger.info("="*80)
        
        from scipy import stats
        
        # Carregar dados de referência
        ref_path = Path(reference_data_path)
        if not ref_path.exists():
            logger.warning(f"⚠️  Dados de referência não encontrados: {ref_path}")
            logger.info("   Criando dados de referência a partir do histórico...")
            
            # Criar referência a partir dos dados históricos
            return self._create_reference_data()
        
        try:
            ref_data = pd.read_csv(ref_path)
            logger.info(f"📊 Dados de referência carregados: {len(ref_data)} amostras")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados de referência: {e}")
            return {'drift_detected': False, 'error': str(e)}
        
        # Buscar dados recentes (últimos 30 dias)
        from data.repositories.db_manager import get_db_manager
        db = get_db_manager()
        
        cutoff_date = (datetime.now() - timedelta(days=30)).date()
        
        query = f'''
            SELECT * FROM predictions
            WHERE date >= '{cutoff_date}'
            AND home_score > 0 AND away_score > 0
        '''
        
        try:
            with db.get_connection() as conn:
                recent_data = pd.read_sql(query, conn)
        except Exception as e:
            logger.error(f"❌ Erro ao buscar dados recentes: {e}")
            return {'drift_detected': False, 'error': str(e)}
        
        if recent_data.empty:
            logger.warning("⚠️  Sem dados recentes para detecção de drift")
            return {'drift_detected': False, 'reason': 'no_recent_data'}
        logger.info(f"📊 Dados recentes: {len(recent_data)} jogos")
        
        # Features para monitorar (numéricas)
        numeric_features = [
            'home_net_rating', 'away_net_rating',
            'home_power_rating', 'away_power_rating',
            'home_pts', 'away_pts',
            'home_fg_pct', 'away_fg_pct'
        ]
        
        # Filtrar features disponíveis
        available_features = [f for f in numeric_features if f in ref_data.columns and f in recent_data.columns]
        
        if not available_features:
            logger.warning("⚠️  Nenhuma feature comum para detecção de drift")
            return {'drift_detected': False, 'reason': 'no_common_features'}
        
        logger.info(f"🔍 Monitorando {len(available_features)} features:")
        for f in available_features:
            logger.info(f"   - {f}")
        
        # Calcular KS test para cada feature
        drift_scores = {}
        drifted_features = []
        
        for feature in available_features:
            try:
                ref_values = ref_data[feature].dropna()
                recent_values = recent_data[feature].dropna()
                
                if len(ref_values) < 30 or len(recent_values) < 30:
                    logger.debug(f"   Skipping {feature}: amostras insuficientes")
                    continue
                
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(ref_values, recent_values)
                
                drift_scores[feature] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'drifted': p_value < 0.05,  # Significância de 5%
                    'severity': 'HIGH' if ks_stat > 0.3 else 'MEDIUM' if ks_stat > 0.15 else 'LOW'
                }
                
                if p_value < 0.05:  # Drift detectado
                    drifted_features.append(feature)
                    logger.warning(
                        f"⚠️  DRIFT DETECTADO: {feature} "
                        f"(KS={ks_stat:.3f}, p={p_value:.4f})"
                    )
                else:
                    logger.info(f"✅ {feature}: OK (KS={ks_stat:.3f})")
                    
            except Exception as e:
                logger.error(f"❌ Erro processando {feature}: {e}")
                continue
        
        # Calcular drift geral
        if drift_scores:
            avg_ks = np.mean([s['ks_statistic'] for s in drift_scores.values()])
            drift_percentage = len(drifted_features) / len(drift_scores) * 100
            
            # Determinar se há drift significativo
            drift_detected = (
                drift_percentage > 30  # Mais de 30% das features driftaram
                or avg_ks > ALERT_THRESHOLDS['drift_max']  # KS médio alto
            )
            
            logger.info(f"\n📈 RESUMO DE DRIFT:")
            logger.info(f"   Features monitoradas: {len(drift_scores)}")
            logger.info(f"   Features com drift: {len(drifted_features)} ({drift_percentage:.1f}%)")
            logger.info(f"   KS médio: {avg_ks:.3f}")
            
            if drift_detected:
                logger.error(f"\n🚨 DRIFT SIGNIFICATIVO DETECTADO!")
                logger.error(f"   Recomendação: RETREINAR MODELO")
                logger.error(f"   Features afetadas: {', '.join(drifted_features)}")
            else:
                logger.info(f"✅ Drift dentro dos limites aceitáveis")
            
            # Salvar resultado
            drift_report = {
                'timestamp': datetime.now().isoformat(),
                'drift_detected': drift_detected,
                'drift_percentage': drift_percentage,
                'avg_ks_statistic': avg_ks,
                'drifted_features': drifted_features,
                'feature_details': drift_scores,
                'reference_period': str(cutoff_date),
                'samples_reference': len(ref_data),
                'samples_recent': len(recent_data)
            }
            
            drift_file = Path('data/monitoring/drift_report.json')
            with open(drift_file, 'w') as f:
                json.dump(drift_report, f, indent=2)
                
            # Adicionar ao histórico (JSON Lines) para o Dashboard
            history_file = Path('data/monitoring/metrics_history.json')
            
            # Preparar registro simplificado para histórico
            history_record = {
                'date': datetime.now().isoformat(),
                'accuracy': 0.0, # Placeholder se não calculado aqui
                'avg_confidence': 0.0, # Placeholder
                'drift_score': avg_ks,
                'drift_detected': drift_detected,
                'features_drifted': len(drifted_features)
            }
            
            with open(history_file, 'a') as f:
                f.write(json.dumps(history_record) + '\n')
            
            logger.info(f"\n💾 Relatório de drift salvo em: {drift_file}")
            
            return drift_report
            
        else:
            logger.warning("⚠️  Nenhuma feature válida para KS test")
            return {'drift_detected': False, 'reason': 'no_valid_features'}
    
    def _create_reference_data(self):
        """Cria dados de referência a partir dos dados históricos."""
        logger.info("🏗️  Criando dados de referência...")
        
        from data.repositories.db_manager import get_db_manager
        db = get_db_manager()
        
        # Usar temporada passada como referência estável
        start_date = '2024-10-01'
        end_date = '2025-06-01'
        
        query = f'''
            SELECT * FROM predictions
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            AND home_score > 0 AND away_score > 0
        '''
        
        try:
            with db.get_connection() as conn:
                ref_data = pd.read_sql(query, conn)
        except Exception as e:
            logger.error(f"❌ Erro ao buscar dados históricos: {e}")
            return {'drift_detected': False, 'error': str(e)}
        
        if ref_data.empty:
            logger.error("❌ Sem dados históricos suficientes")
            return {'drift_detected': False, 'error': 'insufficient_history'}
        
        # Salvar como referência
        ref_path = Path('data/training/reference_data.csv')
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_data.to_csv(ref_path, index=False)
        
        logger.info(f"✅ Dados de referência criados: {len(ref_data)} amostras")
        logger.info(f"   Período: {start_date} a {end_date}")
        logger.info(f"   Salvo em: {ref_path}")
        
        return {'drift_detected': False, 'reason': 'reference_created'}
    

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sistema de Monitoramento Contínuo')
    parser.add_argument('--update-daily', action='store_true',
                       help='Atualizar métricas do dia')
    parser.add_argument('--update-weekly', action='store_true',
                       help='Atualizar métricas da semana')
    parser.add_argument('--check-alerts', action='store_true',
                       help='Verificar alertas')
    parser.add_argument('--generate-report', action='store_true',
                       help='Gerar relatório completo')
    parser.add_argument('--detect-drift', action='store_true',
                       help='Detectar drift de features (KS test)')
    parser.add_argument('--date', type=str,
                       help='Data específica (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor()
    
    if args.update_daily:
        date = args.date if args.date else None
        monitor.update_daily_metrics(date=date)
    
    if args.update_weekly:
        monitor.calculate_weekly_metrics()
    
    if args.check_alerts:
        monitor.check_alerts()
    
    if args.generate_report:
        monitor.generate_report()
    
    if args.detect_drift:
        drift_report = monitor.detect_drift()
        
        # Se drift significativo, sugerir retreinamento
        if drift_report.get('drift_detected'):
            logger.error("\n" + "="*80)
            logger.error("🚨 AÇÃO NECESSÁRIA: RETREINAR MODELO")
            logger.error("="*80)
            logger.error("Comandos sugeridos:")
            logger.error("  1. python ml_pipeline/train.py")
            logger.error("  2. python ml_pipeline/validate_model.py")
            logger.error("="*80)
    
    if not any([args.update_daily, args.update_weekly, args.check_alerts, 
                args.generate_report, args.detect_drift]):
        logger.info("Use --update-daily, --update-weekly, --check-alerts, --generate-report ou --detect-drift")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
