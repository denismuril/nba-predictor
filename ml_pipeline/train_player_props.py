import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminhos
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml_pipeline" / "models"
MODELS_DIR.mkdir(exist_ok=True)

# MLflow Setup
mlflow.set_tracking_uri(f"file://{BASE_DIR}/mlruns")
mlflow.set_experiment("NBA_Player_Props_v1")

def load_and_prepare_data():
    """
    Carrega dados históricos e prepara features para Player Props.
    Idealmente, isso viria de um banco de dados de boxscores históricos.
    Como placeholder, vamos simular a estrutura esperada ou carregar de um CSV se existir.
    """
    # TODO: Implementar carregamento real de histórico de jogadores (boxscores)
    # Por enquanto, vamos criar um script que treina se o arquivo existir, ou avisa que precisa de dados.
    
    history_file = DATA_DIR / "player_boxscores_history.csv"
    
    history_file = DATA_DIR / "player_boxscores_history.csv"
    
    if not history_file.exists():
        logger.warning(f"⚠️  Arquivo de histórico {history_file} não encontrado.")
        logger.info("🛠️  Gerando dados sintéticos para treinamento inicial...")
        
        # Gerar dados sintéticos
        import numpy as np
        from datetime import timedelta
        
        players = [
            ('LeBron James', 'LAL'), ('Anthony Davis', 'LAL'),
            ('Jayson Tatum', 'BOS'), ('Jaylen Brown', 'BOS'),
            ('Stephen Curry', 'GSW'), ('Kevin Durant', 'PHX'),
            ('Nikola Jokic', 'DEN'), ('Jimmy Butler', 'MIA'),
            ('Luka Doncic', 'DAL'), ('Giannis Antetokounmpo', 'MIL')
        ]
        
        data = []
        start_date = datetime.now() - timedelta(days=90)
        for i in range(30): # 30 jogos por jogador
            date = start_date + timedelta(days=i*3)
            for p, team in players:
                data.append({
                    'Player': p, 'Team': team, 'Date': date,
                    'PTS': np.random.normal(25, 8),
                    'REB': np.random.normal(8, 4),
                    'AST': np.random.normal(6, 3),
                    'MIN': np.random.normal(34, 5),
                    'Location': 'Home' if i % 2 == 0 else 'Away',
                    'Opponent': 'OPP'
                })
        
        df = pd.DataFrame(data)
        df.to_csv(history_file, index=False)
        logger.info(f"✅ Dados sintéticos salvos em {history_file}")
    else:
        df = pd.read_csv(history_file)
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Player', 'Date'])
    
    # Feature Engineering
    logger.info("🛠️  Gerando Features (Rolling Averages, Rest Days)...")
    
    features = []
    targets = ['PTS', 'REB', 'AST']
    
    for stat in targets:
        # Média Móvel 5 e 10 jogos
        df[f'{stat}_L5'] = df.groupby('Player')[stat].transform(lambda x: x.shift(1).rolling(5).mean())
        df[f'{stat}_L10'] = df.groupby('Player')[stat].transform(lambda x: x.shift(1).rolling(10).mean())
        features.extend([f'{stat}_L5', f'{stat}_L10'])
        
    # Minutos médios L5
    df['MIN_L5'] = df.groupby('Player')['MIN'].transform(lambda x: x.shift(1).rolling(5).mean())
    features.append('MIN_L5')
    
    # Dias de descanso
    df['Rest_Days'] = df.groupby('Player')['Date'].diff().dt.days.fillna(3)
    features.append('Rest_Days')
    
    # Home/Away (1 = Home, 0 = Away)
    df['Is_Home'] = df['Location'].apply(lambda x: 1 if x == 'Home' else 0)
    features.append('Is_Home')
    
    # Opponent Defense Rank (Placeholder - idealmente viria de um lookup externo)
    # df = df.merge(defense_ranks, on='Opponent')
    
    # Limpar NaNs gerados pelo shift/rolling
    df_clean = df.dropna(subset=features + targets)
    
    return df_clean, features, targets

def train_prop_model(target_col, df, features):
    """
    Treina um modelo XGBoost Regressor para uma estatística específica (PTS, REB, AST).
    """
    logger.info(f"\n🏀 Treinando modelo para: {target_col}")
    
    X = df[features]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name=f"Prop_{target_col}"):
        # Parâmetros otimizados para regressão
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 50
        }
        
        mlflow.log_params(params)
        
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Avaliação
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        logger.info(f"✅ {target_col} Results - MAE: {mae:.2f} | R2: {r2:.3f}")
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Salvar modelo localmente
        model_path = MODELS_DIR / f"xgb_{target_col.lower()}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"💾 Modelo salvo em: {model_path}")
        
        # Logar artefato no MLflow
        mlflow.sklearn.log_model(model, f"model_{target_col}")
        
    return model, mae

if __name__ == "__main__":
    logger.info("🚀 Iniciando Pipeline de Treinamento de Player Props...")
    
    data = load_and_prepare_data()
    
    if data:
        df, features, targets = data
        
        results = {}
        for target in targets:
            model, mae = train_prop_model(target, df, features)
            results[target] = mae
            
        logger.info("\n📊 Resumo do Treinamento:")
        for k, v in results.items():
            logger.info(f"  - {k}: MAE {v:.2f}")
    else:
        logger.info("⏭️  Pulando treinamento (sem dados históricos).")
        # Criar arquivos dummy para evitar erro no frontend se não houver modelos
        for t in ['pts', 'reb', 'ast']:
            dummy_path = MODELS_DIR / f"xgb_{t}.joblib"
            if not dummy_path.exists():
                logger.warning(f"⚠️  Criando placeholder vazio para {dummy_path}")
                # Não criamos arquivo vazio pq joblib vai falhar ao carregar.
                # O frontend deve tratar a ausência do arquivo.
