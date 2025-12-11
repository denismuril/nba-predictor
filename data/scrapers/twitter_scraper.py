import requests
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Configuração (Idealmente viria de variáveis de ambiente)
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAGwV5gEAAAAASkdzjZr2KnRE1HbdytIYErgtD%2B8%3D3DWoQ4DdJxTeNLesMu1Nc2AHRsazD5PKR3JCnJWc1e0q3Zpj1Z"

# IDs dos Insiders (Woj, Shams, Underdog NBA)
# Esses IDs são fixos no Twitter. 
# Woj: 50323173
# Shams: 134756069
# Underdog NBA: 988459323 (Ótimo para updates rápidos)
USER_IDS = ["50323173", "134756069", "988459323"] 

def fetch_latest_injury_tweets():
    """
    Busca os últimos tweets dos insiders sobre lesões.
    Retorna lista de alertas: [{'player': 'LeBron', 'status': 'OUT', 'source': 'Woj'}, ...]
    """
    if not BEARER_TOKEN:
        logger.warning("⚠️  Twitter Bearer Token não configurado.")
        return []

    alerts = []
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    # Keywords para filtrar
    KEYWORDS = ["out", "doubtful", "questionable", "available", "expected to play", "injury", "surgery"]
    
    try:
        logger.info("🐦 Buscando updates de lesão no Twitter (Woj/Shams)...")
        
        for user_id in USER_IDS:
            # Endpoint: User Tweets (v2)
            url = f"https://api.twitter.com/2/users/{user_id}/tweets"
            params = {
                "max_results": 10, # Apenas os mais recentes
                "tweet.fields": "created_at,text",
                "exclude": "retweets,replies"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('data', [])
                for tweet in data:
                    text = tweet['text'].lower()
                    # Verificar keywords
                    if any(k in text for k in KEYWORDS):
                        # Tentar extrair nome do jogador (Simplificado)
                        # Em produção, usaríamos NLP (Spacy/NER). Aqui vamos passar o texto bruto para análise.
                        alerts.append({
                            "text": tweet['text'],
                            "created_at": tweet['created_at'],
                            "source_id": user_id
                        })
            elif response.status_code == 403:
                logger.warning("⚠️  Acesso negado ao Twitter (403). Verifique o nível de acesso da API Key.")
                break # Parar se a chave for inválida
            elif response.status_code == 429:
                logger.warning("⚠️  Rate Limit do Twitter atingido.")
                break
            else:
                logger.warning(f"⚠️  Erro Twitter API ({response.status_code}): {response.text}")
                
        if alerts:
            logger.info(f"🚨 {len(alerts)} tweets relevantes encontrados!")
            return alerts
        else:
            logger.info("✅ Nenhum tweet crítico de lesão recente encontrado.")
            return []
            
    except Exception as e:
        logger.error(f"❌ Erro no Twitter Scraper: {e}")
        return []

if __name__ == "__main__":
    # Teste rápido
    logging.basicConfig(level=logging.INFO)
    alerts = fetch_latest_injury_tweets()
    for a in alerts:
        print(f"🚨 {a['text']}")
