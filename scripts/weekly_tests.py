"""
CI/CD Semanal: Roda testes e envia resultado no Telegram
=========================================================
Executa toda semana automaticamente.
"""

import subprocess
import os
from datetime import datetime
from telegram import Bot
import asyncio

# Config
TOKEN = "8263506662:AAFo-VyHHB4Ocrpvx6hFQfNOB1tFhF0G3Hs"
CHAT_ID = "SEU_CHAT_ID_AQUI"  # Descubra enviando /start pro bot


async def run_tests_and_notif():
    """Roda testes e envia no Telegram"""
    
    print("🧪 Executando testes de integridade...")
    
    # Rodar pytest
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd="/home/denis/nba-predictor"
    )
    
    # Parse resultado
    output = result.stdout + result.stderr
    
    passed = output.count(" PASSED")
    failed = output.count(" FAILED")
    total = passed + failed
    
    # Criar mensagem
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    if failed == 0:
        emoji = "✅"
        status = "APROVADO"
        msg = f"{emoji} *TESTES SEMANAIS - {status}*\n\n"
        msg += f"📅 {now}\n\n"
        msg += f"Passou: {passed}/{total}\n"
        msg += f"Falhou: 0\n\n"
        msg += "🎉 *SISTEMA SEGURO PARA APOSTAS!*"
    else:
        emoji = "❌"
        status = "FALHOU"
        msg = f"{emoji} *TESTES SEMANAIS - {status}*\n\n"
        msg += f"📅 {now}\n\n"
        msg += f"Passou: {passed}/{total}\n"
        msg += f"Falhou: {failed}\n\n"
        msg += "⚠️ *NÃO APOSTAR ATÉ CORRIGIR!*\n\n"
        
        # Adicionar erros
        lines = output.split('\n')
        errors = [l for l in lines if 'FAILED' in l or 'AssertionError' in l]
        
        if errors:
            msg += "```\n"
            msg += '\n'.join(errors[:5])  # Max 5 erros
            msg += "\n```"
    
    # Enviar Telegram
    bot = Bot(token=TOKEN)
    
    try:
        if CHAT_ID != "SEU_CHAT_ID_AQUI":
            await bot.send_message(
                chat_id=CHAT_ID,
                text=msg,
                parse_mode='Markdown'
            )
            print(f"✅ Mensagem enviada para Telegram!")
        else:
            print("⚠️ Configure CHAT_ID primeiro!")
            print(msg)
    except Exception as e:
        print(f"❌ Erro ao enviar Telegram: {e}")
        print(msg)
    
    return failed == 0


if __name__ == "__main__":
    result = asyncio.run(run_tests_and_notify())
    exit(0 if result else 1)
