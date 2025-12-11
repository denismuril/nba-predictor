
import os
import asyncio
from telegram import Bot
from dotenv import load_dotenv

load_dotenv()

async def send_test():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = 1131106666
    if not token:
        print("❌ Token not found")
        return

    bot = Bot(token=token)
    try:
        await bot.send_message(chat_id=chat_id, text="🤖 *Bot Reiniciado!*\n\n✅ Versão 20.6 Ativa\n✅ Totais (Over/Under) Adicionados\n✅ Modelo V6 (Probabilidades Calibradas)\n\nDigite /jogos para ver as novas predições!", parse_mode='Markdown')
        print("✅ Message sent successfully")
    except Exception as e:
        print(f"❌ Error sending message: {e}")

if __name__ == "__main__":
    asyncio.run(send_test())
