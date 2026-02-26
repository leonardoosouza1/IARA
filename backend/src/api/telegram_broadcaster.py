import os
import logging
import asyncio
import aiohttp
from typing import Dict, Any

logger = logging.getLogger("api.telegram_broadcaster")

class TelegramBroadcaster:
    """
    SaaS Broadcaster Phase 5 (Telegram VIP)
    Listens for high-confidence 'Sniper Mode' signals from IARA and dispatches them 
    asynchronously to a Telegram Chat/Group.
    """
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.active = bool(self.bot_token and self.chat_id)
        
        # Sniper Threshold for Public Alerts
        self.broadcast_threshold = 0.82 

        if self.active:
            logger.info(f"📣 SaaS Telegram Broadcaster ON. Dest: {self.chat_id}")
        else:
            logger.warning("Telegram Bot missing credentials. SaaS Broadcasting disabled.")

    async def broadcast_signal(self, symbol: str, direction: str, confidence: float, current_price: float):
        """
        Receives raw JSON from the IARA Model and filters out low-confidence setups.
        If valid, translates to human-readable text and fires to Telegram.
        """
        if not self.active:
            return

        if confidence < self.broadcast_threshold:
            logger.debug(f"Signal {direction} ({confidence:.2f}) below VIP threshold {self.broadcast_threshold}. Ignoring.")
            return

        # Map emojis
        emoji = "🟢 LONG" if direction.upper() == "BUY" else "🔴 SHORT"
        
        # Message Formatting
        message = (
            f"⚡ **SACI BRAIN - ALERTA SNIPER** ⚡\n\n"
            f"🎯 **Ativo:** #{symbol.upper()}\n"
            f"🛡️ **Direção:** {emoji}\n"
            f"🧠 **Confiança da IA:** {confidence * 100:.1f}%\n"
            f"💲 **Preço Estimado:** ${current_price:.2f}\n\n"
            f"⚠️ *Atenção: Gerencie seu risco utilizando stop-loss. Isso é um sinal preditivo autônomo.*"
        )

        # Dispatch Asynchronously
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
             async with aiohttp.ClientSession() as session:
                 async with session.post(url, json=payload) as response:
                     if response.status == 200:
                         logger.info(f"✅ Telegram VIP Alert dispatched! ({symbol} - {direction})")
                     else:
                         error_msg = await response.text()
                         logger.error(f"❌ Telegram API Error ({response.status}): {error_msg}")
        except Exception as e:
             logger.error(f"Failed to post to Telegram: {e}")

# Async Test Trigger
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    async def test_broadcast():
        bot = TelegramBroadcaster()
        # Force active for testing if env vars missing
        if not bot.active:
            print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        else:
            print("Publishing test signal...")
            await bot.broadcast_signal("BTCUSDT", "BUY", 0.88, 97500.50)
            
    asyncio.run(test_broadcast())
