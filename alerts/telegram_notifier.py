"""
Telegram Notifier - сповіщення про сигнали та події
"""

import os
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class NotificationType(Enum):
    """Типи сповіщень"""
    SIGNAL = "signal"
    TRADE_OPEN = "trade_open"
    TRADE_CLOSE = "trade_close"
    TP_HIT = "tp_hit"
    SL_HIT = "sl_hit"
    SLEEPER_READY = "sleeper_ready"
    OB_FORMED = "ob_formed"
    SYSTEM = "system"
    ERROR = "error"


class TelegramNotifier:
    """Telegram bot для сповіщень"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        self.enabled = bool(self.bot_token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        # Emoji для різних типів
        self.emoji = {
            NotificationType.SIGNAL: "🎯",
            NotificationType.TRADE_OPEN: "📈",
            NotificationType.TRADE_CLOSE: "📊",
            NotificationType.TP_HIT: "✅",
            NotificationType.SL_HIT: "❌",
            NotificationType.SLEEPER_READY: "😴➡️🔥",
            NotificationType.OB_FORMED: "📦",
            NotificationType.SYSTEM: "⚙️",
            NotificationType.ERROR: "🚨",
        }
    
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Відправити повідомлення в Telegram"""
        if not self.enabled:
            print(f"[TG DISABLED] {text[:100]}...")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True
                }
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        print(f"Telegram error: {resp.status}")
                        return False
        except Exception as e:
            print(f"Telegram send error: {e}")
            return False
    
    def send_sync(self, text: str) -> bool:
        """Синхронна обгортка для send_message"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Якщо вже в async контексті
                asyncio.create_task(self.send_message(text))
                return True
            else:
                return loop.run_until_complete(self.send_message(text))
        except RuntimeError:
            # Новий event loop якщо немає
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.send_message(text))
            finally:
                loop.close()
    
    # ===== Форматовані сповіщення =====
    
    def notify_signal(self, signal: Dict[str, Any]) -> bool:
        """Сповіщення про новий сигнал"""
        emoji = self.emoji[NotificationType.SIGNAL]
        direction_emoji = "🟢" if signal.get('direction') == 'LONG' else "🔴"
        
        text = f"""
{emoji} <b>НОВИЙ СИГНАЛ</b> {emoji}

{direction_emoji} <b>{signal.get('symbol')}</b> - {signal.get('direction')}

📊 <b>Аналіз:</b>
• Sleeper Score: {signal.get('sleeper_score', 0):.1f}/100
• OB Quality: {signal.get('ob_quality', 0):.1f}/100
• Confidence: {signal.get('confidence', 0):.1f}%

💰 <b>Параметри входу:</b>
• Entry: ${signal.get('entry_price', 0):.4f}
• Stop Loss: ${signal.get('sl', 0):.4f}
• TP1 (1R): ${signal.get('tp1', 0):.4f}
• TP2 (2R): ${signal.get('tp2', 0):.4f}
• TP3 (3R): ${signal.get('tp3', 0):.4f}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send_sync(text.strip())
    
    def notify_trade_open(self, trade: Dict[str, Any]) -> bool:
        """Сповіщення про відкриття позиції"""
        emoji = self.emoji[NotificationType.TRADE_OPEN]
        direction_emoji = "🟢" if trade.get('direction') == 'LONG' else "🔴"
        mode = "📝 PAPER" if trade.get('is_paper') else "💵 LIVE"
        
        text = f"""
{emoji} <b>ПОЗИЦІЯ ВІДКРИТА</b> {mode}

{direction_emoji} <b>{trade.get('symbol')}</b> {trade.get('direction')}

• Entry: ${trade.get('entry_price', 0):.4f}
• Size: {trade.get('position_size', 0):.4f}
• Leverage: {trade.get('leverage', 1)}x
• SL: ${trade.get('sl', 0):.4f}
• TP1: ${trade.get('tp1', 0):.4f}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send_sync(text.strip())
    
    def notify_trade_close(self, trade: Dict[str, Any]) -> bool:
        """Сповіщення про закриття позиції"""
        pnl = trade.get('pnl_usdt', 0)
        pnl_pct = trade.get('pnl_percent', 0)
        
        if pnl >= 0:
            emoji = self.emoji[NotificationType.TP_HIT]
            result = "ПРОФІТ"
        else:
            emoji = self.emoji[NotificationType.SL_HIT]
            result = "ЗБИТОК"
        
        direction_emoji = "🟢" if trade.get('direction') == 'LONG' else "🔴"
        
        text = f"""
{emoji} <b>ПОЗИЦІЯ ЗАКРИТА - {result}</b>

{direction_emoji} <b>{trade.get('symbol')}</b> {trade.get('direction')}

• Entry: ${trade.get('entry_price', 0):.4f}
• Exit: ${trade.get('exit_price', 0):.4f}
• P&L: <b>${pnl:+.2f}</b> ({pnl_pct:+.2f}%)

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send_sync(text.strip())
    
    def notify_sleeper_ready(self, sleeper: Dict[str, Any]) -> bool:
        """Сповіщення про готовий Sleeper"""
        emoji = self.emoji[NotificationType.SLEEPER_READY]
        direction_emoji = "🟢" if sleeper.get('direction') == 'LONG' else "🔴"
        
        text = f"""
{emoji} <b>SLEEPER ГОТОВИЙ!</b>

{direction_emoji} <b>{sleeper.get('symbol')}</b>

📊 <b>Scores:</b>
• Total: {sleeper.get('total_score', 0):.1f}/100
• Fuel: {sleeper.get('fuel_score', 0):.1f}
• Volatility: {sleeper.get('volatility_score', 0):.1f}
• Price: {sleeper.get('price_score', 0):.1f}
• Liquidity: {sleeper.get('liquidity_score', 0):.1f}

❤️ HP: {sleeper.get('hp', 5)}/10
🎯 Direction: {sleeper.get('direction', 'NEUTRAL')}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send_sync(text.strip())
    
    def notify_ob_formed(self, ob: Dict[str, Any]) -> bool:
        """Сповіщення про новий Order Block"""
        emoji = self.emoji[NotificationType.OB_FORMED]
        ob_type = ob.get('ob_type', 'UNKNOWN')
        type_emoji = "🟢" if ob_type == 'BULLISH' else "🔴"
        
        text = f"""
{emoji} <b>ORDER BLOCK DETECTED</b>

{type_emoji} <b>{ob.get('symbol')}</b> - {ob_type}

• Timeframe: {ob.get('timeframe', '?')}
• Zone: ${ob.get('ob_low', 0):.4f} - ${ob.get('ob_high', 0):.4f}
• Quality: {ob.get('quality_score', 0):.1f}/100
• Volume Ratio: {ob.get('volume_ratio', 0):.1f}x

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send_sync(text.strip())
    
    def notify_system(self, message: str, level: str = "INFO") -> bool:
        """Системне сповіщення"""
        emoji = self.emoji[NotificationType.SYSTEM]
        if level == "ERROR":
            emoji = self.emoji[NotificationType.ERROR]
        
        text = f"""
{emoji} <b>SYSTEM [{level}]</b>

{message}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send_sync(text.strip())
    
    def notify_daily_summary(self, stats: Dict[str, Any]) -> bool:
        """Денний звіт"""
        pnl = stats.get('total_pnl', 0)
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        
        text = f"""
📊 <b>ДЕННИЙ ЗВІТ</b>

{pnl_emoji} <b>P&L: ${pnl:+.2f}</b>

📈 Статистика:
• Трейдів: {stats.get('total_trades', 0)}
• Win Rate: {stats.get('win_rate', 0):.1f}%
• Profit Factor: {stats.get('profit_factor', 0):.2f}
• Avg Win: ${stats.get('avg_win', 0):.2f}
• Avg Loss: ${stats.get('avg_loss', 0):.2f}

😴 Sleepers:
• Scanned: {stats.get('sleepers_scanned', 0)}
• Ready: {stats.get('sleepers_ready', 0)}

📦 Order Blocks:
• Detected: {stats.get('obs_detected', 0)}
• Triggered: {stats.get('obs_triggered', 0)}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        return self.send_sync(text.strip())
    
    def send_confirmation_request(self, signal_id: str, signal: Dict[str, Any]) -> bool:
        """Запит на підтвердження сигналу (для SEMI_AUTO режиму)"""
        emoji = "⚡"
        direction_emoji = "🟢" if signal.get('direction') == 'LONG' else "🔴"
        
        text = f"""
{emoji} <b>ПІДТВЕРДЖЕННЯ ПОТРІБНЕ</b> {emoji}

{direction_emoji} <b>{signal.get('symbol')}</b> {signal.get('direction')}

• Entry: ${signal.get('entry_price', 0):.4f}
• Confidence: {signal.get('confidence', 0):.1f}%

🔗 Signal ID: <code>{signal_id}</code>

Підтвердіть через dashboard або відповідайте:
/confirm {signal_id}
/reject {signal_id}

⏰ Expires in 5 min
"""
        return self.send_sync(text.strip())


# ===== Singleton =====
_notifier: Optional[TelegramNotifier] = None

def get_notifier() -> TelegramNotifier:
    """Отримати singleton instance"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
