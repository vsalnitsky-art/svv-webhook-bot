"""
Telegram Notifier - сповіщення про сигнали та події
v5.0: Added per-alert-type toggle settings
"""

import os
import requests
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
    # v4.2 - New alert types
    URGENT_ALERT = "urgent_alert"
    HIGH_ALERT = "high_alert"
    VOLUME_SPIKE = "volume_spike"
    # v5.0 - Additional types
    INTENSIVE_ALERT = "intensive_alert"
    DAILY_SUMMARY = "daily_summary"


# Default alert settings (all enabled by default)
DEFAULT_ALERT_SETTINGS = {
    'alert_sleeper_ready': True,      # Sleeper став READY
    'alert_ob_formed': True,          # Order Block виявлено
    'alert_signal': True,             # Торговий сигнал
    'alert_trade_open': True,         # Позиція відкрита
    'alert_trade_close': True,        # Позиція закрита
    'alert_intensive': True,          # Intensive monitoring (volume spikes)
    'alert_daily_summary': True,      # Денний звіт
    'alert_system': True,             # Системні повідомлення
}


class TelegramNotifier:
    """Telegram bot для сповіщень"""
    
    def __init__(self):
        self._load_config()
        self._alert_settings = DEFAULT_ALERT_SETTINGS.copy()
        
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
            # v4.2 - New alert emojis
            NotificationType.URGENT_ALERT: "⚡⚡⚡",
            NotificationType.HIGH_ALERT: "🚀🔥",
            NotificationType.VOLUME_SPIKE: "📊💥",
            NotificationType.INTENSIVE_ALERT: "👀📊",
            NotificationType.DAILY_SUMMARY: "📋",
        }
    
    def load_alert_settings(self, db):
        """Load alert settings from database"""
        for key in DEFAULT_ALERT_SETTINGS:
            value = db.get_setting(key, DEFAULT_ALERT_SETTINGS[key])
            # Handle string 'true'/'false' from DB
            if isinstance(value, str):
                self._alert_settings[key] = value.lower() == 'true'
            else:
                self._alert_settings[key] = bool(value)
    
    def is_alert_enabled(self, alert_type: str) -> bool:
        """Check if specific alert type is enabled"""
        setting_key = f'alert_{alert_type}'
        return self._alert_settings.get(setting_key, True)
    
    def get_alert_settings(self) -> Dict[str, bool]:
        """Get all alert settings"""
        return self._alert_settings.copy()
    
    def _load_config(self):
        """Load/reload configuration from environment"""
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        self.enabled = bool(self.bot_token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        if self.enabled:
            print(f"[TG] Telegram notifier enabled (chat_id: {self.chat_id[:4]}...)")
        else:
            print("[TG] Telegram notifier DISABLED - missing BOT_TOKEN or CHAT_ID")
    
    def reload(self):
        """Reload configuration (call after env vars change)"""
        self._load_config()
        return self.enabled
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Відправити повідомлення в Telegram (синхронно)"""
        if not self.enabled:
            print(f"[TG DISABLED] {text[:100]}...")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            resp = requests.post(url, json=payload, timeout=10)
            
            if resp.status_code == 200:
                print(f"[TG] Message sent successfully")
                return True
            else:
                # Log detailed error
                error_data = resp.json() if resp.text else {}
                error_desc = error_data.get('description', 'Unknown error')
                print(f"[TG ERROR] Status {resp.status_code}: {error_desc}")
                return False
                
        except Exception as e:
            print(f"[TG ERROR] Send failed: {e}")
            return False
    
    def send_sync(self, text: str) -> bool:
        """Синхронна обгортка для send_message (alias)"""
        return self.send_message(text)
    
    # ===== Форматовані сповіщення =====
    
    def notify_signal(self, signal: Dict[str, Any]) -> bool:
        """Сповіщення про новий сигнал"""
        if not self.is_alert_enabled('signal'):
            return False
        
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
        if not self.is_alert_enabled('trade_open'):
            return False
        
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
        if not self.is_alert_enabled('trade_close'):
            return False
        
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
        """Сповіщення про готовий Sleeper - v5 з phase/exhaustion інформацією"""
        if not self.is_alert_enabled('sleeper_ready'):
            return False
        
        emoji = self.emoji[NotificationType.SLEEPER_READY]
        direction = sleeper.get('direction', 'NEUTRAL')
        direction_emoji = "🟢" if direction == 'LONG' else ("🔴" if direction == 'SHORT' else "⚪")
        
        # v5: Phase and reversal info
        market_phase = sleeper.get('market_phase', 'UNKNOWN')
        phase_maturity = sleeper.get('phase_maturity', 'MIDDLE')
        is_reversal = sleeper.get('is_reversal_setup', False)
        exhaustion_score = sleeper.get('exhaustion_score', 0)
        
        # Phase emoji
        phase_emoji = {
            'ACCUMULATION': '📥',  # Buying at bottom
            'MARKUP': '📈',        # Uptrend
            'DISTRIBUTION': '📤', # Selling at top
            'MARKDOWN': '📉',      # Downtrend
            'UNKNOWN': '❓'
        }.get(market_phase, '❓')
        
        # Maturity indicator
        maturity_indicator = {
            'EARLY': '🌱',
            'MIDDLE': '🌿',
            'LATE': '🍂',
            'EXHAUSTED': '💀'
        }.get(phase_maturity, '❓')
        
        # Reversal badge
        reversal_badge = "🔄 REVERSAL SETUP!" if is_reversal else ""
        
        # Build message
        text = f"""
{emoji} <b>SLEEPER ГОТОВИЙ!</b>
{reversal_badge}

{direction_emoji} <b>{sleeper.get('symbol')}</b>

📊 <b>Scores:</b>
• Total: {sleeper.get('total_score', 0):.1f}/100
• Direction Score: {sleeper.get('direction_score', 0):+.2f}
• Confidence: {sleeper.get('direction_confidence', 'LOW')}

{phase_emoji} <b>Phase:</b> {market_phase} {maturity_indicator}
• Maturity: {phase_maturity}
• Exhaustion: {exhaustion_score*100:.0f}%

📍 <b>Position:</b>
• From High: -{sleeper.get('distance_from_high', 0):.1f}%
• From Low: +{sleeper.get('distance_from_low', 0):.1f}%

❤️ HP: {sleeper.get('hp', 5)}/10
🎯 Direction: <b>{direction}</b>
💡 {sleeper.get('direction_reason', 'No specific reason')[:50]}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send_sync(text.strip())
    
    def notify_ob_formed(self, ob: Dict[str, Any]) -> bool:
        """Сповіщення про новий Order Block"""
        if not self.is_alert_enabled('ob_formed'):
            return False
        
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
        if not self.is_alert_enabled('system'):
            return False
        
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
        if not self.is_alert_enabled('daily_summary'):
            return False
        
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
