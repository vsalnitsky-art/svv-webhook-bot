#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import threading
import time
import logging
import pandas as pd
import pandas_ta as ta
from settings_manager import settings
from bot import bot_instance

logger = logging.getLogger(__name__)

class EnhancedMarketScanner:
    def __init__(self, bot_instance, config):
        self.bot = bot_instance
        self.config = config
        self.data = {}

    def start(self):
        """Запускає фоновий потік моніторингу"""
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("✅ Enhanced Market Scanner Started")

    def loop(self):
        while True:
            try:
                self.monitor()
            except Exception as e:
                # logger.error(f"Scanner Loop Error: {e}")
                pass
            time.sleep(10)

    def get_active(self):
        try:
            r = self.bot.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode'] == 0:
                return [p for p in r['result']['list'] if float(p['size']) > 0]
        except: pass
        return []

    def fetch_candles(self, symbol, timeframe, limit=50):
        try:
            tf_map = {'5':'5','15':'15','30':'30','45':'15','60':'60','240':'240','D':'D'}
            req_tf = tf_map.get(str(timeframe), '240')
            r = self.bot.session.get_kline(category="linear", symbol=symbol, interval=req_tf, limit=limit)
            if r['retCode'] == 0:
                df = pd.DataFrame(r['result']['list'], columns=['time','open','high','low','close','vol','to'])
                df['close'] = df['close'].astype(float)
                return df.iloc[::-1].reset_index(drop=True)
        except: pass
        return None

    def monitor(self):
        active_pos = self.get_active()
        active_syms = [p['symbol'] for p in active_pos]
        
        # Чистка кешу
        for k in list(self.data.keys()):
            if k not in active_syms: del self.data[k]
        
        if not active_pos: return

        # Використовуємо глобальний таймфрейм для моніторингу
        tf = settings.get("htfSelection", "240")

        for p in active_pos:
            s = p['symbol']
            side = p['side']
            
            if s not in self.data: 
                self.data[s] = {'rsi': 0, 'exit_status': 'Safe', 'exit_details': '-'}

            df = self.fetch_candles(s, tf)
            
            if df is not None and len(df) > 20:
                # Локальний розрахунок RSI
                rsi_val = ta.rsi(df['close'], length=14).iloc[-1]
                self.data[s]['rsi'] = round(rsi_val, 1)

                # Проста візуальна індикація (Без авто-закриття)
                status = "Safe"
                details = f"RSI: {round(rsi_val, 1)}"
                
                lb = float(settings.get('obt_entryRsiOversold', 30))
                ub = float(settings.get('obt_entryRsiOverbought', 70))
                
                if side == "Buy" and rsi_val >= ub: 
                    status = "Warning"
                    details += " (High)"
                elif side == "Sell" and rsi_val <= lb: 
                    status = "Warning"
                    details += " (Low)"
                
                self.data[s]['exit_status'] = status
                self.data[s]['exit_details'] = details
            
            time.sleep(0.5)

    def get_coin_data(self, s): return self.data.get(s, {})
    def get_current_rsi(self, s): return self.data.get(s, {}).get('rsi', 0)
    def get_market_pressure(self, s): return 0
    def get_active_symbols(self): return self.get_active()