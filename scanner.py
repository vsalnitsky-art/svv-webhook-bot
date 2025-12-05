#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
from smart_exit_strategy import smart_exit

logger = logging.getLogger(__name__)

class EnhancedMarketScanner:
    """Сканер ринку з інтегрованим Smart Exit"""
    
    def __init__(self, bot):
        self.bot = bot
        self.active_coins_data = {}
        self.last_check = {}
    
    def get_active_symbols(self):
        """Отримати активні позиції"""
        positions = []
        try:
            resp = self.bot.session.get_positions(category="linear", settleCoin="USDT")
            if resp['retCode'] == 0:
                for p in resp['result']['list']:
                    if float(p['size']) > 0:
                        positions.append(p)
        except Exception as e:
            logger.error(f"Active symbols error: {e}")
        return positions
    
    def get_active_symbols_list(self):
        """Отримати тільки символи"""
        return [p['symbol'] for p in self.get_active_symbols()]
    
    def fetch_htf_candles(self, symbol, timeframe='1h', limit=100):
        """Завантажити HTF свічки"""
        try:
            interval_map = {
                '1h': '60', '4h': '240', '1d': 'D'
            }
            interval = interval_map.get(timeframe, '60')
            
            resp = self.bot.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if resp['retCode'] != 0:
                return None
            
            data = resp['result']['list']
            if not data:
                return None
            
            df = pd.DataFrame([
                {
                    'open': float(x[1]),
                    'high': float(x[2]),
                    'low': float(x[3]),
                    'close': float(x[4]),
                    'volume': float(x[5])
                }
                for x in reversed(data)
            ])
            
            # Обчислюємо індикатори
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['hma_fast'] = ta.hma(df['close'], length=9)
            df['hma_slow'] = ta.hma(df['close'], length=21)
            
            return df
        
        except Exception as e:
            logger.error(f"Fetch candles error: {e}")
            return None
    
    def monitor_positions(self):
        """✅ ІНТЕГРОВАНИЙ SMART EXIT"""
        
        try:
            active_positions = self.get_active_symbols()
            
            if not active_positions:
                return
            
            for pos in active_positions:
                symbol = pos['symbol']
                side = pos['side']  # 'Buy' або 'Sell'
                current_price = float(pos['markPrice'])
                
                # Завантажуємо HTF свічки
                df_htf = self.fetch_htf_candles(symbol)
                
                if df_htf is None or len(df_htf) < 2:
                    continue
                
                # Отримуємо RSI
                rsi_value = df_htf['rsi'].iloc[-1]
                
                # ✅ SMART EXIT: Оновлюємо позицію
                exit_signal = smart_exit.update_position(
                    symbol=symbol,
                    current_price=current_price,
                    rsi_value=rsi_value,
                    side='Long' if side == 'Buy' else 'Short'
                )
                
                # ✅ Перевіряємо сигнал на закриття
                if exit_signal['should_close']:
                    logger.warning(
                        f"🚨 SMART EXIT: {symbol}\n"
                        f"   Reason: {exit_signal['reason']}\n"
                        f"   Price: {exit_signal['exit_price']}\n"
                        f"   Profit: {exit_signal['profit_potential']:.2f}%"
                    )
                    
                    # Закриваємо позицію
                    result = self.bot.place_order({
                        "action": "Close",
                        "symbol": symbol,
                        "direction": "Long" if side == "Buy" else "Short"
                    })
                    
                    if result.get("status") == "ok":
                        logger.info(f"✅ Closed: {symbol}")
                        smart_exit.reset_position(symbol)
        
        except Exception as e:
            logger.error(f"Monitor error: {e}")
    
    def loop(self):
        """Основний цикл сканера"""
        self.monitor_positions()

scanner = EnhancedMarketScanner(None)
