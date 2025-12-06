#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import threading
import time
import pandas as pd
import logging
from bot import bot_instance
from settings_manager import settings
from models import db_manager, AnalysisResult
# ✅ Імпорт локальних індикаторів
from indicators import simple_rsi

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status_message = "Ready"

    def get_top_tickers(self, limit=100):
        """Отримує тікери з урахуванням фільтру об'єму (USDT)"""
        try:
            quote_coin = settings.get("scanner_quote_coin", "USDT")
            use_vol_filter = settings.get("scan_use_min_volume")
            
            # Отримуємо всі тікери
            all_tickers = bot_instance.get_all_tickers()
            # Фільтр за базовою валютою
            target_tickers = [t for t in all_tickers if t['symbol'].endswith(quote_coin)]
            
            # Фільтрація по мін. об'єму
            if use_vol_filter:
                try:
                    min_vol_mln = float(settings.get("scan_min_volume", 10))
                    min_vol_raw = min_vol_mln * 1_000_000
                    target_tickers = [t for t in target_tickers if float(t.get('turnover24h', 0) or 0) >= min_vol_raw]
                except: pass

            # Сортуємо за об'ємом (найліквідніші спочатку)
            return sorted(target_tickers, key=lambda x: float(x.get('turnover24h', 0) or 0), reverse=True)[:int(limit)]
        except Exception as e:
            logger.error(f"Error getting tickers: {e}")
            return []

    def fetch_candles(self, symbol, timeframe, limit=50):
        """
        Автономне завантаження свічок.
        ✅ ДОДАНО: Ресемплінг для 45-хвилинного таймфрейму (з 15м даних)
        """
        try:
            # Мапінг таймфреймів для Bybit API
            # Якщо просять 45, ми беремо 15 (бо 45 немає на біржі)
            tf_map = {'5':'5','15':'15','30':'30','45':'15','60':'60','120':'120','240':'240','D':'D','W':'W'}
            req_tf = tf_map.get(str(timeframe), '240')
            
            # Якщо потрібен 45хв ТФ, нам треба в 3 рази більше свічок 15хв
            req_limit = limit * 3 if str(timeframe) == '45' else limit
            
            r = bot_instance.session.get_kline(category="linear", symbol=symbol, interval=req_tf, limit=req_limit)
            
            if r['retCode'] == 0 and r['result']['list']:
                df = pd.DataFrame(r['result']['list'], columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                
                # Конвертуємо типи даних
                cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
                df[cols] = df[cols].astype(float)
                
                # Обробка часу для ресемплінгу
                df['time'] = pd.to_numeric(df['time'])
                df['datetime'] = pd.to_datetime(df['time'], unit='ms')
                
                # Сортуємо: Старі -> Нові (важливо для pandas resample)
                df = df.sort_values('datetime').reset_index(drop=True)
                
                # === ЛОГІКА РЕСЕМПЛІНГУ 45 ХВИЛИН ===
                if str(timeframe) == '45':
                    # Встановлюємо час як індекс
                    df.set_index('datetime', inplace=True)
                    
                    # Склеюємо по 45 хвилин
                    df_45 = df.resample('45min').agg({
                        'open': 'first',   # Ціна відкриття першої свічки
                        'high': 'max',     # Макс ціна за 3 свічки
                        'low': 'min',      # Мін ціна за 3 свічки
                        'close': 'last',   # Ціна закриття останньої свічки
                        'volume': 'sum',   # Сума об'єму
                        'turnover': 'sum',
                        'time': 'first'    # Timestamp початку
                    })
                    
                    # Видаляємо порожні (якщо є дірки в історії)
                    df_45.dropna(inplace=True)
                    
                    # Повертаємо індекс назад
                    df = df_45.reset_index(drop=True)
                
                return df
                
        except Exception as e:
            # logger.error(f"Error fetching candles for {symbol}: {e}")
            pass
        return None

    def run_scan_thread(self):
        if not self.is_scanning: 
            threading.Thread(target=self._scan_process, daemon=True).start()

    def _scan_process(self):
        self.is_scanning = True
        self.progress = 0
        self.status_message = "🚀 Starting Autonomous Scan..."
        session = db_manager.get_session()
        
        try:
            # Очищаємо таблицю результатів перед новим сканом
            db_manager.recreate_analysis_table()
            
            # 1. Зчитуємо налаштування
            limit = settings.get("scan_limit", 100)
            tickers = self.get_top_tickers(limit)
            
            # Глобальний таймфрейм
            htf = settings.get("htfSelection", "240") 
            
            # Параметри RSI
            rsi_len = int(settings.get("obt_rsiLength", 14))
            rsi_buy_level = float(settings.get("obt_entryRsiOversold", 30))
            rsi_sell_level = float(settings.get("obt_entryRsiOverbought", 70))

            total = len(tickers)
            print(f"🔎 Scanning {total} tickers on TF {htf}...")

            for i, t in enumerate(tickers):
                if not self.is_scanning: break
                
                sym = t['symbol']
                vol_24h = float(t.get('turnover24h', 0))
                
                self.status_message = f"Scanning {sym} ({i+1}/{total})"
                self.progress = int((i / total) * 100)

                try:
                    # 2. Отримуємо дані (вже з підтримкою 45 хв)
                    # Беремо трохи більше свічок для розігріву індикатора
                    df = self.fetch_candles(sym, htf, limit=rsi_len + 50)
                    
                    if df is None or len(df) < rsi_len: 
                        time.sleep(0.05)
                        continue

                    # 3. Розрахунок RSI (використовуємо виправлений indicators.py)
                    current_rsi = simple_rsi(df['close'], period=rsi_len)
                    current_price = df['close'].iloc[-1]

                    # 4. Логіка Сигналів (RSI Extremes)
                    signal = None
                    details = ""
                    score = 0 

                    # BUY SIGNAL (Перепроданість)
                    if current_rsi <= rsi_buy_level:
                        signal = "Buy"
                        details = f"RSI Oversold ({round(current_rsi, 1)})"
                        # Score: база 50 + наскільки сильно впав
                        dist = rsi_buy_level - current_rsi
                        score = min(50 + int(dist * 2), 100)

                    # SELL SIGNAL (Перекупленість)
                    elif current_rsi >= rsi_sell_level:
                        signal = "Sell"
                        details = f"RSI Overbought ({round(current_rsi, 1)})"
                        # Score: база 50 + наскільки сильно виріс
                        dist = current_rsi - rsi_sell_level
                        score = min(50 + int(dist * 2), 100)

                    # 5. Збереження результату
                    if signal:
                        res = AnalysisResult(
                            symbol=sym,
                            signal_type=signal,
                            status="New",
                            score=score,
                            price=float(current_price),
                            htf_rsi=float(current_rsi), 
                            ltf_rsi=0.0,
                            volume_24h=vol_24h,
                            details=details
                        )
                        session.add(res)
                        session.commit()
                        logger.info(f"✅ FOUND: {sym} {signal} RSI={current_rsi}")
                        
                except Exception as e:
                    # logger.error(f"Scan error on {sym}: {e}")
                    pass
                
                time.sleep(0.05) # Захист від Rate Limit

            self.progress = 100
            self.status_message = "✅ Scan Completed"

        except Exception as e:
            logger.error(f"Global Scan Error: {e}")
            self.status_message = "❌ Error"
        finally:
            self.is_scanning = False
            session.close()

    def get_results(self):
        s = db_manager.get_session()
        try:
            results = s.query(AnalysisResult).all()
            return [
                {
                    'symbol': r.symbol,
                    'signal': r.signal_type,
                    'score': r.score,
                    'price': r.price,
                    'rsi_htf': round(r.htf_rsi, 1),
                    'volume': r.volume_24h,
                    'time': r.found_at.strftime('%H:%M'),
                    'details': r.details
                }
                for r in results
            ]
        finally:
            s.close()

market_analyzer = MarketAnalyzer()