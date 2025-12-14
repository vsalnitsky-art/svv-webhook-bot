#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import threading
import time
import logging
import pandas as pd
from datetime import datetime
from flask import render_template, request, jsonify

# Imports from main project
from bot import bot_instance
from settings_manager import settings
from models import db_manager, Base
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean

# Imports from PRO indicators
from indicators_pro import calculate_rvol, check_ttm_squeeze, calculate_adx
from indicators import calculate_ema, calculate_obv, calculate_slope, simple_rsi

logger = logging.getLogger("WhalePro")

# === MODEL DEFINITION (Append Only) ===
# Ми визначаємо модель тут, але вона буде створена динамічно при першому запуску
class WhaleProSignal(Base):
    __tablename__ = 'whale_pro_signals'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    price = Column(Float)
    score = Column(Integer)
    
    # Нові метрики
    rvol = Column(Float)
    adx = Column(Float)
    is_squeeze = Column(Boolean)
    btc_trend = Column(String(10)) # BULL / BEAR
    
    details = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

class WhaleProCore:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status = "Ready"
        self.last_scan_time = None
        self.btc_trend_status = "UNKNOWN"
        
        # Налаштування PRO
        self.CONFIG = {
            "timeframe": "240",        # 4H (рекомендовано для Pro)
            "limit_coins": 50,         # Топ 50 за ліквідністю
            "min_vol_usdt": 20000000,  # 20M USDT (фільтр сміття)
            "rvol_min": 1.5,           # Об'єм в 1.5 рази вище середнього
            "adx_min": 20              # Сила тренду
        }

    def ensure_table(self):
        """Створює таблицю для Pro сигналів якщо не існує"""
        try:
            WhaleProSignal.__table__.create(db_manager.engine, checkfirst=True)
        except Exception as e:
            logger.error(f"Table creation error: {e}")

    def fetch_data(self, symbol, limit=300):
        """Завантажує дані (сумісність з indicators.py)"""
        try:
            # Мапінг
            tf_map = {'15': '15', '60': '60', '240': '240', 'D': 'D'}
            interval = tf_map.get(str(self.CONFIG['timeframe']), '240')
            
            res = bot_instance.session.get_kline(
                category="linear", symbol=symbol, interval=interval, limit=limit
            )
            
            if res['retCode'] == 0 and res['result']['list']:
                # Reverse to Oldest -> Newest
                data = res['result']['list'][::-1]
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.astype(float)
                
                # Exclude last unfinished candle
                if len(df) > 1:
                    df = df.iloc[:-1].reset_index(drop=True)
                return df
        except Exception as e:
            # logger.error(f"Fetch error {symbol}: {e}")
            pass
        return None

    def analyze_btc_trend(self):
        """
        КРОК 1: Перевірка здоров'я ринку (BTC).
        Якщо BTC під EMA 200 на 4H - це небезпечний ринок.
        """
        try:
            # logger.info("🔍 Analyzing BTC Trend...")
            df = self.fetch_data("BTCUSDT", limit=300)
            if df is None: return "NEUTRAL"
            
            close = df['close']
            ema200 = calculate_ema(close, 200).iloc[-1]
            price = close.iloc[-1]
            
            if price > ema200:
                self.btc_trend_status = "BULLISH"
            else:
                self.btc_trend_status = "BEARISH"
                
            logger.info(f"✅ BTC Trend: {self.btc_trend_status} (Price: {price}, EMA200: {round(ema200)})")
            return self.btc_trend_status
        except Exception as e:
            logger.error(f"BTC Trend Error: {e}")
            return "NEUTRAL"

    def analyze_ticker_pro(self, symbol, df):
        """
        КРОК 2: Глибокий аналіз (RVOL, Squeeze, ADX)
        """
        try:
            close = df['close']
            volume = df['volume']
            
            # --- 1. RVOL (Відносний об'єм) ---
            rvol_series = calculate_rvol(volume, 20)
            curr_rvol = rvol_series.iloc[-1]
            
            # --- 2. TTM Squeeze (Справжній) ---
            squeeze_series = check_ttm_squeeze(df)
            is_squeeze = bool(squeeze_series.iloc[-1])
            
            # --- 3. ADX (Сила тренду) ---
            adx_series, _, _ = calculate_adx(df['high'], df['low'], close, 14)
            curr_adx = adx_series.iloc[-1]
            
            # --- 4. OBV Slope ---
            obv = calculate_obv(close, volume)
            obv_slope = calculate_slope(obv, 10)
            price_slope = calculate_slope(close, 10)
            
            # === SCORING LOGIC ===
            score = 0
            reasons = []
            
            # Логіка для BULL ринку
            if self.btc_trend_status in ["BULLISH", "NEUTRAL"]:
                # RVOL Filter
                if curr_rvol >= self.CONFIG['rvol_min']:
                    score += 30
                    reasons.append(f"High Vol (x{round(curr_rvol, 1)})")
                
                # Squeeze Filter
                if is_squeeze:
                    score += 20
                    reasons.append("TTM Squeeze")
                
                # Accumulation (Price Flat/Down, OBV Up)
                # Нормалізуємо нахил ціни
                if close.iloc[-1] > 0:
                    norm_p_slope = (price_slope / close.iloc[-1]) * 100
                else:
                    norm_p_slope = 0
                    
                if norm_p_slope < 0.1 and obv_slope > 0:
                    score += 30
                    reasons.append("Accumulation")
                
                # ADX Check (Trend start)
                if curr_adx < 25: # Low ADX = Consolidation (good for entry)
                    score += 10
                elif curr_adx > 25 and len(adx_series) > 1 and curr_adx > adx_series.iloc[-2]: # Trend gaining strength
                    score += 15
                    reasons.append("Trend Start")

            # Якщо Score високий, повертаємо результат
            if score >= 40:
                return {
                    "score": min(score, 100),
                    "rvol": round(curr_rvol, 2),
                    "is_squeeze": is_squeeze,
                    "adx": round(curr_adx, 1),
                    "price": close.iloc[-1],
                    "reason": " + ".join(reasons)
                }
            
            return None

        except Exception as e:
            # logger.error(f"Analyze {symbol} error: {e}") 
            return None

    def save_result(self, symbol, data):
        session = db_manager.get_session()
        try:
            sig = WhaleProSignal(
                symbol=symbol,
                price=data['price'],
                score=data['score'],
                rvol=data['rvol'],
                adx=data['adx'],
                is_squeeze=data['is_squeeze'],
                btc_trend=self.btc_trend_status,
                details=data['reason'],
                created_at=datetime.utcnow()
            )
            session.add(sig)
            session.commit()
        except Exception as e:
            session.rollback()
        finally:
            session.close()

    def get_history(self):
        """Отримує історію для UI"""
        self.ensure_table()
        session = db_manager.get_session()
        try:
            res = session.query(WhaleProSignal).order_by(WhaleProSignal.created_at.desc()).limit(100).all()
            return [{
                'symbol': r.symbol,
                'price': r.price,
                'score': r.score,
                'rvol': r.rvol,
                'adx': r.adx,
                'is_squeeze': r.is_squeeze,
                'btc_trend': r.btc_trend,
                'details': r.details,
                'time': r.created_at.strftime('%H:%M')
            } for r in res]
        finally:
            session.close()

    def _scan_thread(self):
        self.is_scanning = True
        self.progress = 0
        self.status = "Initializing Pro Scan..."
        
        # 1. Clear old results
        self.ensure_table()
        session = db_manager.get_session()
        try:
            session.query(WhaleProSignal).delete()
            session.commit()
        except:
            session.rollback()
        finally:
            session.close()
        
        try:
            # 2. Check BTC
            self.status = "Checking BTC Trend..."
            self.analyze_btc_trend()
            self.progress = 10
            
            # 3. Get Tickers
            self.status = "Filtering Liquidity..."
            all_tickers = bot_instance.get_all_tickers()
            targets = [t for t in all_tickers if t['symbol'].endswith('USDT')]
            
            # Ліквідність > 20M (Pro Filter)
            targets = [t for t in targets if float(t.get('turnover24h', 0)) > self.CONFIG['min_vol_usdt']]
            targets.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            
            # Беремо топ ліквідних
            targets = targets[:self.CONFIG['limit_coins']]
            total = len(targets)
            
            # 4. Scan Loop
            for i, t in enumerate(targets):
                if not self.is_scanning: break
                
                sym = t['symbol']
                self.status = f"Analysing {sym} (RVOL/ADX)..."
                self.progress = 10 + int((i / total) * 90)
                
                df = self.fetch_data(sym)
                if df is not None:
                    res = self.analyze_ticker_pro(sym, df)
                    if res:
                        self.save_result(sym, res)
                
                time.sleep(0.05)
            
            self.progress = 100
            self.status = "Pro Scan Completed"
            self.last_scan_time = datetime.now().strftime("%H:%M")
            
        except Exception as e:
            logger.error(f"Pro Scan Error: {e}")
            self.status = "Error"
        finally:
            self.is_scanning = False

    def start_scan(self, config=None):
        if self.is_scanning: return False
        if config:
            # Оновлюємо конфіг, конвертуємо типи
            if 'min_vol_usdt' in config:
                try: self.CONFIG['min_vol_usdt'] = int(config['min_vol_usdt'])
                except: pass
            if 'rvol_min' in config:
                try: self.CONFIG['rvol_min'] = float(config['rvol_min'])
                except: pass
            if 'timeframe' in config:
                self.CONFIG['timeframe'] = str(config['timeframe'])
                
        threading.Thread(target=self._scan_thread).start()
        return True

whale_pro = WhaleProCore()

# === ROUTES (To be registered in main_app) ===
def register_routes(app):
    @app.route('/whale_pro')
    def whale_pro_page():
        history = whale_pro.get_history()
        return render_template('whale_pro.html', 
                             history=history,
                             status=whale_pro.status,
                             progress=whale_pro.progress,
                             is_scanning=whale_pro.is_scanning,
                             btc_trend=whale_pro.btc_trend_status,
                             last_time=whale_pro.last_scan_time,
                             conf=settings._cache)

    @app.route('/whale_pro/scan', methods=['POST'])
    def whale_pro_scan():
        data = request.json or {}
        whale_pro.start_scan(data)
        return jsonify({'status': 'started'})
