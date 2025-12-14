#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import threading
import time
import logging
import pandas as pd
from datetime import datetime
from flask import render_template, request, jsonify

# Imports
from bot import bot_instance
from settings_manager import settings
from models import db_manager, Base
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean

# Pro Indicators
from indicators_pro import calculate_rvol, check_ttm_squeeze, calculate_adx
from indicators import calculate_ema, calculate_obv, calculate_slope

logger = logging.getLogger("WhalePro")

# Model
class WhaleProSignal(Base):
    __tablename__ = 'whale_pro_signals'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    price = Column(Float)
    score = Column(Integer)
    rvol = Column(Float)
    adx = Column(Float)
    is_squeeze = Column(Boolean)
    btc_trend = Column(String(10)) 
    details = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

class WhaleProCore:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status = "Ready"
        self.last_scan_time = None
        self.btc_trend_status = "UNKNOWN"
        self.CONFIG = {
            "timeframe": "240",
            "limit_coins": 50,
            "min_vol_usdt": 20000000,
            "rvol_min": 1.5,
            "adx_min": 20
        }

    def ensure_table(self):
        try:
            WhaleProSignal.__table__.create(db_manager.engine, checkfirst=True)
        except: pass

    def fetch_data(self, symbol, limit=300):
        try:
            tf_map = {'15': '15', '60': '60', '240': '240', 'D': 'D'}
            interval = tf_map.get(str(self.CONFIG['timeframe']), '240')
            res = bot_instance.session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
            if res['retCode'] == 0 and res['result']['list']:
                data = res['result']['list'][::-1]
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.astype(float)
                if len(df) > 1: df = df.iloc[:-1].reset_index(drop=True)
                return df
        except: pass
        return None

    def analyze_btc_trend(self):
        try:
            df = self.fetch_data("BTCUSDT", limit=300)
            if df is None: return "NEUTRAL"
            close = df['close']
            ema200 = calculate_ema(close, 200).iloc[-1]
            if close.iloc[-1] > ema200: self.btc_trend_status = "BULLISH"
            else: self.btc_trend_status = "BEARISH"
            return self.btc_trend_status
        except: return "NEUTRAL"

    def analyze_ticker_pro(self, symbol, df):
        try:
            close = df['close']
            vol = df['volume']
            
            # Indicators
            rvol_series = calculate_rvol(vol, 20)
            curr_rvol = rvol_series.iloc[-1]
            
            sqz_series = check_ttm_squeeze(df)
            is_sqz = bool(sqz_series.iloc[-1])
            
            adx_series, _, _ = calculate_adx(df['high'], df['low'], close, 14)
            curr_adx = adx_series.iloc[-1]
            
            obv = calculate_obv(close, vol)
            obv_slope = calculate_slope(obv, 10)
            p_slope = calculate_slope(close, 10)
            
            score = 0
            reasons = []
            
            if self.btc_trend_status in ["BULLISH", "NEUTRAL"]:
                if curr_rvol >= self.CONFIG['rvol_min']:
                    score += 30; reasons.append(f"Vol x{round(curr_rvol,1)}")
                if is_sqz:
                    score += 20; reasons.append("Squeeze")
                
                norm_p_slope = (p_slope / close.iloc[-1]) * 100
                if norm_p_slope < 0.1 and obv_slope > 0:
                    score += 30; reasons.append("Accumulation")
                
                if curr_adx < 25: score += 10
                elif curr_adx > 25 and curr_adx > adx_series.iloc[-2]:
                    score += 15; reasons.append("Trend Start")

            if score >= 40:
                return {
                    "score": min(score, 100),
                    "rvol": round(curr_rvol, 2),
                    "is_squeeze": is_sqz,
                    "adx": round(curr_adx, 1),
                    "price": close.iloc[-1],
                    "reason": " + ".join(reasons)
                }
        except: pass
        return None

    def save_result(self, symbol, data):
        session = db_manager.get_session()
        try:
            s = WhaleProSignal(symbol=symbol, price=data['price'], score=data['score'], rvol=data['rvol'], adx=data['adx'], is_squeeze=data['is_squeeze'], btc_trend=self.btc_trend_status, details=data['reason'])
            session.add(s); session.commit()
        except: session.rollback()
        finally: session.close()

    def get_history(self):
        self.ensure_table()
        s = db_manager.get_session()
        try:
            res = s.query(WhaleProSignal).order_by(WhaleProSignal.created_at.desc()).limit(100).all()
            return [{'symbol':r.symbol, 'price':r.price, 'score':r.score, 'rvol':r.rvol, 'adx':r.adx, 'is_squeeze':r.is_squeeze, 'btc_trend':r.btc_trend, 'details':r.details, 'time':r.created_at.strftime('%H:%M')} for r in res]
        finally: s.close()

    def _scan_thread(self):
        self.is_scanning = True; self.progress = 0; self.status = "Init..."
        self.ensure_table()
        session = db_manager.get_session()
        session.query(WhaleProSignal).delete(); session.commit(); session.close()
        
        try:
            self.analyze_btc_trend()
            tickers = bot_instance.get_all_tickers()
            targets = [t for t in tickers if t['symbol'].endswith('USDT') and float(t.get('turnover24h',0)) > self.CONFIG['min_vol_usdt']]
            targets.sort(key=lambda x: float(x.get('turnover24h',0)), reverse=True)
            targets = targets[:self.CONFIG['limit_coins']]
            
            for i, t in enumerate(targets):
                if not self.is_scanning: break
                self.status = f"Analysing {t['symbol']}..."
                self.progress = int((i/len(targets))*100)
                df = self.fetch_data(t['symbol'])
                if df is not None:
                    res = self.analyze_ticker_pro(t['symbol'], df)
                    if res: self.save_result(t['symbol'], res)
                time.sleep(0.05)
            self.status = "Completed"; self.progress = 100; self.last_scan_time = datetime.now().strftime("%H:%M")
        except: self.status = "Error"
        finally: self.is_scanning = False

    def start_scan(self, cfg=None):
        if self.is_scanning: return
        if cfg: self.CONFIG.update(cfg)
        threading.Thread(target=self._scan_thread).start()

whale_pro = WhaleProCore()

def register_routes(app):
    @app.route('/whale_pro')
    def whale_pro_page():
        return render_template('whale_pro.html', history=whale_pro.get_history(), status=whale_pro.status, progress=whale_pro.progress, is_scanning=whale_pro.is_scanning, btc_trend=whale_pro.btc_trend_status, last_time=whale_pro.last_scan_time, conf=settings._cache)

    @app.route('/whale_pro/scan', methods=['POST'])
    def whale_pro_scan():
        whale_pro.start_scan(request.json or {})
        return jsonify({'status': 'started'})
