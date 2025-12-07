#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import threading
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import desc

# Імпорти з нашого проекту
from bot import bot_instance
from models import db_manager, WhaleSignal
from indicators import (
    calculate_ema, calculate_bollinger_bands, 
    calculate_obv, calculate_ichimoku, calculate_slope
)

logger = logging.getLogger("WhaleCore")

class WhaleCore:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status = "Ready"
        self.last_scan_time = None
        
        # Налаштування стратегії
        self.CONFIG = {
            "timeframe": "60",       # 1H (60 min)
            "limit_coins": 50,       # Скільки монет сканувати
            "min_vol_usdt": 5000000, # Мін об'єм 5 млн
            "bb_squeeze_max": 0.15,  # Максимальна ширина каналу (15%)
            "ema_period": 200,       # Тренд
            "obv_slope_min": 0.0     # Дивергенція (OBV росте)
        }

    def fetch_data(self, symbol):
        """Отримує свічки"""
        try:
            # Мапінг для Bybit
            tf_map = {'15': '15', '60': '60', '240': '240', 'D': 'D'}
            interval = tf_map.get(str(self.CONFIG['timeframe']), '60')
            
            res = bot_instance.session.get_kline(
                category="linear", symbol=symbol, interval=interval, limit=300
            )
            
            if res['retCode'] == 0 and res['result']['list']:
                # Bybit повертає Newest -> Oldest. Перевертаємо
                data = res['result']['list'][::-1]
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                return df.astype(float)
        except Exception as e:
            # logger.error(f"Fetch error for {symbol}: {e}")
            pass
        return None

    def analyze_coin(self, df):
        """
        Аналіз монети: Squeeze + OBV Divergence + Ichimoku Spring
        """
        try:
            close = df['close']
            
            # --- 1. Фільтр Тренду (EMA 200) ---
            # Рахуємо EMA 200 (використовуючи функцію з indicators.py, яка повертає series)
            ema_series = calculate_ema(close, self.CONFIG['ema_period'])
            ema_val = ema_series.iloc[-1]
            price = close.iloc[-1]
            
            # Відстань до EMA. Якщо ціна глибоко під EMA (>15%), це небезпечний даунтренд.
            dist_to_ema = (price - ema_val) / ema_val
            if dist_to_ema < -0.15: return None

            # --- 2. Фільтр Волатильності (Squeeze) ---
            _, _, _, bandwidth = calculate_bollinger_bands(close)
            curr_sqz = bandwidth.iloc[-1]
            
            # Якщо канал ширший за поріг (0.15), пружина розслаблена
            if curr_sqz > self.CONFIG['bb_squeeze_max']:
                return None 

            # --- 3. Акумуляція (OBV) ---
            obv = calculate_obv(close, df['volume'])
            
            # Рахуємо нахил за останні 15 свічок
            p_slope = calculate_slope(close, 15)
            obv_slope = calculate_slope(obv, 15)
            
            # Нормалізація нахилу ціни у відсотках
            norm_p_slope = (p_slope / price) * 100
            
            is_whale = False
            reason = ""
            score = 50
            
            # Сигнал 1: Дивергенція (Ціна стоїть/падає, а OBV росте)
            if norm_p_slope <= 0.05 and obv_slope > self.CONFIG['obv_slope_min']:
                is_whale = True
                reason = "Volume Divergence (Accumulation)"
                score += 30
            
            if not is_whale: return None

            # --- 4. Ichimoku Check (Підтвердження) ---
            tenkan, kijun, sa, sb = calculate_ichimoku(df['high'], df['low'], close)
            cloud_top = max(sa.iloc[-1], sb.iloc[-1])
            cloud_bottom = min(sa.iloc[-1], sb.iloc[-1])
            
            # Ідеально: Ціна над хмарою або всередині
            if price > cloud_top:
                score += 20
                reason += " + Above Cloud (Ready)"
            elif price >= cloud_bottom:
                score += 10
                reason += " + In Cloud (Coiling)"
            else:
                # Під хмарою допускаємо тільки якщо є Golden Cross (Tenkan > Kijun)
                if tenkan.iloc[-1] > kijun.iloc[-1]:
                    score += 5
                    reason += " + TK Cross"
                else:
                    return None # Слабкий сигнал

            return {
                "score": min(score, 100),
                "squeeze": round(curr_sqz, 4),
                "obv_slope": round(obv_slope, 2),
                "reason": reason,
                "price": price
            }

        except Exception as e:
            logger.error(f"Analysis calc error: {e}")
            return None

    def save_signal(self, symbol, data):
        """Збереження результату в БД"""
        session = db_manager.get_session()
        try:
            # Можна додати перевірку, чи не сканували ми цей актив нещодавно
            sig = WhaleSignal(
                symbol=symbol,
                price=data['price'],
                score=data['score'],
                squeeze_val=data['squeeze'],
                obv_slope=data['obv_slope'],
                details=data['reason'],
                created_at=datetime.utcnow()
            )
            session.add(sig)
            session.commit()
        except Exception as e:
            logger.error(f"DB Save error: {e}")
        finally:
            session.close()

    def get_history(self, limit=50):
        """Отримання історії з БД для UI"""
        session = db_manager.get_session()
        try:
            # Перевірка наявності таблиці (створення, якщо немає - safety check)
            WhaleSignal.__table__.create(db_manager.engine, checkfirst=True)
            
            res = session.query(WhaleSignal).order_by(desc(WhaleSignal.created_at)).limit(limit).all()
            return [{
                'id': r.id,
                'symbol': r.symbol,
                'price': r.price,
                'score': r.score,
                'squeeze': r.squeeze_val,
                'details': r.details,
                'time': r.created_at.strftime('%d.%m %H:%M')
            } for r in res]
        except Exception as e:
            logger.error(f"Get history error: {e}")
            return []
        finally:
            session.close()

    def start_scan(self, override_cfg=None):
        """Запуск потоку сканування"""
        if self.is_scanning: return False
        
        if override_cfg: 
            # Оновлюємо налаштування (конвертуємо типи)
            if 'limit' in override_cfg: self.CONFIG['limit_coins'] = int(override_cfg['limit'])
            if 'timeframe' in override_cfg: self.CONFIG['timeframe'] = str(override_cfg['timeframe'])
            
        threading.Thread(target=self._run).start()
        return True

    def _run(self):
        self.is_scanning = True
        self.status = "Fetching Markets..."
        self.progress = 0
        
        try:
            # 1. Отримуємо список
            tickers = bot_instance.get_all_tickers()
            targets = [t for t in tickers if t['symbol'].endswith('USDT')]
            
            # Фільтр по ліквідності
            targets = [t for t in targets if float(t.get('turnover24h', 0)) > self.CONFIG['min_vol_usdt']]
            
            # Сортуємо
            targets.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            targets = targets[:int(self.CONFIG['limit_coins'])]
            
            total = len(targets)
            
            # 2. Скануємо
            for i, t in enumerate(targets):
                self.status = f"Analyzing {t['symbol']}..."
                self.progress = int((i / total) * 100)
                
                df = self.fetch_data(t['symbol'])
                if df is not None and len(df) > 200:
                    res = self.analyze_coin(df)
                    if res:
                        self.save_signal(t['symbol'], res)
                
                time.sleep(0.1) # API protection
                
            self.progress = 100
            self.status = "Completed"
            self.last_scan_time = datetime.now().strftime("%H:%M")
            
        except Exception as e:
            logger.error(f"Scan fatal error: {e}")
            self.status = "Error"
        finally:
            self.is_scanning = False

whale_core = WhaleCore()