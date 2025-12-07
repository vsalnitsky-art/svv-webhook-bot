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
from settings_manager import settings
from indicators import (
    calculate_ema, calculate_bollinger_bands, 
    calculate_obv, calculate_ichimoku, calculate_slope,
    simple_rsi
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
        """Завантажує свічки - 100% сумісність з TradingView"""
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
                df = df.astype(float)
                
                # ✅ Виключаємо останню незакриту свічку (як TradingView)
                if len(df) > 1:
                    df = df.iloc[:-1].reset_index(drop=True)
                
                return df
        except Exception as e:
            pass
        return None

    def analyze_coin(self, df):
        """
        Аналіз монети: Squeeze + OBV Divergence + Ichimoku Spring + RSI Filter
        """
        try:
            close = df['close']
            
            # ✅ RSI ФІЛЬТР (якщо увімкнено)
            rsi_filter_on = settings.get("whale_rsi_filter_enabled", False)
            if rsi_filter_on:
                rsi_val = simple_rsi(close, period=14)
                rsi_min = float(settings.get("whale_rsi_min", 30))
                rsi_max = float(settings.get("whale_rsi_max", 70))
                
                # Пропускаємо монети де RSI в "нормальній" зоні
                # Шукаємо тільки екстремальні значення (перекупленість/перепроданість)
                if rsi_min < rsi_val < rsi_max:
                    return None
            
            # 1. Фільтр Тренду (EMA 200)
            ema = calculate_ema(close, self.CONFIG['ema_period'])
            dist_to_ema = (close.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1]
            
            # Якщо ціна > 15% під EMA - це сильний даунтренд
            if dist_to_ema < -0.15: return None

            # 2. Фільтр Волатильності (Squeeze)
            _, _, _, bandwidth = calculate_bollinger_bands(close)
            curr_sqz = bandwidth.iloc[-1]
            
            if curr_sqz > self.CONFIG['bb_squeeze_max']:
                return None 

            # 3. Акумуляція (OBV)
            obv = calculate_obv(close, df['volume'])
            p_slope = calculate_slope(close, 15)
            obv_slope = calculate_slope(obv, 15)
            
            # Нормалізація нахилу ціни (%)
            norm_p_slope = (p_slope / close.iloc[-1]) * 100
            
            is_whale = False
            reason = ""
            score = 50
            
            # Сигнал 1: Дивергенція
            if abs(norm_p_slope) < 0.05 and obv_slope > 0:
                is_whale = True
                reason = "Accumulation (Flat Price, OBV Up)"
                score += 30
            elif norm_p_slope < -0.05 and obv_slope > 0:
                is_whale = True
                reason = "Divergence (Price Drop, OBV Up)"
                score += 20

            if not is_whale: return None

            # 4. Ichimoku Check
            tenkan, kijun, sa, sb = calculate_ichimoku(df['high'], df['low'], close)
            cloud_top = max(sa.iloc[-1], sb.iloc[-1])
            
            if close.iloc[-1] > cloud_top:
                score += 20
                reason += " + Above Cloud"
            
            # ✅ Додаємо RSI до результату
            current_rsi = simple_rsi(close, period=14) if rsi_filter_on else 0
            
            return {
                "score": min(score, 100),
                "squeeze": round(curr_sqz, 4),
                "obv_slope": round(obv_slope, 2),
                "reason": reason,
                "price": close.iloc[-1],
                "rsi": round(current_rsi, 1)
            }

        except Exception as e:
            return None

    def save_signal(self, symbol, data):
        """Збереження в БД"""
        session = db_manager.get_session()
        try:
            sig = WhaleSignal(
                symbol=symbol,
                price=data['price'],
                score=data['score'],
                squeeze_val=data['squeeze'],
                obv_slope=data['obv_slope'],
                rsi=data.get('rsi', 0),  # ✅ RSI
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
        """Отримання історії з БД"""
        session = db_manager.get_session()
        try:
            # Гарантуємо створення таблиці
            WhaleSignal.__table__.create(db_manager.engine, checkfirst=True)
            
            res = session.query(WhaleSignal).order_by(desc(WhaleSignal.created_at)).limit(limit).all()
            return [{
                'id': r.id,
                'symbol': r.symbol,
                'price': r.price,
                'score': r.score,
                'squeeze': r.squeeze_val,
                'rsi': getattr(r, 'rsi', 0) or 0,  # ✅ RSI (з fallback для старих записів)
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
            if 'limit' in override_cfg: self.CONFIG['limit_coins'] = int(override_cfg['limit'])
            if 'timeframe' in override_cfg: self.CONFIG['timeframe'] = str(override_cfg['timeframe'])
            
            # ✅ RSI параметри - зберігаємо в settings
            if 'rsi_filter_enabled' in override_cfg:
                settings.save_settings({'whale_rsi_filter_enabled': override_cfg['rsi_filter_enabled']})
            if 'rsi_min' in override_cfg:
                settings.save_settings({'whale_rsi_min': int(override_cfg['rsi_min'])})
            if 'rsi_max' in override_cfg:
                settings.save_settings({'whale_rsi_max': int(override_cfg['rsi_max'])})
            
        threading.Thread(target=self._run).start()
        return True

    def clear_old_results(self):
        """✨ Видалення старих результатів перед новим скануванням"""
        session = db_manager.get_session()
        try:
            # Видаляємо ВСІ результати (або за часом якщо потрібно)
            session.query(WhaleSignal).delete()
            session.commit()
            logger.info("✅ Cleared old whale results")
        except Exception as e:
            logger.error(f"Clear results error: {e}")
        finally:
            session.close()

    def _run(self):
        self.is_scanning = True
        self.status = "Clearing old results..."
        self.progress = 0
        
        # ✨ ВИДАЛЯЄМО СТАРІ РЕЗУЛЬТАТИ перед новим скануванням
        self.clear_old_results()
        
        try:
            self.status = "Fetching Markets..."
            tickers = bot_instance.get_all_tickers()
            targets = [t for t in tickers if t['symbol'].endswith('USDT')]
            targets = [t for t in targets if float(t.get('turnover24h', 0)) > self.CONFIG['min_vol_usdt']]
            targets.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            targets = targets[:int(self.CONFIG['limit_coins'])]
            
            total = len(targets)
            
            for i, t in enumerate(targets):
                self.status = f"Analyzing {t['symbol']}..."
                self.progress = int((i / total) * 100)
                
                df = self.fetch_data(t['symbol'])
                if df is not None and len(df) > 200:
                    res = self.analyze_coin(df)
                    if res:
                        self.save_signal(t['symbol'], res)
                
                time.sleep(0.1) 
                
            self.progress = 100
            self.status = "Completed"
            self.last_scan_time = datetime.now().strftime("%H:%M")
            
        except Exception as e:
            logger.error(f"Scan fatal error: {e}")
            self.status = "Error"
        finally:
            self.is_scanning = False

whale_core = WhaleCore()
