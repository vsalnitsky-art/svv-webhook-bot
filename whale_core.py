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
        Аналіз монети: RSI Filter + Squeeze + OBV Divergence + Ichimoku
        """
        try:
            close = df['close']
            current_rsi = simple_rsi(close, period=14)
            
            # ✅ RSI ФІЛЬТР (якщо увімкнено)
            rsi_filter_on = settings.get("whale_rsi_filter_enabled", False)
            if rsi_filter_on:
                rsi_min = float(settings.get("whale_rsi_min", 30))
                rsi_max = float(settings.get("whale_rsi_max", 70))
                
                # Пропускаємо монети де RSI в "нормальній" зоні
                if rsi_min < current_rsi < rsi_max:
                    return None
            
            # 1. Фільтр Тренду (EMA 200) - ОПЦІОНАЛЬНО, м'якший
            ema = calculate_ema(close, self.CONFIG['ema_period'])
            dist_to_ema = (close.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1]
            
            # Якщо ціна > 20% під EMA - це дуже сильний даунтренд (було 15%)
            if dist_to_ema < -0.20: 
                return None

            # 2. Фільтр Волатильності (Squeeze) - ОПЦІОНАЛЬНО
            _, _, _, bandwidth = calculate_bollinger_bands(close)
            curr_sqz = bandwidth.iloc[-1] if bandwidth is not None and len(bandwidth) > 0 else 0.1
            
            # 3. Акумуляція (OBV)
            obv = calculate_obv(close, df['volume'])
            p_slope = calculate_slope(close, 15)
            obv_slope = calculate_slope(obv, 15)
            
            # Нормалізація нахилу ціни (%)
            norm_p_slope = (p_slope / close.iloc[-1]) * 100 if close.iloc[-1] > 0 else 0
            
            # === ЛОГІКА СИГНАЛІВ ===
            is_signal = False
            reason = ""
            score = 50
            
            # Сигнал 1: RSI екстремум (якщо фільтр увімкнено)
            if rsi_filter_on:
                rsi_min = float(settings.get("whale_rsi_min", 30))
                rsi_max = float(settings.get("whale_rsi_max", 70))
                
                if current_rsi <= rsi_min:
                    is_signal = True
                    reason = f"RSI Oversold ({round(current_rsi, 1)})"
                    score += 25
                elif current_rsi >= rsi_max:
                    is_signal = True
                    reason = f"RSI Overbought ({round(current_rsi, 1)})"
                    score += 25
            
            # Сигнал 2: OBV Дивергенція (класична whale логіка)
            if abs(norm_p_slope) < 0.1 and obv_slope > 0:  # Розширив з 0.05 до 0.1
                is_signal = True
                if reason:
                    reason += " + Accumulation"
                else:
                    reason = "Accumulation (Flat Price, OBV Up)"
                score += 20
            elif norm_p_slope < -0.05 and obv_slope > 0:
                is_signal = True
                if reason:
                    reason += " + Divergence"
                else:
                    reason = "Divergence (Price Drop, OBV Up)"
                score += 15

            # Сигнал 3: Squeeze (низька волатильність)
            if curr_sqz < self.CONFIG['bb_squeeze_max']:
                score += 10
                if is_signal:
                    reason += " + Squeeze"

            if not is_signal: 
                return None

            # 4. Ichimoku Check (бонус)
            try:
                tenkan, kijun, sa, sb = calculate_ichimoku(df['high'], df['low'], close)
                if sa is not None and sb is not None:
                    cloud_top = max(sa.iloc[-1], sb.iloc[-1])
                    if close.iloc[-1] > cloud_top:
                        score += 15
                        reason += " + Above Cloud"
            except:
                pass
            
            return {
                "score": min(score, 100),
                "squeeze": round(curr_sqz, 4),
                "obv_slope": round(obv_slope, 2),
                "reason": reason,
                "price": close.iloc[-1],
                "rsi": round(current_rsi, 1)
            }

        except Exception as e:
            logger.error(f"Analyze error: {e}")
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
                details=data['reason'],
                created_at=datetime.utcnow()
            )
            # Додаємо RSI якщо колонка існує
            try:
                sig.rsi = data.get('rsi', 0)
            except:
                pass
            
            session.add(sig)
            session.commit()
            logger.info(f"✅ Saved: {symbol} Score={data['score']} RSI={data.get('rsi', '-')}")
        except Exception as e:
            logger.error(f"DB Save error: {e}")
            session.rollback()
        finally:
            session.close()

    def get_history(self, limit=50):
        """Отримання історії з БД"""
        session = db_manager.get_session()
        try:
            # Гарантуємо створення таблиці
            WhaleSignal.__table__.create(db_manager.engine, checkfirst=True)
            
            res = session.query(WhaleSignal).order_by(desc(WhaleSignal.created_at)).limit(limit).all()
            results = []
            for r in res:
                # Безпечне отримання rsi
                try:
                    rsi_val = r.rsi if hasattr(r, 'rsi') and r.rsi else 0
                except:
                    rsi_val = 0
                    
                results.append({
                    'id': r.id,
                    'symbol': r.symbol,
                    'price': r.price,
                    'score': r.score,
                    'squeeze': r.squeeze_val,
                    'rsi': rsi_val,
                    'details': r.details,
                    'time': r.created_at.strftime('%d.%m %H:%M')
                })
            return results
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
        """Видалення старих результатів + міграція БД"""
        session = db_manager.get_session()
        try:
            # ✅ Міграція: додаємо колонку rsi якщо не існує
            try:
                from sqlalchemy import text
                session.execute(text("ALTER TABLE whale_signals ADD COLUMN IF NOT EXISTS rsi FLOAT DEFAULT 0"))
                session.commit()
                logger.info("✅ Migration: rsi column ensured")
            except Exception as migration_err:
                session.rollback()
                logger.warning(f"Migration note: {migration_err}")
            
            # Видаляємо ВСІ результати
            session.query(WhaleSignal).delete()
            session.commit()
            logger.info("✅ Cleared old whale results")
        except Exception as e:
            logger.error(f"Clear results error: {e}")
            session.rollback()
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
