#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SMART MONEY ENGINE v2.0
==========================
Повна система торгівлі на основі Order Blocks.

Функціонал:
- Watchlist з timestamp фільтрацією
- Детекція нових OB (Immediate/Retest)
- Виконання угод (Paper/Real)
- Закриття по протилежному OB
- Execution Log

Автор: SVV Webhook Bot Team
Версія: 2.0.0
"""

import threading
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from models import (
    db_manager, 
    SmartMoneyTicker, 
    DetectedOrderBlock,
    SmartMoneyExecutionLog
)
from settings_manager import settings
from bot import bot_instance
from order_block_scanner import OrderBlockScanner, OrderBlockDetector, OBStatus, EntryMode

logger = logging.getLogger("SmartMoneyEngine")


# ============================================================================
#                              CONSTANTS
# ============================================================================

class ExitMode(Enum):
    FULL_CLOSE = "Full Close"
    PARTIAL_TRAIL = "Partial + Trail"
    MOVE_SL = "Move SL Only"


# Пояснення параметрів
PARAM_HELP = {
    # Detection
    "ob_source_tf": "Таймфрейм для пошуку Order Blocks",
    "ob_swing_length": "Кількість свічок для визначення swing point. Більше = менше OB, але надійніші",
    "ob_zone_count": "Максимальна кількість OB зон для відстеження",
    "ob_max_atr_mult": "Максимальний розмір OB відносно ATR. Фільтрує занадто великі зони",
    "ob_invalidation_method": "Метод інвалідації: Wick (тінь пробила) або Close (закриття пробило)",
    "ob_combine_obs": "Об'єднувати близькі OB в одну зону",
    
    # Entry
    "ob_entry_mode": "Immediate - одразу при появі OB. Retest - чекати вхід/вихід ціни із зони",
    "ob_selection": "Який OB використовувати: Newest (найновіший) або Closest (найближчий до ціни)",
    "ob_sl_atr_mult": "Відступ SL від краю OB в одиницях ATR",
    
    # Exit
    "ob_exit_enabled": "Закривати позицію при появі протилежного Order Block",
    "ob_exit_tf": "Таймфрейм для пошуку Exit OB. 'same' = той самий що Entry",
    "ob_exit_mode": "Full Close - 100%, Partial+Trail - 50%+trailing, Move SL - тільки перемістити SL",
    "ob_exit_only_profit": "Закривати по OB тільки якщо позиція в плюсі",
    "ob_exit_min_hold": "Мінімальний час утримання позиції (хвилини) перед OB Exit",
    "ob_exit_require_rsi": "Вимагати підтвердження RSI для Exit",
    
    # Trading
    "ob_execute_trades": "Головний перемикач виконання угод",
    "ob_paper_trading": "Paper Trading - угоди не відкриваються на біржі, тільки логуються",
    "ob_auto_scan": "Автоматичне сканування watchlist",
    "ob_scan_interval": "Інтервал автоматичного сканування (секунди)",
    "ob_watchlist_limit": "Максимальна кількість монет у watchlist",
    "ob_timeout": "Час життя OB в очікуванні. 'No' = без таймауту",
}


# ============================================================================
#                           SMART MONEY ENGINE
# ============================================================================

class SmartMoneyEngine:
    """
    🎯 Головний двигун Smart Money системи
    
    Відповідає за:
    1. Сканування watchlist на нові OB
    2. Фільтрацію OB по timestamp додавання монети
    3. Моніторинг Retest умов
    4. Виконання угод (Paper/Real)
    5. Моніторинг Exit OB для відкритих позицій
    6. Ведення Execution Log
    """
    
    def __init__(self):
        self.is_running = False
        self.is_scanning = False
        self.scan_progress = 0
        self.scan_status = "Idle"
        self.last_scan_time = None
        self.last_scan_found = 0
        self.last_scan_total = 0
        
        # Exit monitor
        self.exit_monitor_running = False
        
        # Stop events
        self._stop_scan = threading.Event()
        self._stop_exit = threading.Event()
        
        # Створюємо таблиці
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Створює таблиці якщо не існують"""
        try:
            SmartMoneyExecutionLog.__table__.create(db_manager.engine, checkfirst=True)
            logger.info("✅ SmartMoneyExecutionLog table ready")
        except Exception as e:
            logger.error(f"Table creation error: {e}")
    
    # ========================================================================
    #                           CONFIGURATION
    # ========================================================================
    
    def get_config(self) -> Dict:
        """Повертає поточну конфігурацію"""
        return {
            # Detection
            'ob_source_tf': settings.get('ob_source_tf', '15'),
            'ob_swing_length': int(settings.get('ob_swing_length', 10)),
            'ob_zone_count': settings.get('ob_zone_count', 'High'),
            'ob_max_atr_mult': float(settings.get('ob_max_atr_mult', 3.5)),
            'ob_invalidation_method': settings.get('ob_invalidation_method', 'Wick'),
            'ob_combine_obs': settings.get('ob_combine_obs', True),
            
            # Entry
            'ob_entry_mode': settings.get('ob_entry_mode', 'Immediate'),
            'ob_selection': settings.get('ob_selection', 'Newest'),
            'ob_sl_atr_mult': float(settings.get('ob_sl_atr_mult', 0.3)),
            
            # Exit
            'ob_exit_enabled': settings.get('ob_exit_enabled', False),
            'ob_exit_tf': settings.get('ob_exit_tf', 'same'),
            'ob_exit_mode': settings.get('ob_exit_mode', 'Full Close'),
            'ob_exit_only_profit': settings.get('ob_exit_only_profit', False),
            'ob_exit_min_hold': int(settings.get('ob_exit_min_hold', 0)),
            'ob_exit_require_rsi': settings.get('ob_exit_require_rsi', False),
            
            # Trading
            'ob_execute_trades': settings.get('ob_execute_trades', False),
            'ob_paper_trading': settings.get('ob_paper_trading', True),
            'ob_auto_scan': settings.get('ob_auto_scan', False),
            'ob_scan_interval': int(settings.get('ob_scan_interval', 60)),
            'ob_watchlist_limit': int(settings.get('ob_watchlist_limit', 50)),
            'ob_timeout': settings.get('ob_timeout', '24h'),
            
            # RSI/MFI Integration
            'ob_auto_add_from_screener': settings.get('ob_auto_add_from_screener', False),
        }
    
    def save_config(self, config: Dict):
        """Зберігає конфігурацію"""
        settings.save_settings(config)
        logger.info("SmartMoney config saved")
    
    def get_param_help(self) -> Dict:
        """Повертає довідку по параметрах"""
        return PARAM_HELP
    
    # ========================================================================
    #                           WATCHLIST
    # ========================================================================
    
    def get_watchlist(self) -> List[Dict]:
        """Отримує watchlist"""
        session = db_manager.get_session()
        try:
            items = session.query(SmartMoneyTicker).order_by(SmartMoneyTicker.added_at.desc()).all()
            return [{
                'id': item.id,
                'symbol': item.symbol,
                'direction': item.direction or 'BUY',
                'source': item.source or 'Manual',
                'added_at': item.added_at.isoformat() if item.added_at else None
            } for item in items]
        finally:
            session.close()
    
    def add_to_watchlist(self, symbol: str, direction: str, source: str = 'Manual') -> Dict:
        """Додає монету до watchlist"""
        config = self.get_config()
        session = db_manager.get_session()
        
        try:
            # Перевірка ліміту
            count = session.query(SmartMoneyTicker).count()
            if count >= config['ob_watchlist_limit']:
                return {'status': 'error', 'error': f"Watchlist limit reached ({config['ob_watchlist_limit']})"}
            
            # Перевірка дублікату
            existing = session.query(SmartMoneyTicker).filter_by(symbol=symbol).first()
            if existing:
                return {'status': 'error', 'error': 'Symbol already in watchlist'}
            
            # Додаємо
            item = SmartMoneyTicker(
                symbol=symbol.upper(),
                direction=direction,
                source=source,
                added_at=datetime.utcnow()
            )
            session.add(item)
            session.commit()
            
            logger.info(f"✅ Added to watchlist: {symbol} {direction} ({source})")
            return {'status': 'ok', 'added_at': item.added_at.isoformat()}
            
        except Exception as e:
            session.rollback()
            logger.error(f"Add to watchlist error: {e}")
            return {'status': 'error', 'error': str(e)}
        finally:
            session.close()
    
    def remove_from_watchlist(self, symbol: str) -> Dict:
        """Видаляє монету з watchlist"""
        session = db_manager.get_session()
        try:
            item = session.query(SmartMoneyTicker).filter_by(symbol=symbol).first()
            if item:
                session.delete(item)
                session.commit()
                logger.info(f"Removed from watchlist: {symbol}")
            return {'status': 'ok'}
        except Exception as e:
            session.rollback()
            return {'status': 'error', 'error': str(e)}
        finally:
            session.close()
    
    def clear_watchlist(self) -> Dict:
        """Очищає watchlist"""
        session = db_manager.get_session()
        try:
            session.query(SmartMoneyTicker).delete()
            session.commit()
            logger.info("Watchlist cleared")
            return {'status': 'ok'}
        except Exception as e:
            session.rollback()
            return {'status': 'error', 'error': str(e)}
        finally:
            session.close()
    
    # ========================================================================
    #                        DETECTED ORDER BLOCKS
    # ========================================================================
    
    def get_detected_obs(self) -> List[Dict]:
        """Отримує знайдені OB"""
        session = db_manager.get_session()
        try:
            obs = session.query(DetectedOrderBlock)\
                .filter(DetectedOrderBlock.status.in_(['Valid', 'Waiting Retest', 'Triggered']))\
                .order_by(DetectedOrderBlock.detected_at.desc()).all()
            
            return [{
                'id': ob.id,
                'symbol': ob.symbol,
                'direction': ob.direction,
                'ob_type': ob.ob_type,
                'ob_top': ob.ob_top,
                'ob_bottom': ob.ob_bottom,
                'entry_price': ob.entry_price,
                'sl_price': ob.sl_price,
                'current_price': ob.current_price,
                'status': ob.status,
                'timeframe': ob.timeframe,
                'detected_at': ob.detected_at.isoformat() if ob.detected_at else None
            } for ob in obs]
        finally:
            session.close()
    
    def delete_detected_ob(self, ob_id: int) -> Dict:
        """Видаляє OB"""
        session = db_manager.get_session()
        try:
            ob = session.query(DetectedOrderBlock).filter_by(id=ob_id).first()
            if ob:
                session.delete(ob)
                session.commit()
            return {'status': 'ok'}
        except Exception as e:
            session.rollback()
            return {'status': 'error', 'error': str(e)}
        finally:
            session.close()
    
    def clear_detected_obs(self) -> Dict:
        """Очищає всі OB"""
        session = db_manager.get_session()
        try:
            session.query(DetectedOrderBlock).delete()
            session.commit()
            return {'status': 'ok'}
        except Exception as e:
            session.rollback()
            return {'status': 'error', 'error': str(e)}
        finally:
            session.close()
    
    def trigger_retest_trade(self, ob_id: int) -> Dict:
        """
        Виконати угоду для OB з Waiting Retest
        Використовується коли користувач натискає Execute Now
        """
        config = self.get_config()
        session = db_manager.get_session()
        
        try:
            ob = session.query(DetectedOrderBlock).filter_by(id=ob_id).first()
            if not ob:
                return {'status': 'error', 'error': 'OB not found'}
            
            trade_direction = 'LONG' if ob.direction == 'BUY' else 'SHORT'
            
            # Створюємо запис в Execution Log
            log_entry = SmartMoneyExecutionLog(
                symbol=ob.symbol,
                direction=trade_direction,
                entry_price=ob.current_price or ob.entry_price,
                entry_time=datetime.utcnow(),
                sl_price=ob.sl_price,
                ob_top=ob.ob_top,
                ob_bottom=ob.ob_bottom,
                ob_timeframe=ob.timeframe,
                entry_mode='Retest (Manual)',
                paper_trade=config.get('ob_paper_trading', True)
            )
            
            # Перевіряємо Trading ON/OFF
            if config.get('ob_execute_trades', False):
                log_entry.status = 'OPEN'
                
                if not config.get('ob_paper_trading', True):
                    # Реальна угода
                    try:
                        action = "Buy" if ob.direction == "BUY" else "Sell"
                        trade_result = bot_instance.place_order({
                            "symbol": ob.symbol,
                            "action": action,
                            "sl_price": ob.sl_price
                        })
                        
                        if trade_result.get('status') == 'ok':
                            log_entry.order_id = trade_result.get('order_id')
                            log_entry.qty = trade_result.get('qty')
                            logger.info(f"✅ Retest trade opened: {ob.symbol} {trade_direction}")
                        else:
                            log_entry.status = 'FAILED'
                            log_entry.exit_reason = trade_result.get('error', 'Order failed')
                    except Exception as e:
                        log_entry.status = 'FAILED'
                        log_entry.exit_reason = str(e)
                else:
                    logger.info(f"📝 Retest paper trade: {ob.symbol} {trade_direction}")
            else:
                log_entry.status = 'SKIPPED'
                log_entry.exit_reason = 'Trading disabled (manual trigger)'
            
            session.add(log_entry)
            
            # Видаляємо з Waiting Retest
            session.delete(ob)
            session.commit()
            
            return {'status': 'ok'}
            
        except Exception as e:
            session.rollback()
            logger.error(f"Trigger retest error: {e}")
            return {'status': 'error', 'error': str(e)}
        finally:
            session.close()
    
    # ========================================================================
    #                         EXECUTION LOG
    # ========================================================================
    
    def get_execution_log(self, limit: int = 100) -> List[Dict]:
        """Отримує Execution Log"""
        session = db_manager.get_session()
        try:
            logs = session.query(SmartMoneyExecutionLog)\
                .order_by(SmartMoneyExecutionLog.created_at.desc())\
                .limit(limit).all()
            
            return [{
                'id': log.id,
                'symbol': log.symbol,
                'direction': log.direction,
                'entry_price': log.entry_price,
                'entry_time': log.entry_time.strftime('%d.%m %H:%M') if log.entry_time else None,
                'sl_price': log.sl_price,
                'ob_top': log.ob_top,
                'ob_bottom': log.ob_bottom,
                'ob_timeframe': log.ob_timeframe,
                'entry_mode': log.entry_mode,
                'exit_price': log.exit_price,
                'exit_time': log.exit_time.strftime('%d.%m %H:%M') if log.exit_time else None,
                'exit_reason': log.exit_reason,
                'pnl': log.pnl,
                'pnl_percent': log.pnl_percent,
                'is_win': log.is_win,
                'status': log.status,
                'paper_trade': log.paper_trade,
                'order_id': log.order_id,
                'qty': log.qty
            } for log in logs]
        finally:
            session.close()
    
    def get_execution_stats(self) -> Dict:
        """Статистика виконання"""
        session = db_manager.get_session()
        try:
            closed = session.query(SmartMoneyExecutionLog)\
                .filter_by(status='CLOSED').all()
            
            if not closed:
                return {
                    'total': 0, 'wins': 0, 'losses': 0,
                    'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0
                }
            
            wins = [t for t in closed if t.is_win]
            losses = [t for t in closed if not t.is_win]
            total_pnl = sum(t.pnl or 0 for t in closed)
            
            return {
                'total': len(closed),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': round(len(wins) / len(closed) * 100, 1) if closed else 0,
                'total_pnl': round(total_pnl, 2),
                'avg_pnl': round(total_pnl / len(closed), 2) if closed else 0
            }
        finally:
            session.close()
    
    def delete_execution_log(self, log_id: int) -> Dict:
        """Видалити запис з Execution Log"""
        session = db_manager.get_session()
        try:
            log = session.query(SmartMoneyExecutionLog).filter_by(id=log_id).first()
            if log:
                session.delete(log)
                session.commit()
                logger.info(f"🗑️ Deleted execution log: {log_id}")
                return {'status': 'ok'}
            return {'status': 'error', 'error': 'Not found'}
        except Exception as e:
            session.rollback()
            return {'status': 'error', 'error': str(e)}
        finally:
            session.close()
    
    def clear_execution_log(self) -> Dict:
        """Очистити весь Execution Log"""
        session = db_manager.get_session()
        try:
            count = session.query(SmartMoneyExecutionLog).delete()
            session.commit()
            logger.info(f"🗑️ Cleared execution log: {count} records")
            return {'status': 'ok', 'deleted': count}
        except Exception as e:
            session.rollback()
            return {'status': 'error', 'error': str(e)}
        finally:
            session.close()
    
    # ========================================================================
    #                           SCANNING
    # ========================================================================
    
    def _get_bybit_session(self):
        """Отримує Bybit session"""
        import os
        from pybit.unified_trading import HTTP
        
        api_key = os.environ.get("BYBIT_API_KEY", "")
        api_secret = os.environ.get("BYBIT_API_SECRET", "")
        testnet = os.environ.get("TESTNET", "false").lower() == "true"
        
        return HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)
    
    def _fetch_klines(self, bybit_session, symbol: str, interval: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Завантажує свічки"""
        try:
            response = bybit_session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if response.get('retCode') != 0:
                return None
            
            klines = response.get('result', {}).get('list', [])
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.astype({
                'timestamp': 'int64',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.error(f"Fetch klines error {symbol}: {e}")
            return None
    
    def scan_watchlist(self) -> Dict:
        """
        Сканує watchlist на нові Order Blocks
        
        Логіка:
        1. Для кожної монети з watchlist
        2. Завантажуємо свічки
        3. Знаходимо OB створені ПІСЛЯ added_at
        4. Обробляємо згідно Entry Mode
        """
        if self.is_scanning:
            return {'status': 'error', 'error': 'Scan already running'}
        
        self.is_scanning = True
        self.scan_progress = 0
        self.scan_status = "Starting..."
        self._stop_scan.clear()
        
        config = self.get_config()
        session = db_manager.get_session()
        found_count = 0
        executed_count = 0
        retest_count = 0
        
        try:
            # Отримуємо watchlist
            watchlist = session.query(SmartMoneyTicker).all()
            if not watchlist:
                self.scan_status = "Watchlist empty"
                return {'status': 'ok', 'found': 0, 'message': 'Watchlist is empty'}
            
            total = len(watchlist)
            self.last_scan_total = total
            
            # Bybit session
            bybit_session = self._get_bybit_session()
            
            # Scanner
            scanner = OrderBlockScanner(session=bybit_session, settings=config)
            
            for i, item in enumerate(watchlist):
                if self._stop_scan.is_set():
                    break
                
                symbol = item.symbol
                direction = item.direction or 'BUY'
                added_at = item.added_at
                
                self.scan_status = f"Scanning {symbol}..."
                self.scan_progress = int((i / total) * 100)
                
                try:
                    # Отримуємо дані
                    df = self._fetch_klines(bybit_session, symbol, config['ob_source_tf'], 500)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Знаходимо НОВИЙ OB (утворений після added_at)
                    result = self._scan_symbol_with_timestamp(
                        df, symbol, direction, added_at, config, scanner
                    )
                    
                    if result:
                        found_count += 1
                        current_price = result['current_price']
                        trade_direction = 'LONG' if direction == 'BUY' else 'SHORT'
                        entry_mode = config.get('ob_entry_mode', 'Immediate')
                        
                        logger.info(f"🎯 NEW OB Found: {symbol} {result['ob_type']} @ {current_price:.6f} | Mode: {entry_mode}")
                        
                        # ========================================
                        # ENTRY MODE: IMMEDIATE
                        # ========================================
                        if entry_mode == 'Immediate':
                            # Створюємо запис в Execution Log
                            log_entry = SmartMoneyExecutionLog(
                                symbol=symbol,
                                direction=trade_direction,
                                entry_price=current_price,
                                entry_time=datetime.utcnow(),
                                sl_price=result['sl_price'],
                                ob_top=result['ob_top'],
                                ob_bottom=result['ob_bottom'],
                                ob_timeframe=config['ob_source_tf'],
                                entry_mode='Immediate',
                                paper_trade=config.get('ob_paper_trading', True)
                            )
                            
                            # Перевіряємо Trading ON/OFF
                            if config.get('ob_execute_trades', False):
                                # Trading ON - виконуємо угоду
                                log_entry.status = 'OPEN'
                                
                                if not config.get('ob_paper_trading', True):
                                    # Реальна угода через Bybit
                                    try:
                                        action = "Buy" if direction == "BUY" else "Sell"
                                        trade_result = bot_instance.place_order({
                                            "symbol": symbol,
                                            "action": action,
                                            "sl_price": result['sl_price']
                                        })
                                        
                                        if trade_result.get('status') == 'ok':
                                            log_entry.order_id = trade_result.get('order_id')
                                            log_entry.qty = trade_result.get('qty')
                                            executed_count += 1
                                            logger.info(f"✅ Real trade opened: {symbol} {trade_direction}")
                                        else:
                                            log_entry.status = 'FAILED'
                                            log_entry.exit_reason = trade_result.get('error', 'Order failed')
                                            logger.error(f"❌ Trade failed: {symbol}")
                                    except Exception as e:
                                        log_entry.status = 'FAILED'
                                        log_entry.exit_reason = str(e)
                                        logger.error(f"❌ Trade error: {symbol} - {e}")
                                else:
                                    # Paper trade
                                    executed_count += 1
                                    logger.info(f"📝 Paper trade opened: {symbol} {trade_direction}")
                            else:
                                # Trading OFF - записуємо як пропущено
                                log_entry.status = 'SKIPPED'
                                log_entry.exit_reason = 'Trading disabled'
                                logger.info(f"⏸️ Trade skipped (Trading OFF): {symbol}")
                            
                            # Зберігаємо запис в Execution Log
                            session.add(log_entry)
                            
                            # Видаляємо монету з Watchlist
                            session.delete(item)
                            logger.info(f"🗑️ Removed from Watchlist: {symbol}")
                        
                        # ========================================
                        # ENTRY MODE: RETEST
                        # ========================================
                        else:  # Retest mode
                            # Зберігаємо в DetectedOrderBlock для моніторингу
                            detected_ob = DetectedOrderBlock(
                                symbol=symbol,
                                direction=direction,
                                ob_type=result['ob_type'],
                                ob_top=result['ob_top'],
                                ob_bottom=result['ob_bottom'],
                                entry_price=result['entry_price'],
                                sl_price=result['sl_price'],
                                current_price=current_price,
                                atr=result['atr'],
                                status='Waiting Retest',
                                timeframe=config['ob_source_tf'],
                                detected_at=datetime.utcnow()
                            )
                            session.add(detected_ob)
                            retest_count += 1
                            logger.info(f"⏳ Added to Waiting Retest: {symbol}")
                            
                            # Видаляємо монету з Watchlist
                            session.delete(item)
                            logger.info(f"🗑️ Removed from Watchlist: {symbol}")
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Scan {symbol} error: {e}")
                    continue
            
            session.commit()
            
            self.scan_progress = 100
            self.scan_status = f"Done! Found {found_count}, Executed {executed_count}, Retest {retest_count}"
            self.last_scan_time = datetime.now()
            self.last_scan_found = found_count
            
            logger.info(f"✅ Scan complete: {found_count} found, {executed_count} executed, {retest_count} waiting retest")
            
            return {
                'status': 'ok',
                'found': found_count,
                'executed': executed_count,
                'retest': retest_count,
                'scanned': total
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Scan error: {e}", exc_info=True)
            self.scan_status = f"Error: {str(e)}"
            return {'status': 'error', 'error': str(e)}
        finally:
            self.is_scanning = False
            session.close()
    
    def _scan_symbol_with_timestamp(
        self,
        df: pd.DataFrame,
        symbol: str,
        direction: str,
        added_at: datetime,
        config: Dict,
        scanner: OrderBlockScanner
    ) -> Optional[Dict]:
        """
        Шукає НОВИЙ OB утворений ПІСЛЯ added_at
        
        КРИТИЧНА ЛОГІКА:
        1. Знаходимо всі валідні OB
        2. Фільтруємо по timestamp: ob.start_time >= added_at
        3. Повертаємо ОДИН найновіший/найближчий OB
        
        Якщо немає нових OB - повертаємо None
        """
        # Конвертуємо added_at в timestamp мілісекунди
        if added_at:
            added_ts = int(added_at.timestamp() * 1000)
        else:
            # Якщо немає added_at - поточний час (нічого не знайдемо)
            added_ts = int(datetime.utcnow().timestamp() * 1000)
        
        logger.debug(f"🔍 Scanning {symbol}: looking for OB after ts={added_ts}")
        
        # Детекція OB
        bullish_obs, bearish_obs = scanner.detector.detect_order_blocks(df, direction)
        
        # Вибираємо потрібні OB по напрямку
        obs = bullish_obs if direction == "BUY" else bearish_obs
        
        logger.debug(f"   Found {len(obs)} total OBs for {symbol}")
        
        # КРИТИЧНА ФІЛЬТРАЦІЯ:
        # 1. Тільки валідні (is_valid() - не breaker, не disabled)
        # 2. Утворені ПІСЛЯ added_at (start_time >= added_ts)
        new_obs = []
        for ob in obs:
            if not ob.is_valid():
                logger.debug(f"   Skip invalid OB: breaker={ob.breaker}, disabled={ob.disabled}")
                continue
            
            if ob.start_time < added_ts:
                logger.debug(f"   Skip OLD OB: start_time={ob.start_time} < added_ts={added_ts}")
                continue
            
            new_obs.append(ob)
            logger.debug(f"   ✅ NEW OB: start_time={ob.start_time}, top={ob.top:.6f}, bottom={ob.bottom:.6f}")
        
        logger.debug(f"   After timestamp filter: {len(new_obs)} NEW OBs")
        
        if not new_obs:
            return None  # Немає нових OB
        
        # Вибираємо ОДИН OB за налаштуванням
        current_price = df['close'].iloc[-1]
        
        if config['ob_selection'] == 'Newest':
            # Newest = найбільший start_time
            selected_ob = max(new_obs, key=lambda ob: ob.start_time)
        else:  # Closest
            selected_ob = min(new_obs, key=lambda ob: abs((ob.top + ob.bottom) / 2 - current_price))
        
        # ATR для SL
        atr = scanner.detector.calculate_atr(df, 10).iloc[-1]
        sl_mult = config['ob_sl_atr_mult']
        
        # Entry та SL
        if direction == "BUY":
            entry_price = selected_ob.top
            sl_price = selected_ob.bottom - (atr * sl_mult)
        else:
            entry_price = selected_ob.bottom
            sl_price = selected_ob.top + (atr * sl_mult)
        
        logger.info(f"🎯 NEW OB for {symbol}: {selected_ob.ob_type.value} top={selected_ob.top:.6f} bottom={selected_ob.bottom:.6f}")
        
        return {
            'symbol': symbol,
            'direction': direction,
            'ob_type': selected_ob.ob_type.value,
            'ob_top': selected_ob.top,
            'ob_bottom': selected_ob.bottom,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'current_price': current_price,
            'atr': atr,
            'status': 'Valid',  # Завжди Valid для нових OB
            'ob_start_time': selected_ob.start_time
        }
    
    def _execute_trade(
        self,
        session,
        watchlist_item: SmartMoneyTicker,
        detected_ob: DetectedOrderBlock,
        result: Dict,
        config: Dict
    ) -> Dict:
        """Виконує угоду"""
        symbol = watchlist_item.symbol
        direction = 'LONG' if watchlist_item.direction == 'BUY' else 'SHORT'
        paper_trade = config['ob_paper_trading']
        
        try:
            current_price = result['current_price']
            
            # Створюємо запис в Execution Log
            log_entry = SmartMoneyExecutionLog(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                entry_time=datetime.utcnow(),
                sl_price=result['sl_price'],
                ob_top=result['ob_top'],
                ob_bottom=result['ob_bottom'],
                ob_timeframe=config['ob_source_tf'],
                entry_mode=config['ob_entry_mode'],
                status='OPEN',
                paper_trade=paper_trade
            )
            
            # Якщо реальна торгівля
            if not paper_trade:
                # Виконуємо угоду через bot_instance
                action = "Buy" if direction == "LONG" else "Sell"
                
                trade_result = bot_instance.place_order({
                    "symbol": symbol,
                    "action": action,
                    "sl_price": result['sl_price']
                })
                
                if trade_result.get('status') == 'ok':
                    log_entry.order_id = trade_result.get('order_id')
                    log_entry.qty = trade_result.get('qty')
                    logger.info(f"✅ Real trade opened: {symbol} {direction}")
                else:
                    logger.error(f"❌ Trade failed: {trade_result}")
                    return {'status': 'error', 'error': trade_result.get('error')}
            else:
                logger.info(f"📝 Paper trade opened: {symbol} {direction}")
            
            session.add(log_entry)
            
            # Оновлюємо статус OB
            detected_ob.status = 'Executed'
            detected_ob.executed_at = datetime.utcnow()
            
            # Видаляємо з watchlist
            session.delete(watchlist_item)
            
            return {'status': 'ok'}
            
        except Exception as e:
            logger.error(f"Execute trade error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # ========================================================================
    #                          EXIT MONITORING
    # ========================================================================
    
    def start_exit_monitor(self):
        """Запускає моніторинг Exit OB"""
        if self.exit_monitor_running:
            return
        
        self.exit_monitor_running = True
        self._stop_exit.clear()
        threading.Thread(target=self._exit_monitor_loop, daemon=True).start()
        logger.info("🎯 Exit OB Monitor started")
    
    def stop_exit_monitor(self):
        """Зупиняє моніторинг Exit OB"""
        self.exit_monitor_running = False
        self._stop_exit.set()
        logger.info("Exit OB Monitor stopped")
    
    def _exit_monitor_loop(self):
        """Цикл моніторингу Exit OB"""
        while not self._stop_exit.is_set():
            try:
                config = self.get_config()
                
                if config['ob_exit_enabled']:
                    self._check_exit_signals(config)
                
                # Інтервал перевірки - 1/4 від scan_interval
                interval = max(15, config['ob_scan_interval'] // 4)
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Exit monitor error: {e}")
                time.sleep(30)
    
    def _check_exit_signals(self, config: Dict):
        """Перевіряє Exit OB для відкритих позицій"""
        session = db_manager.get_session()
        
        try:
            # Отримуємо відкриті позиції з Execution Log
            open_positions = session.query(SmartMoneyExecutionLog)\
                .filter_by(status='OPEN').all()
            
            if not open_positions:
                return
            
            bybit_session = self._get_bybit_session()
            
            # Exit TF
            exit_tf = config['ob_exit_tf']
            if exit_tf == 'same':
                exit_tf = config['ob_source_tf']
            
            for pos in open_positions:
                try:
                    self._check_position_exit(session, bybit_session, pos, exit_tf, config)
                except Exception as e:
                    logger.error(f"Check exit {pos.symbol} error: {e}")
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Check exit signals error: {e}")
            session.rollback()
        finally:
            session.close()
    
    def _check_position_exit(
        self,
        session,
        bybit_session,
        position: SmartMoneyExecutionLog,
        exit_tf: str,
        config: Dict
    ):
        """Перевіряє одну позицію на Exit OB"""
        symbol = position.symbol
        direction = position.direction
        
        # Отримуємо дані
        df = self._fetch_klines(bybit_session, symbol, exit_tf, 200)
        if df is None or len(df) < 50:
            return
        
        current_price = df['close'].iloc[-1]
        
        # Перевіряємо SL
        if direction == 'LONG' and current_price <= position.sl_price:
            self._close_position(session, position, current_price, 'SL Hit', config)
            return
        elif direction == 'SHORT' and current_price >= position.sl_price:
            self._close_position(session, position, current_price, 'SL Hit', config)
            return
        
        # Шукаємо протилежний OB
        opposite_direction = 'SELL' if direction == 'LONG' else 'BUY'
        
        detector = OrderBlockDetector(swing_length=3)
        bullish_obs, bearish_obs = detector.detect_order_blocks(df, opposite_direction)
        
        obs = bearish_obs if direction == 'LONG' else bullish_obs
        valid_obs = [ob for ob in obs if ob.is_valid()]
        
        if not valid_obs:
            return
        
        # Знайдено протилежний OB!
        exit_ob = valid_obs[0]
        
        # Перевірка фільтрів
        # 1. Only in profit
        if config['ob_exit_only_profit']:
            if direction == 'LONG' and current_price <= position.entry_price:
                return
            if direction == 'SHORT' and current_price >= position.entry_price:
                return
        
        # 2. Min hold time
        min_hold = config['ob_exit_min_hold']
        if min_hold > 0:
            held_minutes = (datetime.utcnow() - position.entry_time).total_seconds() / 60
            if held_minutes < min_hold:
                return
        
        # Закриваємо позицію
        exit_mode = config['ob_exit_mode']
        
        if exit_mode == 'Full Close':
            self._close_position(
                session, position, current_price, 'Opposite OB', config,
                exit_ob_top=exit_ob.top, exit_ob_bottom=exit_ob.bottom
            )
        elif exit_mode == 'Move SL Only':
            # Переміщуємо SL на рівень OB
            if direction == 'LONG':
                new_sl = exit_ob.bottom
            else:
                new_sl = exit_ob.top
            
            position.sl_price = new_sl
            logger.info(f"📍 SL moved for {symbol}: {new_sl}")
            
            # Оновлюємо на біржі якщо реальна угода
            if not position.paper_trade:
                # TODO: bot_instance.modify_sl(...)
                pass
        else:  # Partial + Trail
            # TODO: Implement partial close + trailing
            self._close_position(
                session, position, current_price, 'Opposite OB', config,
                exit_ob_top=exit_ob.top, exit_ob_bottom=exit_ob.bottom
            )
    
    def _close_position(
        self,
        session,
        position: SmartMoneyExecutionLog,
        exit_price: float,
        exit_reason: str,
        config: Dict,
        exit_ob_top: float = None,
        exit_ob_bottom: float = None
    ):
        """Закриває позицію"""
        # Розрахунок P&L
        if position.direction == 'LONG':
            pnl_percent = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            pnl_percent = (position.entry_price - exit_price) / position.entry_price * 100
        
        pnl = pnl_percent  # Для простоти, реальний PnL залежить від qty
        
        # Оновлюємо запис
        position.exit_price = exit_price
        position.exit_time = datetime.utcnow()
        position.exit_reason = exit_reason
        position.exit_ob_top = exit_ob_top
        position.exit_ob_bottom = exit_ob_bottom
        position.pnl = round(pnl, 4)
        position.pnl_percent = round(pnl_percent, 2)
        position.is_win = pnl > 0
        position.status = 'CLOSED'
        
        logger.info(f"✅ Position closed: {position.symbol} {exit_reason} PnL={pnl_percent:.2f}%")
        
        # Закриваємо на біржі якщо реальна угода
        if not position.paper_trade and position.order_id:
            try:
                action = "Sell" if position.direction == "LONG" else "Buy"
                bot_instance.close_position(position.symbol, action)
            except Exception as e:
                logger.error(f"Close position on exchange error: {e}")
    
    # ========================================================================
    #                          AUTO SCAN
    # ========================================================================
    
    def start_auto_scan(self):
        """Запускає автоматичне сканування"""
        if self.is_running:
            return
        
        self.is_running = True
        threading.Thread(target=self._auto_scan_loop, daemon=True).start()
        logger.info("🔄 Auto scan started")
    
    def stop_auto_scan(self):
        """Зупиняє автоматичне сканування"""
        self.is_running = False
        self._stop_scan.set()
        logger.info("Auto scan stopped")
    
    def _auto_scan_loop(self):
        """Цикл автоматичного сканування"""
        while self.is_running:
            config = self.get_config()
            
            if config['ob_auto_scan'] and not self.is_scanning:
                self.scan_watchlist()
            
            # Чекаємо інтервал
            interval = config['ob_scan_interval']
            for _ in range(interval):
                if not self.is_running:
                    break
                time.sleep(1)
    
    # ========================================================================
    #                            STATUS
    # ========================================================================
    
    def get_status(self) -> Dict:
        """Повертає статус системи"""
        config = self.get_config()
        
        return {
            'is_scanning': self.is_scanning,
            'scan_progress': self.scan_progress,
            'scan_status': self.scan_status,
            'last_scan_time': self.last_scan_time.strftime('%H:%M:%S') if self.last_scan_time else None,
            'last_scan_found': self.last_scan_found,
            'last_scan_total': self.last_scan_total,
            'auto_scan_running': self.is_running and config['ob_auto_scan'],
            'exit_monitor_running': self.exit_monitor_running,
            'config': config
        }


# Глобальний екземпляр
smart_money_engine = SmartMoneyEngine()


# ============================================================================
#                    COORDINATOR INTEGRATION
# ============================================================================

def register_with_coordinator():
    """Реєструє Smart Money Engine з координатором сканерів"""
    try:
        from scanner_coordinator import scanner_coordinator, ScannerType
        
        def scan_wrapper():
            """Обгортка для сканування"""
            smart_money_engine.scan_watchlist()
        
        scanner_coordinator.set_scan_function(ScannerType.SMART_MONEY, scan_wrapper)
        logger.info("✅ Smart Money registered with Coordinator")
        
    except ImportError:
        logger.warning("Scanner Coordinator not available")
    except Exception as e:
        logger.error(f"Coordinator registration error: {e}")


# Автореєстрація при імпорті
register_with_coordinator()
