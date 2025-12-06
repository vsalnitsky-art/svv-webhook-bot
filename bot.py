#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import logging
import decimal
from datetime import datetime
from decimal import Decimal
from pybit.unified_trading import HTTP
from config import get_api_credentials
from settings_manager import settings
from statistics_service import stats_service
from utils import (
    get_logger, with_retry, validate_webhook_data, validate_stop_loss,
    safe_float, safe_int, metrics, config as app_config
)

logger = get_logger()

class BybitTradingBot:
    def __init__(self):
        """Ініціалізація бота з API сесією"""
        try:
            api_key, api_secret = get_api_credentials()
            # Використовуємо налаштування з app_config або default False
            is_testnet = getattr(app_config, 'BYBIT_TESTNET', False)
            
            self.session = HTTP(
                testnet=is_testnet,
                api_key=api_key,
                api_secret=api_secret
            )
            logger.info("bot_initialized", testnet=is_testnet)
        except Exception as e:
            logger.error("bot_init_failed", error=str(e))
            raise

    def normalize(self, s): 
        """Нормалізує символ (видаляє '.P')"""
        if not s: return ""
        return s.replace('.P', '')

    @with_retry(max_retries=3, exceptions=(Exception,))
    def get_bal(self):
        """Отримує баланс USDT з обробкою ошибок"""
        try:
            b = self.session.get_wallet_balance(accountType="UNIFIED")
            if b.get('retCode') != 0:
                logger.warning("get_balance_failed", retCode=b.get('retCode'))
                return 0.0
            
            for acc in b['result']['list']:
                for c in acc['coin']:
                    if c['coin'] == "USDT": 
                        return safe_float(c['walletBalance'], 0.0)
            return 0.0
        except Exception as e:
            logger.error("get_balance_error", error=str(e), exc_info=True)
            return 0.0
            
    def get_available_balance(self): return self.get_bal()

    @with_retry(max_retries=3, exceptions=(Exception,))
    def get_price(self, s):
        """Отримує поточну ціну символу"""
        try:
            # Нормалізуємо символ
            r = self.session.get_tickers(category="linear", symbol=self.normalize(s))
            if r.get('retCode') != 0:
                logger.warning("get_price_failed", symbol=s, retCode=r.get('retCode'))
                return 0.0
            return safe_float(r['result']['list'][0]['lastPrice'], 0.0)
        except Exception as e:
            logger.error("get_price_error", symbol=s, error=str(e))
            return 0.0

    def get_all_tickers(self):
        try: return self.session.get_tickers(category="linear")['result']['list']
        except: return []

    def get_instr(self, s):
        try:
            r = self.session.get_instruments_info(category="linear", symbol=self.normalize(s))
            return r['result']['list'][0]['lotSizeFilter'], r['result']['list'][0]['priceFilter']
        except: return None, None

    def round_val(self, val, step):
        try:
            d = abs(Decimal(str(step)).as_tuple().exponent)
            return round(val // step * step, d)
        except: return val

    def set_lev(self, s, l):
        try: 
            self.session.set_leverage(
                category="linear", 
                symbol=self.normalize(s), 
                buyLeverage=str(l), 
                sellLeverage=str(l)
            )
        except: pass

    @with_retry(max_retries=2, exceptions=(Exception,))
    def sync_trades(self, days=7):
        """Синхронізує закриті торги з Bybit за останні N днів"""
        try:
            now_ms = int(time.time() * 1000)
            total_start = now_ms - (days * 86400000)
            chunk = 7 * 86400000
            curr_end = now_ms
            synced_count = 0
            
            while curr_end > total_start:
                curr_start = max(curr_end - chunk, total_start)
                r = self.session.get_closed_pnl(
                    category="linear",
                    startTime=int(curr_start),
                    endTime=int(curr_end),
                    limit=100
                )
                
                if r.get('retCode') != 0:
                    logger.warning("sync_trades_failed", retCode=r.get('retCode'), period=f"{curr_start}-{curr_end}")
                    curr_end = curr_start
                    time.sleep(0.5)
                    continue
                
                for t in r['result']['list']:
                    try:
                        side = 'Long' if t['side'] == 'Buy' else 'Short'
                        stats_service.save_trade({
                            'order_id': t['orderId'],
                            'symbol': t['symbol'],
                            'side': side,
                            'qty': safe_float(t['qty']),
                            'entry_price': safe_float(t['avgEntryPrice']),
                            'exit_price': safe_float(t['avgExitPrice']),
                            'pnl': safe_float(t['closedPnl']),
                            'exit_time': datetime.fromtimestamp(int(t['updatedTime']) / 1000),
                            'exit_reason': 'Synced'
                        })
                        synced_count += 1
                    except Exception as e:
                        logger.warning("sync_trade_save_error", trade_id=t.get('orderId'), error=str(e))
                
                curr_end = curr_start
                time.sleep(0.2)
            
            logger.info("trades_synced", count=synced_count, days=days)
        except Exception as e:
            logger.error("sync_trades_error", error=str(e), exc_info=True)
    
    @with_retry(max_retries=2, exceptions=(Exception,))
    def update_sl(self, symbol, new_sl_price):
        """Оновлює Stop Loss для відкритої позиції з валідацією"""
        try:
            norm = self.normalize(symbol)
            _, tick = self.get_instr(norm)
            
            if not tick:
                logger.warning("update_sl_no_instrument", symbol=symbol)
                return False
            
            # Валідуємо SL
            sl_float = safe_float(new_sl_price, 0)
            if sl_float <= 0:
                logger.warning("update_sl_invalid", symbol=symbol, sl_price=sl_float)
                return False
            
            sl_rounded = self.round_val(sl_float, safe_float(tick.get('tickSize', 0.01)))
            
            if sl_rounded <= 0:
                logger.warning("update_sl_rounded_invalid", symbol=symbol, sl_rounded=sl_rounded)
                return False
            
            # Встановлюємо SL
            r = self.session.set_trading_stop(
                category="linear",
                symbol=norm,
                stopLoss=str(sl_rounded),
                positionIdx=0
            )
            
            if r.get('retCode') == 0:
                logger.info("stop_loss_updated", symbol=symbol, new_sl=sl_rounded)
                return True
            else:
                logger.warning("update_sl_api_error", symbol=symbol, retCode=r.get('retCode'))
                return False
        
        except Exception as e:
            logger.error("update_sl_error", symbol=symbol, error=str(e), exc_info=True)
            return False

    # === ВИКОНАННЯ ОРДЕРІВ ===
    def place_order(self, data):
        """
        Розміщує ордер на основі вебхука з валідацією вхідних даних
        """
        try:
            # === ВИПРАВЛЕННЯ: ПОПЕРЕДНЯ НОРМАЛІЗАЦІЯ ===
            # Якщо символ приходить як "ENAUSDT.P", ми робимо його "ENAUSDT"
            # ДО того, як спрацює сувора валідація validate_webhook_data
            if 'symbol' in data and isinstance(data['symbol'], str):
                if data['symbol'].endswith('.P'):
                    data['symbol'] = data['symbol'].replace('.P', '')

            # === ВАЛІДАЦІЯ ВХОДУ ===
            validated_data = validate_webhook_data(data)
            action = validated_data['action']
            symbol = validated_data['symbol']
            norm = self.normalize(symbol)
            
            logger.info("order_request", action=action, symbol=symbol)
            
            # === ЛОГІКА ЗАКРИТТЯ (Close Signal) ===
            if action == "Close":
                return self._handle_close_order(norm, validated_data['direction'])
            
            # === ЛОГІКА ВІДКРИТТЯ (Open Signal) ===
            return self._handle_open_order(norm, validated_data)
            
        except ValueError as e:
            # Помилка валідації - клієнт помилився
            logger.warning("validation_error", error=str(e), data=data)
            return {"status": "error", "reason": f"Invalid data: {str(e)}", "code": "VALIDATION_ERROR"}
        except Exception as e:
            # Неочікувана помилка
            logger.error("place_order_error", error=str(e), exc_info=True, data=data)
            return {"status": "error", "reason": str(e), "code": "UNEXPECTED_ERROR"}
    
    def _handle_close_order(self, symbol: str, direction: str) -> dict:
        """Закриває позицію якщо напрямок збігається"""
        try:
            r = self.session.get_positions(category="linear", symbol=symbol)
            if r.get('retCode') != 0:
                return {"status": "error", "reason": "Failed to get positions"}
            
            pos_list = r['result']['list']
            target = next((p for p in pos_list if safe_float(p.get('size', 0), 0) > 0), None)
            
            if not target:
                logger.info("close_ignored", symbol=symbol, reason="no_position")
                return {"status": "ignored", "reason": "No open position"}
            
            curr_side = target['side']  # "Buy" або "Sell"
            
            # Перевіряємо напрямок
            if (direction == "Long" and curr_side == "Buy") or (direction == "Short" and curr_side == "Sell"):
                # Напрямок збігається - закриваємо
                close_side = "Sell" if curr_side == "Buy" else "Buy"
                qty = str(target['size'])
                
                # Розміщуємо ордер на закриття
                close_result = self.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=close_side,
                    orderType="Market",
                    qty=qty,
                    reduceOnly=True
                )
                
                if close_result.get('retCode') != 0:
                    logger.error("close_order_failed", symbol=symbol, retCode=close_result.get('retCode'))
                    return {"status": "error", "reason": "Failed to close position"}
                
                # Скасовуємо старі ордери (TP/SL)
                try:
                    self.session.cancel_all_orders(category="linear", symbol=symbol)
                except Exception as e:
                    logger.warning("cancel_orders_failed", symbol=symbol, error=str(e))
                
                logger.info("order_closed", symbol=symbol, direction=direction)
                metrics.log_trade_closed(symbol, pnl=0.0)  # PnL визначимо потім
                return {"status": "ok", "message": f"Closed {direction}"}
            else:
                logger.warning("close_mismatch", symbol=symbol, expected=direction, actual=curr_side)
                return {"status": "ignored", "reason": f"Direction mismatch: expected {direction}, got {curr_side}"}
        
        except Exception as e:
            logger.error("close_order_error", symbol=symbol, error=str(e), exc_info=True)
            return {"status": "error", "reason": str(e)}
    
    def _handle_open_order(self, symbol: str, data: dict) -> dict:
        """Відкриває нову позицію або змінює поточну"""
        try:
            action = data['action']  # "Buy" або "Sell"
            risk = data['riskPercent']
            lev = data['leverage']
            
            # Отримання даних інструменту
            price = self.get_price(symbol)
            lot, tick = self.get_instr(symbol)
            
            if not price or not lot or not tick:
                logger.error("instrument_error", symbol=symbol, price=price, lot=lot, tick=tick)
                return {"status": "error", "reason": "Cannot get instrument data"}
            
            # Перевірка баланса
            bal = self.get_bal()
            # MIN_BALANCE можна брати з config.MIN_BALANCE якщо він там є, або хардкод
            min_bal = getattr(app_config, 'MIN_BALANCE', 5.0) 
            if bal < min_bal:
                logger.warning("insufficient_balance", symbol=symbol, balance=bal, required=min_bal)
                return {"status": "no_balance", "balance": bal, "reason": f"Minimum {min_bal} USDT required"}
            
            # === ПЕРЕВІРКА ПОТОЧНИХ ПОЗИЦІЙ (для REVERSAL) ===
            r = self.session.get_positions(category="linear", symbol=symbol)
            if r.get('retCode') != 0:
                logger.error("get_positions_failed", symbol=symbol, retCode=r.get('retCode'))
                return {"status": "error", "reason": "Cannot get current positions"}
            
            pos_list = r['result']['list']
            existing_pos = next((p for p in pos_list if safe_float(p.get('size', 0), 0) > 0), None)
            
            if existing_pos:
                current_side = existing_pos['side']
                
                # Якщо той же напрямок - ігноруємо
                if current_side == action:
                    logger.info("already_in_position", symbol=symbol, side=action)
                    return {"status": "ignored", "reason": f"Already in {action}"}
                
                # REVERSAL: закриваємо позицію перед новим входом
                logger.info("reversal_detected", symbol=symbol, old_side=current_side, new_side=action)
                close_side = "Sell" if current_side == "Buy" else "Buy"
                
                self.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=close_side,
                    orderType="Market",
                    qty=str(existing_pos['size']),
                    reduceOnly=True
                )
                
                # Скасовуємо старі ордери
                try:
                    self.session.cancel_all_orders(category="linear", symbol=symbol)
                except:
                    pass
                
                # Чекаємо на оновлення баланса
                time.sleep(1)
                # Оновлюємо баланс
                bal = self.get_bal()
            
            # === РОЗРАХУНОК ОБ'ЄМУ ===
            qty_step = safe_float(lot.get('qtyStep', 0.01))
            min_qty = safe_float(lot.get('minOrderQty', 1))
            tick_size = safe_float(tick.get('tickSize', 0.01))
            
            # Розраховуємо кількість: (баланс × risk × leverage) / ціна
            raw_qty = (bal * (risk / 100) * 0.98 * lev) / price
            qty = self.round_val(raw_qty, qty_step)
            if qty < min_qty:
                qty = min_qty
            
            # Встановлюємо левередж
            self.set_lev(symbol, lev)
            
            # === РОЗМІЩЕННЯ ОРДЕРА ===
            logger.info("placing_order", symbol=symbol, action=action, qty=qty, price=price, leverage=lev)
            
            order_result = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=action,
                orderType="Market",
                qty=str(qty)
            )
            
            if order_result.get('retCode') != 0:
                logger.error("order_placement_failed", symbol=symbol, retCode=order_result.get('retCode'))
                return {"status": "error", "reason": f"Order failed: {order_result.get('retMsg', 'Unknown error')}"}
            
            metrics.log_trade_opened(symbol, qty, price)
            
            # === ВСТАНОВЛЕННЯ STOP LOSS (З ВІДСОТКІВ) ===
            if data.get('stopLossPercent') and data.get('entryPrice'):
                sl_percent = safe_float(data['stopLossPercent'])
                entry_price = safe_float(data['entryPrice'])
                
                # Розраховуємо абсолютну ціну SL на основі напрямку
                if action == "Buy":
                    # Для Long: SL нижче за entry (entry * (1 - percent/100))
                    sl_raw = entry_price * (1 - sl_percent / 100)
                else:
                    # Для Short: SL вище за entry (entry * (1 + percent/100))
                    sl_raw = entry_price * (1 + sl_percent / 100)
                
                # Валідуємо SL
                if validate_stop_loss(sl_raw, entry_price, action):
                    sl_rounded = self.round_val(sl_raw, tick_size)
                    try:
                        self.session.set_trading_stop(
                            category="linear",
                            symbol=symbol,
                            stopLoss=str(sl_rounded),
                            positionIdx=0
                        )
                        logger.info("stop_loss_set", symbol=symbol, sl_price=sl_rounded, sl_percent=sl_percent)
                    except Exception as e:
                        logger.error("sl_set_error", symbol=symbol, error=str(e))
                else:
                    logger.warning("invalid_sl", symbol=symbol, sl_price=sl_raw, entry_price=entry_price, side=action)
            else:
                logger.warning("no_stop_loss", symbol=symbol, message="Trade is unprotected!")
            
            # === ВСТАНОВЛЕННЯ TAKE PROFIT (Внутрішній метод) ===
            self._tp(symbol, action, price, qty, data, tick_size, qty_step)
            
            logger.info("order_success", symbol=symbol, action=action, qty=qty)
            return {"status": "ok", "qty": qty, "price": price, "leverage": lev}
        
        except Exception as e:
            logger.error("open_order_error", symbol=symbol, error=str(e), exc_info=True)
            return {"status": "error", "reason": str(e)}

    def _tp(self, s, side, ep, qty, d, tick, step):
        """Розрахунок та встановлення TP"""
        try:
            mode = settings.get("tp_mode")
            exit_side = "Sell" if side=="Buy" else "Buy"
            
            if mode == "Fixed_1_50":
                q = self.round_val(qty*0.5, step)
                p = self.round_val(ep*1.01 if side=="Buy" else ep*0.99, tick)
                if q>0: 
                    self.session.place_order(
                        category="linear", symbol=s, side=exit_side, 
                        orderType="Limit", qty=str(q), price=str(p), reduceOnly=True
                    )
            elif mode == "Ladder_3":
                tp = float(d.get('takeProfitPercent', settings.get('fixedTP')))/100
                q_step = self.round_val(qty*0.33, step)
                for i, mult in enumerate([1/3, 2/3, 1]):
                    pct = tp*mult
                    p = self.round_val(ep*(1+pct) if side=="Buy" else ep*(1-pct), tick)
                    q = self.round_val(qty - q_step*2, step) if i==2 else q_step
                    if q>0: 
                        self.session.place_order(
                            category="linear", symbol=s, side=exit_side, 
                            orderType="Limit", qty=str(q), price=str(p), reduceOnly=True
                        )
        except Exception as e:
            logger.warning("tp_error", error=str(e))

bot_instance = BybitTradingBot()