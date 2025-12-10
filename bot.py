#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from datetime import datetime
from decimal import Decimal
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError  # ✅ Імпорт для обробки 34040

# === ІМПОРТИ ===
from bot_config import config
from config import get_api_credentials
from settings_manager import settings
from statistics_service import stats_service
from utils import (
    get_logger, with_retry, validate_webhook_data, validate_stop_loss,
    safe_float, safe_int, metrics
)

logger = get_logger()

class BybitTradingBot:
    def __init__(self):
        """Ініціалізація бота з API сесією"""
        try:
            api_key, api_secret = get_api_credentials()
            self.session = HTTP(
                testnet=False,
                api_key=api_key,
                api_secret=api_secret
            )
            logger.info("bot_initialized", testnet=False)
        except Exception as e:
            logger.error("bot_init_failed", error=str(e))
            raise

    def normalize(self, s): 
        """Нормалізує символ (видаляє '.P')"""
        return s.replace('.P', '')

    @with_retry(max_retries=3, exceptions=(Exception,))
    def get_bal(self):
        """Отримує баланс USDT з обробкою помилок"""
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
            
    def get_available_balance(self): 
        return self.get_bal()

    @with_retry(max_retries=3, exceptions=(Exception,))
    def get_price(self, s):
        """Отримує поточну ціну символу"""
        try:
            r = self.session.get_tickers(category="linear", symbol=self.normalize(s))
            if r.get('retCode') != 0:
                logger.warning("get_price_failed", symbol=s, retCode=r.get('retCode'))
                return 0.0
            return safe_float(r['result']['list'][0]['lastPrice'], 0.0)
        except Exception as e:
            logger.error("get_price_error", symbol=s, error=str(e))
            return 0.0

    def get_all_tickers(self):
        """Отримує список всіх тікерів"""
        try: 
            return self.session.get_tickers(category="linear")['result']['list']
        except: 
            return []

    def get_instr(self, s):
        """Отримує інформацію про інструмент (розмір лота, крок ціни)"""
        try:
            r = self.session.get_instruments_info(category="linear", symbol=self.normalize(s))
            return r['result']['list'][0]['lotSizeFilter'], r['result']['list'][0]['priceFilter']
        except: 
            return None, None

    def round_val(self, val, step):
        """Округлює значення до кроку"""
        try:
            d = abs(Decimal(str(step)).as_tuple().exponent)
            return round(val // step * step, d)
        except: 
            return val

    def set_lev(self, s, l):
        """Встановлює кредитне плече"""
        try: 
            self.session.set_leverage(category="linear", symbol=self.normalize(s), buyLeverage=str(l), sellLeverage=str(l))
        except: 
            pass

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
                        symbol = t['symbol']
                        qty = safe_float(t['qty'])
                        entry_price = safe_float(t['avgEntryPrice'])
                        exit_price = safe_float(t['avgExitPrice'])
                        
                        # ✨ РОЗРАХОВУЄМО КОМІСІЇ
                        TAKER_RATE = 0.000275
                        opening_fee = qty * entry_price * TAKER_RATE
                        closing_fee = qty * exit_price * TAKER_RATE
                        funding_fee = 0.0
                        
                        stats_service.save_trade({
                            'order_id': t['orderId'],
                            'symbol': symbol,
                            'side': side,
                            'qty': qty,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': safe_float(t['closedPnl']),
                            'exit_time': datetime.fromtimestamp(int(t['updatedTime']) / 1000),
                            'exit_reason': 'Synced',
                            'opening_fee': opening_fee,
                            'closing_fee': closing_fee,
                            'funding_fee': funding_fee
                        })
                        synced_count += 1
                    except Exception as e:
                        logger.warning("sync_trade_save_error", trade_id=t.get('orderId'), error=str(e))
                
                curr_end = curr_start
                time.sleep(0.2)
            
            logger.info("trades_synced", count=synced_count, days=days)
        except Exception as e:
            logger.error("sync_trades_error", error=str(e), exc_info=True)
    
    def update_sl(self, symbol, new_sl_price, position_idx=0):
        """
        Оновлює Stop Loss для відкритої позиції.
        
        ✅ Без retry - 34040 не є помилкою що потребує повтору!
        """
        try:
            norm = self.normalize(symbol)
            _, tick = self.get_instr(norm)
            
            if not tick:
                logger.warning("update_sl_no_instrument", symbol=symbol)
                return False
            
            sl_float = safe_float(new_sl_price, 0)
            if sl_float <= 0:
                logger.warning("update_sl_invalid", symbol=symbol, sl_price=sl_float)
                return False
            
            sl_rounded = self.round_val(sl_float, safe_float(tick.get('tickSize', 0.01)))
            
            if sl_rounded <= 0:
                logger.warning("update_sl_rounded_invalid", symbol=symbol, sl_rounded=sl_rounded)
                return False
            
            # ✅ ВИКОРИСТОВУЄМО position_idx, переданий зі сканера
            r = self.session.set_trading_stop(
                category="linear",
                symbol=norm,
                stopLoss=str(sl_rounded),
                positionIdx=position_idx 
            )
            
            ret_code = r.get('retCode')
            
            if ret_code == 0:
                logger.info("stop_loss_updated", symbol=symbol, new_sl=sl_rounded, idx=position_idx)
                return True
            elif ret_code == 34040:
                # Position not modified - позиція закрита або SL вже на цьому рівні
                # Це нормальна ситуація, не помилка - логуємо як INFO
                logger.info("sl_not_modified", symbol=symbol, sl=sl_rounded, reason="position closed or SL unchanged")
                return False
            else:
                logger.warning("update_sl_api_error", symbol=symbol, retCode=ret_code, msg=r.get('retMsg'))
                return False
        
        except InvalidRequestError as e:
            # ✅ Спеціальна обробка для pybit InvalidRequestError
            error_str = str(e)
            if "34040" in error_str or "not modified" in error_str.lower():
                # Це НЕ помилка - просто позиція закрита або SL не змінився
                # Логуємо тихо без traceback
                logger.info("sl_position_unchanged", symbol=symbol, sl=sl_rounded)
                return False
            # Інші InvalidRequestError - логуємо як warning
            logger.warning("update_sl_invalid_request", symbol=symbol, error=error_str)
            return False
            
        except Exception as e:
            error_str = str(e)
            # Ігноруємо 34040 якщо прийшло як інший exception
            if "34040" in error_str or "not modified" in error_str.lower():
                logger.info("sl_not_modified_exc", symbol=symbol, reason="position closed or SL unchanged")
                return False
            logger.error("update_sl_error", symbol=symbol, error=error_str, exc_info=True)
            return False

    def place_order(self, data):
        """Розміщує ордер на основі вебхука"""
        try:
            validated_data = validate_webhook_data(data)
            action = validated_data['action']
            symbol = validated_data['symbol']
            norm = self.normalize(symbol)
            
            logger.info("order_request", action=action, symbol=symbol)
            
            if action == "Close":
                return self._handle_close_order(norm, validated_data['direction'])
            
            return self._handle_open_order(norm, validated_data)
            
        except ValueError as e:
            logger.warning("validation_error", error=str(e), data=data)
            return {"status": "error", "reason": f"Invalid data: {str(e)}", "code": "VALIDATION_ERROR"}
        except Exception as e:
            logger.error("place_order_error", error=str(e), exc_info=True, data=data)
            return {"status": "error", "reason": str(e), "code": "UNEXPECTED_ERROR"}
    
    def _handle_close_order(self, symbol: str, direction: str) -> dict:
        """Закриває позицію"""
        try:
            r = self.session.get_positions(category="linear", symbol=symbol)
            if r.get('retCode') != 0:
                return {"status": "error", "reason": "Failed to get positions"}
            
            pos_list = r['result']['list']
            target = next((p for p in pos_list if safe_float(p.get('size', 0), 0) > 0), None)
            
            if not target:
                logger.info("close_ignored", symbol=symbol, reason="no_position")
                return {"status": "ignored", "reason": "No open position"}
            
            curr_side = target['side']
            
            if (direction == "Long" and curr_side == "Buy") or (direction == "Short" and curr_side == "Sell"):
                close_side = "Sell" if curr_side == "Buy" else "Buy"
                qty = str(target['size'])
                
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
                
                try:
                    self.session.cancel_all_orders(category="linear", symbol=symbol)
                except Exception as e:
                    logger.warning("cancel_orders_failed", symbol=symbol, error=str(e))
                
                logger.info("order_closed", symbol=symbol, direction=direction)
                metrics.log_trade_closed(symbol, pnl=0.0)
                return {"status": "ok", "message": f"Closed {direction}"}
            else:
                logger.warning("close_mismatch", symbol=symbol, expected=direction, actual=curr_side)
                return {"status": "ignored", "reason": f"Direction mismatch: expected {direction}, got {curr_side}"}
        
        except Exception as e:
            logger.error("close_order_error", symbol=symbol, error=str(e), exc_info=True)
            return {"status": "error", "reason": str(e)}
    
    def _handle_open_order(self, symbol: str, data: dict) -> dict:
        """Відкриває нову позицію"""
        try:
            action = data['action']
            risk = data['riskPercent']
            lev = data['leverage']
            
            price = self.get_price(symbol)
            lot, tick = self.get_instr(symbol)
            
            if not price or not lot or not tick:
                logger.error("instrument_error", symbol=symbol, price=price, lot=lot, tick=tick)
                return {"status": "error", "reason": "Cannot get instrument data"}
            
            bal = self.get_bal()
            
            min_bal = getattr(config, 'MIN_BALANCE', 5.0) 
            
            if bal < min_bal:
                logger.warning("insufficient_balance", symbol=symbol, balance=bal, required=min_bal)
                return {"status": "no_balance", "balance": bal, "reason": f"Minimum {min_bal} USDT required"}
            
            r = self.session.get_positions(category="linear", symbol=symbol)
            if r.get('retCode') != 0:
                logger.error("get_positions_failed", symbol=symbol, retCode=r.get('retCode'))
                return {"status": "error", "reason": "Cannot get current positions"}
            
            pos_list = r['result']['list']
            existing_pos = next((p for p in pos_list if safe_float(p.get('size', 0), 0) > 0), None)
            
            if existing_pos:
                current_side = existing_pos['side']
                if current_side == action:
                    logger.info("already_in_position", symbol=symbol, side=action)
                    return {"status": "ignored", "reason": f"Already in {action}"}
                
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
                
                try:
                    self.session.cancel_all_orders(category="linear", symbol=symbol)
                except: pass
                time.sleep(1)
            
            qty_step = safe_float(lot.get('qtyStep', 0.01))
            min_qty = safe_float(lot.get('minOrderQty', 1))
            tick_size = safe_float(tick.get('tickSize', 0.01))
            
            raw_qty = (bal * (risk / 100) * 0.98 * lev) / price
            qty = self.round_val(raw_qty, qty_step)
            if qty < min_qty:
                qty = min_qty
            
            self.set_lev(symbol, lev)
            
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
            
            if data.get('stopLossPercent') and data.get('entryPrice'):
                sl_percent = safe_float(data['stopLossPercent'])
                entry_price = safe_float(data['entryPrice'])
                
                if action == "Buy":
                    sl_raw = entry_price * (1 - sl_percent / 100)
                else:
                    sl_raw = entry_price * (1 + sl_percent / 100)
                
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
            
            # === Take Profit Strategy ===
            use_tp = settings.get("use_tp", False)
            tp_mode = settings.get("tp_mode", "Fixed_1_50")
            
            if use_tp and tp_mode != "None":
                self._tp(symbol, action, price, qty, data, tick_size, qty_step)
            else:
                logger.info("tp_disabled_by_settings", symbol=symbol, use_tp=use_tp, tp_mode=tp_mode)
            
            logger.info("order_success", symbol=symbol, action=action, qty=qty)
            return {"status": "ok", "qty": qty, "price": price, "leverage": lev}
        
        except Exception as e:
            logger.error("open_order_error", symbol=symbol, error=str(e), exc_info=True)
            return {"status": "error", "reason": str(e)}

    # === TAKE PROFIT LOGIC (Smart TP: 50/25/25 з Trailing) ===
    def _tp(self, s, side, ep, qty, d, tick, step):
        """
        🎯 Smart Take Profit Strategy (50/25/25)
        
        Алгоритм:
        1. TP1 (+1%): Закриває 50% позиції
        2. TP2 (+2%): Закриває 25% позиції
        3. Залишок 25%: Для Trailing Stop (без TP ордера)
        
        Моніторинг (scanner.py) відстежує:
        - Коли TP1 виконався → SL в Break-Even
        - Коли TP2 виконався → Активує Trailing
        
        Bybit нюанси:
        - reduceOnly=True для закриття частини позиції
        - Округлення qty до qtyStep, price до tickSize
        - Перевірка minOrderQty
        """
        try:
            mode = settings.get("tp_mode")
            exit_side = "Sell" if side == "Buy" else "Buy"
            
            # Отримуємо мінімальний розмір ордера
            lot, _ = self.get_instr(s)
            min_qty = safe_float(lot.get('minOrderQty', 0.001)) if lot else 0.001
            
            # 1. Пріоритет: Якщо TP прийшов у сигналі
            if d.get('takeProfitPercent') and d.get('entryPrice'):
                tp_percent = safe_float(d['takeProfitPercent'])
                if tp_percent > 0:
                    if side == "Buy":
                        tp_price = ep * (1 + tp_percent / 100)
                    else:
                        tp_price = ep * (1 - tp_percent / 100)
                        
                    tp_rounded = self.round_val(tp_price, tick)
                    
                    self.session.place_order(
                        category="linear",
                        symbol=s,
                        side=exit_side,
                        orderType="Limit",
                        qty=str(qty),
                        price=str(tp_rounded),
                        reduceOnly=True
                    )
                    logger.info("tp_signal_set", symbol=s, price=tp_rounded)
                    return

            # 2. Режим Smart_TP (50/25/25): Головний режим з Trailing
            if mode == "Smart_TP" or mode == "Fixed_1_50":
                # === РОЗРАХУНОК ОБ'ЄМІВ ===
                # TP1: 50% позиції
                q1 = self.round_val(qty * 0.50, step)
                # TP2: 25% позиції  
                q2 = self.round_val(qty * 0.25, step)
                # Залишок 25%: для Trailing (не ставимо TP ордер!)
                q_trailing = self.round_val(qty - q1 - q2, step)
                
                # Перевірка мінімальних розмірів
                if q1 < min_qty:
                    logger.warning("tp1_qty_too_small", symbol=s, qty=q1, min=min_qty)
                    q1 = 0
                if q2 < min_qty:
                    logger.warning("tp2_qty_too_small", symbol=s, qty=q2, min=min_qty)
                    # Додаємо до trailing частини
                    q_trailing = self.round_val(q_trailing + q2, step)
                    q2 = 0
                
                # === РОЗРАХУНОК ЦІН ===
                if side == "Buy":
                    p1 = self.round_val(ep * 1.01, tick)  # TP1: +1%
                    p2 = self.round_val(ep * 1.02, tick)  # TP2: +2% 
                else:
                    p1 = self.round_val(ep * 0.99, tick)  # TP1: -1%
                    p2 = self.round_val(ep * 0.98, tick)  # TP2: -2%
                
                # === РОЗМІЩЕННЯ ОРДЕРІВ ===
                
                # TP1: 50% на +1%
                if q1 > 0:
                    try:
                        r1 = self.session.place_order(
                            category="linear",
                            symbol=s,
                            side=exit_side,
                            orderType="Limit",
                            qty=str(q1),
                            price=str(p1),
                            reduceOnly=True
                        )
                        if r1.get('retCode') == 0:
                            logger.info("tp1_set", symbol=s, price=p1, qty=q1, pct="50%")
                        else:
                            logger.warning("tp1_failed", symbol=s, error=r1.get('retMsg'))
                    except Exception as e:
                        logger.error("tp1_error", symbol=s, error=str(e))
                
                # TP2: 25% на +2%
                if q2 > 0:
                    try:
                        r2 = self.session.place_order(
                            category="linear",
                            symbol=s,
                            side=exit_side,
                            orderType="Limit",
                            qty=str(q2),
                            price=str(p2),
                            reduceOnly=True
                        )
                        if r2.get('retCode') == 0:
                            logger.info("tp2_set", symbol=s, price=p2, qty=q2, pct="25%")
                        else:
                            logger.warning("tp2_failed", symbol=s, error=r2.get('retMsg'))
                    except Exception as e:
                        logger.error("tp2_error", symbol=s, error=str(e))
                
                # Логуємо що 25% залишається для Trailing
                if q_trailing > 0:
                    logger.info("trailing_reserved", symbol=s, qty=q_trailing, pct="25%", 
                               note="Will activate after TP2")
                
                return

            # 3. Режим Ladder_3: Драбинка на 3 рівні (без trailing)
            elif mode == "Ladder_3":
                base_tp = float(d.get('takeProfitPercent', settings.get('fixedTP', 3.0))) / 100
                q_step = self.round_val(qty * 0.33, step)
                
                multipliers = [1/3, 2/3, 1.0]
                
                for i, mult in enumerate(multipliers):
                    pct = base_tp * mult
                    if side == "Buy":
                        p = self.round_val(ep * (1 + pct), tick)
                    else:
                        p = self.round_val(ep * (1 - pct), tick)
                    
                    if i == 2:
                        current_q = self.round_val(qty - q_step * 2, step)
                    else:
                        current_q = q_step
                    
                    if current_q >= min_qty:
                        self.session.place_order(
                            category="linear",
                            symbol=s,
                            side=exit_side,
                            orderType="Limit",
                            qty=str(current_q),
                            price=str(p),
                            reduceOnly=True
                        )
                        logger.info(f"tp_ladder_{i+1}_set", symbol=s, price=p, qty=current_q)

        except Exception as e:
            logger.error("tp_set_error", symbol=s, error=str(e))

    def normalize(self, s):
        """Нормалізує символ (видаляє '.P')"""
        return s.replace('.P', '')

bot_instance = BybitTradingBot()
