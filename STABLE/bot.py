#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import decimal
import time
from datetime import datetime
from pybit.unified_trading import HTTP
from bot_config import config
from config import get_api_credentials
from settings_manager import settings
from statistics_service import stats_service

logger = logging.getLogger(__name__)

class BybitTradingBot:
    def __init__(self):
        k, s = get_api_credentials()
        self.session = HTTP(testnet=False, api_key=k, api_secret=s)

    def normalize(self, s): return s.replace('.P', '')

    def get_bal(self):
        try:
            b = self.session.get_wallet_balance(accountType="UNIFIED")
            for acc in b['result']['list']:
                for c in acc['coin']:
                    if c['coin'] == "USDT": return float(c['walletBalance'])
            return 0.0
        except: return 0.0
            
    def get_available_balance(self): return self.get_bal()

    def get_price(self, s):
        try: 
            return float(self.session.get_tickers(category="linear", symbol=self.normalize(s))['result']['list'][0]['lastPrice'])
        except: return 0.0

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
            d = abs(decimal.Decimal(str(step)).as_tuple().exponent)
            return round(val // step * step, d)
        except: return val

    def set_lev(self, s, l):
        try: self.session.set_leverage(category="linear", symbol=self.normalize(s), buyLeverage=str(l), sellLeverage=str(l))
        except: pass

    # === ІСТОРІЯ (Синхронізація) ===
    def sync_trades(self, days=7):
        try:
            now_ms = int(time.time() * 1000)
            total_start = now_ms - (days * 86400000)
            chunk = 7 * 86400000
            curr_end = now_ms
            
            while curr_end > total_start:
                curr_start = max(curr_end - chunk, total_start)
                r = self.session.get_closed_pnl(category="linear", startTime=int(curr_start), endTime=int(curr_end), limit=100)
                if r['retCode'] == 0:
                    for t in r['result']['list']:
                        side = 'Long' if t['side'] == 'Buy' else 'Short'
                        stats_service.save_trade({
                            'order_id': t['orderId'], 'symbol': t['symbol'], 'side': side,
                            'qty': float(t['qty']), 'entry_price': float(t['avgEntryPrice']),
                            'exit_price': float(t['avgExitPrice']), 'pnl': float(t['closedPnl']),
                            'exit_time': datetime.fromtimestamp(int(t['updatedTime'])/1000),
                            'exit_reason': 'Signal/TP/SL'
                        })
                curr_end = curr_start
                time.sleep(0.2)
        except Exception as e: logger.error(f"Sync error: {e}")
    
    # === TRAILING STOP UPDATE ===
    def update_sl(self, symbol, new_sl_price):
        """Оновлює Stop Loss для відкритої позиції"""
        try:
            norm = self.normalize(symbol)
            _, tick = self.get_instr(norm)
            if not tick: return False
            
            sl_rounded = self.round_val(float(new_sl_price), float(tick['tickSize']))
            if sl_rounded <= 0: return False

            self.session.set_trading_stop(category="linear", symbol=norm, stopLoss=str(sl_rounded), positionIdx=0)
            return True
        except Exception as e:
            logger.error(f"Update SL Error ({symbol}): {e}")
            return False

    # === ВИКОНАННЯ ОРДЕРІВ ===
    def place_order(self, data):
        try:
            action = data.get('action') # "Buy", "Sell" або "Close"
            symbol = data.get('symbol')
            norm = self.normalize(symbol)
            
            # --- 1. ЛОГІКА ЗАКРИТТЯ (Close Signal) ---
            if action == "Close":
                direction = data.get('direction') # "Long" or "Short"
                pos_list = self.session.get_positions(category="linear", symbol=norm)['result']['list']
                target = next((p for p in pos_list if float(p['size']) > 0), None)
                
                if not target: return {"status": "ignored", "reason": "No position to close"}
                
                curr_side = target['side'] # "Buy" or "Sell"
                # Перевіряємо, чи співпадає напрямок закриття
                if (direction == "Long" and curr_side == "Buy") or (direction == "Short" and curr_side == "Sell"):
                    close_side = "Sell" if curr_side == "Buy" else "Buy"
                    self.session.place_order(category="linear", symbol=norm, side=close_side, orderType="Market", qty=target['size'], reduceOnly=True)
                    try: self.session.cancel_all_orders(category="linear", symbol=norm)
                    except: pass
                    return {"status": "ok", "message": f"Closed {direction}"}
                return {"status": "ignored", "reason": "Direction mismatch"}

            # --- 2. ЛОГІКА ВІДКРИТТЯ (Open Signal) ---
            
            # Отримання даних інструменту
            price = self.get_price(norm)
            lot, tick = self.get_instr(norm)
            if not price or not lot: return {"status": "error", "reason": "No price/instrument data"}

            # ПЕРЕВІРКА ПОТОЧНИХ ПОЗИЦІЙ (ЛОГІКА ПЕРЕВОРОТУ)
            pos_list = self.session.get_positions(category="linear", symbol=norm)['result']['list']
            existing_pos = next((p for p in pos_list if float(p['size']) > 0), None)

            if existing_pos:
                current_side = existing_pos['side'] # "Buy" or "Sell"
                
                # Якщо напрямок той самий — ігноруємо (вже в ринку)
                if current_side == action:
                    return {"status": "ignored", "reason": f"Already in {action}"}
                
                # Якщо напрямок протилежний — ПЕРЕВОРОТ (Закриття + Відкриття)
                else:
                    logger.info(f"🔄 REVERSAL: Closing {current_side} to open {action} on {symbol}")
                    
                    # а) Закриваємо поточну позицію по ринку
                    close_side = "Sell" if current_side == "Buy" else "Buy"
                    self.session.place_order(
                        category="linear", symbol=norm, side=close_side, 
                        orderType="Market", qty=existing_pos['size'], reduceOnly=True
                    )
                    
                    # б) Скасовуємо всі старі ордери (TP/SL)
                    try: self.session.cancel_all_orders(category="linear", symbol=norm)
                    except: pass
                    
                    # в) Чекаємо оновлення маржі (важливо!)
                    time.sleep(0.5)

            # РОЗРАХУНОК ОБ'ЄМУ (Вже на чистий баланс)
            bal = self.get_bal()
            if bal < 5: return {"status": "no_balance", "balance": bal}

            risk = float(data.get('riskPercent', settings.get('riskPercent')))
            lev = int(data.get('leverage', settings.get('leverage')))
            
            qty_step = float(lot['qtyStep'])
            min_qty = float(lot['minOrderQty'])
            tick_size = float(tick['tickSize'])
            
            raw_qty = (bal * (risk/100) * 0.98 * lev) / price
            qty = self.round_val(raw_qty, qty_step)
            if qty < min_qty: qty = min_qty
            
            self.set_lev(norm, lev)
            
            # ВХІД У РИНОК
            logger.info(f"🚀 OPEN {action} {symbol} | Qty: {qty}")
            self.session.place_order(category="linear", symbol=norm, side=action, orderType="Market", qty=str(qty))
            
            # СТОП-ЛОСС (ТІЛЬКИ З JSON)
            if data.get('sl_price'):
                sl_raw = float(data['sl_price'])
                sl_price_rounded = self.round_val(sl_raw, tick_size)
                
                if sl_price_rounded > 0:
                    try:
                        self.session.set_trading_stop(
                            category="linear", symbol=norm, 
                            stopLoss=str(sl_price_rounded), positionIdx=0
                        )
                        logger.info(f"✅ SL Set from JSON: {sl_price_rounded}")
                    except Exception as e:
                        logger.error(f"❌ SL Set Error: {e}")
            else:
                logger.warning(f"⚠️ NO SL in JSON for {symbol}. Trade is unprotected!")

            # TAKE PROFIT (ВИМКНЕНО - функція пуста)
            self._place_take_profits(norm, action, price, qty, data, tick_size, qty_step)

            return {"status": "ok", "qty": qty}

        except Exception as e:
            logger.error(f"Trading Error: {e}")
            return {"status": "error", "reason": str(e)}

    def _place_take_profits(self, symbol, side, entry_price, total_qty, data, tick_size, qty_step):
        # TP вимкнено згідно з новою стратегією супроводу
        pass 

bot_instance = BybitTradingBot()