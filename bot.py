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
            
            # Округлюємо ціну згідно з кроком інструмента
            sl_rounded = self.round_val(float(new_sl_price), float(tick['tickSize']))
            
            if sl_rounded <= 0: return False

            # Відправляємо запит на зміну SL
            self.session.set_trading_stop(
                category="linear",
                symbol=norm,
                stopLoss=str(sl_rounded),
                positionIdx=0 # 0 для One-Way Mode
            )
            return True
        except Exception as e:
            logger.error(f"Update SL Error ({symbol}): {e}")
            return False

    # === ВИКОНАННЯ ОРДЕРІВ ===
    def place_order(self, data):
        try:
            action = data.get('action') 
            symbol = data.get('symbol')
            norm = self.normalize(symbol)
            
            # --- CLOSE LOGIC ---
            if action == "Close":
                direction = data.get('direction')
                pos_list = self.session.get_positions(category="linear", symbol=norm)['result']['list']
                target = next((p for p in pos_list if float(p['size']) > 0), None)
                
                if not target: return {"status": "ignored", "reason": "No position"}
                
                curr_side = target['side']
                if (direction == "Long" and curr_side == "Buy") or (direction == "Short" and curr_side == "Sell"):
                    close_side = "Sell" if curr_side == "Buy" else "Buy"
                    self.session.place_order(category="linear", symbol=norm, side=close_side, orderType="Market", qty=target['size'], reduceOnly=True)
                    try: self.session.cancel_all_orders(category="linear", symbol=norm)
                    except: pass
                    return {"status": "ok", "message": f"Closed {direction}"}
                return {"status": "ignored"}

            # --- OPEN LOGIC ---
            pos_list = self.session.get_positions(category="linear", symbol=norm)['result']['list']
            if any(float(p['size']) > 0 for p in pos_list):
                return {"status": "ignored", "reason": "Position exists"}
            
            risk = float(data.get('riskPercent', settings.get('riskPercent')))
            lev = int(data.get('leverage', settings.get('leverage')))
            
            price = self.get_price(norm)
            lot, tick = self.get_instr(norm)
            if not price or not lot: return {"status": "error", "reason": "No price data"}
            
            bal = self.get_bal()
            if bal < 5: return {"status": "no_balance", "balance": bal}

            qty_step = float(lot['qtyStep'])
            min_qty = float(lot['minOrderQty'])
            tick_size = float(tick['tickSize'])
            
            raw_qty = (bal * (risk/100) * 0.98 * lev) / price
            qty = self.round_val(raw_qty, qty_step)
            if qty < min_qty: qty = min_qty
            
            self.set_lev(norm, lev)
            
            logger.info(f"🚀 OPEN {action} {symbol} | Qty: {qty}")
            self.session.place_order(category="linear", symbol=norm, side=action, orderType="Market", qty=str(qty))
            
            # === STOP LOSS LOGIC (OB EXTREMITY) ===
            sl_price = 0.0
            
            if data.get('sl_price'):
                sl_price = float(data['sl_price'])
            else:
                sl_pct = float(data.get('stopLossPercent', settings.get('fixedSL')))
                if sl_pct > 0:
                    dist = price * (sl_pct / 100)
                    sl_price = price - dist if action == "Buy" else price + dist

            if sl_price > 0:
                sl_price_rounded = self.round_val(sl_price, tick_size)
                if sl_price_rounded > 0:
                    try:
                        self.session.set_trading_stop(category="linear", symbol=norm, stopLoss=str(sl_price_rounded), positionIdx=0)
                    except Exception as e:
                        logger.error(f"❌ Помилка установки SL: {e}")

            # === TAKE PROFIT LOGIC ===
            self._place_take_profits(norm, action, price, qty, data, tick_size, qty_step)

            return {"status": "ok", "qty": qty}

        except Exception as e:
            logger.error(f"Trading Error: {e}")
            return {"status": "error", "reason": str(e)}

    def _place_take_profits(self, symbol, side, entry_price, total_qty, data, tick_size, qty_step):
        try:
            tp_mode = settings.get("tp_mode")
            exit_side = "Sell" if side == "Buy" else "Buy"
            
            if tp_mode == "Fixed_1_50":
                percent = 0.01 
                qty_to_close = self.round_val(total_qty * 0.5, qty_step)
                tp_price = entry_price * (1 + percent) if side == "Buy" else entry_price * (1 - percent)
                tp_price = self.round_val(tp_price, tick_size)
                
                if qty_to_close > 0:
                    self.session.place_order(category="linear", symbol=symbol, side=exit_side, orderType="Limit", qty=str(qty_to_close), price=str(tp_price), reduceOnly=True)

            elif tp_mode == "Ladder_3":
                total_tp_pct = float(data.get('takeProfitPercent', settings.get('fixedTP'))) / 100
                step_pct = total_tp_pct / 3
                q_part = self.round_val(total_qty * 0.33, qty_step)
                
                targets = [(step_pct, q_part), (step_pct * 2, q_part), (total_tp_pct, total_qty - q_part*2)]
                
                for pct, q in targets:
                    if q <= 0: continue
                    tp_price = entry_price * (1 + pct) if side == "Buy" else entry_price * (1 - pct)
                    tp_price = self.round_val(tp_price, tick_size)
                    self.session.place_order(category="linear", symbol=symbol, side=exit_side, orderType="Limit", qty=str(q), price=str(tp_price), reduceOnly=True)

        except Exception as e:
            logger.error(f"TP Placement Error: {e}")

bot_instance = BybitTradingBot()