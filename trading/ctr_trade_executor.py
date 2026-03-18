"""
CTR Trade Executor v1.0 — Bybit Futures Trading

Автоматичне виконання угод на Bybit Linear Futures
по сигналам CTR Scanner.

Логіка:
1. Сигнал BUY/SELL → перевіряємо чи є відкрита позиція
2. Якщо є протилежна → закриваємо + відкриваємо нову
3. Якщо та ж сторона → ігноруємо
4. Якщо немає → відкриваємо нову

Безпека:
- Auto-Trade вимкнений за замовчуванням
- Тільки символи з міткою "trade" в watchlist
- Ліміт одночасних позицій (max 5)
- Position sizing на основі % від депозиту
- Isolated margin
"""

import time
import math
import json
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone


class CTRTradeExecutor:
    """Виконання торгів на Bybit Futures по CTR сигналам"""
    
    # DB settings keys
    SETTINGS_KEYS = {
        'enabled': 'ctr_trade_enabled',
        'leverage': 'ctr_trade_leverage',
        'deposit_pct': 'ctr_trade_deposit_pct',
        'sizing_mode': 'ctr_trade_sizing_mode',
        'fixed_margin': 'ctr_trade_fixed_margin',
        'tp_pct': 'ctr_trade_tp_pct',
        'sl_pct': 'ctr_trade_sl_pct',
        'max_positions': 'ctr_trade_max_positions',
        'trade_symbols': 'ctr_trade_symbols',  # Comma-separated: BTCUSDT,ETHUSDT
    }
    
    DEFAULTS = {
        'enabled': '0',
        'leverage': '10',
        'deposit_pct': '5',
        'sizing_mode': 'percent',
        'fixed_margin': '10',
        'tp_pct': '0',
        'sl_pct': '0',
        'max_positions': '5',
        'trade_symbols': '',
    }
    
    def __init__(self, db, bybit_connector):
        self.db = db
        self.bybit = bybit_connector
        self._instrument_cache = {}  # symbol -> {qtyStep, minQty, tickSize}
        self._cache_ttl = 3600  # 1 hour
        self._cache_time = {}
        
        # Status cache — prevents API spam on page refresh
        self._status_cache = None
        self._status_cache_time = 0
        self._status_cache_ttl = 60  # seconds
        
        # Auth error tracking — stops calling private API after 401
        self._auth_ok = None  # None = not tested yet
        self._auth_error_msg = ""
        
        # Test auth on init
        self._test_auth()
    
    def _test_auth(self):
        """Пряма перевірка API авторизації через session (обходить connector error handling)"""
        if not self.bybit.api_key:
            self._auth_ok = False
            self._auth_error_msg = "API keys not configured"
            print("[CTR Trade] ⚠️ Bybit API keys not configured")
            return
        
        try:
            # Call session directly to catch 401
            response = self.bybit.session.get_wallet_balance(accountType="UNIFIED")
            if response.get('retCode') == 0:
                result = response.get('result', {})
                balance = 0.0
                for acc in result.get('list', []):
                    for coin in acc.get('coin', []):
                        if coin['coin'] == 'USDT':
                            balance = float(coin.get('walletBalance', 0))
                self._auth_ok = True
                print(f"[CTR Trade] ✅ Bybit auth OK — Balance: ${balance:.2f} USDT")
            else:
                self._auth_ok = False
                self._auth_error_msg = f"API error: {response.get('retMsg', 'unknown')}"
                print(f"[CTR Trade] ❌ Bybit auth failed: {self._auth_error_msg}")
        except Exception as e:
            err_str = str(e)
            self._auth_ok = False
            if '401' in err_str or 'Unauthorized' in err_str:
                self._auth_error_msg = "Invalid API keys (401 Unauthorized)"
            elif '403' in err_str:
                self._auth_error_msg = "API key forbidden (403) — check IP whitelist"
            else:
                self._auth_error_msg = str(e)
            print(f"[CTR Trade] ❌ Bybit auth failed: {self._auth_error_msg}")
        
    # =============================================
    # SETTINGS
    # =============================================
    
    def get_settings(self) -> Dict:
        """Завантажити всі торговельні налаштування з БД"""
        settings = {}
        for key, db_key in self.SETTINGS_KEYS.items():
            settings[key] = self.db.get_setting(db_key, self.DEFAULTS[key])
        
        return {
            'enabled': settings['enabled'] in ('1', 'true', 'True'),
            'leverage': int(settings['leverage']),
            'deposit_pct': float(settings['deposit_pct']),
            'sizing_mode': settings['sizing_mode'],
            'fixed_margin': float(settings['fixed_margin']),
            'tp_pct': float(settings['tp_pct']),
            'sl_pct': float(settings['sl_pct']),
            'max_positions': int(settings['max_positions']),
            'trade_symbols': [s.strip() for s in settings['trade_symbols'].split(',') if s.strip()],
        }
    
    def save_settings(self, settings: Dict) -> bool:
        """Зберегти торговельні налаштування в БД"""
        try:
            if 'enabled' in settings:
                self.db.set_setting(self.SETTINGS_KEYS['enabled'],
                                    '1' if settings['enabled'] else '0')
            if 'leverage' in settings:
                lev = max(1, min(100, int(settings['leverage'])))
                self.db.set_setting(self.SETTINGS_KEYS['leverage'], str(lev))
            if 'deposit_pct' in settings:
                pct = max(0.1, min(50, float(settings['deposit_pct'])))
                self.db.set_setting(self.SETTINGS_KEYS['deposit_pct'], str(pct))
            if 'sizing_mode' in settings:
                mode = settings['sizing_mode'] if settings['sizing_mode'] in ('percent', 'fixed') else 'percent'
                self.db.set_setting(self.SETTINGS_KEYS['sizing_mode'], mode)
            if 'fixed_margin' in settings:
                fm = max(1, min(100000, float(settings['fixed_margin'])))
                self.db.set_setting(self.SETTINGS_KEYS['fixed_margin'], str(fm))
            if 'tp_pct' in settings:
                tp = max(0, min(100, float(settings['tp_pct'])))
                self.db.set_setting(self.SETTINGS_KEYS['tp_pct'], str(tp))
            if 'sl_pct' in settings:
                sl = max(0, min(50, float(settings['sl_pct'])))
                self.db.set_setting(self.SETTINGS_KEYS['sl_pct'], str(sl))
            if 'max_positions' in settings:
                mp = max(1, min(20, int(settings['max_positions'])))
                self.db.set_setting(self.SETTINGS_KEYS['max_positions'], str(mp))
            return True
        except Exception as e:
            print(f"[CTR Trade] Error saving settings: {e}")
            return False
    
    def get_trade_symbols(self) -> List[str]:
        """Отримати список символів з міткою для торгівлі"""
        raw = self.db.get_setting(self.SETTINGS_KEYS['trade_symbols'], '')
        return [s.strip() for s in raw.split(',') if s.strip()]
    
    def set_trade_symbol(self, symbol: str, enabled: bool) -> List[str]:
        """Додати/видалити символ з торгового списку"""
        symbols = self.get_trade_symbols()
        if enabled and symbol not in symbols:
            symbols.append(symbol)
        elif not enabled and symbol in symbols:
            symbols.remove(symbol)
        self.db.set_setting(self.SETTINGS_KEYS['trade_symbols'], ','.join(symbols))
        return symbols
    
    # =============================================
    # INSTRUMENT SPECS
    # =============================================
    
    def _get_instrument_specs(self, symbol: str) -> Optional[Dict]:
        """Отримати специфікацію інструменту (min qty, step size, tick size)"""
        now = time.time()
        
        # Check cache
        if symbol in self._instrument_cache:
            if (now - self._cache_time.get(symbol, 0)) < self._cache_ttl:
                return self._instrument_cache[symbol]
        
        try:
            info = self.bybit.get_instrument_info(symbol)
            if not info:
                print(f"[CTR Trade] ❌ Cannot get instrument info for {symbol}")
                return None
            
            lot_filter = info.get('lotSizeFilter', {})
            price_filter = info.get('priceFilter', {})
            
            specs = {
                'min_qty': float(lot_filter.get('minOrderQty', '0.001')),
                'max_qty': float(lot_filter.get('maxOrderQty', '1000000')),
                'qty_step': float(lot_filter.get('qtyStep', '0.001')),
                'min_notional': float(lot_filter.get('minNotionalValue', '5')),
                'tick_size': float(price_filter.get('tickSize', '0.01')),
            }
            
            # Calculate decimal precision from step
            qty_str = lot_filter.get('qtyStep', '0.001')
            if '.' in qty_str:
                specs['qty_decimals'] = len(qty_str.rstrip('0').split('.')[1])
            else:
                specs['qty_decimals'] = 0
            
            self._instrument_cache[symbol] = specs
            self._cache_time[symbol] = now
            return specs
            
        except Exception as e:
            print(f"[CTR Trade] Error getting instrument specs for {symbol}: {e}")
            return None
    
    def _round_qty(self, qty: float, specs: Dict) -> float:
        """Округлити qty до stepSize"""
        step = specs['qty_step']
        decimals = specs['qty_decimals']
        # Floor to step
        rounded = math.floor(qty / step) * step
        return round(rounded, decimals)
    
    # =============================================
    # BALANCE & POSITIONS
    # =============================================
    
    def get_balance(self) -> float:
        """Отримати баланс USDT"""
        if not self._auth_ok:
            return 0.0
        try:
            return self.bybit.get_wallet_balance()
        except Exception as e:
            print(f"[CTR Trade] Error getting balance: {e}")
            return 0.0
    
    def get_open_positions(self) -> List[Dict]:
        """Отримати всі відкриті позиції"""
        if not self._auth_ok:
            return []
        try:
            return self.bybit.get_positions()
        except Exception as e:
            print(f"[CTR Trade] Error getting positions: {e}")
            return []
    
    def get_position_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Отримати позицію для конкретного символу"""
        if not self._auth_ok:
            return None
        try:
            positions = self.bybit.get_positions(symbol=symbol)
            if positions:
                return positions[0]
            return None
        except Exception as e:
            print(f"[CTR Trade] Error getting position for {symbol}: {e}")
            return None
    
    def get_cached_status(self) -> Dict:
        """Отримати статус з кешем (не спамити API при рефреші)"""
        now = time.time()
        
        # Return cache if fresh
        if self._status_cache and (now - self._status_cache_time) < self._status_cache_ttl:
            return self._status_cache
        
        # Auth failed — return error without calling API
        if not self._auth_ok:
            self._status_cache = {
                'available': True,
                'auth_ok': False,
                'error': self._auth_error_msg,
                'enabled': self.get_settings()['enabled'],
                'balance': 0,
                'positions_count': 0,
                'max_positions': self.get_settings()['max_positions'],
                'positions': [],
                'trade_symbols': self.get_settings()['trade_symbols'],
            }
            self._status_cache_time = now
            return self._status_cache
        
        # Fetch fresh data
        try:
            settings = self.get_settings()
            positions = self.get_open_positions()
            balance = self.get_balance()
            
            self._status_cache = {
                'available': True,
                'auth_ok': self._auth_ok,
                'balance': balance,
                'positions_count': len(positions),
                'max_positions': settings['max_positions'],
                'positions': positions,
                'leverage': settings['leverage'],
                'deposit_pct': settings['deposit_pct'],
                'trade_symbols': settings['trade_symbols'],
                'enabled': settings['enabled'],
            }
            self._status_cache_time = now
            return self._status_cache
            
        except Exception as e:
            print(f"[CTR Trade] Status error: {e}")
            return {'available': True, 'auth_ok': False, 'error': str(e)}
    
    # =============================================
    # POSITION SIZING
    # =============================================
    
    def _calculate_position_size(self, symbol: str, price: float, settings: Dict) -> Optional[Dict]:
        """
        Розрахунок розміру позиції
        
        percent mode: deposit_pct% від балансу × leverage = notional
        fixed mode:   fixed_margin × leverage = notional
        notional / price = qty
        
        Returns: {qty, notional, margin, leverage} або None
        """
        balance = self.get_balance()
        if balance <= 0:
            print(f"[CTR Trade] ❌ Zero balance, cannot open position")
            return None
        
        specs = self._get_instrument_specs(symbol)
        if not specs:
            return None
        
        leverage = settings['leverage']
        sizing_mode = settings.get('sizing_mode', 'percent')
        
        # Calculate margin based on mode
        if sizing_mode == 'fixed':
            margin = settings.get('fixed_margin', 10)
            if margin > balance:
                print(f"[CTR Trade] ⚠️ Fixed margin ${margin:.2f} > balance ${balance:.2f}, using balance")
                margin = balance
        else:
            deposit_pct = settings['deposit_pct']
            margin = balance * (deposit_pct / 100)
        
        # Notional = margin * leverage
        notional = margin * leverage
        
        # Qty = notional / price
        raw_qty = notional / price
        qty = self._round_qty(raw_qty, specs)
        
        # Перевірки
        if qty < specs['min_qty']:
            print(f"[CTR Trade] ❌ {symbol}: qty {qty} < min {specs['min_qty']} "
                  f"(balance=${balance:.2f}, margin=${margin:.2f}, notional=${notional:.2f})")
            return None
        
        actual_notional = qty * price
        if actual_notional < specs['min_notional']:
            print(f"[CTR Trade] ❌ {symbol}: notional ${actual_notional:.2f} < min ${specs['min_notional']}")
            return None
        
        return {
            'qty': qty,
            'notional': actual_notional,
            'margin': margin,
            'leverage': leverage,
            'balance': balance,
        }
    
    # =============================================
    # TRADE EXECUTION
    # =============================================
    
    def _set_isolated_margin(self, symbol: str, leverage: int) -> bool:
        """Встановити Isolated margin mode + leverage"""
        try:
            # Switch to Isolated mode
            try:
                self.bybit.session.switch_margin_mode(
                    category="linear",
                    symbol=symbol,
                    tradeMode=1,  # 0=Cross, 1=Isolated
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage)
                )
            except Exception as e:
                err_str = str(e)
                # 110026 = "margin mode is not modified" (already isolated)
                if '110026' not in err_str:
                    print(f"[CTR Trade] Margin mode switch note: {e}")
            
            # Set leverage (idempotent)
            self.bybit.set_leverage(symbol, leverage)
            return True
            
        except Exception as e:
            print(f"[CTR Trade] Error setting margin/leverage for {symbol}: {e}")
            return False
    
    def _close_existing_position(self, symbol: str, position: Dict) -> bool:
        """Закрити існуючу позицію market ордером"""
        try:
            side = position['side']  # "Buy" or "Sell"
            size = position['size']
            
            print(f"[CTR Trade] 🔄 Closing {side} position {symbol}: qty={size}")
            
            result = self.bybit.close_position(symbol, side, size)
            
            if result:
                pnl = position.get('unrealized_pnl', 0)
                print(f"[CTR Trade] ✅ Closed {symbol} {side} position "
                      f"(PnL: ${pnl:+.4f}, OrderID: {result.get('order_id', '?')})")
                
                self._log_trade(symbol, 'CLOSE', side, size,
                                position.get('entry_price', 0), pnl, result.get('order_id', ''))
                return True
            else:
                print(f"[CTR Trade] ❌ Failed to close {symbol} {side} position")
                return False
                
        except Exception as e:
            print(f"[CTR Trade] Error closing position {symbol}: {e}")
            return False
    
    def _open_position(self, symbol: str, signal_type: str, price: float,
                       settings: Dict) -> Optional[Dict]:
        """Відкрити нову позицію"""
        try:
            # Side: BUY signal → Buy side (Long), SELL signal → Sell side (Short)
            side = "Buy" if signal_type == "BUY" else "Sell"
            
            # Calculate position size
            sizing = self._calculate_position_size(symbol, price, settings)
            if not sizing:
                return None
            
            # Set Isolated margin + leverage
            self._set_isolated_margin(symbol, sizing['leverage'])
            
            # TP/SL
            tp_price = None
            sl_price = None
            
            if settings['tp_pct'] > 0:
                if signal_type == "BUY":
                    tp_price = round(price * (1 + settings['tp_pct'] / 100), 8)
                else:
                    tp_price = round(price * (1 - settings['tp_pct'] / 100), 8)
            
            if settings['sl_pct'] > 0:
                if signal_type == "BUY":
                    sl_price = round(price * (1 - settings['sl_pct'] / 100), 8)
                else:
                    sl_price = round(price * (1 + settings['sl_pct'] / 100), 8)
            
            print(f"[CTR Trade] 📈 Opening {side} {symbol}: "
                  f"qty={sizing['qty']}, price≈${price:.4f}, "
                  f"notional=${sizing['notional']:.2f}, margin=${sizing['margin']:.2f}, "
                  f"leverage={sizing['leverage']}x"
                  f"{f', TP=${tp_price:.4f}' if tp_price else ''}"
                  f"{f', SL=${sl_price:.4f}' if sl_price else ''}")
            
            # Place Market order
            result = self.bybit.place_order(
                symbol=symbol,
                side=side,
                qty=sizing['qty'],
                order_type="Market",
                take_profit=tp_price,
                stop_loss=sl_price
            )
            
            if result:
                print(f"[CTR Trade] ✅ Opened {side} {symbol} "
                      f"(qty={sizing['qty']}, OrderID: {result.get('order_id', '?')})")
                
                self._log_trade(symbol, 'OPEN', side, sizing['qty'],
                                price, 0, result.get('order_id', ''),
                                leverage=sizing['leverage'],
                                margin=sizing['margin'])
                return result
            else:
                print(f"[CTR Trade] ❌ Failed to open {side} {symbol}")
                return None
                
        except Exception as e:
            print(f"[CTR Trade] Error opening position {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # =============================================
    # MAIN ENTRY POINT
    # =============================================
    
    def execute_signal(self, symbol: str, signal_type: str, price: float,
                       reason: str = "") -> Dict:
        """
        Головна функція: обробити CTR сигнал і виконати угоду
        
        Returns: {success, action, details}
        """
        result = {
            'success': False,
            'action': 'none',
            'symbol': symbol,
            'signal_type': signal_type,
            'details': ''
        }
        
        try:
            # 1. Load settings
            settings = self.get_settings()
            
            # 2. Check if trading enabled
            if not settings['enabled']:
                result['details'] = 'Auto-Trade disabled'
                return result
            
            # 2b. Check API auth
            if not self._auth_ok:
                result['details'] = f'API auth failed: {self._auth_error_msg}'
                print(f"[CTR Trade] ⚠️ {symbol}: skipped — {self._auth_error_msg}")
                return result
            
            # 3. Check if symbol marked for trading
            if symbol not in settings['trade_symbols']:
                result['details'] = f'{symbol} not marked for trading'
                return result
            
            # 4. Check existing position for this symbol
            position = self.get_position_for_symbol(symbol)
            
            if position:
                pos_side = position['side']  # "Buy" or "Sell"
                expected_side = "Buy" if signal_type == "BUY" else "Sell"
                
                if pos_side == expected_side:
                    # Same direction — skip
                    result['action'] = 'skip_same_direction'
                    result['details'] = f'Already in {pos_side} position'
                    print(f"[CTR Trade] ⏭️ {symbol}: already {pos_side}, skip {signal_type}")
                    return result
                else:
                    # Opposite direction — close old + open new
                    print(f"[CTR Trade] 🔄 {symbol}: reversing {pos_side} → {expected_side}")
                    
                    closed = self._close_existing_position(symbol, position)
                    if not closed:
                        result['details'] = f'Failed to close {pos_side} position'
                        return result
                    
                    result['action'] = 'reverse'
                    time.sleep(0.3)  # Small delay between close and open
            else:
                # No position — check max positions limit
                all_positions = self.get_open_positions()
                if len(all_positions) >= settings['max_positions']:
                    result['action'] = 'skip_max_positions'
                    result['details'] = f"Max positions ({settings['max_positions']}) reached"
                    print(f"[CTR Trade] ⚠️ {symbol}: max positions "
                          f"({len(all_positions)}/{settings['max_positions']}), skip")
                    return result
                
                result['action'] = 'open_new'
            
            # 5. Open new position
            order = self._open_position(symbol, signal_type, price, settings)
            
            if order:
                result['success'] = True
                if result['action'] == 'reverse':
                    result['details'] = f"Reversed to {signal_type}"
                else:
                    result['action'] = 'opened'
                    result['details'] = f"Opened {signal_type}"
                result['order_id'] = order.get('order_id', '')
            else:
                result['details'] = 'Failed to open position'
            
            return result
            
        except Exception as e:
            print(f"[CTR Trade] ❌ Execute error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            result['details'] = str(e)
            return result
    
    # =============================================
    # TRADE LOG
    # =============================================
    
    def _log_trade(self, symbol: str, action: str, side: str, qty: float,
                   price: float, pnl: float, order_id: str, **extra):
        """Зберегти запис про угоду в БД"""
        try:
            log_str = self.db.get_setting('ctr_trade_log', '[]')
            log = json.loads(log_str)
            
            entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'action': action,  # OPEN, CLOSE
                'side': side,
                'qty': qty,
                'price': price,
                'pnl': pnl,
                'order_id': order_id,
            }
            entry.update(extra)
            
            log.append(entry)
            # Keep last 200 entries
            log = log[-200:]
            
            self.db.set_setting('ctr_trade_log', json.dumps(log))
            
        except Exception as e:
            print(f"[CTR Trade] Error logging trade: {e}")
    
    def get_trade_log(self, limit: int = 50) -> List[Dict]:
        """Отримати історію торгів"""
        try:
            log_str = self.db.get_setting('ctr_trade_log', '[]')
            log = json.loads(log_str)
            return sorted(log, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
        except:
            return []
