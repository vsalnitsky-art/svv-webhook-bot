"""
Bybit Connector - використовує офіційну бібліотеку pybit
"""

import os
import time
from typing import Optional, Dict, Any, List
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

from config.bot_settings import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_CONFIG


class BybitConnector:
    """Клієнт для роботи з Bybit API через pybit"""
    
    def __init__(self):
        self.api_key = BYBIT_API_KEY
        self.api_secret = BYBIT_API_SECRET
        self.testnet = BYBIT_CONFIG.get('testnet', False)
        self.recv_window = BYBIT_CONFIG.get('recv_window', 20000)
        
        # Створити сесію
        self.session = self._create_session()
    
    def _create_session(self) -> HTTP:
        """Створити HTTP сесію pybit"""
        if self.api_key and self.api_secret:
            return HTTP(
                testnet=self.testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
                recv_window=self.recv_window
            )
        else:
            # Public API only (без ключів)
            return HTTP(testnet=self.testnet)
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """Безпечне перетворення в float"""
        try:
            return float(value) if value else default
        except (ValueError, TypeError):
            return default
    
    def _handle_response(self, response: Dict) -> Optional[Dict]:
        """Обробка відповіді API"""
        if response.get('retCode') == 0:
            return response.get('result')
        else:
            print(f"[BYBIT ERROR] Code: {response.get('retCode')}, Msg: {response.get('retMsg')}")
            return None
    
    # ===== PUBLIC API =====
    
    def get_tickers(self, category: str = "linear") -> List[Dict]:
        """Отримати всі тікери"""
        try:
            response = self.session.get_tickers(category=category)
            result = self._handle_response(response)
            return result.get('list', []) if result else []
        except Exception as e:
            print(f"[BYBIT] get_tickers error: {e}")
            return []
    
    def get_ticker(self, symbol: str, category: str = "linear") -> Optional[Dict]:
        """Отримати тікер для одного символу"""
        try:
            response = self.session.get_tickers(category=category, symbol=symbol)
            result = self._handle_response(response)
            if result and result.get('list'):
                return result['list'][0]
            return None
        except Exception as e:
            print(f"[BYBIT] get_ticker error: {e}")
            return None
    
    def get_price(self, symbol: str) -> float:
        """Отримати поточну ціну"""
        ticker = self.get_ticker(symbol)
        if ticker:
            return self._safe_float(ticker.get('lastPrice'))
        return 0.0
    
    def get_klines(self, symbol: str, interval: str = "60", limit: int = 200) -> List[Dict]:
        """
        Отримати свічки (klines)
        interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        """
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            result = self._handle_response(response)
            if result and result.get('list'):
                # Bybit повертає у зворотному порядку, конвертуємо
                klines = []
                for k in reversed(result['list']):
                    klines.append({
                        'timestamp': int(k[0]),
                        'open': self._safe_float(k[1]),
                        'high': self._safe_float(k[2]),
                        'low': self._safe_float(k[3]),
                        'close': self._safe_float(k[4]),
                        'volume': self._safe_float(k[5]),
                        'turnover': self._safe_float(k[6]) if len(k) > 6 else 0
                    })
                return klines
            return []
        except Exception as e:
            print(f"[BYBIT] get_klines error: {e}")
            return []
    
    def get_orderbook(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """Отримати orderbook"""
        try:
            response = self.session.get_orderbook(
                category="linear",
                symbol=symbol,
                limit=limit
            )
            result = self._handle_response(response)
            return result
        except Exception as e:
            print(f"[BYBIT] get_orderbook error: {e}")
            return None
    
    def get_instrument_info(self, symbol: str) -> Optional[Dict]:
        """Отримати інформацію про інструмент"""
        try:
            response = self.session.get_instruments_info(
                category="linear",
                symbol=symbol
            )
            result = self._handle_response(response)
            if result and result.get('list'):
                return result['list'][0]
            return None
        except Exception as e:
            print(f"[BYBIT] get_instrument_info error: {e}")
            return None
    
    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Отримати funding rate"""
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=symbol
            )
            result = self._handle_response(response)
            if result and result.get('list'):
                ticker = result['list'][0]
                return {
                    'symbol': symbol,
                    'funding_rate': self._safe_float(ticker.get('fundingRate')),
                    'next_funding_time': ticker.get('nextFundingTime')
                }
            return None
        except Exception as e:
            print(f"[BYBIT] get_funding_rate error: {e}")
            return None
    
    def get_open_interest(self, symbol: str, interval: str = "1h", limit: int = 50) -> List[Dict]:
        """Отримати історію open interest"""
        try:
            response = self.session.get_open_interest(
                category="linear",
                symbol=symbol,
                intervalTime=interval,
                limit=limit
            )
            result = self._handle_response(response)
            if result and result.get('list'):
                return [
                    {
                        'timestamp': int(item['timestamp']),
                        'open_interest': self._safe_float(item['openInterest'])
                    }
                    for item in result['list']
                ]
            return []
        except Exception as e:
            print(f"[BYBIT] get_open_interest error: {e}")
            return []
    
    # ===== PRIVATE API =====
    
    def get_wallet_balance(self) -> float:
        """Отримати баланс USDT"""
        if not self.api_key:
            return 0.0
        
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED")
            result = self._handle_response(response)
            if result and result.get('list'):
                for acc in result['list']:
                    for coin in acc.get('coin', []):
                        if coin['coin'] == 'USDT':
                            return self._safe_float(coin.get('walletBalance'))
            return 0.0
        except Exception as e:
            print(f"[BYBIT] get_wallet_balance error: {e}")
            return 0.0
    
    def get_positions(self, symbol: str = None) -> List[Dict]:
        """Отримати відкриті позиції"""
        if not self.api_key:
            return []
        
        try:
            params = {"category": "linear", "settleCoin": "USDT"}
            if symbol:
                params["symbol"] = symbol
            
            response = self.session.get_positions(**params)
            result = self._handle_response(response)
            if result and result.get('list'):
                positions = []
                for pos in result['list']:
                    size = self._safe_float(pos.get('size'))
                    if size > 0:
                        positions.append({
                            'symbol': pos.get('symbol'),
                            'side': pos.get('side'),
                            'size': size,
                            'entry_price': self._safe_float(pos.get('avgPrice')),
                            'mark_price': self._safe_float(pos.get('markPrice')),
                            'unrealized_pnl': self._safe_float(pos.get('unrealisedPnl')),
                            'leverage': self._safe_float(pos.get('leverage')),
                            'position_value': self._safe_float(pos.get('positionValue')),
                            'liq_price': self._safe_float(pos.get('liqPrice')),
                            'take_profit': self._safe_float(pos.get('takeProfit')),
                            'stop_loss': self._safe_float(pos.get('stopLoss'))
                        })
                return positions
            return []
        except Exception as e:
            print(f"[BYBIT] get_positions error: {e}")
            return []
    
    def place_order(
        self,
        symbol: str,
        side: str,  # "Buy" or "Sell"
        qty: float,
        order_type: str = "Market",
        price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """Розмістити ордер"""
        if not self.api_key:
            print("[BYBIT] No API key - cannot place order")
            return None
        
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
                "reduceOnly": reduce_only
            }
            
            if order_type == "Limit" and price:
                params["price"] = str(price)
            
            if stop_loss:
                params["stopLoss"] = str(stop_loss)
            
            if take_profit:
                params["takeProfit"] = str(take_profit)
            
            response = self.session.place_order(**params)
            result = self._handle_response(response)
            
            if result:
                return {
                    'order_id': result.get('orderId'),
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                    'order_type': order_type
                }
            return None
        except InvalidRequestError as e:
            print(f"[BYBIT] Invalid request: {e}")
            return None
        except Exception as e:
            print(f"[BYBIT] place_order error: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Скасувати ордер"""
        if not self.api_key:
            return False
        
        try:
            response = self.session.cancel_order(
                category="linear",
                symbol=symbol,
                orderId=order_id
            )
            return response.get('retCode') == 0
        except Exception as e:
            print(f"[BYBIT] cancel_order error: {e}")
            return False
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Встановити leverage"""
        if not self.api_key:
            return False
        
        try:
            response = self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            # 110043 = leverage not modified (вже такий)
            return response.get('retCode') in [0, 110043]
        except Exception as e:
            print(f"[BYBIT] set_leverage error: {e}")
            return False
    
    def set_trading_stop(
        self,
        symbol: str,
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop: float = None,
        position_idx: int = 0
    ) -> bool:
        """Встановити SL/TP/Trailing"""
        if not self.api_key:
            return False
        
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "positionIdx": position_idx
            }
            
            if stop_loss:
                params["stopLoss"] = str(stop_loss)
            if take_profit:
                params["takeProfit"] = str(take_profit)
            if trailing_stop:
                params["trailingStop"] = str(trailing_stop)
            
            response = self.session.set_trading_stop(**params)
            return response.get('retCode') == 0
        except Exception as e:
            print(f"[BYBIT] set_trading_stop error: {e}")
            return False
    
    def close_position(self, symbol: str, side: str, qty: float) -> Optional[Dict]:
        """Закрити позицію"""
        # Для закриття позиції потрібно виставити протилежний ордер
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.place_order(
            symbol=symbol,
            side=close_side,
            qty=qty,
            order_type="Market",
            reduce_only=True
        )


# ===== Singleton =====
_connector: Optional[BybitConnector] = None

def get_connector() -> BybitConnector:
    """Отримати singleton instance"""
    global _connector
    if _connector is None:
        _connector = BybitConnector()
    return _connector
