"""
ZLT Bot Strategy v2.0 — Zero Lag Trend Pullback Entry (Professional)

PHILOSOPHY:
  Entry: M5 micro-precision (jeweler's entry on pullback)
  Target: H1/H4 macro movement (ride the impulse 2-8 hours)
  Exit: M15 structure break ONLY (stable intraday TF)
  
  M5 is a MICROSCOPE for entry — NEVER used for exit management.
  After entry, ONLY M15 controls the position.

STATE MACHINE:
  IDLE → SETUP → PULLBACK → ENTRY_READY → IN_TRADE → IDLE

  IDLE:         No macro alignment
  SETUP:        H4+H1 aligned, waiting for M15 pullback
  PULLBACK:     M15 pulled back, waiting for M5 confirmation  
  ENTRY_READY:  M5 confirmed pullback (M15+M5 against trend),
                waiting for M5 to reverse back = ENTRY
  IN_TRADE:     Position open, only M15 or macro break exits

EXIT RULES (in priority order):
  1. MACRO BREAK: H4 or H1 reverses → immediate EXIT (safety)
  2. M15 STRUCTURE BREAK: M15 aligned with trade → then reversed → EXIT
  3. Nothing else exits. M5 noise is ignored after entry.
"""

import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from enum import Enum


class BotState(Enum):
    IDLE = 'idle'
    SETUP = 'setup'
    PULLBACK = 'pullback'
    ENTRY_READY = 'entry_ready'
    IN_TRADE = 'in_trade'


class SymbolState:
    """Track state machine for one symbol."""
    __slots__ = [
        'state', 'direction', 'macro_since', 'pullback_since',
        'entry_price', 'entry_time', 'trade_count',
        'last_exit_time', 'm15_aligned',
    ]
    
    def __init__(self):
        self.state: BotState = BotState.IDLE
        self.direction: str = ''
        self.macro_since: float = 0
        self.pullback_since: float = 0
        self.entry_price: float = 0
        self.entry_time: float = 0
        self.trade_count: int = 0
        self.last_exit_time: float = 0
        self.m15_aligned: bool = False
    
    def reset(self):
        self.state = BotState.IDLE
        self.direction = ''
        self.macro_since = 0
        self.pullback_since = 0
        self.entry_price = 0
        self.entry_time = 0
        self.m15_aligned = False
    
    def to_dict(self) -> Dict:
        return {
            'state': self.state.value,
            'direction': self.direction,
            'macro_since': self.macro_since,
            'pullback_since': self.pullback_since,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'trade_count': self.trade_count,
            'last_exit_time': self.last_exit_time,
            'm15_aligned': self.m15_aligned,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SymbolState':
        s = cls()
        try:
            s.state = BotState(data.get('state', 'idle'))
        except ValueError:
            s.state = BotState.IDLE
        s.direction = data.get('direction', '')
        s.macro_since = data.get('macro_since', 0)
        s.pullback_since = data.get('pullback_since', 0)
        s.entry_price = data.get('entry_price', 0)
        s.entry_time = data.get('entry_time', 0)
        s.trade_count = data.get('trade_count', 0)
        s.last_exit_time = data.get('last_exit_time', 0)
        s.m15_aligned = data.get('m15_aligned', False)
        return s


class ZLTBot:
    """
    Zero Lag Trend Bot v2.0 — Professional Pullback Strategy.
    
    M5 = entry only. M15 = exit only. H4+H1 = direction + safety.
    No partial/reload. Trades last 2-8 hours.
    """
    
    def __init__(
        self,
        zl_service,
        enabled: bool = False,
        exit_cooldown_sec: int = 1800,
        on_trade: Optional[Callable] = None,
        on_notify: Optional[Callable] = None,
        get_price: Optional[Callable] = None,
        on_save: Optional[Callable] = None,
        # Legacy params (accepted but ignored for backward compat)
        partial_close_pct: float = 50.0,
        min_trade_sec: int = 1800,
        partial_cooldown_sec: int = 300,
    ):
        self.zl_service = zl_service
        self.enabled = enabled
        self.exit_cooldown_sec = exit_cooldown_sec
        self.on_trade = on_trade
        self.on_notify = on_notify
        self.get_price = get_price
        self.on_save = on_save
        
        self._states: Dict[str, SymbolState] = {}
        self._lock = threading.RLock()
        self._stats = {'entries': 0, 'exits': 0, 'macro_formed': 0}
        
        if self.zl_service:
            self.zl_service.set_on_transition(self._on_trend_change)
    
    # ========================================
    # CORE: TREND TRANSITION HANDLER
    # ========================================
    
    def _on_trend_change(self, symbol: str, tf_key: str, old_trend: str, new_trend: str):
        if not self.enabled or symbol not in self._states:
            return
        self._process(symbol)
    
    def _process(self, symbol: str):
        with self._lock:
            s = self._states.get(symbol)
            if not s:
                return
            
            trends = self.zl_service.get_all_trends(symbol) if self.zl_service else {}
            h4 = trends.get('240', 'neutral')
            h1 = trends.get('60', 'neutral')
            m15 = trends.get('15', 'neutral')
            m5 = trends.get('5', 'neutral')
            
            # === SAFETY: Macro break ===
            if s.state == BotState.IN_TRADE:
                if self._macro_broken(s.direction, h4, h1):
                    self._do_exit(symbol, s, f"Macro broken (H4={h4}, H1={h1})")
                    return
            
            if s.state in (BotState.SETUP, BotState.PULLBACK, BotState.ENTRY_READY):
                if self._macro_broken(s.direction, h4, h1):
                    self._transition(symbol, s, BotState.IDLE, "Macro broken")
                    s.reset()
                    return
            
            # === STATE MACHINE ===
            if s.state == BotState.IDLE:
                self._handle_idle(symbol, s, h4, h1, m15, m5)
            elif s.state == BotState.SETUP:
                self._handle_setup(symbol, s, m15, m5)
            elif s.state == BotState.PULLBACK:
                self._handle_pullback(symbol, s, m15, m5)
            elif s.state == BotState.ENTRY_READY:
                self._handle_entry_ready(symbol, s, m15, m5)
            elif s.state == BotState.IN_TRADE:
                self._handle_in_trade(symbol, s, m15)
    
    # ========================================
    # STATE HANDLERS
    # ========================================
    
    def _handle_idle(self, symbol: str, s: SymbolState, h4: str, h1: str, m15: str, m5: str):
        """IDLE → detect macro alignment."""
        if s.last_exit_time > 0:
            if time.time() - s.last_exit_time < self.exit_cooldown_sec:
                return
        
        direction = ''
        if h4 == 'bullish' and h1 == 'bullish':
            direction = 'LONG'
        elif h4 == 'bearish' and h1 == 'bearish':
            direction = 'SHORT'
        
        if not direction:
            return
        
        s.direction = direction
        s.macro_since = time.time()
        self._stats['macro_formed'] += 1
        
        pullback_dir = 'bearish' if direction == 'LONG' else 'bullish'
        
        if m15 == pullback_dir:
            s.pullback_since = time.time()
            if m5 == pullback_dir:
                self._transition(symbol, s, BotState.ENTRY_READY,
                                 f"Macro {direction} + pullback ready")
            else:
                self._transition(symbol, s, BotState.PULLBACK,
                                 f"Macro {direction} + M15 pullback started")
        else:
            self._transition(symbol, s, BotState.SETUP,
                             f"Macro {direction} — waiting for pullback")
    
    def _handle_setup(self, symbol: str, s: SymbolState, m15: str, m5: str):
        """SETUP → wait for M15 pullback to start."""
        pullback_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        
        if m15 == pullback_dir:
            s.pullback_since = time.time()
            if m5 == pullback_dir:
                self._transition(symbol, s, BotState.ENTRY_READY,
                                 f"Pullback deep (M15+M5 {pullback_dir})")
            else:
                self._transition(symbol, s, BotState.PULLBACK,
                                 f"M15 pullback started ({pullback_dir})")
    
    def _handle_pullback(self, symbol: str, s: SymbolState, m15: str, m5: str):
        """PULLBACK → wait for M5 to also pull back."""
        pullback_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        
        if m5 == pullback_dir:
            self._transition(symbol, s, BotState.ENTRY_READY,
                             f"M5 pullback confirmed")
        elif m15 == entry_dir:
            self._transition(symbol, s, BotState.SETUP,
                             "M15 returned — false pullback")
    
    def _handle_entry_ready(self, symbol: str, s: SymbolState, m15: str, m5: str):
        """ENTRY_READY → M5 flips back to trend direction = ENTRY."""
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        
        if m5 == entry_dir:
            self._do_entry(symbol, s)
        elif m15 == entry_dir:
            self._transition(symbol, s, BotState.SETUP,
                             "M15 returned — pullback ended without M5 entry")
    
    def _handle_in_trade(self, symbol: str, s: SymbolState, m15: str):
        """IN_TRADE → ONLY M15 manages exit. M5 completely ignored.
        
        Ride the H1/H4 impulse for 2-8 hours.
        Exit ONLY when M15 first aligned, then broke structure.
        If M15 never aligned → hold until macro breaks.
        """
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        exit_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        
        # Track M15 alignment (locked once set)
        if m15 == entry_dir and not s.m15_aligned:
            s.m15_aligned = True
            self._save_state()
            print(f"[ZLT Bot] ✅ {symbol}: M15 aligned — exit armed")
        
        # EXIT: M15 aligned → then reversed
        if m15 == exit_dir and s.m15_aligned:
            dur = time.time() - s.entry_time if s.entry_time else 0
            self._do_exit(symbol, s,
                          f"M15 structure break ({int(dur/3600)}h{int((dur%3600)/60)}m)")
    
    # ========================================
    # TRADE ACTIONS
    # ========================================
    
    def _do_entry(self, symbol: str, s: SymbolState):
        signal_type = 'BUY' if s.direction == 'LONG' else 'SELL'
        price = self._get_price(symbol)
        
        trends = self.zl_service.get_all_trends(symbol) if self.zl_service else {}
        m15_now = trends.get('15', 'neutral')
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        
        s.state = BotState.IN_TRADE
        s.entry_price = price
        s.entry_time = time.time()
        s.m15_aligned = (m15_now == entry_dir)
        s.trade_count += 1
        self._stats['entries'] += 1
        
        label = '🟢 LONG' if s.direction == 'LONG' else '🔴 SHORT'
        price_str = f"${price:,.4f}" if price else "market"
        aligned = "M15 aligned ✅" if s.m15_aligned else "M15 pending ⏳"
        
        msg = (f"⚡ ZLT Bot ENTRY | {symbol} {label}\n"
               f"Price: {price_str}\n"
               f"{aligned} | Trade #{s.trade_count}\n"
               f"Target: H1 impulse (2-8h hold)")
        
        print(f"[ZLT Bot] ⚡ ENTRY: {symbol} {signal_type} @ {price_str} ({aligned})")
        self._notify(msg)
        
        if self.on_trade:
            self.on_trade(symbol, 'entry', {
                'signal_type': signal_type,
                'direction': s.direction,
                'price': price,
                'reason': 'ZLT Bot Pullback Entry',
            })
        self._save_state()
    
    def _do_exit(self, symbol: str, s: SymbolState, reason: str):
        price = self._get_price(symbol)
        self._stats['exits'] += 1
        
        pnl_str = ""
        if s.entry_price and price:
            if s.direction == 'LONG':
                pnl_pct = (price - s.entry_price) / s.entry_price * 100
            else:
                pnl_pct = (s.entry_price - price) / s.entry_price * 100
            pnl_str = f"\nP&L: {'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
        
        duration = ""
        if s.entry_time:
            dur = time.time() - s.entry_time
            duration = f"\nDuration: {int(dur/3600)}h {int((dur%3600)/60)}m"
        
        price_str = f"${price:,.4f}" if price else "market"
        entry_str = f"${s.entry_price:,.4f}" if s.entry_price else "?"
        
        msg = (f"❌ ZLT Bot EXIT | {symbol}\n"
               f"Price: {price_str} (entry: {entry_str}){pnl_str}{duration}\n"
               f"{reason}")
        
        print(f"[ZLT Bot] ❌ EXIT: {symbol} @ {price_str} ({reason})")
        self._notify(msg)
        
        if self.on_trade:
            self.on_trade(symbol, 'full_exit', {
                'direction': s.direction,
                'price': price,
                'entry_price': s.entry_price,
                'reason': f'ZLT Bot Exit: {reason}',
                'was_partial': False,
            })
        
        s.last_exit_time = time.time()
        s.reset()
        self._save_state()
    
    # ========================================
    # HELPERS
    # ========================================
    
    def _get_price(self, symbol: str) -> float:
        if self.get_price:
            try:
                return self.get_price(symbol)
            except:
                pass
        return 0.0
    
    def _macro_broken(self, direction: str, h4: str, h1: str) -> bool:
        if direction == 'LONG':
            return h4 != 'bullish' or h1 != 'bullish'
        elif direction == 'SHORT':
            return h4 != 'bearish' or h1 != 'bearish'
        return False
    
    def _transition(self, symbol: str, s: SymbolState, new_state: BotState, reason: str):
        old = s.state.value
        s.state = new_state
        print(f"[ZLT Bot] {symbol}: {old} → {new_state.value} ({reason})")
        self._save_state()
    
    def _notify(self, msg: str):
        if self.on_notify:
            try:
                self.on_notify(msg)
            except Exception as e:
                print(f"[ZLT Bot] Notify error: {e}")
    
    # ========================================
    # PERIODIC SAFETY CHECK
    # ========================================
    
    def check_all(self):
        if not self.enabled or not self.zl_service:
            return
        with self._lock:
            for symbol, s in self._states.items():
                if s.state == BotState.IDLE:
                    continue
                trends = self.zl_service.get_all_trends(symbol)
                h4 = trends.get('240', 'neutral')
                h1 = trends.get('60', 'neutral')
                m15 = trends.get('15', 'neutral')
                m5 = trends.get('5', 'neutral')
                
                if s.state == BotState.IN_TRADE:
                    if self._macro_broken(s.direction, h4, h1):
                        self._do_exit(symbol, s, "Macro broken (periodic)")
                        continue
                    self._handle_in_trade(symbol, s, m15)
                elif s.state in (BotState.SETUP, BotState.PULLBACK, BotState.ENTRY_READY):
                    if self._macro_broken(s.direction, h4, h1):
                        self._transition(symbol, s, BotState.IDLE, "Macro broken (periodic)")
                        s.reset()
                        continue
                    if s.state == BotState.SETUP:
                        self._handle_setup(symbol, s, m15, m5)
                    elif s.state == BotState.PULLBACK:
                        self._handle_pullback(symbol, s, m15, m5)
                    elif s.state == BotState.ENTRY_READY:
                        self._handle_entry_ready(symbol, s, m15, m5)
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def set_watchlist(self, watchlist: List[str]):
        with self._lock:
            for sym in watchlist:
                if sym not in self._states:
                    self._states[sym] = SymbolState()
            for sym in list(self._states.keys()):
                if sym not in watchlist:
                    self._states.pop(sym, None)
    
    def get_states(self) -> Dict:
        with self._lock:
            return {sym: s.to_dict() for sym, s in self._states.items()}
    
    def get_stats(self) -> Dict:
        with self._lock:
            active = sum(1 for s in self._states.values() if s.state != BotState.IDLE)
            in_trade = sum(1 for s in self._states.values() if s.state == BotState.IN_TRADE)
            return {
                'enabled': self.enabled,
                'active_states': active,
                'in_trade': in_trade,
                'symbols': len(self._states),
                'states': self.get_states(),
                **self._stats,
            }
    
    def reset_symbol(self, symbol: str):
        with self._lock:
            if symbol in self._states:
                self._states[symbol] = SymbolState()
                self._save_state()
    
    def reset_all(self):
        with self._lock:
            for sym in self._states:
                self._states[sym] = SymbolState()
            self._save_state()
    
    # ========================================
    # PERSISTENCE
    # ========================================
    
    def _save_state(self):
        if not self.on_save:
            return
        try:
            data = {}
            for sym, s in self._states.items():
                d = s.to_dict()
                if d['state'] != 'idle' or d.get('last_exit_time', 0) > 0:
                    data[sym] = d
            self.on_save(data)
        except Exception as e:
            print(f"[ZLT Bot] ⚠️ Save error: {e}")
    
    def restore_states(self, saved_data: Dict):
        if not saved_data:
            return 0
        restored = 0
        with self._lock:
            for sym, data in saved_data.items():
                if sym not in self._states:
                    self._states[sym] = SymbolState()
                try:
                    self._states[sym] = SymbolState.from_dict(data)
                    restored += 1
                except Exception as e:
                    print(f"[ZLT Bot] ⚠️ Restore error {sym}: {e}")
        
        in_trade = sum(1 for s in self._states.values() if s.state == BotState.IN_TRADE)
        if restored:
            print(f"[ZLT Bot] 🔄 Restored {restored} states ({in_trade} in trade)")
        return restored
