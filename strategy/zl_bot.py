"""
ZLT Bot Strategy v1.0 — Zero Lag Trend Pullback Entry

State machine for professional intraday pullback trading.
Uses ZeroLagTrendService for multi-timeframe trend detection.

Strategy:
  LONG Entry:
    1. Macro: H4=BULL + H1=BULL
    2. Wait pullback: M15→BEAR, then M5→BEAR
    3. Entry: M5→BULL (pullback complete, trend continues)
    4. Exit partial: M5→BEAR (close partial_pct%)
    5. Re-entry: M5→BULL again (if M15 still LONG → reload)
    6. Exit full: M15→BEAR or H4/H1→BEAR (close 100%)

  SHORT: mirror of LONG

Architecture:
  - Receives trend transitions from ZLT Service via callback
  - Manages state per symbol independently
  - Delegates trade execution to external executor
"""

import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from enum import Enum


class BotState(Enum):
    IDLE = 'idle'
    MACRO_OK = 'macro_ok'
    WAITING_PULLBACK = 'waiting_pullback'
    PULLBACK_ACTIVE = 'pullback_active'
    IN_TRADE = 'in_trade'
    IN_TRADE_PARTIAL = 'in_trade_partial'


class SymbolState:
    """Track state machine for one symbol."""
    __slots__ = [
        'state', 'direction', 'macro_since', 'pullback_since',
        'entry_price', 'entry_time', 'partial_done', 'trade_count',
        'last_exit_time', 'entry_m15', 'm15_aligned', 'last_partial_time',
    ]
    
    def __init__(self):
        self.state: BotState = BotState.IDLE
        self.direction: str = ''
        self.macro_since: float = 0
        self.pullback_since: float = 0
        self.entry_price: float = 0
        self.entry_time: float = 0
        self.partial_done: bool = False
        self.trade_count: int = 0
        self.last_exit_time: float = 0
        self.entry_m15: str = ''       # M15 trend at entry time
        self.m15_aligned: bool = False  # True once M15 aligned with trade after entry
        self.last_partial_time: float = 0  # For partial/reload cooldown
    
    def reset(self):
        self.state = BotState.IDLE
        self.direction = ''
        self.macro_since = 0
        self.pullback_since = 0
        self.entry_price = 0
        self.entry_time = 0
        self.partial_done = False
        self.entry_m15 = ''
        self.m15_aligned = False
        self.last_partial_time = 0
        # last_exit_time NOT reset — needed for cooldown
    
    def to_dict(self) -> Dict:
        return {
            'state': self.state.value,
            'direction': self.direction,
            'macro_since': self.macro_since,
            'pullback_since': self.pullback_since,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'partial_done': self.partial_done,
            'trade_count': self.trade_count,
            'last_exit_time': self.last_exit_time,
            'entry_m15': self.entry_m15,
            'm15_aligned': self.m15_aligned,
        }


class ZLTBot:
    """
    Zero Lag Trend Bot — Pullback Entry Strategy.
    
    Monitors ZLT Service trends and manages entry/exit state machine
    for each symbol in the watchlist.
    """
    
    def __init__(
        self,
        zl_service,
        enabled: bool = False,
        partial_close_pct: float = 50.0,
        exit_cooldown_sec: int = 900,  # 15 min cooldown after full exit
        min_trade_sec: int = 300,      # 5 min minimum trade duration before any exit
        partial_cooldown_sec: int = 300,  # 5 min cooldown between partial/reload
        on_trade: Optional[Callable] = None,
        on_notify: Optional[Callable] = None,
        get_price: Optional[Callable] = None,
    ):
        self.zl_service = zl_service
        self.enabled = enabled
        self.partial_close_pct = partial_close_pct
        self.exit_cooldown_sec = exit_cooldown_sec
        self.min_trade_sec = min_trade_sec
        self.partial_cooldown_sec = partial_cooldown_sec
        self.on_trade = on_trade
        self.on_notify = on_notify
        self.get_price = get_price
        
        # State per symbol
        self._states: Dict[str, SymbolState] = {}
        self._lock = threading.RLock()
        
        # Stats
        self._stats = {
            'entries': 0,
            'partial_exits': 0,
            'full_exits': 0,
            'reloads': 0,
            'macro_formed': 0,
        }
        
        # Register transition callback on ZLT Service
        if self.zl_service:
            self.zl_service.set_on_transition(self._on_trend_change)
    
    # ========================================
    # CORE: TREND TRANSITION HANDLER
    # ========================================
    
    def _on_trend_change(self, symbol: str, tf_key: str, old_trend: str, new_trend: str):
        """Called by ZLT Service when any TF changes trend."""
        if not self.enabled:
            return
        if symbol not in self._states:
            return
        
        self._process_transition(symbol, tf_key, old_trend, new_trend)
    
    def _process_transition(self, symbol: str, tf_key: str, old_trend: str, new_trend: str):
        """Main state machine logic."""
        with self._lock:
            s = self._states.get(symbol)
            if not s:
                return
            
            trends = self.zl_service.get_all_trends(symbol)
            h4 = trends.get('240', 'neutral')
            h1 = trends.get('60', 'neutral')
            m15 = trends.get('15', 'neutral')
            m5 = trends.get('5', 'neutral')
            
            old_state = s.state
            
            # ========================================
            # GLOBAL SAFETY: Macro break → exit everything
            # ========================================
            if s.state in (BotState.IN_TRADE, BotState.IN_TRADE_PARTIAL):
                if self._macro_broken(s.direction, h4, h1):
                    self._do_full_exit(symbol, s, f"Macro broken (H4={h4}, H1={h1})")
                    return
            
            # If macro breaks during any waiting state → back to IDLE
            if s.state in (BotState.MACRO_OK, BotState.WAITING_PULLBACK, BotState.PULLBACK_ACTIVE):
                if self._macro_broken(s.direction, h4, h1):
                    self._transition(symbol, s, BotState.IDLE, "Macro broken")
                    s.reset()
                    return
            
            # ========================================
            # STATE MACHINE
            # ========================================
            
            if s.state == BotState.IDLE:
                self._handle_idle(symbol, s, h4, h1, m15, m5)
            
            elif s.state == BotState.MACRO_OK:
                self._handle_macro_ok(symbol, s, m15, m5)
            
            elif s.state == BotState.WAITING_PULLBACK:
                self._handle_waiting_pullback(symbol, s, m15, m5)
            
            elif s.state == BotState.PULLBACK_ACTIVE:
                self._handle_pullback_active(symbol, s, m15, m5)
            
            elif s.state == BotState.IN_TRADE:
                self._handle_in_trade(symbol, s, tf_key, m15, m5)
            
            elif s.state == BotState.IN_TRADE_PARTIAL:
                self._handle_in_trade_partial(symbol, s, tf_key, m15, m5)
    
    # ========================================
    # STATE HANDLERS
    # ========================================
    
    def _handle_idle(self, symbol: str, s: SymbolState, h4: str, h1: str, m15: str, m5: str):
        """IDLE → check if macro forms."""
        # Cooldown after exit — prevent rapid re-entry (ping-pong)
        if s.last_exit_time > 0:
            elapsed = time.time() - s.last_exit_time
            if elapsed < self.exit_cooldown_sec:
                return  # Still in cooldown, skip
        
        # LONG macro
        if h4 == 'bullish' and h1 == 'bullish':
            s.direction = 'LONG'
            s.macro_since = time.time()
            self._stats['macro_formed'] += 1
            
            # Check if already in pullback or need to wait
            if m15 == 'bearish':
                if m5 == 'bearish':
                    self._transition(symbol, s, BotState.PULLBACK_ACTIVE, "Macro LONG + pullback active")
                else:
                    self._transition(symbol, s, BotState.WAITING_PULLBACK, "Macro LONG + M15 pullback started")
            elif m15 == 'bullish' and m5 == 'bullish':
                # All aligned already — overextended, wait for pullback
                self._transition(symbol, s, BotState.MACRO_OK, "Macro LONG — waiting for pullback")
            else:
                self._transition(symbol, s, BotState.MACRO_OK, "Macro LONG formed")
        
        # SHORT macro
        elif h4 == 'bearish' and h1 == 'bearish':
            s.direction = 'SHORT'
            s.macro_since = time.time()
            self._stats['macro_formed'] += 1
            
            if m15 == 'bullish':
                if m5 == 'bullish':
                    self._transition(symbol, s, BotState.PULLBACK_ACTIVE, "Macro SHORT + pullback active")
                else:
                    self._transition(symbol, s, BotState.WAITING_PULLBACK, "Macro SHORT + M15 pullback started")
            elif m15 == 'bearish' and m5 == 'bearish':
                self._transition(symbol, s, BotState.MACRO_OK, "Macro SHORT — waiting for pullback")
            else:
                self._transition(symbol, s, BotState.MACRO_OK, "Macro SHORT formed")
    
    def _handle_macro_ok(self, symbol: str, s: SymbolState, m15: str, m5: str):
        """MACRO_OK → wait for pullback to start (M15 reverses)."""
        pullback_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        
        if m15 == pullback_dir:
            s.pullback_since = time.time()
            if m5 == pullback_dir:
                self._transition(symbol, s, BotState.PULLBACK_ACTIVE, f"Pullback active (M15+M5={pullback_dir})")
            else:
                self._transition(symbol, s, BotState.WAITING_PULLBACK, f"M15 pullback started ({pullback_dir})")
    
    def _handle_waiting_pullback(self, symbol: str, s: SymbolState, m15: str, m5: str):
        """WAITING_PULLBACK → wait for M5 to also reverse (deeper pullback)."""
        pullback_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        
        if m5 == pullback_dir:
            self._transition(symbol, s, BotState.PULLBACK_ACTIVE, f"M5 pullback ({pullback_dir}) — ready for entry trigger")
        elif m15 == entry_dir:
            # M15 reversed back without M5 pullback — false pullback
            self._transition(symbol, s, BotState.MACRO_OK, "M15 reversed back — false pullback")
    
    def _handle_pullback_active(self, symbol: str, s: SymbolState, m15: str, m5: str):
        """PULLBACK_ACTIVE → M5 reverses back to trend direction = ENTRY!"""
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        
        if m5 == entry_dir:
            self._do_entry(symbol, s)
    
    def _handle_in_trade(self, symbol: str, s: SymbolState, tf_key: str, m15: str, m5: str):
        """IN_TRADE → monitor for partial/full exit."""
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        exit_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        trade_age = time.time() - s.entry_time if s.entry_time else 0
        
        # Track M15 alignment: once M15 aligns with trade, set flag
        if m15 == entry_dir and not s.m15_aligned:
            s.m15_aligned = True
        
        # M15 exit logic — with grace period for pullback entries
        if m15 == exit_dir:
            if s.m15_aligned:
                # M15 was aligned → now reversed = real exit signal
                self._do_full_exit(symbol, s, f"M15 → {exit_dir}")
                return
            elif trade_age > self.min_trade_sec:
                # Grace period expired — M15 never aligned, exit anyway
                self._do_full_exit(symbol, s, f"M15 → {exit_dir} (grace expired)")
                return
            # else: still in grace period, M15 hasn't aligned yet — hold
        
        # M5 partial exit — only after min trade duration
        if m5 == exit_dir and tf_key == '5':
            if trade_age >= self.min_trade_sec:
                self._do_partial_exit(symbol, s)
            # else: too early, skip partial
    
    def _handle_in_trade_partial(self, symbol: str, s: SymbolState, tf_key: str, m15: str, m5: str):
        """IN_TRADE_PARTIAL → monitor for reload or full exit."""
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        exit_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        
        # Track M15 alignment
        if m15 == entry_dir and not s.m15_aligned:
            s.m15_aligned = True
        
        # M15 → exit direction = FULL EXIT remaining
        if m15 == exit_dir:
            if s.m15_aligned:
                self._do_full_exit(symbol, s, f"M15 → {exit_dir}")
                return
            elif (time.time() - s.entry_time) > self.min_trade_sec:
                self._do_full_exit(symbol, s, f"M15 → {exit_dir} (grace expired)")
                return
        
        # M5 → entry direction = RELOAD (with cooldown)
        if m5 == entry_dir and m15 == entry_dir and tf_key == '5':
            # Check cooldown since last partial
            if s.last_partial_time > 0:
                elapsed = time.time() - s.last_partial_time
                if elapsed < self.partial_cooldown_sec:
                    return  # Too soon after partial, wait
            self._do_reload(symbol, s)
    
    # ========================================
    # TRADE ACTIONS
    # ========================================
    
    def _do_entry(self, symbol: str, s: SymbolState):
        """Execute entry trade."""
        signal_type = 'BUY' if s.direction == 'LONG' else 'SELL'
        price = self._get_price(symbol)
        
        # Get M15 trend at entry for grace period logic
        trends = self.zl_service.get_all_trends(symbol) if self.zl_service else {}
        m15_at_entry = trends.get('15', 'neutral')
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        
        s.state = BotState.IN_TRADE
        s.entry_price = price
        s.entry_time = time.time()
        s.partial_done = False
        s.entry_m15 = m15_at_entry
        s.m15_aligned = (m15_at_entry == entry_dir)  # Already aligned if M15 matches
        s.last_partial_time = 0
        s.trade_count += 1
        self._stats['entries'] += 1
        
        grace = "" if s.m15_aligned else " (M15 grace active)"
        label = '🟢 LONG' if s.direction == 'LONG' else '🔴 SHORT'
        price_str = f"${price:,.4f}" if price else "market"
        msg = (f"⚡ ZLT Bot ENTRY | {symbol} {label}\n"
               f"Price: {price_str}{grace}\n"
               f"Pullback complete — M5 reversed to trend\n"
               f"Trade #{s.trade_count}")
        
        print(f"[ZLT Bot] ⚡ ENTRY: {symbol} {signal_type} @ {price_str}")
        self._notify(msg)
        
        if self.on_trade:
            self.on_trade(symbol, 'entry', {
                'signal_type': signal_type,
                'direction': s.direction,
                'price': price,
                'reason': 'ZLT Bot Pullback Entry',
            })
    
    def _do_partial_exit(self, symbol: str, s: SymbolState):
        """Execute partial close."""
        price = self._get_price(symbol)
        
        s.state = BotState.IN_TRADE_PARTIAL
        s.partial_done = True
        s.last_partial_time = time.time()
        self._stats['partial_exits'] += 1
        
        pnl_str = ""
        if s.entry_price and price:
            pnl_pct = ((price - s.entry_price) / s.entry_price * 100) if s.direction == 'LONG' else ((s.entry_price - price) / s.entry_price * 100)
            pnl_str = f"\nP&L: {'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
        
        exit_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        price_str = f"${price:,.4f}" if price else "market"
        msg = (f"🔻 ZLT Bot PARTIAL | {symbol}\n"
               f"Price: {price_str} (entry: ${s.entry_price:,.4f}){pnl_str}\n"
               f"M5 → {exit_dir} — closing {self.partial_close_pct}%")
        
        print(f"[ZLT Bot] 🔻 PARTIAL: {symbol} @ {price_str} (close {self.partial_close_pct}%)")
        self._notify(msg)
        
        if self.on_trade:
            self.on_trade(symbol, 'partial_exit', {
                'close_pct': self.partial_close_pct,
                'direction': s.direction,
                'price': price,
                'entry_price': s.entry_price,
                'reason': 'ZLT Bot M5 Exit',
            })
    
    def _do_reload(self, symbol: str, s: SymbolState):
        """Reload position back to 100%."""
        price = self._get_price(symbol)
        
        s.state = BotState.IN_TRADE
        s.partial_done = False
        self._stats['reloads'] += 1
        
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        signal_type = 'BUY' if s.direction == 'LONG' else 'SELL'
        price_str = f"${price:,.4f}" if price else "market"
        msg = (f"🔄 ZLT Bot RELOAD | {symbol}\n"
               f"Price: {price_str}\n"
               f"M5 → {entry_dir} + M15 intact — reloading to 100%")
        
        print(f"[ZLT Bot] 🔄 RELOAD: {symbol} {signal_type} @ {price_str}")
        self._notify(msg)
        
        if self.on_trade:
            self.on_trade(symbol, 'reload', {
                'signal_type': signal_type,
                'direction': s.direction,
                'price': price,
                'reason': 'ZLT Bot Reload',
            })
    
    def _do_full_exit(self, symbol: str, s: SymbolState, reason: str):
        """Full position close."""
        price = self._get_price(symbol)
        was_partial = s.state == BotState.IN_TRADE_PARTIAL
        remaining = f"{100 - self.partial_close_pct}%" if was_partial else "100%"
        
        pnl_str = ""
        if s.entry_price and price:
            pnl_pct = ((price - s.entry_price) / s.entry_price * 100) if s.direction == 'LONG' else ((s.entry_price - price) / s.entry_price * 100)
            pnl_str = f"\nP&L: {'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
        
        self._stats['full_exits'] += 1
        
        price_str = f"${price:,.4f}" if price else "market"
        entry_str = f"${s.entry_price:,.4f}" if s.entry_price else "?"
        duration = ""
        if s.entry_time:
            dur_sec = time.time() - s.entry_time
            dur_min = int(dur_sec / 60)
            duration = f"\nDuration: {dur_min}m"
        
        msg = (f"❌ ZLT Bot EXIT | {symbol}\n"
               f"Price: {price_str} (entry: {entry_str}){pnl_str}{duration}\n"
               f"{reason} — closing {remaining}")
        
        print(f"[ZLT Bot] ❌ EXIT: {symbol} @ {price_str} ({reason}, {remaining})")
        self._notify(msg)
        
        if self.on_trade:
            self.on_trade(symbol, 'full_exit', {
                'direction': s.direction,
                'price': price,
                'entry_price': s.entry_price,
                'reason': f'ZLT Bot Exit: {reason}',
                'was_partial': was_partial,
            })
        
        s.last_exit_time = time.time()
        s.reset()
    
    # ========================================
    # HELPERS
    # ========================================
    
    def _get_price(self, symbol: str) -> float:
        """Get current market price from scanner cache."""
        if self.get_price:
            try:
                return self.get_price(symbol)
            except:
                pass
        return 0.0
    
    def _macro_broken(self, direction: str, h4: str, h1: str) -> bool:
        """Check if macro trend is broken."""
        if direction == 'LONG':
            return h4 != 'bullish' or h1 != 'bullish'
        elif direction == 'SHORT':
            return h4 != 'bearish' or h1 != 'bearish'
        return False
    
    def _transition(self, symbol: str, s: SymbolState, new_state: BotState, reason: str):
        """Log state transition."""
        old = s.state.value
        s.state = new_state
        print(f"[ZLT Bot] {symbol}: {old} → {new_state.value} ({reason})")
    
    def _notify(self, msg: str):
        """Send notification."""
        if self.on_notify:
            try:
                self.on_notify(msg)
            except Exception:
                pass
    
    # ========================================
    # PERIODIC CHECK (fallback for missed transitions)
    # ========================================
    
    def check_all(self):
        """
        Periodic check: evaluate state machine for all symbols.
        Called by ctr_job every ~30s as safety net for missed real-time transitions.
        """
        if not self.enabled or not self.zl_service:
            return
        
        for symbol in list(self._states.keys()):
            trends = self.zl_service.get_all_trends(symbol)
            if not trends:
                continue
            
            s = self._states[symbol]
            h4 = trends.get('240', 'neutral')
            h1 = trends.get('60', 'neutral')
            m15 = trends.get('15', 'neutral')
            m5 = trends.get('5', 'neutral')
            
            with self._lock:
                # Re-evaluate current state
                if s.state == BotState.IDLE:
                    self._handle_idle(symbol, s, h4, h1, m15, m5)
                elif s.state == BotState.MACRO_OK:
                    if self._macro_broken(s.direction, h4, h1):
                        self._transition(symbol, s, BotState.IDLE, "Macro broken (periodic)")
                        s.reset()
                    else:
                        self._handle_macro_ok(symbol, s, m15, m5)
                elif s.state == BotState.WAITING_PULLBACK:
                    if self._macro_broken(s.direction, h4, h1):
                        self._transition(symbol, s, BotState.IDLE, "Macro broken (periodic)")
                        s.reset()
                    else:
                        self._handle_waiting_pullback(symbol, s, m15, m5)
                elif s.state == BotState.PULLBACK_ACTIVE:
                    if self._macro_broken(s.direction, h4, h1):
                        self._transition(symbol, s, BotState.IDLE, "Macro broken (periodic)")
                        s.reset()
                    else:
                        self._handle_pullback_active(symbol, s, m15, m5)
                elif s.state in (BotState.IN_TRADE, BotState.IN_TRADE_PARTIAL):
                    if self._macro_broken(s.direction, h4, h1):
                        self._do_full_exit(symbol, s, "Macro broken (periodic)")
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def set_watchlist(self, watchlist: List[str]):
        """Set symbols to monitor."""
        with self._lock:
            for sym in watchlist:
                if sym not in self._states:
                    self._states[sym] = SymbolState()
            # Remove symbols no longer in watchlist
            for sym in list(self._states.keys()):
                if sym not in watchlist:
                    self._states.pop(sym, None)
    
    def get_states(self) -> Dict:
        """Get all symbol states for UI."""
        with self._lock:
            return {sym: s.to_dict() for sym, s in self._states.items()}
    
    def get_stats(self) -> Dict:
        """Get bot statistics."""
        with self._lock:
            active = sum(1 for s in self._states.values() if s.state != BotState.IDLE)
            in_trade = sum(1 for s in self._states.values() if s.state in (BotState.IN_TRADE, BotState.IN_TRADE_PARTIAL))
            return {
                'enabled': self.enabled,
                'partial_close_pct': self.partial_close_pct,
                'active_states': active,
                'in_trade': in_trade,
                'symbols': len(self._states),
                'states': self.get_states(),
                **self._stats,
            }
    
    def reset_symbol(self, symbol: str):
        """Reset state for one symbol."""
        with self._lock:
            if symbol in self._states:
                self._states[symbol].reset()
    
    def reset_all(self):
        """Reset all states."""
        with self._lock:
            for s in self._states.values():
                s.reset()
