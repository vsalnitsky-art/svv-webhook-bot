"""
ZLT Bot Strategy v3.0 — Professional Multi-Timeframe Pullback Entry

PHILOSOPHY:
  "Enter when lower TFs have CAUGHT UP to higher TFs"
  At entry moment: H4, H1, M15, M5 — ALL aligned in same direction.
  
  M5 is the microscope for precision entry.
  M15 must FIRST pull back, THEN recover — confirming the impulse.
  Only AFTER M15 recovery, M5 gives the final trigger.

STATE MACHINE:
  IDLE → SETUP → PULLBACK → RECOVERED → DIP → IN_TRADE → IDLE

  IDLE:       No macro alignment (H4+H1 disagree)
  SETUP:      H4+H1 aligned. Waiting for M15 to pull back.
  PULLBACK:   M15 reversed against trend = pullback in progress.
  RECOVERED:  M15 returned to trend direction = pullback over!
              Now waiting for M5 micro-dip for precision entry.
  DIP:        M5 pulled back (micro-dip within M15 impulse).
              Waiting for M5 to return = ENTRY trigger.
  IN_TRADE:   All TFs aligned at entry. Hold for H1 impulse (2-8h).
              Exit ONLY on M15 structure break or macro break.

AT ENTRY MOMENT (LONG example):
  H4: ▲  H1: ▲  M15: ▲  M5: ▲  — ALL ALIGNED

EXIT RULES:
  1. M15 → bearish (structure break) = EXIT
  2. H4 or H1 reverses (macro break) = EXIT (safety)
  3. Nothing else. No M5 exit. No partial. No grace period.
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from enum import Enum


class BotState(Enum):
    IDLE = 'idle'
    SETUP = 'setup'          # Macro OK, waiting M15 pullback
    PULLBACK = 'pullback'    # M15 pulled back against trend
    RECOVERED = 'recovered'  # M15 returned to trend, waiting M5 dip
    DIP = 'dip'              # M5 dipped, waiting M5 return = ENTRY
    IN_TRADE = 'in_trade'    # Position open, all TFs were aligned


class SymbolState:
    __slots__ = [
        'state', 'direction', 'macro_since', 'pullback_since',
        'recovered_since', 'entry_price', 'entry_time', 'trade_count',
        'last_exit_time', 'peak_price',
    ]
    
    def __init__(self):
        self.state: BotState = BotState.IDLE
        self.direction: str = ''
        self.macro_since: float = 0
        self.pullback_since: float = 0
        self.recovered_since: float = 0
        self.entry_price: float = 0
        self.entry_time: float = 0
        self.trade_count: int = 0
        self.last_exit_time: float = 0
        self.peak_price: float = 0
    
    def reset(self):
        self.state = BotState.IDLE
        self.direction = ''
        self.macro_since = 0
        self.pullback_since = 0
        self.recovered_since = 0
        self.entry_price = 0
        self.entry_time = 0
        self.peak_price = 0
        # last_exit_time NOT reset
    
    def to_dict(self) -> Dict:
        return {
            'state': self.state.value,
            'direction': self.direction,
            'macro_since': self.macro_since,
            'pullback_since': self.pullback_since,
            'recovered_since': self.recovered_since,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'trade_count': self.trade_count,
            'last_exit_time': self.last_exit_time,
            'peak_price': self.peak_price,
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
        s.recovered_since = data.get('recovered_since', 0)
        s.entry_price = data.get('entry_price', 0)
        s.entry_time = data.get('entry_time', 0)
        s.trade_count = data.get('trade_count', 0)
        s.last_exit_time = data.get('last_exit_time', 0)
        s.peak_price = data.get('peak_price', 0)
        return s


class ZLTBot:
    """
    ZLT Bot v3.0 — Professional Pullback Strategy.
    
    Key difference from v2: entry ONLY when ALL TFs aligned.
    M15 must pull back AND recover BEFORE M5 entry trigger.
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
        # Legacy params (ignored)
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
    # CORE
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
            
            if s.state in (BotState.SETUP, BotState.PULLBACK, BotState.RECOVERED, BotState.DIP):
                if self._macro_broken(s.direction, h4, h1):
                    self._transition(symbol, s, BotState.IDLE, "Macro broken")
                    s.reset()
                    return
            
            # === STATE MACHINE ===
            if s.state == BotState.IDLE:
                self._handle_idle(symbol, s, h4, h1, m15)
            elif s.state == BotState.SETUP:
                self._handle_setup(symbol, s, m15)
            elif s.state == BotState.PULLBACK:
                self._handle_pullback(symbol, s, m15)
            elif s.state == BotState.RECOVERED:
                self._handle_recovered(symbol, s, m15, m5)
            elif s.state == BotState.DIP:
                self._handle_dip(symbol, s, m15, m5)
            elif s.state == BotState.IN_TRADE:
                self._handle_in_trade(symbol, s, m15)
    
    # ========================================
    # STATE HANDLERS
    # ========================================
    
    def _handle_idle(self, symbol: str, s: SymbolState, h4: str, h1: str, m15: str):
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
            # M15 already in pullback
            s.pullback_since = time.time()
            self._transition(symbol, s, BotState.PULLBACK,
                             f"Macro {direction} + M15 already pulling back")
        else:
            self._transition(symbol, s, BotState.SETUP,
                             f"Macro {direction} — waiting for M15 pullback")
    
    def _handle_setup(self, symbol: str, s: SymbolState, m15: str):
        """SETUP → wait for M15 to pull back."""
        pullback_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        
        if m15 == pullback_dir:
            s.pullback_since = time.time()
            self._transition(symbol, s, BotState.PULLBACK,
                             f"M15 pullback started ({pullback_dir})")
    
    def _handle_pullback(self, symbol: str, s: SymbolState, m15: str):
        """PULLBACK → wait for M15 to RECOVER (return to trend direction).
        
        This is the KEY step that was missing in v1/v2.
        M15 must complete its pullback and return BEFORE we look at M5.
        """
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        
        if m15 == entry_dir:
            # M15 recovered! Pullback is OVER.
            s.recovered_since = time.time()
            self._transition(symbol, s, BotState.RECOVERED,
                             f"M15 recovered to {entry_dir} — waiting M5 dip")
    
    def _handle_recovered(self, symbol: str, s: SymbolState, m15: str, m5: str):
        """RECOVERED → M15 is with trend. Wait for M5 micro-dip.
        
        Now M15 is aligned. We want M5 to dip briefly (micro-pullback)
        so we enter at a better price within the M15 impulse.
        
        If M15 breaks again before M5 dips → back to PULLBACK.
        """
        pullback_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        
        # M15 broke again → new pullback cycle
        if m15 == pullback_dir:
            s.pullback_since = time.time()
            self._transition(symbol, s, BotState.PULLBACK,
                             "M15 broke again — new pullback")
            return
        
        # M5 dipped against trend = micro-pullback for precision entry
        if m5 == pullback_dir:
            self._transition(symbol, s, BotState.DIP,
                             f"M5 dip ({pullback_dir}) — waiting for entry trigger")
    
    def _handle_dip(self, symbol: str, s: SymbolState, m15: str, m5: str):
        """DIP → M5 returns to trend direction = ENTRY!
        
        At this moment: H4 ✅ H1 ✅ M15 ✅ M5 ✅ — ALL ALIGNED.
        This is the precision entry point.
        
        If M15 breaks during dip → back to PULLBACK.
        """
        entry_dir = 'bullish' if s.direction == 'LONG' else 'bearish'
        pullback_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        
        # M15 broke during M5 dip → back to pullback
        if m15 == pullback_dir:
            s.pullback_since = time.time()
            self._transition(symbol, s, BotState.PULLBACK,
                             "M15 broke during M5 dip — back to pullback")
            return
        
        # M5 returned to trend = ALL TFs ALIGNED = ENTRY!
        if m5 == entry_dir:
            self._do_entry(symbol, s)
    
    def _handle_in_trade(self, symbol: str, s: SymbolState, m15: str):
        """IN_TRADE → ONLY M15 manages exit.
        
        At entry, all TFs were aligned. Now ride the H1 impulse.
        M15 is stable enough for intraday. M5 noise is ignored.
        Exit when M15 structure breaks.
        """
        # Track peak price (from scanner cache, zero REST cost)
        current = self._get_price(symbol)
        if current and s.entry_price:
            if s.direction == 'LONG':
                if current > s.peak_price:
                    s.peak_price = current
            else:
                if s.peak_price == 0 or current < s.peak_price:
                    s.peak_price = current
        
        exit_dir = 'bearish' if s.direction == 'LONG' else 'bullish'
        
        if m15 == exit_dir:
            dur = time.time() - s.entry_time if s.entry_time else 0
            self._do_exit(symbol, s,
                          f"M15 structure break ({int(dur/3600)}h{int((dur%3600)/60)}m)")
    
    # ========================================
    # TRADE ACTIONS
    # ========================================
    
    def _do_entry(self, symbol: str, s: SymbolState):
        signal_type = 'BUY' if s.direction == 'LONG' else 'SELL'
        price = self._get_price(symbol)
        
        s.state = BotState.IN_TRADE
        s.entry_price = price
        s.peak_price = price  # Initialize peak at entry price
        s.entry_time = time.time()
        s.trade_count += 1
        self._stats['entries'] += 1
        
        label = '🟢 LONG' if s.direction == 'LONG' else '🔴 SHORT'
        price_str = f"${price:,.4f}" if price else "market"
        
        msg = (f"⚡ ZLT Bot ENTRY | {symbol} {label}\n"
               f"Price: {price_str}\n"
               f"All TFs aligned ✅ | Trade #{s.trade_count}\n"
               f"Target: H1 impulse (2-8h hold)")
        
        print(f"[ZLT Bot] ⚡ ENTRY: {symbol} {signal_type} @ {price_str} (ALL TFs aligned)")
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
        
        # Final peak update
        if price and s.entry_price:
            if s.direction == 'LONG' and price > s.peak_price:
                s.peak_price = price
            elif s.direction == 'SHORT' and (s.peak_price == 0 or price < s.peak_price):
                s.peak_price = price
        
        pnl_str = ""
        peak_str = ""
        if s.entry_price and price:
            if s.direction == 'LONG':
                pnl_pct = (price - s.entry_price) / s.entry_price * 100
            else:
                pnl_pct = (s.entry_price - price) / s.entry_price * 100
            pnl_str = f"\nP&L: {'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
            
            # Peak calculation
            if s.peak_price and s.peak_price != s.entry_price:
                if s.direction == 'LONG':
                    peak_pct = (s.peak_price - s.entry_price) / s.entry_price * 100
                else:
                    peak_pct = (s.entry_price - s.peak_price) / s.entry_price * 100
                peak_price_str = f"${s.peak_price:,.4f}" if s.peak_price < 100 else f"${s.peak_price:,.2f}"
                peak_str = f"\nPeak: +{peak_pct:.2f}% ({peak_price_str})"
        
        duration = ""
        if s.entry_time:
            dur = time.time() - s.entry_time
            duration = f"\nDuration: {int(dur/3600)}h {int((dur%3600)/60)}m"
        
        price_str = f"${price:,.4f}" if price else "market"
        entry_str = f"${s.entry_price:,.4f}" if s.entry_price else "?"
        
        msg = (f"❌ ZLT Bot EXIT | {symbol}\n"
               f"Price: {price_str} (entry: {entry_str}){pnl_str}{peak_str}{duration}\n"
               f"{reason}")
        
        print(f"[ZLT Bot] ❌ EXIT: {symbol} @ {price_str} ({reason})")
        self._notify(msg)
        
        if self.on_trade:
            self.on_trade(symbol, 'full_exit', {
                'direction': s.direction,
                'price': price,
                'entry_price': s.entry_price,
                'peak_price': s.peak_price,
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
                elif s.state in (BotState.SETUP, BotState.PULLBACK, BotState.RECOVERED, BotState.DIP):
                    if self._macro_broken(s.direction, h4, h1):
                        self._transition(symbol, s, BotState.IDLE, "Macro broken (periodic)")
                        s.reset()
                        continue
                    if s.state == BotState.SETUP:
                        self._handle_setup(symbol, s, m15)
                    elif s.state == BotState.PULLBACK:
                        self._handle_pullback(symbol, s, m15)
                    elif s.state == BotState.RECOVERED:
                        self._handle_recovered(symbol, s, m15, m5)
                    elif s.state == BotState.DIP:
                        self._handle_dip(symbol, s, m15, m5)
    
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
