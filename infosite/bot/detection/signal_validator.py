"""
Signal Validator v1.0 — TradingView Webhook Signal Analysis

Receives TradingView JSON → analyzes coin via Volume Flow → stores result.
Same analysis as BTC Volume Flow but for any coin on-demand.

Flow:
  1. /webhook receives JSON from TradingView
  2. Fetch klines + OI + sentiment for the coin
  3. Score signal (same 8-factor engine as Volume Flow)
  4. Store result in DB (auto-cleanup 5 days)
  5. Future: if approved → execute trade
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

DB_KEY = 'signal_validator_log'
KEEP_DAYS = 5


class SignalValidator:
    
    def __init__(self, db=None, notifier=None):
        self.db = db
        self.notifier = notifier
    
    def is_enabled(self) -> bool:
        if not self.db:
            return True
        return self.db.get_setting('validator_enabled', '1') == '1'
    
    def set_enabled(self, enabled: bool) -> bool:
        if self.db:
            self.db.set_setting('validator_enabled', '1' if enabled else '0')
        print(f"[VALIDATOR] {'✅ Enabled' if enabled else '⏸️ Disabled'}")
        return True
    
    def validate(self, data: Dict) -> Dict:
        """
        Validate a TradingView signal.
        Supports formats:
          v38 SMC_PRO:      {"strategy","action","symbol","entry","sl","tp"}
          v38 CTR/MOM:      {"strategy","action","symbol","entry"}
          v38 Close:        {"strategy","action":"Close","symbol","direction","reason","entry"}
          v38 TREND_BREAK:  {"strategy","event":"TREND_BREAK",...}  ← IGNORED
          Manual:           {"action":"Buy","symbol","leverage","riskPercent","stopLossPercent"}
        """
        # Check if module is enabled
        if not self.is_enabled():
            print(f"[VALIDATOR] ⏸️ Disabled — webhook ignored: {data.get('symbol','')} {data.get('action','')}")
            return {'status': 'disabled', 'reason': 'Validator disabled'}
        
        # Ignore TREND_BREAK events — informational, no action needed
        event = data.get('event', '')
        if event == 'TREND_BREAK':
            print(f"[VALIDATOR] 🔸 TREND_BREAK ignored: {data.get('symbol','')} {data.get('direction','')}")
            return {'status': 'ignored', 'reason': 'TREND_BREAK event'}
        
        action = data.get('action', '').capitalize()
        symbol = data.get('symbol', '')
        
        if not symbol:
            return {'status': 'error', 'reason': 'No symbol'}
        
        # Normalize symbol
        symbol = symbol.upper().replace('.P', '')
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        
        # Parse strategy type
        strategy = data.get('strategy', 'MANUAL')
        
        # Parse entry/SL/TP (Pine sends absolute prices as strings)
        entry_price = float(data.get('entry', 0) or 0)
        sl_price = float(data.get('sl', 0) or 0)
        tp_price = float(data.get('tp', 0) or 0)
        tp1_price = float(data.get('tp1', 0) or 0)
        rr_ratio = float(data.get('rr', 0) or 0)
        
        # Parse risk (Pine: "1%" string, Manual: riskPercent number)
        risk_str = data.get('risk', '')
        risk_pct = float(data.get('riskPercent', 0) or 0)
        if not risk_pct and risk_str:
            try:
                risk_pct = float(risk_str.replace('%', ''))
            except:
                pass
        
        # Parse Pine-specific fields
        pine_tf = data.get('tf', '')
        pine_bias = data.get('bias', '')
        pine_htf = data.get('htf_bias', '')
        pine_htf2 = data.get('htf2_bias', '')
        pine_trigger = data.get('trigger', '')
        pine_zone = data.get('zone', '')
        pine_sweep = data.get('sweep', '')
        pine_fvg = data.get('fvg', '')
        pine_ob = data.get('ob', '')
        pine_mtf = data.get('mtf_strength', '')
        pine_mtf_conf = data.get('mtf_confidence', '')
        
        leverage = int(data.get('leverage', 0) or 0)
        sl_pct = float(data.get('stopLossPercent', 0) or 0)
        
        # Close signals — just log
        if action == 'Close':
            result = {
                'id': self._gen_id(),
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'action': 'Close',
                'direction': data.get('direction', data.get('bias', 'N/A')),
                'reason': data.get('reason', ''),
                'status': 'CLOSE',
                'approved': False,
                'signal': {},
                'cvd': {'value': 0, 'direction': 'NEUTRAL'},
                'pine': {'strategy': strategy, 'trigger': pine_trigger},
                'raw_json': data,
            }
            self._store(result)
            return result
        
        # Buy/Sell — full analysis
        tv_direction = 'LONG' if action == 'Buy' else 'SHORT'
        
        print(f"[VALIDATOR] 📥 Analyzing {symbol} {action}...")
        
        # Fetch data
        klines = self._fetch_klines(symbol)
        oi = self._fetch_oi(symbol, klines)
        sentiment = self._fetch_sentiment(symbol)
        
        # Calculate signal (same engine as Volume Flow)
        signal = self._calc_signal(klines, oi, sentiment) if klines else {}
        
        # Calculate CVD for the coin (15-min cumulative volume delta)
        coin_cvd = self._calc_cvd(klines, 15) if klines else 0
        cvd_direction = 'LONG' if coin_cvd > 0 else 'SHORT' if coin_cvd < 0 else 'NEUTRAL'
        
        # Determine approval
        coin_dir = signal.get('direction', 'NEUTRAL')
        coin_conf = signal.get('confidence', 0)
        
        # Approval logic:
        # 1. Vol Flow direction = TV direction
        # 2. Confidence ≥ 50%
        # 3. CVD of coin confirms TV direction (positive for LONG, negative for SHORT)
        direction_match = (coin_dir == tv_direction)
        conf_ok = (coin_conf >= 50)
        cvd_ok = (cvd_direction == tv_direction) or cvd_direction == 'NEUTRAL'
        
        approved = direction_match and conf_ok and cvd_ok
        
        # Status text
        if approved:
            status = f'✅ APPROVED'
        elif not direction_match:
            status = f'❌ REJECTED (Vol says {coin_dir})'
        elif not conf_ok:
            status = f'⚠️ WEAK ({coin_conf}%)'
        else:
            status = f'❌ CVD OPPOSING ({cvd_direction})'
        
        price = entry_price if entry_price else (klines[-1]['p'] if klines else 0)
        
        result = {
            'id': self._gen_id(),
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'action': action,
            'direction': tv_direction,
            'price': price,
            'strategy': strategy,
            'risk_pct': risk_pct,
            'leverage': leverage,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'tp1_price': tp1_price,
            'rr_ratio': rr_ratio,
            'sl_pct': sl_pct,
            'status': status,
            'approved': approved,
            'signal': {
                'direction': coin_dir,
                'confidence': coin_conf,
                'reasons': signal.get('reasons', []),
                'long_score': signal.get('long_score', 0),
                'short_score': signal.get('short_score', 0),
            },
            'cvd': {
                'value': coin_cvd,
                'direction': cvd_direction,
            },
            'pine': {
                'strategy': strategy,
                'trigger': pine_trigger,
                'tf': pine_tf,
                'bias': pine_bias,
                'htf': pine_htf,
                'htf2': pine_htf2,
                'zone': pine_zone,
                'sweep': pine_sweep,
                'fvg': pine_fvg,
                'ob': pine_ob,
                'mtf': pine_mtf,
                'mtf_conf': pine_mtf_conf,
            },
            'raw_json': data,
        }
        
        self._store(result)
        self._send_notification(result)
        
        print(f"[VALIDATOR] {status} | {symbol} {tv_direction} | "
              f"Vol: {coin_dir} {coin_conf}% | CVD: {cvd_direction} ({coin_cvd:+,.0f})")
        
        return result
    
    # ========================================
    # DATA FETCHING (same as coin_flow.py)
    # ========================================
    
    def _fetch_klines(self, symbol):
        from detection.market_data import get_market_data
        return get_market_data().fetch_klines(symbol, 60)
    
    def _fetch_oi(self, symbol, klines):
        from detection.market_data import get_market_data
        price = klines[-1]['p'] if klines else 0
        oi, _ = get_market_data().fetch_oi(symbol, price)
        return oi or 0
    
    def _fetch_sentiment(self, symbol):
        from detection.market_data import get_market_data
        return get_market_data().fetch_sentiment(symbol)
    
    def _calc_cvd(self, klines, minutes=15):
        """Cumulative Volume Delta over last N minutes (taker buy - taker sell)."""
        if not klines:
            return 0
        recent = klines[-minutes:] if len(klines) >= minutes else klines
        return round(sum(c.get('b', 0) - c.get('s', 0) for c in recent))
    
    # ========================================
    # SIGNAL ENGINE (same as volume_flow.py)
    # ========================================
    
    def _calc_signal(self, klines, oi, sent):
        if not klines or len(klines) < 5:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reasons': [],
                    'long_score': 0, 'short_score': 0}
        
        def calc_w(candles, mins):
            r = candles[-mins:] if len(candles) >= mins else candles
            tb = sum(c['b'] for c in r); ts = sum(c['s'] for c in r)
            t = tb + ts; bp = (tb/t*100) if t > 0 else 50
            return {'buy_pct': round(bp,1), 'sell_pct': round(100-bp,1), 'cvd': round(tb-ts)}
        
        w5 = calc_w(klines, 5); w15 = calc_w(klines, 15); w60 = calc_w(klines, 60)
        ls = 0; ss = 0; reasons = []
        
        for label, w, pts in [('5m', w5, 18), ('15m', w15, 18), ('1h', w60, 14)]:
            if w['buy_pct'] >= 60: ls += pts; reasons.append(f"{label} Buy {w['buy_pct']:.0f}%")
            elif w['sell_pct'] >= 60: ss += pts; reasons.append(f"{label} Sell {w['sell_pct']:.0f}%")
        
        if len(klines) >= 15:
            cv = []; run = 0
            for c in klines[-15:]: run += (c['b']-c['s']); cv.append(run)
            if cv[-1] > cv[0] + abs(cv[0])*0.1: ls += 20; reasons.append("CVD rising ↑")
            elif cv[-1] < cv[0] - abs(cv[0])*0.1: ss += 20; reasons.append("CVD falling ↓")
            
            pc = (klines[-1]['p']-klines[-15]['p'])/klines[-15]['p']*100
            cvd15 = sum(c['b']-c['s'] for c in klines[-15:])
            if pc < -0.1 and cvd15 > 0: ls += 25; reasons.append(f"⚡ Accumulation")
            elif pc > 0.1 and cvd15 < 0: ss += 25; reasons.append(f"⚡ Distribution")
        
        lsr = sent.get('ls_ratio', 0)
        if lsr >= 2.0: ss += 20; reasons.append(f"L/S {lsr:.2f} crowd LONG")
        elif lsr <= 0.7: ls += 20; reasons.append(f"L/S {lsr:.2f} crowd SHORT")
        
        top = sent.get('top_ls', 0)
        if top >= 1.3: ls += 20; reasons.append(f"Top LONG {sent.get('top_long',0):.0f}%")
        elif top <= 0.75: ss += 20; reasons.append("Top SHORT")
        
        taker = sent.get('taker', 0)
        if taker >= 1.15: ls += 15; reasons.append(f"Taker Buy {taker:.2f}")
        elif taker <= 0.85: ss += 15; reasons.append(f"Taker Sell {taker:.2f}")
        
        total = ls + ss
        if total == 0: return {'direction': 'NEUTRAL', 'confidence': 0, 'reasons': [], 'long_score': 0, 'short_score': 0}
        if ls > ss: return {'direction': 'LONG', 'confidence': min(95, round(ls/1.7)), 'reasons': reasons, 'long_score': ls, 'short_score': ss}
        if ss > ls: return {'direction': 'SHORT', 'confidence': min(95, round(ss/1.7)), 'reasons': reasons, 'long_score': ls, 'short_score': ss}
        return {'direction': 'NEUTRAL', 'confidence': 0, 'reasons': reasons, 'long_score': ls, 'short_score': ss}
    
    # ========================================
    # STORAGE
    # ========================================
    
    def _store(self, result):
        if not self.db: return
        try:
            log = self.db.get_setting(DB_KEY, [])
            if not isinstance(log, list): log = []
            log.append(result)
            if len(log) > 500: log = log[-500:]
            self.db.set_setting(DB_KEY, log)
        except Exception as e:
            print(f"[VALIDATOR] Store error: {e}")
    
    def get_log(self):
        if not self.db: return []
        try:
            log = self.db.get_setting(DB_KEY, [])
            if not isinstance(log, list): return []
            # Auto-cleanup > KEEP_DAYS
            cutoff = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).strftime('%Y-%m-%d')
            cleaned = [r for r in log if r.get('timestamp', '')[:10] >= cutoff]
            if len(cleaned) != len(log):
                self.db.set_setting(DB_KEY, cleaned)
            return list(reversed(cleaned))  # newest first
        except: return []
    
    def delete_entry(self, entry_id):
        if not self.db: return False
        try:
            log = self.db.get_setting(DB_KEY, [])
            if not isinstance(log, list): return False
            log = [r for r in log if r.get('id') != entry_id]
            self.db.set_setting(DB_KEY, log)
            return True
        except: return False
    
    def clear_all(self):
        if not self.db: return False
        try:
            self.db.set_setting(DB_KEY, [])
            return True
        except: return False
    
    def _gen_id(self):
        import random
        return f"{int(time.time())}{random.randint(100,999)}"
    
    def _send_notification(self, result):
        if not self.notifier: return
        try:
            s = result.get('signal', {})
            c = result.get('cvd', {})
            icon = '✅' if result['approved'] else '❌'
            dir_icon = '🟢' if result['direction'] == 'LONG' else '🔴'
            
            # Vol Flow direction icon
            vol_icon = '🟢' if s.get('direction') == 'LONG' else '🔴' if s.get('direction') == 'SHORT' else '⚪'
            
            # CVD direction icon + formatted value
            cvd_val = c.get('value', 0)
            cvd_dir = c.get('direction', 'NEUTRAL')
            cvd_icon = '🟢' if cvd_dir == 'LONG' else '🔴' if cvd_dir == 'SHORT' else '⚪'
            cvd_str = f"{cvd_val:+,.0f}" if cvd_val else '0'
            
            # === MAIN ===
            main = (
                f"{icon} <b>TV: {result['symbol']} {result['action']}</b>  "
                f"💰 <b>${result.get('price', 0):,.6g}</b>\n"
                f"{dir_icon} TV:{result['direction']}  "
                f"{vol_icon} Vol:{s.get('confidence',0)}%  "
                f"{cvd_icon} CVD:{cvd_str}\n"
                f"<b>{result['status']}</b>"
            )
            
            msg = f"{main}\n⏱ {result['timestamp'][11:16]} UTC"
            
            self.notifier.send_message(msg)
        except Exception as e:
            print(f"[VALIDATOR] Notification error: {e}")


_instance: Optional[SignalValidator] = None
def get_validator(): return _instance
def init_validator(db=None, notifier=None):
    global _instance
    _instance = SignalValidator(db=db, notifier=notifier)
    return _instance
