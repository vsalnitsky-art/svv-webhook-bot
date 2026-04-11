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
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

BINANCE_KLINE_URL = 'https://fapi.binance.com/fapi/v1/klines'
BINANCE_OI_URL = 'https://fapi.binance.com/fapi/v1/openInterest'
BINANCE_LS_URL = 'https://fapi.binance.com/futures/data/globalLongShortAccountRatio'
BINANCE_TOP_LS_URL = 'https://fapi.binance.com/futures/data/topLongShortAccountRatio'
BINANCE_TAKER_URL = 'https://fapi.binance.com/futures/data/takerlongshortRatio'

DB_KEY = 'signal_validator_log'
KEEP_DAYS = 5
REQUEST_DELAY = 0.3


class SignalValidator:
    
    def __init__(self, db=None, notifier=None):
        self.db = db
        self.notifier = notifier
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'SVV-Bot/1.0'})
    
    def validate(self, data: Dict) -> Dict:
        """
        Validate a TradingView signal.
        Supports both formats:
          Pine Script: {"action":"buy","symbol":"BTCUSDT","entry":"67500","sl":"67200","tp":"68400","strategy":"SMC_PRO",...}
          Manual:      {"action":"Buy","symbol":"BTCUSDT","leverage":10,"riskPercent":1,"stopLossPercent":2}
        """
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
                'btc_signal': {},
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
        
        # Get current BTC signal for context
        btc_signal = self._get_btc_signal()
        
        # Determine approval
        coin_dir = signal.get('direction', 'NEUTRAL')
        coin_conf = signal.get('confidence', 0)
        btc_dir = btc_signal.get('direction', 'NEUTRAL')
        btc_conf = btc_signal.get('confidence', 0)
        
        # Approval logic:
        # 1. Coin signal must agree with TV direction
        # 2. Coin confidence ≥ 50%
        # 3. BTC not strongly opposing (if BTC SHORT 70%+ don't open LONG)
        direction_match = (coin_dir == tv_direction)
        conf_ok = (coin_conf >= 50)
        btc_ok = not (btc_dir != tv_direction and btc_conf >= 70)
        
        approved = direction_match and conf_ok and btc_ok
        
        # Status text
        if approved:
            status = f'✅ APPROVED'
        elif not direction_match:
            status = f'❌ REJECTED (Vol says {coin_dir})'
        elif not conf_ok:
            status = f'⚠️ WEAK ({coin_conf}%)'
        else:
            status = f'❌ BTC OPPOSING ({btc_dir} {btc_conf}%)'
        
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
            'btc_signal': {
                'direction': btc_dir,
                'confidence': btc_conf,
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
              f"Vol: {coin_dir} {coin_conf}% | BTC: {btc_dir} {btc_conf}%")
        
        return result
    
    # ========================================
    # DATA FETCHING (same as coin_flow.py)
    # ========================================
    
    def _fetch_klines(self, symbol):
        try:
            r = self._session.get(BINANCE_KLINE_URL,
                params={'symbol': symbol, 'interval': '1m', 'limit': 60}, timeout=10)
            if r.status_code != 200:
                return None
            candles = []
            for k in r.json():
                try:
                    tv = float(k[7]); tb = float(k[10])
                    candles.append({'p': float(k[4]), 'v': round(tv), 'b': round(tb), 's': round(tv-tb)})
                except: continue
            return candles if candles else None
        except: return None
    
    def _fetch_oi(self, symbol, klines):
        try:
            time.sleep(REQUEST_DELAY)
            r = self._session.get(BINANCE_OI_URL, params={'symbol': symbol}, timeout=10)
            if r.status_code == 200:
                qty = float(r.json().get('openInterest', 0))
                price = klines[-1]['p'] if klines else 0
                return qty * price if price else qty
        except: pass
        return 0
    
    def _fetch_sentiment(self, symbol):
        sent = {}
        try:
            time.sleep(REQUEST_DELAY)
            r = self._session.get(BINANCE_LS_URL,
                params={'symbol': symbol, 'period': '5m', 'limit': 6}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    sent['ls_ratio'] = float(data[-1].get('longShortRatio', 1))
                    sent['ls_long'] = round(float(data[-1].get('longAccount', 0.5)) * 100, 1)
        except: pass
        try:
            time.sleep(REQUEST_DELAY)
            r = self._session.get(BINANCE_TOP_LS_URL,
                params={'symbol': symbol, 'period': '5m', 'limit': 6}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    sent['top_ls'] = float(data[-1].get('longShortRatio', 1))
                    sent['top_long'] = round(float(data[-1].get('longAccount', 0.5)) * 100, 1)
        except: pass
        try:
            time.sleep(REQUEST_DELAY)
            r = self._session.get(BINANCE_TAKER_URL,
                params={'symbol': symbol, 'period': '5m', 'limit': 6}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    sent['taker'] = float(data[-1].get('buySellRatio', 1))
        except: pass
        return sent
    
    def _get_btc_signal(self):
        try:
            from detection.volume_flow import get_volume_flow
            vf = get_volume_flow()
            if vf:
                s = vf.get_summary()
                return s.get('signal', {})
        except: pass
        return {}
    
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
            b = result.get('btc_signal', {})
            p = result.get('pine', {})
            icon = '✅' if result['approved'] else '❌'
            dir_icon = '🟢' if result['direction'] == 'LONG' else '🔴'
            
            # Vol Flow direction
            vol_icon = '🟢' if s.get('direction') == 'LONG' else '🔴' if s.get('direction') == 'SHORT' else '⚪'
            btc_icon = '🟢' if b.get('direction') == 'LONG' else '🔴' if b.get('direction') == 'SHORT' else '⚪'
            
            # === MAIN (always visible) ===
            main = (
                f"{icon} <b>TV: {result['symbol']} {result['action']}</b>  "
                f"💰 <b>${result.get('price', 0):,.6g}</b>\n"
                f"{dir_icon} TV:{result['direction']}  "
                f"{vol_icon} Vol:{s.get('confidence',0)}%  "
                f"{btc_icon} BTC:{b.get('confidence',0)}%\n"
                f"<b>{result['status']}</b>"
            )
            
            # === DETAILS (in spoiler — tap to reveal) ===
            details_parts = []
            
            # Strategy + trigger
            strat = p.get('strategy', result.get('strategy', ''))
            if strat and strat != 'MANUAL':
                trig = f" ({p['trigger']})" if p.get('trigger') else ''
                details_parts.append(f"🔧 {strat}{trig}")
            
            # SL/TP
            sltp = []
            if result.get('sl_price'):
                sltp.append(f"SL:${result['sl_price']:,.6g}")
            if result.get('tp_price'):
                sltp.append(f"TP:${result['tp_price']:,.6g}")
            if result.get('rr_ratio'):
                sltp.append(f"RR:{result['rr_ratio']:.1f}")
            if sltp:
                details_parts.append('🛡 ' + '  '.join(sltp))
            
            # Pine filters
            filters = []
            if p.get('htf'): filters.append(f"HTF:{p['htf']}")
            if p.get('htf2'): filters.append(f"HTF2:{p['htf2']}")
            if p.get('zone'): filters.append(f"Zone:{p['zone']}")
            if p.get('mtf'): filters.append(f"MTF:{p['mtf']}/7")
            if filters:
                details_parts.append('📋 ' + ' '.join(filters))
            
            # Vol Flow reasons
            reasons = s.get('reasons', [])[:5]
            if reasons:
                details_parts.append('📊 ' + ' · '.join(reasons))
            
            details_text = '\n'.join(details_parts)
            spoiler = f"\n<tg-spoiler>{details_text}</tg-spoiler>" if details_text else ''
            
            msg = f"{main}{spoiler}\n⏱ {result['timestamp'][11:16]} UTC"
            
            self.notifier.send_message(msg)
        except Exception as e:
            print(f"[VALIDATOR] Notification error: {e}")


_instance: Optional[SignalValidator] = None
def get_validator(): return _instance
def init_validator(db=None, notifier=None):
    global _instance
    _instance = SignalValidator(db=db, notifier=notifier)
    return _instance
