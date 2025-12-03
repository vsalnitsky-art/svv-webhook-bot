import threading, time, logging, pandas as pd, pandas_ta as ta
from settings_manager import settings
from bot import bot_instance
from strategy_ob_trend import ob_trend_strategy
logger = logging.getLogger(__name__)
class EnhancedMarketScanner:
    def __init__(self, bot_instance, config):
        self.bot = bot_instance; self.config = config; self.running = True; self.active_coins_data = {}; self.scan_interval = 10
    def start(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try: self.monitor_positions()
            except: pass
            time.sleep(self.scan_interval)
    def get_active_symbols(self):
        try:
            r = self.bot.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode'] == 0: return [p for p in r['result']['list'] if float(p['size']) > 0]
        except: pass
        return []
    def fetch_htf_candles(self, symbol):
        try:
            htf = settings.get("htfSelection"); tf_map = {'60':'60','240':'240','D':'D'}; req_tf = tf_map.get(str(htf), '240')
            r = self.bot.session.get_kline(category="linear", symbol=symbol, interval=req_tf, limit=50)
            if r['retCode']==0: 
                df = pd.DataFrame(r['result']['list'], columns=['time','open','high','low','close','vol','to'])
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
                return df.sort_values('time').reset_index(drop=True)
        except: pass
        return None
    def monitor_positions(self):
        active = self.get_active_symbols(); target = [p['symbol'] for p in active]
        for k in list(self.active_coins_data.keys()):
            if k not in target: del self.active_coins_data[k]
        if not active: return
        smart_exit = settings.get("exit_enableStrategy", False)
        for pos in active:
            sym = pos['symbol']; side = pos['side']
            if sym not in self.active_coins_data: self.active_coins_data[sym] = {'rsi':0,'pressure':0,'exit_status':'Safe','exit_details':'-'}
            df = self.fetch_htf_candles(sym)
            if df is not None:
                info = ob_trend_strategy.check_exit_signal(df, side)
                self.active_coins_data[sym]['rsi'] = info['details'].get('rsi',0)
                if info['close']:
                    self.active_coins_data[sym]['exit_status'] = 'EXIT NOW'
                    self.active_coins_data[sym]['exit_details'] = info['reason']
                    if smart_exit:
                        logger.info(f"SMART EXIT: {sym}"); self.bot.place_order({"action":"Close","symbol":sym,"direction":"Long" if side=="Buy" else "Short"})
                else:
                    rsi = info['details'].get('rsi',50)
                    ll = float(settings.get('exit_rsiOverbought',70)); ls = float(settings.get('exit_rsiOversold',30))
                    stat = "Safe"
                    if side=="Buy" and rsi>=(ll-5): stat="Warning"
                    if side=="Sell" and rsi<=(ls+5): stat="Warning"
                    self.active_coins_data[sym]['exit_status'] = stat
                    self.active_coins_data[sym]['exit_details'] = f"RSI: {rsi}"
            time.sleep(0.5)
    def get_coin_data(self, s): return self.active_coins_data.get(s, {})
    def get_current_rsi(self, s): return self.active_coins_data.get(s,{}).get('rsi',0)
    def get_market_pressure(self, s): return self.active_coins_data.get(s,{}).get('pressure',0)