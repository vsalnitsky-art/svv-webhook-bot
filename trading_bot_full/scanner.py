import threading, time, logging, pandas as pd, pandas_ta as ta
from settings_manager import settings
logger = logging.getLogger(__name__)
class EnhancedMarketScanner:
    def __init__(self, bot, cfg): self.bot = bot; self.data = {}; threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while True:
            try:
                pos = self.bot.session.get_positions(category="linear", settleCoin="USDT")['result']['list']
                actives = [p['symbol'] for p in pos if float(p['size']) > 0]
                if not actives: self.data = {}; time.sleep(5); continue
                tickers = self.bot.get_all_tickers()
                for t in tickers:
                    sym = t['symbol']
                    if sym not in actives: continue
                    if sym not in self.data: self.data[sym] = {'rsi': 50, 'pressure': 0}
                    self.data[sym]['rsi'] = self.fetch_rsi(sym)
                    price, turn = float(t['lastPrice']), float(t['turnover24h'])
                    diff = turn - self.data[sym].get('prev_turn', turn)
                    if diff > 0:
                        d = 1 if price >= self.data[sym].get('prev_price', price) else -1
                        self.data[sym]['pressure'] = self.data[sym]['pressure']*0.9 + (diff*d)
                    self.data[sym]['prev_turn'] = turn; self.data[sym]['prev_price'] = price
                    time.sleep(0.2)
            except: pass
            time.sleep(5)
    def fetch_rsi(self, sym):
        try:
            tf = settings.get("ltfSelection") or "15"
            r = self.bot.session.get_kline(category="linear", symbol=sym, interval=str(tf), limit=30)
            df = pd.DataFrame(r['result']['list'], columns=['t','o','h','l','c','v','to'])
            return round(ta.rsi(df.iloc[::-1]['c'].astype(float), length=14).iloc[-1], 1)
        except: return 50
    def get_current_rsi(self, s): return self.data.get(s, {}).get('rsi', 50)
    def get_market_pressure(self, s): return self.data.get(s, {}).get('pressure', 0)
