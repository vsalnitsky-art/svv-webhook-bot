"""🔄 Реверс протилежним сигналом (вимога 30.10, черги вимкнені).

Дослівно: «протилежна угода, яка вже пройшла всі ворота і має відкритись, а є
відкрита протилежна — потрібно повністю все закрити на біржі вже відкритої угоди
і аж потім продовжити відкриття нової (при умові, що тумблер увімкнений)».

Правила, які стереже тест:
  • стару позицію закриваємо НА БІРЖІ кількістю З БІРЖІ і чекаємо, поки біржа
    підтвердить нуль; лише потім — відкриття нової;
  • біржа не підтвердила → нову НЕ відкриваємо, стара лишається в книзі;
  • ордер закриття не дублюється (облік іде з exchange_done=True);
  • стеля Max Open Positions реверс не блокує;
  • тумблер вимкнено → стара тримається, нічого не закриваємо.
"""
import sys, types, importlib.util
sys.path.insert(0, '.')


class _DB:
    def __init__(self): self.store = {}
    def get_setting(self, k, default=None): return self.store.get(k, default)
    def set_setting(self, k, v): self.store[k] = v


storage_pkg = types.ModuleType('storage'); storage_pkg.__path__ = ['./storage']
sys.modules.setdefault('storage', storage_pkg)
db_mod = types.ModuleType('storage.db_operations')
db_mod.get_db = lambda: _DB()
sys.modules['storage.db_operations'] = db_mod
det = types.ModuleType('detection'); det.__path__ = ['./detection']
sys.modules['detection'] = det
for name in ['detection.market_data', 'detection.smc_scanner', 'detection.smc_structure',
             'detection.position_evaluator', 'detection.ob_detector',
             'core.bybit_connector', 'core.telegram_notifier', 'core', 'web']:
    sys.modules.setdefault(name, types.ModuleType(name))
_LOG = []
al = types.ModuleType('detection.activity_log')
al.log_activity = lambda sym, kind, text, **kw: _LOG.append((kind, text))
sys.modules['detection.activity_log'] = al
ffm = types.ModuleType('detection.fuel_filter')
ffm.get_fuel_filter = lambda: None
sys.modules['detection.fuel_filter'] = ffm

spec = importlib.util.spec_from_file_location('detection.trade_manager', 'detection/trade_manager.py')
tm_mod = importlib.util.module_from_spec(spec)
sys.modules['detection.trade_manager'] = tm_mod
spec.loader.exec_module(tm_mod)
TradeManager = tm_mod.TradeManager


class _Bybit:
    """Біржа: позиція зникає після N ордерів закриття (або ніколи)."""
    def __init__(self, size=3.0, side='Buy', closes_needed=1):
        self.size, self.side, self.need = size, side, closes_needed
        self.events = []
    def get_wallet_balance(self): return 1000
    def get_price(self, s): return None
    def set_leverage(self, *a, **kw): pass
    def place_order(self, **kw):
        self.events.append(('order', kw)); return {'order_id': 'X'}
    def get_positions_checked(self, symbol=None):
        self.events.append(('check', symbol))
        if self.size > 0:
            return [{'symbol': 'BTCUSDT', 'side': self.side, 'size': self.size}], True
        return [], True
    def close_position(self, symbol, side, qty):
        self.events.append(('close', side, qty))
        self.need -= 1
        if self.need <= 0:
            self.size = 0.0


def _mk(bybit, reverse=True, at_limit=False):
    _LOG.clear()
    tm = TradeManager(db=_DB(), notifier=None, bybit=bybit)
    tm._settings.update({'enabled': True, 'use_reverse_signal': reverse, 'test_mode': False})
    tm.REVERSE_FLAT_WAIT_SEC = 0
    tm._positions['BTCUSDT'] = {'symbol': 'BTCUSDT', 'side': 'LONG', 'entry_price': 100.0,
                                'qty': 2.0, 'remaining_qty': 2.0,   # книга бота ≠ біржа
                                'opened_at': 1, 'opened_by': 'choch'}
    tm.is_enabled = lambda: True
    tm._is_tradeable = lambda s: True
    tm._at_max_positions = lambda: at_limit
    tm.real_open_possible = lambda s: True
    tm._get_current_price = lambda s, **kw: 101.0
    tm._notify = lambda *a, **kw: None
    opened = []
    def _open(symbol, side, price, opened_by=None, **kw):
        opened.append((symbol, side, bybit.size))
        return {'ok': True}
    tm._open_position = _open
    return tm, opened


def test_closes_fully_on_exchange_then_opens():
    b = _Bybit(size=3.0)
    tm, opened = _mk(b)
    r = tm.on_signal('BTCUSDT', side='SHORT', entry_price=100.0, opened_by='choch')
    closes = [e for e in b.events if e[0] == 'close']
    assert closes == [('close', 'Buy', 3.0)], closes          # кількість — З БІРЖІ (3), не з книги (2)
    assert opened == [('BTCUSDT', 'SHORT', 0.0)], opened      # відкриття — коли на біржі вже 0
    assert r.get('real_opened') is True, r
    assert 'BTCUSDT' not in tm._positions or tm._positions['BTCUSDT'].get('side') != 'LONG'
    assert not [e for e in b.events if e[0] == 'order'], 'ордер закриття не дублюється'
    assert tm._closed_trades[-1]['reason'] == 'reverse_signal'


def test_waits_until_exchange_confirms():
    b = _Bybit(size=3.0, closes_needed=3)                     # біржа «дотягує» за 3 спроби
    tm, opened = _mk(b)
    tm.on_signal('BTCUSDT', side='SHORT', entry_price=100.0, opened_by='choch')
    assert len([e for e in b.events if e[0] == 'close']) == 3
    assert opened and opened[0][2] == 0.0


def test_no_open_when_exchange_never_confirms():
    b = _Bybit(size=3.0, closes_needed=10 ** 6)
    tm, opened = _mk(b)
    r = tm.on_signal('BTCUSDT', side='SHORT', entry_price=100.0, opened_by='choch')
    assert opened == [], 'нову угоду НЕ відкриваємо'
    assert r['status'] == 'rejected', r
    assert tm._positions['BTCUSDT']['side'] == 'LONG', 'стара лишається в книзі'
    assert any('Реверс зупинено' in t for _, t in _LOG), _LOG


def test_position_limit_does_not_block_reverse():
    b = _Bybit(size=3.0)
    tm, opened = _mk(b, at_limit=True)
    tm.on_signal('BTCUSDT', side='SHORT', entry_price=100.0, opened_by='choch')
    assert opened and opened[0][1] == 'SHORT'


def test_switch_off_holds_the_position():
    b = _Bybit(size=3.0)
    tm, opened = _mk(b, reverse=False)
    r = tm.on_signal('BTCUSDT', side='SHORT', entry_price=100.0, opened_by='choch')
    assert opened == [] and not [e for e in b.events if e[0] == 'close']
    assert r['status'] == 'duplicate' and tm._positions['BTCUSDT']['side'] == 'LONG'


def test_flat_state_is_pure():
    f = TradeManager.reverse_flat_state
    assert f([], 'BTCUSDT') == []
    assert f([{'symbol': 'ETHUSDT', 'side': 'Buy', 'size': 1}], 'BTCUSDT') == []
    assert f([{'symbol': 'btcusdt', 'side': 'Sell', 'size': '2.5'}], 'BTCUSDT') == [('Sell', 2.5)]
    assert f([{'symbol': 'BTCUSDT', 'side': 'Buy', 'size': 0}], 'BTCUSDT') == []


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); print('✓', k); n += 1
    print(f'OK — {n} тестів')
