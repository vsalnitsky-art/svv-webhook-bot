"""✋ Кнопка «Закрити всі» над таблицею Open Paper Positions (вимога 30.10)
+ 💵 у Telegram немає сум прибутку/збитку.

  • закриваються ВСІ paper-позиції, реальні не чіпаються;
  • маршрут і кнопка існують і звʼязані;
  • текст External close не несе суми в $.
"""
import re, sys, types, importlib.util
sys.path.insert(0, '.')


class _DB:
    def __init__(self): self.store = {}
    def get_setting(self, k, default=None): return self.store.get(k, default)
    def set_setting(self, k, v): self.store[k] = v


storage_pkg = types.ModuleType('storage'); storage_pkg.__path__ = ['./storage']
sys.modules.setdefault('storage', storage_pkg)
db_mod = types.ModuleType('storage.db_operations'); db_mod.get_db = lambda: _DB()
sys.modules['storage.db_operations'] = db_mod
det = types.ModuleType('detection'); det.__path__ = ['./detection']
sys.modules['detection'] = det
for name in ['detection.market_data', 'detection.smc_scanner', 'detection.smc_structure',
             'detection.position_evaluator', 'detection.ob_detector',
             'core.bybit_connector', 'core.telegram_notifier', 'core', 'web']:
    sys.modules.setdefault(name, types.ModuleType(name))
spec = importlib.util.spec_from_file_location('detection.trade_manager', 'detection/trade_manager.py')
tm_mod = importlib.util.module_from_spec(spec)
sys.modules['detection.trade_manager'] = tm_mod
spec.loader.exec_module(tm_mod)
TradeManager = tm_mod.TradeManager


def test_close_all_closes_only_paper():
    tm = TradeManager(db=_DB(), notifier=None, bybit=None)
    tm._positions['BTCUSDT'] = {'symbol': 'BTCUSDT', 'side': 'LONG', 'entry_price': 1.0}
    for s in ('ETHUSDT', 'XRPUSDT'):
        tm._shadow_positions[s] = {'symbol': s, 'side': 'SHORT', 'entry_price': 2.0}
    closed = []
    tm._get_current_price = lambda s, **kw: 2.0
    def _cs(sym, px, reason=''):
        closed.append((sym, reason)); tm._shadow_positions.pop(sym, None)
    tm._close_shadow = _cs
    r = tm.manual_close_all_shadow()
    assert r['ok'] and sorted(r['closed']) == ['ETHUSDT', 'XRPUSDT'] and not r['failed']
    assert all(x[1] == 'manual' for x in closed)
    assert 'BTCUSDT' in tm._positions, 'реальна угода не чіпається'


def test_route_and_button_exist():
    app = open('web/flask_app.py', encoding='utf-8').read()
    html = open('templates/smart_money.html', encoding='utf-8').read()
    assert "'/api/tm/shadow/close-all'" in app and 'manual_close_all_shadow' in app
    assert 'onclick="closeAllTMShadow()"' in html
    assert "fetch('/api/tm/shadow/close-all'" in html


def test_external_close_message_has_no_usd_amount():
    src = open('detection/trade_manager.py', encoding='utf-8').read()
    i = src.index('def _ext_notify')
    body = src[i:src.index('_finalize_close_async(symbol, closed', i)]
    assert 'pnl_usd' not in body and '$' not in body


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); print('✓', k); n += 1
    print(f'OK — {n} тестів')
