"""👤 Особистий Trade Manager користувача + розділення ключів (вимога 24.11).

Дослівно: «Ключі адміністратора бота — то для роботи і торгівлі бота, а ключі
користувачів — це зовсім окремі ключі… працюють лише на цього і для цього
користувача» · «Trade Manager — має бути у кожного користувача свій… дзеркалом
бота, але виконувати торгівлю саме під конкретного користувача» · «Користувач
сам контролює відкриття та додавання угод. Можливість вмикати дублювання угод
бота (контроль доступу від адміністратора)» · «зʼявиться на інфосайті тільки
після підключення ключів апі» · «спочатку це має одобрити адміністратор».

python3 test_user_tm.py
"""
import ast
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


class _DB:
    def __init__(self):
        self.store = {}
        self.archived = []

    def get_setting(self, k, d=None):
        return self.store.get(k, d)

    def set_setting(self, k, v):
        self.store[k] = v

    def archive_trade(self, *a, **k):
        self.archived.append(a)


_BASE = _DB()
for name, path in (('storage', 'storage'), ('detection', 'detection'), ('web', 'web'),
                   ('core', 'core'), ('config', 'config')):
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(ROOT, path)]
        sys.modules[name] = m
dbm = types.ModuleType('storage.db_operations')
dbm.get_db = lambda: _BASE
sys.modules['storage.db_operations'] = dbm
for n in ['detection.market_data', 'detection.smc_scanner', 'detection.smc_structure',
          'detection.position_evaluator', 'detection.ob_detector', 'core.telegram_notifier']:
    sys.modules.setdefault(n, types.ModuleType(n))


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


AL = _load('detection.activity_log', 'detection/activity_log.py')
TMM = _load('detection.trade_manager', 'detection/trade_manager.py')
UTM = _load('detection.user_tm', 'detection/user_tm.py')
UP = _load('web.user_perms', 'web/user_perms.py')


def _src(rel):
    return open(os.path.join(ROOT, rel), encoding='utf-8').read()


def _fn(rel, name):
    tree = ast.parse(_src(rel))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return ast.get_source_segment(_src(rel), n)
    raise AssertionError(name)


class _Bybit:
    def __init__(self, key):
        self.api_key = key
        self.session = None
        self.orders = []

    def get_positions(self, symbol=None):
        return []

    def get_wallet_balance(self):
        return 1000

    def get_price(self, s):
        return None

    def set_leverage(self, *a):
        pass

    def place_order(self, **kw):
        self.orders.append(kw)
        return {'order_id': 'X'}


def _utm(uid=7, key='USERKEY'):
    sent = []
    t = UTM.UserTradeManager(uid, 'u@x', db=_BASE, scanner=None, bybit=_Bybit(key))
    t._send_personal = lambda m: sent.append(m)
    t._personal_notify = t._send_personal
    t._get_current_price = lambda s, cache_only=False: 100.0
    t._perms = lambda: {'books': ('paper', 'real'), 'mirror': True}
    t._settings.update({'enabled': True, 'test_mode': True, 'sizing_mode': 'fixed_usd',
                        'fixed_usd_amount': 50.0, 'leverage': 5})
    UTM._reg[int(uid)] = t
    return t, sent


# ── 🔐 ключі ───────────────────────────────────────────────────────────────
def test_connector_for_user_never_reloads_bot_keys():
    src = _src('core/bybit_connector.py')
    assert 'def for_keys(' in src
    rk = src[src.index('def reload_keys'):src.index('def reload_keys') + 900]
    assert "getattr(self, '_owner', 'bot') == 'user'" in rk
    assert rk.index("== 'user'") < rk.index('get_bybit_keys')


def test_user_keys_are_per_owner_and_refuse_bot_keys():
    src = _src('web/user_keys.py')
    assert "KEY_FMT = 'user_exkeys_v1_{uid}'" in src
    assert "int(rec.get('uid')) != int(uid)" in src          # чужий запис не читається
    assert 'is_bot_key(api_key)' in _fn('web/user_keys.py', 'save')
    assert 'bybit_api_key_encrypted' not in src               # ключі бота тут не пишуться
    assert 'get_connector' not in src                         # конектор бота не чіпає


def test_keys_route_split_by_owner():
    src = _src('web/user_trading.py')
    ko = _fn('web/user_trading.py', 'keys_owner')
    assert "return 'bot'" in ko and "'user' if user_books(u)" in ko
    km = src[src.index('def _keys_mod'):src.index('def _out')]
    assert 'exchange_keys' in km and 'user_keys' in km
    # ключі користувача викликаються ЛИШЕ з його uid
    assert 'm.save(u.id, k, sc)' in src and 'm.status(u.id)' in src


def test_user_trades_never_go_to_bot_tm():
    plan = _fn('web/user_trading.py', 'trade_plan')
    assert 'is_admin_sub(t)' in plan
    import importlib
    UT = _load('web.user_trading_t2', 'web/user_trading.py')
    subs = [(1, 'adm@x', {'enabled': True, 'coins': {'BTCUSDT': 'real'}, 'admin': True}),
            (2, 'u@x', {'enabled': True, 'coins': {'BTCUSDT': 'real'}, 'admin': False})]
    assert UT.trade_plan(subs, 'BTCUSDT') == {'real': ['adm@x']}
    assert UT.user_plan(subs, 'BTCUSDT') == [(2, 'u@x', 'real')]
    assert 'admin=bool(u.is_admin)' in _src('web/auth.py')


# ── 👤 ізоляція TM ─────────────────────────────────────────────────────────
def test_scoped_db_and_no_bot_archive():
    t, _ = _utm(11)
    t._persist_settings()
    assert 'u11:tm_settings' in _BASE.store
    assert 'tm_settings' not in _BASE.store or _BASE.store.get('tm_settings') is not t._settings
    n0 = len(_BASE.archived)
    t.db.archive_trade({'x': 1})
    assert len(_BASE.archived) == n0                          # архів бота не наповнюється


def test_user_open_logs_privately_and_notifies_only_owner():
    t, sent = _utm(12)
    bot_log = AL.get_activity_log()
    bot_log._enabled = True
    bot_log._loaded = True
    bot_log._flush = lambda *a, **k: None
    n0 = bot_log.count()
    r = UTM.execute(12, 'u@x', 'BTCUSDT', 'LONG', 'paper')
    assert r.get('ok'), r
    assert 'BTCUSDT' in t._shadow_positions
    assert any('Кабінет' in (e.get('detail') or '') for e in t.user_log(50))
    assert bot_log.count() == n0                              # спільний 🧾 Лог бота не чіпали
    sent.clear()
    t._notify('груповий текст', category='trades')            # у групу — ніколи
    t._notify('службове')                                     # адміну — ніколи
    assert sent == ['груповий текст', 'службове']
    assert "'👤 <b>Мій Trade Manager</b>" in _fn('detection/user_tm.py', '_send_personal')


def test_real_trade_uses_user_connector():
    t, _ = _utm(13, key='OWNKEY13')
    t._settings['test_mode'] = False
    r = UTM.execute(13, 'u@x', 'ETHUSDT', 'SHORT', 'real')
    assert r.get('ok'), r
    assert t.bybit.api_key == 'OWNKEY13' and t.bybit.orders
    assert t.bybit.orders[0]['side'] == 'Sell'


def test_manual_add_and_close_by_user():
    t, _ = _utm(14)
    UTM.execute(14, 'u@x', 'SOLUSDT', 'LONG', 'paper')
    q0 = t._shadow_positions['SOLUSDT'].get('cab_avg_n') or 0
    r = UTM.hand_add(14, 'u@x', 'SOLUSDT', 'paper')
    assert r.get('ok'), r
    assert t._shadow_positions['SOLUSDT']['cab_avg_n'] == q0 + 1
    r = UTM.hand_close(14, 'u@x', 'SOLUSDT', 'paper')
    assert 'SOLUSDT' not in t._shadow_positions


def test_mirror_needs_admin_permission():
    assert UTM.mirror_plan({'mirror_signals': True, 'mirror_book': 'paper'}, ('paper',),
                           'open', False, allowed=False) is None
    assert UTM.mirror_plan({'mirror_signals': True, 'mirror_book': 'paper'}, ('paper',),
                           'open', False, allowed=True) == 'paper'
    assert UTM.mirror_plan({'mirror_signals': False}, ('paper',), 'open', False) is None
    assert UTM.mirror_plan({'mirror_signals': True, 'mirror_book': 'real'}, ('paper',),
                           'open', False) is None
    assert UP.mirror_ok({'trade_mirror': True, 'trade_paper': True})
    assert not UP.mirror_ok({'trade_mirror': True})
    assert 'trade_mirror' in UP.KEYS


def test_mirror_opens_and_closes_copy():
    t, _ = _utm(15)
    t.set_meta({'mirror_signals': True, 'mirror_book': 'paper'})
    UTM._mirror_one(t, 'open', {'symbol': 'ADAUSDT', 'side': 'LONG', 'opened_by': 'choch → Q4'}, False)
    p = t._shadow_positions['ADAUSDT']
    assert p['mirror_bot']['src'] == 'real' and p['opened_by'].endswith('MIR')
    UTM._mirror_one(t, 'close', {'symbol': 'ADAUSDT', 'side': 'LONG'}, True)   # інша книга бота
    assert 'ADAUSDT' in t._shadow_positions
    UTM._mirror_one(t, 'close', {'symbol': 'ADAUSDT', 'side': 'LONG'}, False)
    assert 'ADAUSDT' not in t._shadow_positions


def test_mirror_settings_keep_personal():
    own = {'enabled': True, 'leverage': 3, 'use_tp': False}
    bot = {'enabled': False, 'leverage': 20, 'use_tp': True, 'tp_pct': 7.0}
    m = UTM.mirror_merge(own, bot)
    assert m['enabled'] is True and m['leverage'] == 3       # особисте сильніше
    assert m['use_tp'] is True and m['tp_pct'] == 7.0         # алгоритм — як у бота


def test_tm_appears_only_after_approval_and_keys():
    en = _fn('detection/user_tm.py', 'ensure')
    assert '(approved(uid) and has_keys(uid)) or has_book_data(uid)' in en
    g = _src('web/user_trading.py')
    assert "'need_keys': True" in g
    js = _src('infosite/js/app.js')
    assert '(TR.tm || {}).need_keys' in js and 'Підключіть <b>свої</b> ключі API' in js


def test_bot_tm_does_not_touch_ff_or_ai_from_user_tm():
    src = _src('detection/trade_manager.py')
    assert src.count("_ff = None if self.owner_uid else get_fuel_filter()") == 2
    assert 'if not self.owner_uid and (self._ai_owned(pos)' in src
    assert "if self.owner_uid:\n            fn = getattr(self, '_personal_notify'" in src
    assert UTM.UserTradeManager._ai_owned({'opened_by': 'ai'}) is False


def test_price_feed_watch_is_per_owner():
    PF = _load('detection.price_feed_t', 'detection/price_feed.py')
    f = PF.PriceFeed(lambda: {})
    f.start = lambda: None
    f.watch({'BTCUSDT'})
    f.watch({'XRPUSDT'}, owner='#u1')
    assert f._watch == {'BTCUSDT', 'XRPUSDT'}
    f.watch(set(), owner='#u1')
    assert f._watch == {'BTCUSDT'}


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for f in fns:
        f()
        print('✓', f.__name__)
    print(f'OK {len(fns)}')
