"""🧪 Віртуальний рахунок тестових угод + 📅 календар днів торгівлі + 🧪 панель
paper на інфосайті (вимоги 08.11).

Дослівно: «Додай в боті для тестових угод налаштування де можна керувати
угодою стосовно суми на угоду плеча і вказати розмір тестового депозиту…» ·
«Додай всю інформацію стосовно панелі і відкритих лише тестових угод в
інфосайт» · «Додай інформацію в календарі… якого дня були угоди… Для бота і
інфосайту».
"""
import json
import os
import subprocess
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_autosl_chain as _tmb  # noqa: E402

TM = _tmb.TM
tm_mod = sys.modules[TM.__module__]
paper_size = tm_mod.paper_size
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()
_FLASK = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
_IJS = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
_IHTML = open(os.path.join(_HERE, 'infosite', 'index.html'), encoding='utf-8').read()
_IPORT = open(os.path.join(_HERE, 'web', 'info_portal.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


S = {'paper_sizing_mode': 'fixed', 'paper_deposit': 1000.0, 'paper_margin_usd': 100.0,
     'paper_leverage': 10, 'paper_fee_pct': 0.05, 'paper_risk_pct': 2.0}


def test_fixed_size_needs_no_sl():
    z = paper_size(S, 50.0, None)
    _check(z['mode'] == 'fixed' and z['margin'] == 100.0 and z['notional'] == 1000.0, z)
    _check(abs(z['qty'] - 20.0) < 1e-9 and z['lev'] == 10 and z['cash'] == 1000.0, z)
    _check(paper_size(S, 0, None) is None, 'без ціни розміру немає')
    big = paper_size(dict(S, paper_margin_usd=5000), 50.0, None)
    _check(big['margin'] == 1000.0 and big['capped'], 'маржа не більша за депозит')
    print('✓ «сума на угоду»: маржа × плече, SL не потрібен (кінець $0.00)')


def test_risk_mode_and_fallback():
    z = paper_size(dict(S, paper_sizing_mode='risk'), 100.0, 98.0)   # SL 2%
    _check(z['mode'] == 'risk' and abs(z['notional'] - 1000.0) < 1e-6, z)   # 2% × 1000 / 2%
    _check(abs(z['risk_at_sl'] - 20.0) < 1e-6, z)
    f = paper_size(dict(S, paper_sizing_mode='risk'), 100.0, None)
    _check(f['mode'] == 'fixed' and f['fallback'] and f['notional'] == 1000.0, 'без SL → «сума на угоду»')
    print('✓ «ризик на SL» від депозиту; без SL — фолбек на суму на угоду')


def test_paper_pnl_uses_size_lev_and_fees():
    z = paper_size(S, 50.0, None)
    pp = TM._paper_pnl(z, 50.0, 51.0, 'LONG')          # +2% руху, qty 20
    _check(pp['gross_usd'] == 20.0, pp)
    _check(abs(pp['fees_usd'] - (20 * 50 * 0.0005 + 20 * 51 * 0.0005)) < 1e-6, pp)
    _check(pp['roi_margin_pct'] is not None and pp['net_pct_cash'] is not None, pp)
    print('✓ віртуальний $: брутто × плече − комісії вхід+вихід')


def _tm(settings, closed=(), opens=None):
    t = TM.__new__(TM)
    t._lock = threading.RLock()
    t._settings = dict(settings)
    t._shadow_closed = list(closed)
    t._closed_trades = []
    t._shadow_positions = dict(opens or {})
    t._get_current_price = lambda sym, cache_only=False: 52.0
    return t


def test_account_and_late_sizing():
    pos = {'side': 'LONG', 'entry_price': 50.0, 'opened_at': 1}
    t = _tm(S, closed=[{'closed_at': 10, 'pnl_pct': 1, 'paper_result': {'net_usd': 12.5}}],
            opens={'XUSDT': pos})
    late = TM._paper_size_of(t, pos)
    _check(late and late.get('late') and late['notional'] == 1000.0, 'стара угода без sz → поточні налаштування')
    acc = TM.paper_account(t)
    _check(acc['deposit'] == 1000.0 and acc['realized_usd'] == 12.5 and acc['balance'] == 1012.5, acc)
    _check(acc['unrealized_usd'] > 0 and acc['equity'] > acc['balance'] and acc['margin_used'] == 100.0, acc)
    _check(acc['leverage'] == 10 and acc['margin_usd'] == 100.0, acc)
    print('✓ віртуальний рахунок: депозит + реалізований + нереалізований')


def test_trade_days_and_route():
    t = _tm(S, closed=[{'closed_at': 100.0, 'pnl_pct': 1.0, 'paper_result': {'net_usd': 2.0}},
                       {'closed_at': 0, 'pnl_pct': 1.0}])
    items = TM.trade_days(t, shadow=True)
    _check(items == [[100.0, 2.0, 1.0]], items)
    _check("@app.route('/api/tm/trade-days')" in _FLASK, 'маршрут днів для бота')
    _check("@app.route('/api/info/paper/days')" in _IPORT and "@app.route('/api/info/paper')" in _IPORT, 'маршрути інфосайту')
    print('✓ дні торгівлі: [час, $, %] з тих самих записів')


def _js_from(src, names):
    out = []
    for n in names:
        key = f'function {n}('
        i = src.index(key)
        end = '\n}' if src[i - 1] == '\n' else '\n  }'
        j = src.index(end, i)
        out.append(src[i:j + len(end)])
    return '\n'.join(out)


def test_calendar_bucket_by_local_day_in_both_pages():
    for src, nm in ((_HTML, 'бот'), (_IJS, 'інфосайт')):
        js = _js_from(src, ('tcalYmd', 'tcalBucket')) + r'''
        const t1 = new Date(2026, 10, 8, 10, 0).getTime() / 1000;
        const t2 = new Date(2026, 10, 8, 23, 59).getTime() / 1000;
        const t3 = new Date(2026, 10, 7, 0, 1).getTime() / 1000;
        console.log(JSON.stringify(tcalBucket([[t1, 2, 1], [t2, -5, -2], [t3, 0, 0.5], [0, 1, 1]])));'''
        r = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=20)
        _check(r.returncode == 0, nm + ': ' + r.stderr)
        o = json.loads(r.stdout)
        d8 = o['2026-11-08']
        _check(d8['n'] == 2 and d8['w'] == 1 and d8['l'] == 1 and d8['usd'] == -3, (nm, o))
        _check(o['2026-11-07']['w'] == 1, (nm, 'без $ — рахуємо за %'))
        _check(len(o) == 2, (nm, 'нульовий час ігноруємо'))
    print('✓ календар: дні за годинником ПК, лічильник і результат дня — в обох сторінках')


def test_pages_use_the_calendar_and_paper_controls():
    _check('type="date"' not in _HTML.split('id="tm-stats-day"')[0][-200:], 'рідний date-input замінено')
    for x in ('tm', 'tmp'):
        for f in ('day', 'from', 'to'):
            _check(f'id="{x}-stats-{f}" onclick="tmCalOpen(this' in _HTML, f'{x}-{f}: календар')
    _check('/api/tm/trade-days?book=' in _HTML, 'бот питає дні торгівлі')
    for i in ('tm-paper-deposit', 'tm-paper-mode', 'tm-paper-margin', 'tm-paper-lev'):
        _check(f'id="{i}"' in _HTML, i)
    _check('paper_deposit:' in _HTML and 'paper_leverage:' in _HTML, 'збереження нових ключів')
    _check("'Тестовий баланс'" in _HTML and 'paperPnlCellHTML(p)' in _HTML, 'картка балансу + PnL $ у таблиці')
    _check('id="pp-card"' in _IHTML and 'class="pp-dateinp"' in _IHTML, 'панель paper на інфосайті')
    _check('/api/info/paper' in _IJS and 'window.tcalPick = tcalPick' in _IJS, 'інфосайт кличе paper і календар')
    print('✓ бот і інфосайт: календар, налаштування paper, панель')


def test_margin_as_percent_of_deposit_0811():
    """🧪 Вимога 08.11: маржа = % депозиту (деф. 3%) · рядок кольорами."""
    _check(tm_mod.DEFAULT_SETTINGS['paper_sizing_mode'] == 'pct'
           and tm_mod.DEFAULT_SETTINGS['paper_margin_pct'] == 3.0, 'дефолт: 3% депозиту')
    z = paper_size({'paper_deposit': 1000.0, 'paper_leverage': 20, 'paper_fee_pct': 0.055}, 50.0)
    _check(z['mode'] == 'pct' and z['margin'] == 30.0 and z['notional'] == 600.0
           and z['margin_pct'] == 3.0, z)
    z = paper_size(dict(S, paper_sizing_mode='pct', paper_margin_pct=10), 50.0, None)
    _check(z['margin'] == 100.0 and z['notional'] == 1000.0, z)
    _check(tm_mod.paper_margin_pct_of({'paper_margin_pct': 0}) == 3.0
           and tm_mod.paper_margin_pct_of({'paper_margin_pct': 500}) == 100.0, 'межі %')
    acc = TM.paper_account(_tm(dict(S, paper_sizing_mode='pct', paper_margin_pct=3)))
    _check(acc['mode'] == 'pct' and acc['margin_usd'] == 30.0 and acc['margin_pct'] == 3.0, acc)
    # міграція збереженого 'fixed' → 'pct' разово; 'risk' не чіпаємо
    src = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()
    _check("paper_pct_migrated_v1" in src.split('def _load_settings')[1].split('def ')[0], 'міграція')
    _check('<option value="pct">% балансу</option>' in _HTML and 'id="tm-paper-mpct"' in _HTML
           and 'paper_margin_pct:' in _HTML, 'UI бота: режим і поле %')
    _check('ppInlineHTML(sa)' in _IJS and 'ppi-usd' in _IJS and 'ppi-pct' in _IJS, 'рядок кольорами')
    for i in ('tmpq-dep', 'tmpq-mpct', 'tmpq-lev', 'tmpq-fee'):
        _check(f'id="{i}"' in _HTML, i)
    _check('function tmpqApply()' in _HTML and 'saveTMSettings()' in _HTML.split('function tmpqApply()')[1][:1500],
           'швидкий рядок пише ті самі налаштування')
    _check(_HTML.index('id="tmpq-row"') < _HTML.index('id="tmp-stats-wrap"'), 'рядок над панеллю Paper')
    print('✓ маржа = % депозиту + кольоровий рядок + вибір у боті')


def test_margin_pct_is_from_balance_at_open_0811():
    """🧪 % угоди рахується від НАЯВНОГО балансу на момент відкриття, а не
    від стартового депозиту (вимога 08.11, друга правка)."""
    base = {'paper_deposit': 1000.0, 'paper_leverage': 20, 'paper_margin_pct': 3}
    z = paper_size(base, 50.0, None, balance=1200.0)
    _check(z['margin'] == 36.0 and z['notional'] == 720.0 and z['cash'] == 1200.0, z)
    z = paper_size(base, 50.0, None, balance=800.0)
    _check(z['margin'] == 24.0, 'збиток зменшує наступну угоду')
    _check(paper_size(base, 50.0, None, balance=None)['margin'] == 30.0, 'без балансу → депозит')
    _check(paper_size(base, 50.0, None, balance=-5)['margin'] == 30.0, 'сміття → депозит')
    t = _tm(dict(S, paper_sizing_mode='pct', paper_margin_pct=3),
            closed=[{'closed_at': 10, 'pnl_pct': 1, 'paper_result': {'net_usd': 200.0}}])
    _check(TM._paper_balance(t) == 1200.0, 'баланс = депозит + закриті')
    _check(TM._paper_sizing(t, 50.0, None)['margin'] == 36.0, 'відкриття бере наявний баланс')
    acc = TM.paper_account(t)
    _check(acc['balance'] == 1200.0 and acc['margin_usd'] == 36.0, acc)
    _check('_tmPaperBase()' in _HTML and 'id="tmpq-bal"' in _HTML, 'бот показує баланс')
    _check('usd(a.balance)' in _IJS, 'інфосайт показує баланс')
    print('✓ % угоди — від наявного балансу на момент відкриття')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
