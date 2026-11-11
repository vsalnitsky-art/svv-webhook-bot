"""💥 Вихід «Пружина вистрілила» + живий процес вистрілу (вимога 10.11).

Запуск: python3 test_spring_fired_exit.py
"""
import importlib.util
import os
import random
import subprocess
import sys
import time
import types

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
_sp = importlib.util.spec_from_file_location('spring_coil_fx', os.path.join(_ROOT, 'detection', 'spring_coil.py'))
S = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(S)
_TM = open(os.path.join(_ROOT, 'detection', 'trade_manager.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
_APP = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
_IP = open(os.path.join(_ROOT, 'web', 'info_portal.py'), encoding='utf-8').read()
_JS = open(os.path.join(_ROOT, 'infosite', 'app.js'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _coil(seed=1, n=240):
    random.seed(seed)
    p, out = 100.0, []
    for i in range(n):
        if i < n - 60:
            p *= 1 + random.gauss(0.001, 0.01)
            amp, vol = 0.012, 1000
        else:
            p *= 1 + 0.0003 + (0.0015 if i % 2 else -0.0015)
            amp, vol = 0.003, 500
        out.append({'t': 1_700_000_000_000 + i * 3600000, 'o': p, 'h': p * (1 + amp), 'l': p * (1 - amp), 'p': p, 'v': vol})
    return out


def _fire(b, up=True, k=1.04):
    last = b[-1]['p']
    if up:
        b[-1] = dict(b[-1], p=last * k, h=last * (k + 0.005), l=last * 0.999, v=5000)
    else:
        b[-1] = dict(b[-1], p=last / k, l=last / (k + 0.005), h=last * 1.001, v=5000)
    return b


def test_live_bar_uses_the_same_rule_as_closed_bars():
    b = _fire(_coil())
    _check(S.analyze(b)['state'] == 'fired', 'закриті бари: вистріл')
    f = S.live_fire(b)
    _check(f and f['side'] == 'LONG' and f['live'] and f['t'] == b[-1]['t'] // 1000, f)
    _check(S.live_fire(_coil()) is None, 'без пробою вистрілу немає')
    d = S.live_fire(_fire(_coil(), up=False))
    _check(d and d['side'] == 'SHORT', d)


def test_live_view_shows_progress_of_each_condition():
    vw = S.live_fire_view(_coil())
    _check(vw and vw['fired'] is None and vw['beyond'] is None, vw)
    _check(vw['range_x'] < vw['range_need'] and vw['pre_sq'] >= 0 and vw['sq_need'] == S.FIRE_MIN_SQ, vw)
    vf = S.live_fire_view(_fire(_coil()))
    _check(vf['fired'] and vf['beyond'] == 'LONG' and vf['range_x'] > vf['range_need']
           and vf['vol_x'] > vf['vol_need'], vf)
    _check(vf['pct'] == 100.0, 'вистріл = 100% виснаження')
    _check(0 <= vw['pct'] < 100 and set(vw['parts']) == {'range', 'vol', 'box', 'sq'}
           and abs(vw['pct'] - sum(vw['parts'].values()) / 4) < 0.2, vw)


def test_analyze_unchanged_by_refactor():
    b = _fire(_coil())
    r = S.analyze(b)
    _check(r['fired']['bars_ago'] == 0 and r['fired']['side'] == 'LONG' and 'pre_sq' in r['fired'], r)


_TMC = []


def _tm_cls():
    if _TMC:
        return _TMC[0]
    for n in ('pybit', 'pybit.unified_trading'):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules['pybit.unified_trading'].HTTP = object
    pkg = types.ModuleType('detection'); pkg.__path__ = [os.path.join(_ROOT, 'detection')]
    sys.modules['detection'] = pkg

    def _load(name, rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
        spec.loader.exec_module(mod); return mod
    _load('detection.signal_labels', 'detection/signal_labels.py')
    _load('detection.setup_grader', 'detection/setup_grader.py')
    lg = types.ModuleType('detection.activity_log'); lg.log_activity = lambda *a, **k: None
    sys.modules['detection.activity_log'] = lg
    db = types.ModuleType('storage.db_operations'); db.get_db = lambda: None
    st = types.ModuleType('storage'); st.__path__ = [os.path.join(_ROOT, 'storage')]
    sys.modules.setdefault('storage', st); sys.modules['storage.db_operations'] = db
    ffm = types.ModuleType('detection.fuel_filter'); ffm.get_fuel_filter = lambda: None
    sys.modules['detection.fuel_filter'] = ffm
    T = _load('detection.trade_manager', 'detection/trade_manager.py')
    _TMC.append(T.TradeManager)
    return T.TradeManager


def test_hit_rules():
    TM = _tm_cls()
    oa = 10_000.0
    f = {'side': 'SHORT', 't': 9_000}           # бар 9000..12600 закінчується після входу
    _check(TM.spring_fire_hit('LONG', oa, f) == 'SHORT|9000', 'вистріл після входу → вихід')
    _check(TM.spring_fire_hit('LONG', oa, {'side': 'SHORT', 't': 6_000}) is None, 'вистріл ДО входу — не подія')
    _check(TM.spring_fire_hit('LONG', oa, f, 'with') is None, 'with: проти угоди не закриває')
    _check(TM.spring_fire_hit('LONG', oa, f, 'against') == 'SHORT|9000', 'against')
    _check(TM.spring_fire_hit('SHORT', oa, f, 'against') is None, 'against: за угодою не закриває')
    _check(TM.spring_fire_hit('LONG', oa, f, base='SHORT|9000') is None, 'база на вході не закриває')
    _check(TM.spring_fire_hit('LONG', 0, f) is None and TM.spring_fire_hit('', oa, f) is None, 'немає входу/боку')
    _check(TM.spring_fire_hit('LONG', oa, None) is None, 'немає вистрілу')


def _tm(fire):
    TM = _tm_cls()
    tm = TM.__new__(TM)
    tm._settings = {'use_spring_fired_exit': True, 'spring_fired_exit_dir': 'any'}
    tm._persist_book = lambda is_shadow: None
    tm._spring_fire_view = lambda sym, fetch=True: {'fired': fire} if fire else {}
    return tm


def test_reason_and_entry_baseline():
    now = time.time()
    # Угода щойно відкрилась, а вистріл уже є → база, не закриваємо.
    f0 = {'side': 'LONG', 't': int(now) - 1800, 'level': 1.0, 'pre_sq': 7, 'live': True}
    tm = _tm(f0)
    pos = {'side': 'SHORT', 'opened_at': now - 30}
    _check(tm._spring_fired_exit_reason('X', pos, True) is None and pos['spring_fire_base'] == 'LONG|' + str(f0['t']), pos)
    tm._spring_fire_at = {}
    _check(tm._spring_fired_exit_reason('X', pos, True) is None, 'той самий вистріл (база) не закриває')
    # Новий вистріл після входу → вихід.
    tm = _tm({'side': 'LONG', 't': int(now), 'level': 1.0, 'pre_sq': 7, 'live': True})
    pos = {'side': 'SHORT', 'opened_at': now - 600}
    r = tm._spring_fired_exit_reason('X', pos, True)
    _check(r and r[0] == 'spring_fired_exit' and 'ПРОТИ' in r[1] and 'живий' in r[1], r)
    # Тротл: друга перевірка одразу — мовчить.
    _check(tm._spring_fired_exit_reason('X', {'side': 'SHORT', 'opened_at': now - 600}, True) is None, 'тротл')
    # Вимкнено → мовчить.
    tm = _tm({'side': 'LONG', 't': int(now)}); tm._settings['use_spring_fired_exit'] = False
    _check(tm._spring_fired_exit_reason('X', {'side': 'SHORT', 'opened_at': now - 600}) is None, 'вимкнено')


def test_wiring():
    TM = _tm_cls()
    T = sys.modules['detection.trade_manager']
    _check(T.DEFAULT_SETTINGS['use_spring_fired_exit'] is False
           and T.DEFAULT_SETTINGS['spring_fired_exit_dir'] == 'any', 'дефолти')
    _check(_TM.count("'spring_fired_exit':") >= 2, 'підпис у _close_detail і бейджі')
    _check("_spring_fired_exit_reason(symbol, pos, is_shadow)" in _TM, 'кличеться з _check_signal_exits')
    _check(_TM.count('            self._spring_fire_view(symbol)\n        except Exception:') == 2, 'обидва монітори тягнуть живий бар')
    _check("'spring_fired_exit': '💥 Пружина вистрілила'" in _HTML, 'JS-мапа причин')
    _check('id="tm-use-spring-fired-exit"' in _HTML and 'id="tm-spring-fired-exit-dir"' in _HTML, 'UI')
    _check(_HTML.count('${springFireCellHTML(p)}') == 2 and '${springTradeCellHTML(p)}' not in _HTML,
           'колонка «💥 Виснаження» замість «🌀 Заряд» у обох таблицях')
    _check(_HTML.count('>🎯 Ціль</th>') == 2, 'заголовок обох таблиць (11.11: «Ціль»)')
    _check('${fmtPriceJS(p.current_price)}${pxAgeHTML(p)}' in _HTML and 'curPxHTML' not in _HTML,
           'Current — як було')
    _check(_HTML.count('pnlUsdChip(') >= 3, '$ у PnL виділено (real + paper)')
    _check("'spring_fire'" in _IP and '_spring_fire_view(p.get(\'symbol\'), fetch=False)' in _IP,
           'інфосайт: той самий кеш TM')
    _check('function ppExh(' in _JS and 'ppExh(p)' in _JS and 'pp-usd' in _JS, 'інфосайт: комірка й $')
    _check("_spring_fire_view(sym, fetch=False)" in _APP, '/api/tm/state лише читає кеш')


def test_js_cell():
    """💥 Комірка «Виснаження» = прогрес угоди від входу до цілі Пружини (10.11)."""
    i = _HTML.index('function exhColor(')
    j = _HTML.index('function springTradeCellHTML(', i)
    js = 'const fmtPriceJS = p => String(p);\n' + _HTML[i:j] + r"""
console.log(springFireCellHTML({}));
console.log(springFireCellHTML({side:'SHORT', exh:{pct:30,prog:30,done_pct:1.5,target:90,target_pct:5,climax:false}, spring_fire:{pct:100,fired:{side:'LONG',live:true}}}));
console.log(springFireCellHTML({side:'LONG', exh:{pct:0,prog:-1,done_pct:-0.24,target:0.0543,target_pct:29.3,climax:false}, spring_fire:{pct:20,fired:null}}));
console.log(springFireCellHTML({side:'LONG', exh:{pct:82,done_pct:8,target:110,target_pct:10,climax:true}}));
"""
    out = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=20)
    _check(out.returncode == 0, out.stderr)
    a, b, c, d = out.stdout.strip().split('\n')
    _check('—' in a, a)
    _check('30%' in b and '⚠' in b and 'tm-mmd-alert-hot' in b, b)
    _check('−1%' in c and 'width:0%' in c and '20%' not in c, ('проти угоди — мінус, смужка порожня', c))
    _check('82%' in d and '💥' in d, d)


def test_trade_exhaustion_is_progress_to_target():
    """💥 Калібровка 10.11 (кейс UAIUSDT): на вході виснаження = 0%, а не 20%
    від стиснення. % = пройдений шлях від входу до цілі «Ймовірного ходу»."""
    TM = _tm_cls()
    ex = TM.trade_exhaustion
    _check(ex('LONG', 0.042, 0.042, 0.0543)['pct'] == 0.0, 'на вході 0%')
    _check(ex('LONG', 0.042, 0.0419, 0.0543)['pct'] == 0.0, 'проти угоди — 0%, не мінус')
    _check(abs(ex('LONG', 100, 105, 110)['pct'] - 50.0) < 1e-9, 'пів шляху = 50%')
    _check(ex('LONG', 100, 120, 110)['pct'] == 100.0, 'ціль пройдена — 100%')
    _check(abs(ex('SHORT', 100, 96, 92)['pct'] - 50.0) < 1e-9, 'SHORT дзеркально')
    _check(ex('LONG', 100, 105, 95) is None and ex('LONG', 100, 105, None) is None, 'немає цілі попереду → None')
    _check(ex('LONG', 100, 105, 110, 'LONG')['climax'] is True
           and ex('LONG', 100, 105, 110, 'SHORT')['climax'] is False, 'сплеск лише у бік угоди')
    # одне джерело: колонка й правило виходу кличуть trade_exh_for
    _check("tm.trade_exh_for(pos, _sf)" in _APP and 'trade_exh_for(p, sf)' in _IP
           and 'self.trade_exh_for(pos, vw)' in _TM, 'одне джерело для показу і виходу')
    _check("'spring_open_tgt' not in pos" in _TM, 'ціль записується на вході (і добирається для старих угод)')


def test_exhaustion_climax_exit():
    """🎯 10.11: вихід на кульмінації — сплеск «Виснаження» у бік угоди в плюсі
    озброює трейл від вершини на k·ATR; не гірше входу + мін. прибуток."""
    b = _fire(_coil())
    vw = S.live_fire_view(b)
    _check(vw['climax'] == 'LONG' and vw['atr'] > 0 and vw['bar_hi'] >= vw['price'], vw)
    _check(S.live_fire_view(_coil())['climax'] is None, 'у коробці кульмінації немає')
    TM = _tm_cls()
    step = TM.exh_exit_step
    view = {'pct': 80, 'atr': 2.0, 'bar_hi': 112.0, 't': 1}
    _check(step('LONG', 100, None, dict(view, pct=60), 110)[0] is None, 'нижче порогу — не озброюємо')
    _check(step('LONG', 100, None, dict(view, pct=10, climax='SHORT'), 110)[0] is None, 'сплеск проти угоди — не наш')
    _check(step('LONG', 100, None, dict(view, pct=10, climax='LONG'), 110)[0] is not None, 'сплеск у бік угоди озброює і без порогу')
    _check(step('LONG', 100, None, view, 100.1)[0] is None, 'не в плюсі — не озброюємо')
    st, why = step('LONG', 100, None, view, 110)
    _check(st and why is None and st['best_px'] == 112.0 and st['stop_px'] == 111.0, st)
    st, why = step('LONG', 100, st, None, 113, lo=112.5, hi=114)
    _check(why is None and st['best_px'] == 114 and st['stop_px'] == 113.0, st)
    st2, why = step('LONG', 100, st, None, 112.8, lo=112.8, hi=113)
    _check(why == 'trail', (st2, why))
    sh, _ = step('SHORT', 100, None, dict(view, bar_lo=88.0), 90)
    _check(sh['best_px'] == 88.0 and sh['stop_px'] == 89.0, sh)
    _, why = step('SHORT', 100, sh, None, 89.5, lo=89.2, hi=89.5)
    _check(why == 'trail', why)
    fl, _ = step('LONG', 100, None, dict(view, atr=50.0), 110)
    _check(abs(fl['stop_px'] - 100.3) < 1e-9, ('стоп не гірший за вхід + комісії', fl))
    _, why = step('LONG', 100, fl, None, 100.2, lo=100.2, hi=100.2)
    _check(why == 'floor', why)
    T = sys.modules['detection.trade_manager']
    _check(T.DEFAULT_SETTINGS['use_exh_exit'] is False, 'дефолт OFF')
    _check(_TM.count('self._exh_exit_tick(symbol, pos, current_price,') == 2, 'обидва монітори')
    _check(_TM.count("'exh_exit':") >= 2 and "'exh_exit': '🎯 Кульмінація'" in _HTML, 'підписи причини')
    _check('id="tm-use-exh-exit"' in _HTML and 'exhTrailBadgeHTML(p)' in _HTML, 'UI бота')
    _check("'exh_trail'" in _IP and 'pp-exh-st' in _JS, 'інфосайт')

def test_spring_rec_fallback_outside_watchlist():
    """💥 Угода поза WATCHLIST (вимога 11.11): «Виснаження» бере ціль із
    🔭 скан-таблиці, інакше — з `analyze` тих самих 1h-барів, що вже тягне
    `_spring_fire_view` (без додаткових запитів)."""
    TM = _tm_cls()
    tm = TM.__new__(TM)
    tm.scanner = types.SimpleNamespace(spring_for=lambda syms: {})
    sys.modules.pop('detection.spring_scan', None)
    _check(tm._spring_rec('ICNTUSDT') is None, 'даних немає — None')
    # 🔭 рядок скан-таблиці
    eng = types.SimpleNamespace(row_for=lambda s: {'charge': 81, 'tgt_short': 0.27} if s == 'ICNTUSDT' else None)
    sm = types.ModuleType('detection.spring_scan'); sm.get_spring_scan = lambda: eng
    sys.modules['detection.spring_scan'] = sm
    _check(tm._spring_rec('ICNTUSDT')['tgt_short'] == 0.27, 'рядок 🔭 скану')
    sys.modules.pop('detection.spring_scan', None)
    # бари, які TM уже тягнув для колонки
    rec = S.analyze(_coil())
    tm._spring_bar_rec = {'UAIUSDT': (time.time(), rec)}
    _check(tm._spring_rec('UAIUSDT') is rec, 'запис із барів TM')
    tm._spring_bar_rec = {'UAIUSDT': (time.time() - 4 * 3600, rec)}
    _check(tm._spring_rec('UAIUSDT') is None, 'застарілий запис не береться')
    # ціль для угоди → «Виснаження» рахується
    tm._spring_bar_rec = {'UAIUSDT': (time.time(), rec)}
    pos = {'side': 'LONG', 'entry_price': rec['price'] * 0.99, 'opened_at': time.time()}
    tm._spring_open_mark('UAIUSDT', pos)
    _check(pos.get('spring_open_tgt') == round(rec['tgt_long'], 10), pos)
    _check("br[S] = (now, rec)" in _TM, '_spring_fire_view кешує запис із тих самих барів')


if __name__ == '__main__':
    n = 0
    for k, fn in list(globals().items()):
        if k.startswith('test_') and callable(fn):
            fn(); n += 1; print('✓', k)
    print(f'OK — {n} тестів')
