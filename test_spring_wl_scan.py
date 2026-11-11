"""🌀 «Пружина WATCHLIST» — ручний скан + автоскан (вимога 11.11).

Той самий розрахунок (`analyze_many` → `spring_coil.analyze`), що 🔭 скан біржі;
результат пишеться в сканер (`spring_put`) — таблиця й ворота читають одне джерело.
Плюс 🎯 знаковий прогрес до цілі в 🚨 і непрозора липка шапка інфосайту.
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_spring_scan as T  # noqa: E402

ss, ROOT, HTML = T.ss, T.ROOT, T.HTML
SRC_SC = open(os.path.join(ROOT, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
APP = open(os.path.join(ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
CSS = open(os.path.join(ROOT, 'infosite', 'style.css'), encoding='utf-8').read()
IJS = open(os.path.join(ROOT, 'infosite', 'app.js'), encoding='utf-8').read()
TM_SRC = open(os.path.join(ROOT, 'detection', 'trade_manager.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


class _Sc:
    def __init__(self, on=True):
        self._settings = {'spring_enabled': on}
        self.put = {}

    def spring_put(self, S, rec):
        self.put[S] = rec
        return True


def _eng(sc, wl=('AAAUSDT', 'BBBUSDT', 'CCCUSDT'), db=None, clock=None):
    now = time.time()
    bars = {'AAAUSDT': T._bars('coil', now=now), 'BBBUSDT': T._bars('wild', now=now)}

    def fetch_factory(ex):
        _check(ex == 'bybit', 'WATCHLIST — біржа бота (Bybit)')
        return lambda s: bars.get(s) or []
    return ss.SpringWlScan(db=db, fetch_factory=fetch_factory,
                           run_queued=lambda n, fn, src: fn(), workers=2,
                           watchlist=lambda: list(wl), scanner=lambda: sc,
                           clock=clock or time.time)


def test_scan_writes_into_the_scanner():
    sc = _Sc()
    e = _eng(sc)
    res = e.scan()
    _check(set(sc.put) == {'AAAUSDT', 'BBBUSDT'}, sc.put)   # CCC — замало барів
    _check(res['scanned'] == 3 and res['analyzed'] == 2 and res['updated'] == 2, res)
    p = e.progress()
    _check(p['phase'] == 'done' and not p['running'] and p['done'] == 3, p)
    _check(p.get('short') == 1, 'монета без історії — «замало історії»')
    print('✓ скан WATCHLIST: той самий analyze → scanner.spring_put')


def test_disabled_spring_refuses():
    e = _eng(_Sc(on=False))
    r = e.start('ui')
    _check(not r['ok'] and not r['started'], r)
    print('✓ 🌀 вимкнено → скан не стартує')


def test_auto_and_persist():
    db = T._DB()
    clk = [1_000_000.0]
    sc = _Sc()
    e = _eng(sc, db=db, clock=lambda: clk[0])
    e.update_settings({'auto_on': True, 'auto_min': 2})
    _check(e.settings() == {'auto_on': True, 'auto_min': 5}, 'інтервал обрізано до 5 хв')
    _check(e.next_auto_at() == 300.0, 'ще не сканували → одразу')
    e.scan()
    _check(abs(e.next_auto_at() - (clk[0] + 300)) < 1, 'наступний — через інтервал від скану')
    e2 = _eng(sc, db=db)
    _check(e2.settings()['auto_on'] and e2.state()['result'].get('restored'), 'налаштування й результат переживають рестарт')
    print('✓ автоскан раз на N хв · налаштування й результат персистяться')


def test_scanner_put_and_routes_and_ui():
    _check('def spring_put(' in SRC_SC and "spring_enabled', True)" in SRC_SC, 'сканер приймає запис')
    for r in ("/api/spring-wl-scan/run", "/api/spring-wl-scan/settings", "'wl_scan': _wl_scan_state()"):
        _check(r in APP, r)
    _check('init_spring_wl_scan(db=get_db(), start_loop=True)' in APP, 'потік автоскану стартує з застосунком')
    for x in ('id="sprw-run"', 'id="sprw-auto"', 'id="sprw-viz" class="xs-viz"', 'function sprwPaint()',
              "p.kind === 'wl'", '_xsSteps(p)'):
        _check(x in HTML, x)
    print('✓ маршрути + UI: той самий вигляд процесу, що в 🔭 скані')


def test_signed_progress_to_target():
    i = TM_SRC.index('    def trade_exhaustion(')
    body = TM_SRC[i:TM_SRC.index('    def trade_exh_for(', i)]
    ns = {}
    exec('class X:\n    @staticmethod\n' + body.replace('\n    ', '\n    '), ns)
    f = ns['X'].trade_exhaustion
    _check(f('LONG', 100, 105, 110)['prog'] == 50.0, 'пів шляху')
    _check(f('LONG', 100, 98, 110)['prog'] == -20.0, 'проти угоди — від\'ємний')
    _check(f('SHORT', 100, 102, 90)['prog'] == -20.0, 'SHORT дзеркально')
    _check(f('LONG', 100, 98, 110)['pct'] == 0.0, '«Виснаження» лишилось 0..100')
    # 11.11 (друга правка): % до цілі переїхав із 🚨 у колонку «🎯 Ціль»
    _check('exhProgHTML' not in HTML and 'ppProg' not in IJS, '🚨 більше не несе %')
    _check('e.prog' in HTML[HTML.index('function springFireCellHTML('):] and 'e.prog' in IJS, 'Ціль = знаковий %')
    print('✓ 🎯 знаковий % до ймовірної цілі — у колонці «Ціль» (мінус — відхиляємось)')


def test_sticky_header_is_opaque():
    i = CSS.index('.rec-wrap .pp-tbl th {')
    rule = CSS[i:CSS.index('}', i)]
    _check('sticky' in rule and 'var(--bg)' in rule, 'липка шапка з суцільним тлом')
    print('✓ шапка таблиці не просвічує рядки')


if __name__ == '__main__':
    n = 0
    for k, fn in list(globals().items()):
        if k.startswith('test_') and callable(fn):
            fn(); n += 1
    print(f'OK — {n} тестів')
