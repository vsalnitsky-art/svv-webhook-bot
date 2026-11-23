"""💰 → 🔭 «Funding — Пружина» передає заряджені монети в «Пружина — скан біржі»
(вимога 24.11). Дослівно: «забирати заряджені монети із "Funding - пружина" і
додавати в таблицю "Пружина - скан біржі" для подальшого контролю… У таблиці
фандінг пружина її більше не відображати і роби помітку… Цей алгоритм
працювати при увімкненому тумблері "Funding - пружина".»

Запуск: python3 test_funding_to_scan.py
"""
import importlib.util
import os
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    sp = importlib.util.spec_from_file_location('_ft_' + name, os.path.join(_ROOT, 'detection', name + '.py'))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


ss = _load('spring_scan')
fs = _load('funding_spring')
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
_APP = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
NOW = (int(time.time()) // 3600) * 3600 + 30
BAR = NOW - 30 - 3600          # відкриття останнього закритого бару


def _rec(state='charged', charge=82.0, bar_t=BAR, d='LONG'):
    return {'ok': True, 'state': state, 'charge': charge, 'dir': d, 'lean': d,
            'pot_long': 60, 'pot_short': 20, 'price': 1.0, 'bar_t': bar_t,
            'box_hi': 1.1, 'box_lo': 0.9, 'atr': 0.01}


def _scan(wl=()):
    calls = []

    def fetch_factory(ex):
        def f(sym):
            calls.append((ex, sym))
            return []
        return f
    eng = ss.SpringScan(db=None, tickers=lambda ex: {}, fetch_factory=fetch_factory,
                        run_queued=lambda n, fn, src: fn(), workers=1,
                        clock=lambda: NOW, watchlist=lambda: list(wl))
    return eng, calls


def test_absorb_takes_charged_and_marks_source():
    eng, _ = _scan(wl=('WLUSDT',))
    add = eng.absorb_funding({'AAAUSDT': _rec(), 'BBBUSDT': _rec(state='building', charge=50),
                              'WLUSDT': _rec()}, NOW)
    assert add == ['AAAUSDT'], add                       # лише заряджена, не WATCHLIST
    row = eng.row_for('AAAUSDT')
    assert row['src'] == 'funding' and row['src_at'] == NOW and row['bybit'] is True
    assert eng.absorb_funding({'AAAUSDT': _rec()}, NOW) == []     # уже в таблиці — без дубля
    assert 'AAAUSDT' in eng.held_symbols()
    # правило ВХОДУ — статус «🌀 заряджена», фільтр заряду скану тут не діє
    assert eng.absorb_funding({'CCCUSDT': _rec(charge=66)}, NOW) == ['CCCUSDT']


def test_source_mark_survives_recalc_and_recalc_uses_bybit():
    eng, calls = _scan()
    eng.update_settings({'exchange': 'binance'})
    eng.absorb_funding({'AAAUSDT': _rec()}, NOW)
    eng._table['SCANUSDT'] = eng._row('SCANUSDT', _rec(), NOW, bybit=True)
    eng._analyze_many = lambda syms, fetch, now: ({s: _rec() for s in syms if fetch(s) is not None}, [])
    eng.recalc(NOW)
    assert ('bybit', 'AAAUSDT') in calls and ('binance', 'SCANUSDT') in calls, calls
    assert eng._table['AAAUSDT']['src'] == 'funding'
    assert 'src' not in eng._table['SCANUSDT']


def test_dropped_coin_is_not_readded_by_a_stale_bar():
    eng, _ = _scan()
    eng.absorb_funding({'AAAUSDT': _rec()}, NOW)
    eng.remove('AAAUSDT', now=NOW + 60)
    assert eng.absorb_funding({'AAAUSDT': _rec()}, NOW + 70) == []          # той самий бар
    assert eng.absorb_funding({'AAAUSDT': _rec(bar_t=BAR + 3600)}, NOW + 3700) == ['AAAUSDT']


class _Net:
    def __call__(self, sym, tf, limit):
        return []


def _fspring(handoff, held, enabled=True):
    f = fs.FundingSpring(None, lambda: ['AAAUSDT', 'BBBUSDT', 'WLUSDT'], _Net(),
                         lambda syms: {'WLUSDT': _rec()}, lambda: {}, handoff, held)
    f._settings['enabled'] = enabled
    with f._lock:
        f._recs['AAAUSDT'] = {**_rec(), 'src': 'own', 'ts': NOW}
        f._recs['BBBUSDT'] = {**_rec(state='building', charge=50), 'src': 'own', 'ts': NOW}
    return f


def test_funding_hands_over_only_own_charged_when_enabled():
    got = []
    f = _fspring(lambda recs: (got.append(sorted(recs)) or list(recs)), lambda: set())
    f.tick(NOW)
    assert got == [['AAAUSDT']], got                     # WATCHLIST і незаряджена — ні
    assert f._stats['handed'] == 1 and f._stats['handed_last'] == ['AAAUSDT']
    got.clear()
    off = _fspring(lambda recs: (got.append(sorted(recs)) or list(recs)), lambda: set(), enabled=False)
    off.tick(NOW)
    assert got == []                                     # тумблер вимкнено — нічого


def test_handed_coin_is_hidden_from_the_funding_table():
    f = _fspring(lambda recs: [], lambda: {'AAAUSDT'})
    st = f.state(NOW)
    syms = [r['symbol'] for r in st['rows']]
    assert 'AAAUSDT' not in syms and 'BBBUSDT' in syms
    assert [h['symbol'] for h in st['handed']] == ['AAAUSDT']


def test_end_to_end_through_the_scan_engine():
    eng, _ = _scan(wl=('WLUSDT',))
    f = _fspring(eng.absorb_funding, eng.held_symbols)
    f.tick(NOW)
    assert eng.row_for('AAAUSDT')['src'] == 'funding'
    assert 'AAAUSDT' not in [r['symbol'] for r in f.state(NOW)['rows']]


def test_page_and_log():
    assert "function _xsSrcBadge(x)" in _HTML and "_xsSrcBadge(x) + _xsBybitBadge(x)" in _HTML
    assert "x.vol_usd, x.src]" in _HTML                  # позначка в сигнатурі таблиці
    assert "function _fsprHandedLine" in _HTML and "handed: d.handed || []" in _HTML
    assert "xspr_in: ['💰'" in _HTML
    assert "SOLO_EVENTS = new Set(['ob_new', 'ai_scan', 'xspr_in'])" in _HTML
    assert "_SOLO = {'ob_new', 'ai_scan', 'xspr_in'}" in _APP
    src = open(os.path.join(_ROOT, 'detection', 'funding_spring.py'), encoding='utf-8').read()
    for bad in ('on_signal', 'intercept', '_open(', 'notify_category', 'manual_open'):
        assert bad not in src, bad                       # funding і далі нічого не відкриває


if __name__ == '__main__':
    n = 0
    for k, fn in list(globals().items()):
        if k.startswith('test_') and callable(fn):
            fn()
            n += 1
    print(f'OK {n}')
