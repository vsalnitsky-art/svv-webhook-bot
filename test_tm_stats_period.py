"""📊 Панель P&L у стилі Bybit + статистика за період (вимога 27.10).

Дослівно: «Зроби цю панель в стилі біржі Bybit. В таблиці 📜 Recent Closed
Trades все розмісти в одну стрічку, поле PnL в кольорах передай текст. І додай
зебру в усіх таблицях. Додай можливість переглядати статистику за певний
період або конкретний день.»
"""
import json
import os
import re
import subprocess
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_autosl_chain as _tmb  # noqa: E402

TM = _tmb.TM
tm_mod = sys.modules[TM.__module__]
closed_stats = tm_mod.closed_stats
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()
_FLASK = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _c(at, pct, usd, src='exchange', fees=None, verdict=None):
    c = {'symbol': 'XUSDT', 'side': 'LONG', 'closed_at': at, 'pnl_pct': pct,
         'pnl_usd': usd, 'pnl_source': src}
    if fees is not None:
        c['exchange'] = {'fees_usd': fees}
    if verdict:
        c['entry_score'] = {'verdict': verdict}
    return c


def test_closed_stats_is_the_single_source():
    st = closed_stats([_c(1, 1.0, 3.0, fees=0.2), _c(2, -0.5, -1.0, src='pending'),
                       _c(3, 2.0, 1.0, fees=0.1)])
    _check(st['total_closed'] == 3 and st['wins'] == 2 and st['losses'] == 1, st)
    _check(st['win_rate'] == 66.7 and st['total_pnl_usd'] == 3.0, st)
    _check(st['gross_profit_usd'] == 4.0 and st['gross_loss_usd'] == -1.0, st)
    _check(st['profit_factor'] == 4.0 and st['avg_win_usd'] == 2.0 and st['avg_loss_usd'] == -1.0, st)
    _check(st['best_usd'] == 3.0 and st['worst_usd'] == -1.0, st)
    _check(st['not_exchange'] == 1 and abs(st['total_fees_usd'] - 0.3) < 1e-9, st)
    _check(closed_stats([_c(1, 1, 1)])['profit_factor'] is None, 'без збитків PF не вигадуємо')
    _check(closed_stats([])['total_closed'] == 0, 'порожньо → нулі, не збій')
    print('✓ closed_stats: усі показники панелі з одного розрахунку')


def test_get_state_stats_come_from_closed_stats():
    src = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()
    i = src.index('    def get_state(self)')
    body = src[i:src.index('\n    def ', i + 10)]
    _check('closed_stats(self._closed_trades)' in body, '«за весь час» мусить рахуватись тією ж функцією')
    print('✓ «Усе» і будь-який період рахує ОДНА closed_stats')


def _tm(closed, shadow=()):
    t = TM.__new__(TM)
    t._lock = threading.RLock()
    t._closed_trades = list(closed)
    t._shadow_closed = list(shadow)
    return t


def test_stats_for_period_filters_by_closed_at():
    t = _tm([_c(100, 1, 5, verdict='good'), _c(200, -1, -2), _c(300, 1, 1, verdict='good')],
            shadow=[_c(150, 1, 9)])
    st = TM.stats_for_period(t, 150, 300)
    _check(st['total_closed'] == 1 and st['total_pnl_usd'] == -2, 'межа [from, to) — to не включно')
    _check(TM.stats_for_period(t, None, None)['total_closed'] == 3, 'без меж — усі')
    _check(TM.stats_for_period(t, '250', '')['total_closed'] == 1, 'рядки з query → числа / порожньо = без межі')
    _check(TM.stats_for_period(t, None, None, shadow=True)['total_pnl_usd'] == 9, 'paper — своя книга')
    eb = TM.stats_for_period(t, 0, 1000)['entry_score_breakdown']
    _check(eb['good']['count'] == 2, 'перевірка оцінки входу теж за період')
    _check(TM.stats_for_period(t, 0, 1)['oldest_at'] == 100, st)
    print('✓ статистика за період: фільтр за часом закриття, обидві книги')


def test_endpoint_exists_and_page_calls_it():
    _check("@app.route('/api/tm/stats')" in _FLASK and 'stats_for_period' in _FLASK, 'маршрут')
    _check('/api/tm/stats?' in _HTML, 'сторінка мусить питати статистику періоду')
    print('✓ /api/tm/stats є і сторінка його кличе')


def _js_fn(name):
    i = _HTML.index(f'function {name}(')
    return _HTML[i:_HTML.index('\n}\n', i) + 2]


def test_period_ranges_use_local_day_boundaries():
    js = '\n'.join(_js_fn(n) for n in ('_tmYmd', '_tmDayStart', '_tmPeriodRange')) + r'''
    const now = new Date(2026, 9, 27, 15, 30).getTime();
    const d0 = new Date(2026, 9, 27).getTime() / 1000;
    const out = {
      today: _tmPeriodRange({mode:'today'}, now),
      d7: _tmPeriodRange({mode:'7d'}, now),
      day: _tmPeriodRange({mode:'day', day:'2026-10-25'}, now),
      range: _tmPeriodRange({mode:'range', from:'2026-10-26', to:'2026-10-20'}, now),
      bad: _tmPeriodRange({mode:'day', day:''}, now),
      all: _tmPeriodRange({mode:'all'}, now),
      d0,
      d25: new Date(2026, 9, 25).getTime() / 1000, d26: new Date(2026, 9, 26).getTime() / 1000,
      d20: new Date(2026, 9, 20).getTime() / 1000, d27: d0,
    };
    console.log(JSON.stringify(out));'''
    r = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=20)
    _check(r.returncode == 0, r.stderr)
    o = json.loads(r.stdout)
    _check(o['today']['from'] == o['d0'] and o['today']['to'] == o['d0'] + 86400, o['today'])
    _check(o['d7']['from'] == o['d0'] - 6 * 86400, '7Д = сьогодні + 6 попередніх днів')
    _check(o['day']['from'] == o['d25'] and o['day']['to'] == o['d26'] and '25.10.2026' in o['day']['label'], o['day'])
    _check(o['range']['from'] == o['d20'] and o['range']['to'] == o['d27'], 'переплутані дати → упорядковано, кінець включно')
    _check(o['bad'] is None, 'без дати — немає діапазону (не вигадуємо)')
    _check(o['all']['from'] is None and o['all']['to'] is None, o['all'])
    print('✓ межі днів — за годинником ПК, кінець періоду включно')


def test_panel_is_bybit_styled_and_has_period_controls():
    for t in ('class="bb-stats"', 'id="tm-stats-period"', 'data-p="today"', 'data-p="7d"',
              'data-p="30d"', 'data-p="all"', 'data-p="day"', 'data-p="range"',
              'id="tm-stats-day"', 'id="tm-stats-from"', 'id="tm-stats-to"'):
        _check(t in _HTML, f'немає {t}')
    css = _HTML[_HTML.index('.bb-stats {'):_HTML.index('.bb-empty')]
    _check('#20b26c' in css and '#ef454a' in css and '#f7a600' in css, 'палітра Bybit')
    _check('.tm-pnl-pos { color: #20b26c; }' in _HTML and '.tm-pnl-neg { color: #ef454a; }' in _HTML,
           'PnL у таблицях — у тих самих кольорах')
    print('✓ панель у стилі Bybit + вибір періоду / дня')


def test_closed_rows_are_one_line_with_colored_pnl():
    _check('#tm-closed-table th, #tm-closed-table td' in _HTML, 'закриті — одна стрічка')
    f = _js_fn('closedRealPnlCellHTML')
    _check('<br>' not in f, 'перенос рядка в комірці PnL')
    _check('tm-pnl-cell' in f and 'opacity:0.85' not in f, 'весь текст PnL — у кольорі результату')
    print('✓ 📜 Recent Closed: рядок в одну стрічку, PnL кольоровий')


def test_zebra_in_every_table():
    _check(':where(.tm-table, .poc-tbl) tbody tr:nth-child(even) > td' in _HTML, 'спільна зебра')
    n_tables = len(re.findall(r'<table class="tm-table"', _HTML))
    _check(n_tables >= 20, 'зебра чіпляється до класу .tm-table усіх таблиць')
    print('✓ зебра — одне правило на всі таблиці')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
