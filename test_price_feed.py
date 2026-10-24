"""⚡ Свіжа ціна для виходів + ґноти (вимога 25.10, кейс XMRUSDT).

Корінь: монітор TP/SL тікав раз на 4с, але брав ціну з кешу SMC-сканера —
закриття бару на момент ОСТАННЬОГО скану монети (скан ~230 монет іде
послідовно, тож ціна могла бути старшою на хвилини), і бачив лише точку, а не
ґноти між тактами. Тепер: окрема стрічка цін (один bulk-запит Bybit), гейт
свіжості з фолбеком на прямий запит, і перевірка рівнів за [мін..макс].
"""
import ast
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_autosl_chain as _base  # noqa: E402  (ставить стаби пакета)
from detection import price_feed as pf  # noqa: E402

TM = _base.TM
_tmmod = sys.modules[TM.__module__]
_TM_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _fn_src(name):
    for n in ast.walk(ast.parse(_TM_SRC)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(_TM_SRC, n)
    raise AssertionError(f'{name} не знайдено')


# ═══════════════════════ стрічка цін ══════════════════════════════════════
def _feed():
    return pf.PriceFeed(lambda: {})


def test_price_is_fresh_or_nothing():
    f = _feed()
    f.ingest({'XMRUSDT': 590.0}, now=1000.0)
    _check(f.price('XMRUSDT', now=1005.0) == (590.0, 1000.0), 'свіжа ціна')
    _check(f.price('XMRUSDT', now=1000.0 + pf.STALE_SEC + 1) is None,
           'стара ціна НЕ видається за поточну')
    print('✓ стрічка віддає лише свіжу ціну')


def test_range_catches_the_wick_between_ticks():
    f = _feed()
    f.watch(['XMRUSDT']); f.stop()
    f.ingest({'XMRUSDT': 590.0}, now=1000.0)
    _check(f.take_range('X|S', 'XMRUSDT', now=1000.0) == (590.0, 590.0),
           'перше взяття — лише поточна точка')
    for t, p in ((1002, 575.0), (1004, 568.0), (1006, 588.0)):
        f.ingest({'XMRUSDT': p}, now=float(t))
    _check(f.take_range('X|S', 'XMRUSDT', now=1006.0) == (568.0, 588.0),
           'ґніт до 568 між тактами мусить потрапити в діапазон')
    _check(f.take_range('X|S', 'XMRUSDT', now=1006.0) == (588.0, 588.0),
           'після взяття діапазон починається заново')
    print('✓ діапазон між тактами ловить ґніт')


def test_books_take_ranges_independently():
    f = _feed()
    f.watch(['XMRUSDT']); f.stop()
    f.ingest({'XMRUSDT': 590.0}, now=1000.0)
    f.take_range('X|R', 'XMRUSDT', now=1000.0)
    f.take_range('X|S', 'XMRUSDT', now=1000.0)
    f.ingest({'XMRUSDT': 560.0}, now=1002.0)
    f.ingest({'XMRUSDT': 589.0}, now=1004.0)
    _check(f.take_range('X|R', 'XMRUSDT', now=1004.0)[0] == 560.0, 'real бачить ґніт')
    _check(f.take_range('X|S', 'XMRUSDT', now=1004.0)[0] == 560.0,
           'paper теж бачить ТОЙ САМИЙ ґніт — книги не зʼїдають одна одну')
    print('✓ книги беруть діапазон незалежно')


def test_range_window_is_capped():
    f = _feed()
    f.watch(['XMRUSDT']); f.stop()
    f.ingest({'XMRUSDT': 590.0}, now=1000.0)
    f.take_range('X|S', 'XMRUSDT', now=1000.0)
    f.ingest({'XMRUSDT': 500.0}, now=1001.0)            # давній прокол
    f.ingest({'XMRUSDT': 589.0}, now=1100.0)
    lo, hi = f.take_range('X|S', 'XMRUSDT', now=1100.0)
    _check(lo == 589.0, f'прокол старший за вікно не має рахуватись: {lo}')
    print('✓ вікно діапазону обмежене')


def test_no_polling_without_open_trades():
    calls = []
    f = pf.PriceFeed(lambda: calls.append(1) or {}, poll_sec=0.01)
    f.watch([])
    _check(f._thread is None, 'без угод потік не стартує і біржу не смикає')
    f.watch(['XMRUSDT'])
    ev = threading.Event()
    for _ in range(100):
        if calls:
            break
        ev.wait(0.01)
    f.stop()
    _check(calls, 'з угодою стрічка опитує біржу')
    print('✓ опитування лише поки є відкриті угоди')


def test_failure_keeps_last_price_and_reports():
    f = pf.PriceFeed(lambda: (_ for _ in ()).throw(RuntimeError('418')))
    _check(not f.poll(), 'збій — False')
    _check('418' in f.health()['err'], 'причина видна в health')
    print('✓ збій біржі не мовчазний')


def test_bybit_fetch_parses_bulk_tickers():
    class S:
        def get_tickers(self, category):
            return {'retCode': 0, 'result': {'list': [
                {'symbol': 'XMRUSDT', 'lastPrice': '610.67'},
                {'symbol': 'BADUSDT', 'lastPrice': ''}]}}
    out = pf.bybit_fetch_all(type('C', (), {'session': S()})())()
    _check(out == {'XMRUSDT': 610.67}, out)
    print('✓ bulk-тікери розбираються')


# ═══════════════════════ рівні з ґнотами ══════════════════════════════════
def test_level_hit_both_sides():
    lh = _tmmod.level_hit
    _check(lh('SHORT', 'tp', 571.06, 565.0, 595.0), 'SHORT TP — ґніт униз')
    _check(not lh('SHORT', 'tp', 571.06, 575.0, 610.0), 'SHORT TP не торкнутий')
    _check(lh('SHORT', 'sl', 612.0, 590.0, 612.5), 'SHORT SL — ґніт угору')
    _check(lh('LONG', 'tp', 110.0, 100.0, 110.0), 'LONG TP')
    _check(lh('LONG', 'sl', 95.0, 94.9, 101.0), 'LONG SL')
    _check(not lh('LONG', 'sl', 0, 0.0, 1.0), 'порожній рівень не спрацьовує')
    print('✓ рівні перевіряються за діапазоном')


def test_paper_fill_is_honest():
    fill = _tmmod.paper_fill
    _check(fill('SHORT', 'tp', 571.06, 588.0) == 571.06,
           'TP торкнули ґнотом → виконання на рівні')
    _check(fill('SHORT', 'sl', 612.0, 615.0) == 615.0,
           'SL пройдено ринком → виконання за поточною (гірше), як на біржі')
    _check(fill('LONG', 'sl', 95.0, 96.0) == 95.0, 'SL ґнотом → на рівні')
    print('✓ paper-виконання не прикрашає SL і не губить TP')


def test_manual_checks_use_the_range():
    pos = {'side': 'SHORT', 'manual_tp': 571.06, 'manual_sl': 640.0}
    _check(TM._check_manual_sl_tp(None, pos, 588.0) is None, 'точка — не спрацьовує')
    _check(TM._check_manual_sl_tp(None, pos, 588.0, 565.0, 590.0) == 'manual_tp',
           'з діапазоном (кейс XMRUSDT) TP спрацьовує')
    pos2 = {'side': 'SHORT', 'manual_sl': 600.0}
    _check(TM._check_manual_sl_tp(None, pos2, 595.0, 594.0, 601.0) == 'manual_sl',
           'ґніт угору вибиває SL — як і стоп на біржі')
    print('✓ Manual SL/TP судяться за діапазоном')


# ═══════════════════════ джерело ціни в TM ════════════════════════════════
class _Scn:
    def __init__(self, p, ts):
        self._cache = {'XMRUSDT': {'updated_at': ts}}
        self._p = p

    def _get_live_price(self, s):
        return self._p


class _BB:
    session = None

    def __init__(self, p):
        self.p, self.calls = p, 0

    def get_price(self, s):
        self.calls += 1
        return self.p


def _tm(scn, bb, feed=None):
    t = TM.__new__(TM)
    t.scanner, t.bybit = scn, bb
    t._feed = (lambda: feed)
    return t


def test_stale_scanner_price_is_replaced_by_the_exchange():
    import time as _t
    bb = _BB(610.67)
    t = _tm(_Scn(590.36, _t.time() - 300), bb)
    _check(t._get_current_price('XMRUSDT') == 610.67,
           'кеш сканера старший за поріг → свіжа ціна з біржі')
    _check(t.price_meta('XMRUSDT')['src'] == 'rest', t.price_meta('XMRUSDT'))
    t._get_current_price('XMRUSDT')
    _check(bb.calls == 1, 'прямий запит кешується на кілька секунд')
    print('✓ застаріла ціна сканера більше не судить виходи')


def test_fresh_scanner_price_needs_no_request():
    import time as _t
    bb = _BB(610.67)
    t = _tm(_Scn(608.0, _t.time() - 5), bb)
    _check(t._get_current_price('XMRUSDT') == 608.0 and bb.calls == 0,
           'свіжий кеш — без мережі')
    print('✓ свіжий кеш сканера без зайвих запитів')


def test_feed_wins_and_cache_only_never_hits_the_network():
    import time as _t
    f = _feed()
    f.ingest({'XMRUSDT': 611.0}, now=_t.time())
    bb = _BB(610.67)
    t = _tm(_Scn(590.36, _t.time() - 300), bb, f)
    _check(t._get_current_price('XMRUSDT') == 611.0, 'стрічка — найперше джерело')
    t2 = _tm(_Scn(590.36, _t.time() - 300), bb)
    t2._get_current_price('XMRUSDT', cache_only=True)
    _check(bb.calls == 0, 'cache_only (під локом get_state) — без мережі')
    print('✓ стрічка першою, get_state без мережі')


def test_monitors_pass_the_range_to_every_level_check():
    for fn, book in (('_monitor_position', 'False'), ('_monitor_shadow_position', 'True')):
        src = _fn_src(fn)
        _check(f'self._px_range(symbol, {book}, current_price)' in src, fn)
        _check('self._check_manual_sl_tp(pos, current_price, _lo, _hi)' in src, fn)
        _check("level_hit(pos['side'], 'sl', pos['sl_price'], _lo, _hi)" in src, fn)
        _check("level_hit(pos['side'], 'tp', pos['tp_price'], _lo, _hi)" in src, fn)
    _check('cache_only=True' in _fn_src('get_state'), 'get_state без мережі')
    _check('_fd.watch(' in _fn_src('_tick'), 'стрічка стереже відкриті угоди')
    print('✓ обидва монітори судять рівні за діапазоном')


def test_table_shows_stale_price_age():
    _check('${pxAgeHTML(p)}' in _HTML and 'function pxAgeHTML' in _HTML,
           'колонка Current показує вік ціни')
    print('✓ вік ціни видно в таблиці')


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for f in fns:
        f()
    print(f'\nOK — {len(fns)} тестів')
