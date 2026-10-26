"""🧮 МММ-Trend + компактний 🧮 МММ LiQ + 🏷 OB 4H у таблицях відкритих угод (12.10).

Вимога дослівно: «Додай до таблиць відкритих угод ще поле "МММ-Trend" і зроби їх
компактнішими ці поля "🧮 МММ LiQ". І додай … помітку, який останній був OB 4Н
(розмалюй відповідним кольором) по монеті. В полі "Side" зроби невеличку
позначку стосовно OB 4Н.»
"""
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon          # noqa: E402  (стаби FF вже там)

_spec = importlib.util.spec_from_file_location(
    "smc_scanner_ob4h_test", os.path.join(_HERE, "detection", "smc_scanner.py"))
_sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sc)

_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()
_FLASK = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
_SC_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


# ⚠️ 22.10: колонки «🧮 МММ LiQ»/«🧮 МММ-Trend» у таблицях угод замінено на
# «🧠 МММ-Decision» + 🚨 сигнал (див. test_trades_mmd.py); `mmn_snapshot_for`
# видалено разом із колонкою.
def test_flask_only_reads_snapshots():
    i = _FLASK.index("_dec_map = ff.dec_trade_for(")
    blk = _FLASK[i - 600:i + 1600]
    _check("pos['mmd'] = _dec_map[_dk]" in blk, 'mmd не кладеться в позицію')
    _check("pos['ob4h'] = _ob4_map[sym]" in blk, 'ob4h не кладеться в позицію')
    _check('ob_mark_for(' in blk and 'get_smc_ob_state' not in blk,
           'OB 4H на гарячому ендпоінті мусить бути ЧИТАННЯМ памʼяті, без БД')
    print('✓ /api/tm/state лише читає знімки (без розрахунків і БД)')


# ═══════════════════════ 🏷 OB 4H у сканері ═════════════════════════════════
def _scanner(ret):
    s = _sc.SMCScanner.__new__(_sc.SMCScanner)
    s._ob_mark = {}
    s._ob_on_htf = lambda sym, md, tf, ob_tf, same: ret
    return s


def test_mark_comes_from_the_same_htf_detector():
    ob = {'bias': 'BEARISH', 'created_by_tag': 'CHOCH', 'bar_time': 1_700_000_000_000,
          'bar_high': 101.0, 'bar_low': 99.0}
    s = _scanner((ob, ''))
    s._ob_mark_update('btcusdt', None)
    m = s.ob_mark_for(['BTCUSDT'])['BTCUSDT']
    _check(m['bias'] == 'BEARISH' and m['tag'] == 'CHOCH' and m['tf'] == '4h'
           and m['high'] == 101.0 and m['low'] == 99.0, m)
    src = _SC_SRC[_SC_SRC.index('def _ob_mark_update('):]
    src = src[:src.index('def ob_mark_for(')]
    _check('self._ob_on_htf(' in src and 'OB_MARK_TF' in src,
           'позначка мусить рахуватись ТИМ САМИМ _ob_on_htf, що й алерт')
    _check(_sc.OB_MARK_TF == '4h', 'TF позначки — 4h')
    print('✓ OB 4H рахує той самий _ob_on_htf, що й алерт 🆕 Новий OB')


def test_failed_fetch_keeps_the_previous_mark():
    s = _scanner((None, 'бари 4h не завантажились'))
    s._ob_mark['BTCUSDT'] = {'bias': 'BULLISH', 'ts': 1}
    s._ob_mark_update('BTCUSDT', None)
    _check(s._ob_mark['BTCUSDT']['bias'] == 'BULLISH',
           '«бари не прийшли» ≠ «блоку немає»')
    s2 = _scanner((None, ''))                     # порахували — блоку немає
    s2._ob_mark_update('BTCUSDT', None)
    _check(s2._ob_mark['BTCUSDT']['bias'] is None, 'відсутність блоку теж факт')
    _check(s2.ob_mark_for(['XRPUSDT']) == {}, 'невідома монета → нічого')
    print('✓ збій мережі не стирає позначку, «блоку немає» — чесне None')


def test_scan_updates_the_mark_each_cycle():
    i = _SC_SRC.index('self._ob_mark_update(symbol, md)')
    j = _SC_SRC.index('# === Compute PD Zone')
    _check(i < j, 'оновлення позначки мусить стояти в проході скану')
    print('✓ скан оновлює позначку OB 4H (кеш до закриття 4h-бару)')


# ═══════════════════════ UI ═════════════════════════════════════════════════
def test_tables_have_the_new_column_and_matching_colspans():
    for tid in ('tm-open-table', 'tm-shadow-open-table'):
        i = _HTML.index(f'id="{tid}"')
        tbl = _HTML[i:_HTML.index('</table>', i)]
        _check('>🧮 МММ-Trend</th>' in tbl, f'{tid}: немає колонки МММ-Trend')
        n = len(re.findall(r'<th[\s>]', tbl))
        m = re.search(r'colspan="(\d+)"', tbl)
        _check(m and int(m.group(1)) == n, f'{tid}: colspan ≠ {n}')
    _check(_HTML.count('colspan="17" class="tm-empty-msg">No open positions') == 2
           and _HTML.count('colspan="16" class="tm-empty-msg">No paper positions') == 2,
           'colspan у JS-заглушках не оновлено')
    # ⚠️ 22.10: позначка OB 4H стоїть ЛІВОРУЧ від сторони угоди.
    _check(_HTML.count('${ob4hMarkHTML(p, null, true)}${p.side}') == 2,
           'позначка OB 4H мусить стояти ЛІВОРУЧ у полі Side обох таблиць')
    print('✓ колонка МММ-Decision + позначка OB 4H ліворуч у Side, colspan 16/15')


def _js(body):
    src = ''
    for f in ('function fmtPriceJS(', 'function ob4hMarkHTML('):
        a = _HTML.index(f)
        src += _HTML[a:_HTML.index('\n}', a) + 2] + '\n'
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as fh:
        fh.write(src + body)
        path = fh.name
    try:
        return subprocess.run(['node', path], capture_output=True, text=True,
                              timeout=30).stdout
    finally:
        os.unlink(path)


def test_js_side_mark_render():
    out = _js(r'''
console.log(ob4hMarkHTML({side:'SHORT', ob4h:{bias:'BEARISH', tag:'CHOCH', bar_time:1700000000000, low:1, high:2}}));
console.log('|');
console.log(ob4hMarkHTML({side:'LONG', ob4h:{bias:'BEARISH'}}));
console.log('|');
console.log(ob4hMarkHTML({side:'LONG'}));
console.log('|');
console.log(JSON.stringify(ob4hMarkHTML({side:'LONG', ob4h:{bias:'BULLISH'}}, null, true)));
''')
    parts = out.split('|')
    _check(len(parts) == 4, out)
    _check('4H▼' in parts[0] and '#ef4444' in parts[0] and 'CHoCH' in parts[0]
           and 'ЗА напрямком' in parts[0], f'ведмежий 4H за шортом: {parts[0]}')
    _check('ПРОТИ напрямку' in parts[1] and '#fbbf24' in parts[1],
           f'проти угоди → бурштинова рамка: {parts[1]}')
    _check(parts[2].strip() == '', 'немає даних → позначки немає')
    left = parts[3].strip()
    _check(left.startswith('"<span') and 'tm-ob4h-l' in left,
           f'ліва позначка: без пробілу попереду і з класом відступу праворуч: {left}')
    print('✓ JS: позначка 4H▲/▼ кольором + «за/проти», ліворуч — без зсуву')


if __name__ == '__main__':
    n = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn(); n += 1
    print(f'\nOK — {n} тестів')
