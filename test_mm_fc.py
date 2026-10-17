"""🔮 МММ-LOCAL — четвертий банер/монітор на 🔮 Forecast 1H + 4H (вимога 15.10).

Дослівно: «Зроби ще один такий самий МММ (монітор) банер, як і "⚡ МММ-CTR"
тільки на основі ось цих даних» (бейджі «1H: 🔴 SHORT −12% · 50%» /
«4H: 🔴 SHORT −41% · 60%»).
"""
import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon   # noqa: E402  (ізольований fuel_filter + стаби)

_m = _mon._m
_HTML = _mon._HTML
_SRC = _mon._SRC


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _f(pct, conf=60):
    side = 1 if pct > 0 else (-1 if pct < 0 else 0)
    return {'side': side, 'pct': pct, 'conf': conf}


# ═══════════ 1. ПРАВИЛО НАПРЯМКУ (чиста функція) ═════════════════════════
def test_screenshot_case_is_a_short():
    c = _m.fc_mm_dir(_f(-12, 50), _f(-41, 60))   # −0.048 − 0.246 = −0.294
    _check(c['status'] == 'SHORT' and c['strength'] == 29, c)
    _check(c['lean_1h'] == -12 and c['lean_4h'] == -41 and not c['conflict'], c)
    print('✓ скрін: 1H −12% · 4H −41% → SHORT 29%')


def test_mirror_gives_long():
    c = _m.fc_mm_dir(_f(12, 50), _f(41, 60))
    _check(c['status'] == 'LONG' and c['strength'] == 29, c)
    print('✓ дзеркало → LONG 29%')


def test_same_weights_as_ctr_block():
    _check(_m.FC_MM_W_1H == _m.CTR_MM_W_1H and _m.FC_MM_W_4H == _m.CTR_MM_W_4H,
           'одне правило зважування двох TF на обидва блоки')
    a = _m.fc_mm_dir(_f(0, 50), _f(40))
    b = _m.fc_mm_dir(_f(40), _f(0, 50))
    _check(a['strength'] > b['strength'], (a, b))
    print('✓ ваги 0.4/0.6 — ті самі, що в ⚡ МММ-CTR; 4H важить більше')


def test_confidence_is_not_counted_twice():
    a = _m.fc_mm_dir(_f(-30, 60), _f(-30, 60))
    b = _m.fc_mm_dir(_f(-30, 90), _f(-30, 90))
    _check(a['strength'] == b['strength'] == 30, (a, b))
    print('✓ впевненість — похідна від pct, у силу вдруге не входить')


def test_balance_and_conflict():
    _check(_m.fc_mm_dir(_f(-5, 50), _f(-8, 50))['status'] is None, 'слабкий → ⚖')
    c = _m.fc_mm_dir(_f(50), _f(-30))      # 1H LONG, 4H SHORT
    _check(c['status'] is None and c['conflict'], c)
    print('✓ ≤ 10% → ⚖ рівновага · 1H проти 4H → ⚖ конфлікт')


def test_missing_or_not_computed_is_unknown():
    _check(_m.fc_mm_dir(None, _f(-41)) is None, 'немає 1H')
    _check(_m.fc_mm_dir({'side': 0, 'pct': 0, 'conf': 0}, _f(-41)) is None,
           '«not computed yet» (conf 0) ≠ рівновага')
    print('✓ немає прогнозу / conf 0 → None («невідомо» ≠ «рівновага»)')


# ═══════════ 2. КНИГА В ТАКТІ ДВИГУНА ══════════════════════════════════
class _FE:
    def __init__(self, data):
        self.data, self.calls = data, []

    def get(self, sym):
        self.calls.append(sym)
        d = self.data.get(sym)
        if not d:
            return None
        return {'forecast_1h': {'side': 1 if d[0] > 0 else -1, 'pct': d[0], 'confidence': 60},
                'forecast_4h': {'side': 1 if d[1] > 0 else -1, 'pct': d[1], 'confidence': 60}}

    def ensure_fresh(self, *a, **k):
        raise AssertionError('МММ-Local не має ходити по свічки')


def _ff_with(data):
    ff = _mon._mk()
    fe = _FE(data)
    ff._mm_forecast_engine = lambda: fe
    for k in ('mm_ctr_enabled',):
        ff._settings[k] = False
    return ff, fe


def test_capture_builds_the_fc_book_and_its_banner():
    ff, fe = _ff_with({'BTCUSDT': (-12, -41), 'ETHUSDT': (60, 70), 'XRPUSDT': (3, -2)})
    ff._legacy = {'BTCUSDT': 0.3, 'ETHUSDT': 0.3, 'XRPUSDT': 0.3}
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3, ETHUSDT=0.3, XRPUSDT=0.3), now=1000.0)
    st = ff.mm_monitor_state(book='fc')
    by = {r['symbol']: r for r in st['rows']}
    _check(st['book'] == 'fc' and st['enabled'], st)
    _check(by['BTCUSDT']['mm'] == 'SHORT' and by['BTCUSDT']['strength'] == 29, by['BTCUSDT'])
    _check(by['ETHUSDT']['mm'] == 'LONG' and by['XRPUSDT']['mm'] is None, by)
    _check(by['BTCUSDT']['f1']['pct'] == -12 and by['BTCUSDT']['f4']['conf'] == 60, by['BTCUSDT'])
    # ETH: 0.4·0.6 + 0.6·0.7 = 0.66 → важіль (66 − 29) / (66 + 29 + 0) → LONG 38.9%
    b = st['bias']
    _check(b['dir'] == 'LONG' and abs(b['pct'] - 38.9) < 0.2, b)
    _check(fe.calls.count('BTCUSDT') == 1, f'кеш читається ОДИН раз на монету: {fe.calls}')
    print('✓ такт будує книгу Прогнозу і її банер (той самий mm_bias_step)')


def test_disabled_block_clears_everything():
    ff, _ = _ff_with({'BTCUSDT': (-12, -41)})
    ff._legacy = {'BTCUSDT': 0.3}
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1000.0)
    ff._settings['mm_fc_enabled'] = False
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1030.0)
    st = ff.mm_monitor_state(book='fc')
    _check(not st['rows'] and not st['bias'] and not st['enabled'], st)
    print('✓ вимкнений блок гасить знімок і банер')


def test_group_open_from_fc_uses_its_own_direction_and_label():
    ff, _ = _ff_with({'BTCUSDT': (-12, -41)})
    ff._legacy = {'BTCUSDT': 0.5}      # 🧮 каже LONG — Прогноз має перемогти
    ff._mm_capture(_mon._fuels(BTCUSDT=0.5), now=1000.0)
    res = ff.group_open(['BTCUSDT'], book='fc')
    _check(res['opened'] == 1 and ff.opened[0]['side'] == 'SHORT', (res, ff.opened))
    _check('MMF' in ff.opened[0]['kw'].get('opened_by', ''), ff.opened[0])
    print('✓ ✋ групове відкриття бере напрямок Прогнозу, мітка MMF')


def test_state_persists_and_exposes_the_section():
    _check("'mmf'" in _SRC.split('HEAVY_SECTIONS = ')[1].split('\n')[0], 'секція mmf')
    _check("'mm_fc': (self.mm_monitor_state(settings, book='fc')" in _SRC, 'поле mm_fc')
    _check("'mmf': _want_mmf" in _SRC, 'served.mmf')
    _check("'mm_bias_fc':" in _SRC and "('mmf_state_since', '_mmf_state_since')" in _SRC,
           'банер і таймери переживають рестарт')
    print('✓ get_state: секція mmf → mm_fc; персист банера')


def test_labels_mirrors():
    sl = open(os.path.join(_HERE, 'detection', 'signal_labels.py'), encoding='utf-8').read()
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check("'MMF':              '🔮 МММ-Local'" in sl and sl.count("'MMF': '🔮'") == 2, 'labels')
    _check('"MMF": "🔮 МММ-Local"' in js and js.count('"MMF": "🔮"') == 2, 'infosite')
    _check("'MMF': '🔮 МММ-Local'" in _HTML and _HTML.count("'MMF': '🔮'") == 2, 'page')
    print('✓ мітка MMF = 🔮 МММ-Local у Python і обох JS-дзеркалах')


# ═══════════ 3. UI ═════════════════════════════════════════════════════
def test_ui_block_is_a_twin_with_its_own_columns():
    a = _HTML.index('id="mmf-panel"')
    blk = _HTML[a:_HTML.index('</table>', a)]
    th = len(re.findall(r'<th[\s>]', blk))
    m = re.search(r"_MMB\.fc\.cols = \[([^\]]*)\]", _HTML)
    ncols = len(m.group(1).split(','))
    _check(th == ncols + 2, f'<th> {th} ≠ cols {ncols}+2')
    _check(f'colspan="{ncols + 2}"' in blk, 'colspan порожнього рядка')
    _check('mmc' not in blk and "'ctr'" not in blk, 'у двійнику лишились id/книга CTR')
    for need in ("togglePanel('mmf')", "togglePanel('mmflist')", 'ff-mm-fc-enabled',
                 'ff-mmf-bias-confirm', "mmSetDir('LONG','fc')", 'mmf-bias-banner',
                 "mmSort('fc4','fc')"):
        _check(need in blk or need in _HTML, need)
    _check("'mmf', 'mmflist'" in _HTML and "'mmflist'])" in _HTML, 'гармошки')
    _check("mmApplyState(d.mm_fc, 'fc')" in _HTML, 'стан доїжджає до книги')
    _check("want.push('mmf')" in _HTML, 'секцію просимо лише розгорнутою')
    _check('mm_fc_enabled: _c(' in _HTML and 'mmf_str_mins:' in _HTML, 'збереження')
    _check('[_MMB.old, _MMB.new, _MMB.ctr, _MMB.fc].forEach' in _HTML, 'банер малює 1с-тікер')
    i = _HTML.index('id="mmc-panel"'); j = _HTML.index('id="mmf-panel"')
    _check(i < j, 'блок 🔮 стоїть під ⚡ МММ-CTR')
    print('✓ UI: блок-двійник під ⚡ CTR, колонки 🔮 1H/4H, гармошки, збереження')


def test_js_cell_matches_the_badge():
    a = _HTML.index('function _mmFcFullCell(')
    fn = _HTML[a:_HTML.index('\n}', a) + 2]
    js = fn + ("\nconsole.log(_mmFcFullCell({side:-1,pct:-12,conf:50},'1H'));console.log('|');"
               "console.log(_mmFcFullCell({side:1,pct:41,conf:60},'4H'));console.log('|');"
               "console.log(_mmFcFullCell({side:0,pct:0,conf:0},'1H'));")
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(js)
        path = f.name
    try:
        out = subprocess.run(['node', path], capture_output=True, text=True, timeout=30).stdout
    finally:
        os.unlink(path)
    p = out.split('|')
    _check('🔴' in p[0] and 'SHORT' in p[0] and '−12%' in p[0] and '50%' in p[0], p[0])
    _check('🟢' in p[1] and 'LONG' in p[1] and '+41%' in p[1] and '60%' in p[1], p[1])
    _check('—' in p[2], 'conf 0 → «—»')
    print('✓ JS: комірка «🔴 SHORT −12% · 50%» як бейдж над графіком')


# ═══════════ 4. TELEGRAM (вимога 15.10) ════════════════════════════════
def test_banner_change_goes_to_telegram_once_per_confirmed_flip():
    ff, _ = _ff_with({'BTCUSDT': (-50, -60)})
    ff._settings['mmf_bias_confirm_sec'] = 0
    sent = []
    ff._broadcast_users = lambda cat, key, msg: (sent.append((cat, msg)) or (True, ''))
    ff._legacy = {'BTCUSDT': 0.3}
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1000.0)      # перший такт — мовчки
    _check(not sent, f'перший стан після старту не шлемо: {sent}')
    ff._mm_forecast_engine().data['BTCUSDT'] = (50, 60)
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1030.0)
    _check(len(sent) == 1 and sent[0][0] == 'btc' and 'МММ-LOCAL' in sent[0][1]
           and 'LONG' in sent[0][1], sent)
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1060.0)
    _check(len(sent) == 1, 'той самий статус — не подія')
    ff._settings['mmf_bias_tg'] = False
    ff._mm_forecast_engine().data['BTCUSDT'] = (-50, -60)
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1090.0)
    _check(len(sent) == 1, 'вимкнений тумблер — тиша')
    print('✓ 🔮 банер → Telegram: лише підтверджена зміна, перший такт мовчки, свій тумблер')


def test_telegram_toggle_on_page():
    for need in ('id="ff-mmf-bias-tg"', "mmf_bias_tg: _c('ff-mmf-bias-tg')",
                 "setIf('ff-mmf-bias-tg'", "'🔮 банер Local'"):
        _check(need in _HTML, need)
    _check("'mmf_bias_tg': True" in _SRC, 'дефолт УВІМК')
    print('✓ UI: тумблер 📨 🔮 банер у гармошці сповіщень')


if __name__ == '__main__':
    n = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn(); n += 1
    print(f'\nOK — {n} тестів')
