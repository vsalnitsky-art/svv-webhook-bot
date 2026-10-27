"""⚡ МММ-CTR — третій банер/монітор на STC (CTR) 1H + 4H (вимога 14.10).

Дослівно: «Створи третій МММ банер (монітор) по такому принципу як і два
попередні МММ, але за основу візьми дані [бейдж ⚡ 1h 🟢 0 · 100% · 4H 🟢 30 ·
40%]. Визнач які відсотки разом 1Н і 4Н максимально відповідають за LONG або
SHORT нахил і рівновагу.»
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


# ═══════════ 1. ПРАВИЛО НАПРЯМКУ (чиста функція) ═════════════════════════
def test_screenshot_case_is_a_strong_long():
    c = _m.ctr_mm_dir(0, 30)          # 1H STC 0 (+1.0), 4H STC 30 (+0.4)
    _check(c['status'] == 'LONG' and c['strength'] == 64, c)
    _check(c['lean_1h'] == 100 and c['lean_4h'] == 40 and not c['conflict'], c)
    print('✓ скрін: 1H 0 · 4H 30 → LONG 64%')


def test_mirror_gives_short():
    c = _m.ctr_mm_dir(100, 70)
    _check(c['status'] == 'SHORT' and c['strength'] == 64, c)
    print('✓ дзеркало → SHORT 64%')


def test_4h_weighs_more_than_1h():
    _check(_m.CTR_MM_W_4H > _m.CTR_MM_W_1H, 'контекст 4H важить більше')
    _check(abs(_m.CTR_MM_W_1H + _m.CTR_MM_W_4H - 1.0) < 1e-9, 'шкала 0-100')
    a = _m.ctr_mm_dir(50, 25)          # лише 4H
    b = _m.ctr_mm_dir(25, 50)          # лише 1H
    _check(a['strength'] > b['strength'], (a, b))
    print('✓ 4H (0.6) важить більше за 1H (0.4)')


def test_balance_threshold_is_the_same_as_other_banners():
    c = _m.ctr_mm_dir(50, 46)          # +0.048 → ⚖
    _check(c['status'] is None, c)
    _check(_m.ctr_mm_dir(45, 44)['status'] == 'LONG', 'вище 10% — напрямок')
    print('✓ ≤ 10% → ⚖ рівновага (той самий поріг MM_BIAS_FLAT)')


def test_opposite_timeframes_are_a_conflict_not_a_direction():
    c = _m.ctr_mm_dir(0, 80)           # 1H +1.0, 4H −0.6 → сума +0.04
    _check(c['status'] is None and c['conflict'], c)
    c2 = _m.ctr_mm_dir(0, 60)          # 1H +1.0, 4H −0.2 → сума +0.28, але конфлікт
    _check(c2['status'] is None and c2['conflict'], c2)
    print('✓ 1H проти 4H → ⚖ рівновага-конфлікт')


def test_missing_timeframe_is_unknown_not_flat():
    _check(_m.ctr_mm_dir(None, 30) is None and _m.ctr_mm_dir(20, 'x') is None, '')
    print('✓ немає TF → None («невідомо» ≠ «рівновага»)')


# ═══════════ 2. ДАНІ: БЮДЖЕТ ЗАПИТІВ ════════════════════════════════════
class _FE:
    def __init__(self, data, fresh=True):
        self.data, self.fresh, self.fetch = data, fresh, []

    def peek_ctr_tf(self, sym, tf):
        v = self.data.get((sym, tf))
        if v is None:
            return None
        return (10.0 if self.fresh else 9999.0, {'stc': v}, 300)

    def get_ctr_tf(self, sym, tf):
        self.fetch.append((sym, tf))
        return {'stc': self.data.get((sym, tf), 50.0)}

    def get(self, sym):
        return {}


def test_fresh_cache_costs_no_requests():
    fe = _FE({('A', '1h'): 10, ('A', '4h'): 20})
    r = _mon._mk()._ctr_mm_readings(['A'], fe)
    _check(r == {'A': (10, 20)} and not fe.fetch, (r, fe.fetch))
    print('✓ свіжий кеш — нуль запитів')


def test_fetch_budget_per_tick_is_respected():
    fe = _FE({})
    syms = [f'C{i}' for i in range(30)]
    _mon._mk()._ctr_mm_readings(syms, fe)
    _check(len(fe.fetch) == _m.CTR_MM_FETCH_PER_TICK, len(fe.fetch))
    print(f'✓ за такт не більше {_m.CTR_MM_FETCH_PER_TICK} запитів')


def test_coins_without_any_value_go_first():
    fe = _FE({('OLD', '1h'): 10, ('OLD', '4h'): 20}, fresh=False)
    _mon._mk()._ctr_mm_readings(['OLD', 'NEW'], fe)
    _check(fe.fetch[:2] == [('NEW', '1h'), ('NEW', '4h')], fe.fetch)
    print('✓ монети без значення — першими в черзі запитів')


# ═══════════ 3. КНИГА В ТАКТІ ДВИГУНА ══════════════════════════════════
def _ff_with(data):
    ff = _mon._mk()
    fe = _FE(data)
    ff._mm_forecast_engine = lambda: fe
    return ff


def test_capture_builds_the_ctr_book_and_its_banner():
    ff = _ff_with({('BTCUSDT', '1h'): 0, ('BTCUSDT', '4h'): 30,
                   ('ETHUSDT', '1h'): 100, ('ETHUSDT', '4h'): 90,
                   ('XRPUSDT', '1h'): 50, ('XRPUSDT', '4h'): 50})
    ff._legacy = {'BTCUSDT': 0.3, 'ETHUSDT': 0.3, 'XRPUSDT': 0.3}
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3, ETHUSDT=0.3, XRPUSDT=0.3), now=1000.0)
    st = ff.mm_monitor_state(book='ctr')
    by = {r['symbol']: r for r in st['rows']}
    _check(st['book'] == 'ctr' and st['enabled'], st)
    _check(by['BTCUSDT']['mm'] == 'LONG' and by['BTCUSDT']['strength'] == 64, by['BTCUSDT'])
    _check(by['ETHUSDT']['mm'] == 'SHORT' and by['XRPUSDT']['mm'] is None, by)
    _check(by['BTCUSDT']['stc_1h'] == 0 and by['BTCUSDT']['stc_4h'] == 30, by['BTCUSDT'])
    b = st['bias']
    # ETH: 0.4·(−1) + 0.6·(−0.8) = −0.88 → важіль (64 − 88) / 152 → SHORT 15.8%
    _check(b['dir'] == 'SHORT' and abs(b['pct'] - 15.8) < 0.1, b)
    print('✓ такт будує книгу CTR і її банер (той самий mm_bias_step)')


def test_disabled_block_clears_everything():
    ff = _ff_with({('BTCUSDT', '1h'): 0, ('BTCUSDT', '4h'): 30})
    ff._legacy = {'BTCUSDT': 0.3}
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1000.0)
    ff._settings['mm_ctr_enabled'] = False
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1030.0)
    st = ff.mm_monitor_state(book='ctr')
    _check(not st['rows'] and not st['bias'] and not st['enabled'], st)
    print('✓ вимкнений блок гасить знімок і банер')


def test_group_open_from_ctr_uses_its_own_direction_and_label():
    ff = _ff_with({('BTCUSDT', '1h'): 0, ('BTCUSDT', '4h'): 30})
    ff._legacy = {'BTCUSDT': -0.5}     # 🧮 каже SHORT — CTR має перемогти
    ff._mm_capture(_mon._fuels(BTCUSDT=-0.5), now=1000.0)
    res = ff.group_open(['BTCUSDT'], book='ctr')
    _check(res['opened'] == 1 and ff.opened[0]['side'] == 'LONG', (res, ff.opened))
    _check('MMC' in ff.opened[0]['kw'].get('opened_by', ''), ff.opened[0])
    _check("'snap': '_mmc_snapshot', 'eng': 'MMC'" in _SRC, 'мітка двигуна MMC')
    print('✓ ✋ групове відкриття бере напрямок CTR, мітка MMC')


def test_state_exposes_the_section():
    _check("'mmc'" in _SRC.split('HEAVY_SECTIONS = ')[1].split('\n')[0], 'секція mmc')
    _check("'mm_ctr': (self.mm_monitor_state(settings, book='ctr')" in _SRC, 'поле mm_ctr')
    _check("'mmc': _want_mmc" in _SRC, 'served.mmc')
    print('✓ get_state: секція mmc → mm_ctr')


# ═══════════ 4. UI ═════════════════════════════════════════════════════
def test_ui_block_is_a_twin_with_its_own_columns():
    a = _HTML.index('id="mmc-panel"')
    blk = _HTML[a:_HTML.index('</table>', a)]
    th = len(re.findall(r'<th[\s>]', blk))
    m = re.search(r"_MMB\.ctr\.cols = \[([^\]]*)\]", _HTML)
    ncols = len(m.group(1).split(','))
    _check(th == ncols + 3, f'<th> {th} ≠ cols {ncols}+3 (TRD + ☑ + символ)')
    _check(f'colspan="{ncols + 3}"' in blk, 'colspan порожнього рядка')
    for need in ("togglePanel('mmc')", "togglePanel('mmclist')", 'ff-mm-ctr-enabled',
                 'ff-mmc-bias-confirm', "mmSetDir('LONG','ctr')", 'mmc-bias-banner',
                 "mmSort('ctr4','ctr')"):
        _check(need in blk or need in _HTML, need)
    _check("'mmc', 'mmclist'" in _HTML and "'mmclist'" in _HTML.split('_PANEL_CLOSED_BY_DEFAULT = ')[1].split(';')[0], 'гармошки')
    _check("mmApplyState(d.mm_ctr, 'ctr')" in _HTML, 'стан доїжджає до книги')
    _check("want.push('mmc')" in _HTML, 'секцію просимо лише розгорнутою')
    _check('mm_ctr_enabled: _c(' in _HTML and 'mmc_str_mins:' in _HTML, 'збереження')
    _check('[_MMB.old, _MMB.new, _MMB.ctr, _MMB.fc, _MMB.pull, _MMB.dec, _MMB.rdy].forEach' in _HTML, 'банер малює 1с-тікер')
    print('✓ UI: блок-двійник, колонки ⚡ 1H/⚡ 4H, гармошки, збереження')


def test_js_stc_cell_follows_the_badge_convention():
    a = _HTML.index('function _mmStcCell(')
    fn = _HTML[a:_HTML.index('\n}', a) + 2]
    js = fn + "\nconsole.log(_mmStcCell(20, 60, '1H'));console.log('|');" \
              "console.log(_mmStcCell(80, -60, '4H'));console.log('|');console.log(_mmStcCell(null,null,'1H'));"
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(js)
        path = f.name
    try:
        out = subprocess.run(['node', path], capture_output=True, text=True, timeout=30).stdout
    finally:
        os.unlink(path)
    p = out.split('|')
    _check('🟢' in p[0] and '#4ade80' in p[0] and '60%' in p[0], p[0])
    _check('🔴' in p[1] and '#f87171' in p[1] and '60%' in p[1], p[1])
    _check('—' in p[2], p[2])
    print('✓ JS: комірка STC — 🟢 <50 LONG / 🔴 >50 SHORT, як бейдж')


# ═══════════ 5. TELEGRAM (вимога 17.10) ════════════════════════════════
def test_ctr_banner_change_goes_to_telegram_once_per_confirmed_flip():
    ff = _ff_with({('BTCUSDT', '1h'): 0, ('BTCUSDT', '4h'): 30})
    ff._settings['mmc_bias_confirm_sec'] = 0
    sent = []
    ff._broadcast_users = lambda cat, key, msg: (sent.append((cat, msg)) or (True, ''))
    ff._legacy = {'BTCUSDT': 0.3}
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1000.0)      # перший такт — мовчки
    ctr = lambda: [m for c, m in sent if 'МММ-CTR' in m]
    _check(not ctr(), f'перший стан після старту не шлемо: {sent}')
    fe = ff._mm_forecast_engine()
    fe.data[('BTCUSDT', '1h')] = 100
    fe.data[('BTCUSDT', '4h')] = 70
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1030.0)
    m = ctr()
    _check(len(m) == 1 and 'SHORT' in m[0] and 'LONG' in m[0] and 'монет:' not in m[0], m)
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1060.0)
    _check(len(ctr()) == 1, 'той самий статус — не подія')
    ff._settings['mmc_bias_tg'] = False
    fe.data[('BTCUSDT', '1h')] = 0
    fe.data[('BTCUSDT', '4h')] = 30
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1090.0)
    _check(len(ctr()) == 1, 'вимкнений тумблер — тиша')
    for need in ('id="ff-mmc-bias-tg"', "mmc_bias_tg: _c('ff-mmc-bias-tg')",
                 "setIf('ff-mmc-bias-tg'", "'⚡ банер CTR'"):
        _check(need in _HTML, need)
    _check("'mmc_bias_tg': True" in _SRC, 'дефолт УВІМК')
    print('✓ ⚡ банер CTR → Telegram: підтверджена зміна, перший такт мовчки, свій тумблер')


if __name__ == '__main__':
    n = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn(); n += 1
    print(f'\nOK — {n} тестів')
