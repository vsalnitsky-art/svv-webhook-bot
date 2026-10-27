"""💰 PnL відкритих реальних угод з ПЛЕЧЕМ і КОМІСІЯМИ (вимога 27.10).

Дослівно: «Інформацію по відкритим реальним угодам відображай в таблиці
відповідно плеча і комісії і розрахунки також це враховуй.»
"""
import os
import re
import sys
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_autosl_chain as _tmb  # noqa: E402  (ізольований trade_manager)

tm_mod = sys.modules[_tmb.TM.__module__]
real_pnl_net = tm_mod.real_pnl_net
_TM_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def test_long_with_leverage_and_fees():
    # 10 монет по $100, ціна +1% → брутто $10; комісії 0.055%: 0.55 + 0.5555
    n = real_pnl_net('LONG', 100, 101, 10, 10, 0.055)
    _check(abs(n['gross_usd'] - 10) < 1e-9, n)
    _check(abs(n['fees_usd'] - (0.55 + 0.5555)) < 1e-4, n)
    _check(abs(n['net_usd'] - (10 - 1.1055)) < 1e-4, n)
    _check(abs(n['margin_usd'] - 100) < 1e-9, n)          # 1000 / ×10
    _check(abs(n['roi_pct'] - 8.8945) < 1e-3, n)          # ≈ рух 1% × плече − комісії
    _check(abs(n['move_pct'] - 1.0) < 1e-9, n)
    print('✓ LONG: ROI = нетто / маржа, з плечем і обома комісіями')


def test_short_and_exchange_gross_wins():
    n = real_pnl_net('SHORT', 100, 99, 10, 5, 0.0, gross_usd=9.5)
    _check(n['gross_usd'] == 9.5, 'брутто Bybit має пріоритет над розрахунком')
    _check(abs(n['roi_pct'] - 9.5 / 200 * 100) < 1e-9, n)
    _check(abs(n['move_pct'] - 1.0) < 1e-9, 'SHORT: падіння ціни = плюс')
    print('✓ SHORT + unrealised PnL біржі як брутто')


def test_flat_price_is_a_loss_by_fees():
    n = real_pnl_net('LONG', 100, 100, 1, 20, 0.055)
    _check(n['net_usd'] < 0 and n['roi_pct'] < 0, 'на вході угода в мінусі на комісії')
    print('✓ ціна на вході = мінус на комісії')


def test_no_data_no_numbers():
    for args in (('LONG', 0, 1, 1, 10, 0.05), ('LONG', 1, 1, 0, 10, 0.05),
                 ('X', 1, 1, 1, 10, 0.05), ('LONG', 'a', 1, 1, 10, 0.05)):
        _check(real_pnl_net(*args) is None, args)
    print('✓ немає даних → None')


def test_state_attaches_net_for_real_book():
    i = _TM_SRC.index('def get_state(')
    body = _TM_SRC[i:_TM_SRC.index('shadow_positions = []', i)]
    _check("pos_dict['pnl_net'] = real_pnl_net(" in body, 'get_state не віддає pnl_net')
    _check("get('leverage')" in body and 'paper_fee_pct' in body, 'плече/комісія не беруться')
    print('✓ get_state кладе pnl_net у реальні позиції')


def test_page_renders_net_cell_and_ukrainian_stats():
    _check('${realPnlCellHTML(p, usd, syncMark)}' in _HTML, 'комірка PnL не підключена')
    _check('Decision validation' not in _HTML, 'лишився англійський підпис')
    _check('Перевірка оцінки входу' in _HTML, 'немає українського підпису')
    # Панель P&L у стилі Bybit (27.10) — підписи карток.
    for w in ('Чистий P&amp;L', 'Відсоток успішних', 'Закрито угод'):
        _check(w in _HTML, w)
    i = _HTML.index('function realPnlCellHTML(')
    fn = _HTML[i:_HTML.index('\n}\n', i) + 2]
    js = fn + "\nconst h = realPnlCellHTML({pnl_pct: 1, pnl_source: 'bybit', pnl_net: " \
        "{roi_pct: 8.89, net_usd: 8.89, gross_usd: 10, fee_in_usd: 0.55, fee_out_usd: 0.56, " \
        "margin_usd: 100, move_pct: 1, lev: 10, fee_pct: 0.055}}, '', '');\nconsole.log(h);"
    out = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=20)
    _check(out.returncode == 0, out.stderr)
    _check('+8.89%' in out.stdout and '×10' in out.stdout and 'нетто' in out.stdout, out.stdout)
    print('✓ сторінка: ROI · нетто · ×плече, підписи українською')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
