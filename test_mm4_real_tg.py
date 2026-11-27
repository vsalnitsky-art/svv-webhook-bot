"""25.11: (1) події РЕАЛЬНИХ угод бота → лише особисто власнику від головного
TG-бота, а не в спільну групу; (2) інфосайт — зведення чотирьох банерів
(🧮 VSV-Trend · 🔮 VSV-Local · 🧠 VSV-Decision · 🧩 VSV-Готовність): загальний
відсоток + додаткова візуалізація, коли всі чотири зійшлись у напрямку."""
import os
import sys
import threading
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_mm_monitor as _mon   # noqa: E402  (ізольований fuel_filter)
import test_autosl_chain as _base   # noqa: E402  (ізольований trade_manager)

FFM = _mon._m
FF = _mon.FF
TM = _base.TM
APP = open(os.path.join(_HERE, 'infosite', 'js', 'app.js'), encoding='utf-8').read()
IDX = open(os.path.join(_HERE, 'infosite', 'index.html'), encoding='utf-8').read()
CSS = open(os.path.join(_HERE, 'infosite', 'css', 'style.css'), encoding='utf-8').read()
TMSRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()


def _it(key, d, net, on=True, coins=10, since=100):
    return {'key': key, 'icon': '·', 'name': key, 'on': on, 'dir': d, 'net': net,
            'pct': abs(net) * 100, 'coins': coins, 'since': since}


def test_all_four_agree_gives_the_extra_state():
    m = FFM.mm4_consensus([_it('new', 'LONG', 0.5, since=100), _it('fc', 'LONG', 0.3, since=400),
                           _it('dec', 'LONG', 0.6, since=200), _it('rdy', 'LONG', 0.2, since=300)])
    assert m['agree'] == 'LONG' and m['agree_since'] == 400, m
    t = m['total']
    assert t['dir'] == 'LONG' and t['pct'] == 40.0 and t['long_pct'] == 70.0 and t['n_long'] == 4, t


def test_no_agreement_unless_all_four():
    base = [_it('new', 'SHORT', -0.5), _it('fc', 'SHORT', -0.3), _it('dec', 'SHORT', -0.6)]
    assert FFM.mm4_consensus(base + [_it('rdy', None, 0.02)])['agree'] is None, '⚖ ламає збіг'
    m = FFM.mm4_consensus(base + [_it('rdy', 'SHORT', -0.4, on=False)])
    assert m['agree'] is None and m['total']['n'] == 3, 'вимкнений блок — не збіг, у відсоток не входить'
    assert FFM.mm4_consensus(base + [_it('rdy', 'LONG', 0.4)])['agree'] is None
    mix = FFM.mm4_consensus([_it('new', 'LONG', 0.2), _it('fc', 'SHORT', -0.15),
                             _it('dec', None, 0.0), _it('rdy', 'LONG', 0.1)])
    assert mix['total']['dir'] is None and mix['total']['pct'] == 3.8, 'слабка перевага → ⚖'
    assert FFM.mm4_consensus([_it('new', None, 0, coins=0)])['total'] is None


def test_view_reads_the_same_banner_snapshots():
    ff = FF.__new__(FF)
    ff._lock = threading.RLock()
    for i, attr in enumerate(('_mm_bias_new', '_mm_bias_fc', '_mm_bias_dec', '_mm_bias_rdy')):
        setattr(ff, attr, {'dir': 'SHORT', 'net': -0.3, 'pct': 30.0, 'coins': 5, 'since': 10 + i})
    ff._mmn_on = ff._mmf_on = ff._mmd_on = True
    ff._mmr_on = None                              # тумблер ще невідомий → судимо за даними
    v = ff.mm4_view()
    assert [i['key'] for i in v['items']] == ['new', 'fc', 'dec', 'rdy']
    assert v['agree'] == 'SHORT' and v['agree_since'] == 13, v
    ff._mmd_on = False
    assert ff.mm4_view()['agree'] is None


def test_infosite_renders_total_chips_and_agreement():
    assert 'id="spb-mm4"' in IDX
    f = APP[APP.index('function mm4Render('):APP.index('setInterval(function () { document.querySelectorAll(".spb-tm')]
    # 26.11 (дизайн): зведення «VSV · СИНХРОН», бейдж ЗБІГ 4/4 з таймером прибрано,
    # збіг показує підсвічування картки.
    for w in ('VSV · СИНХРОН', 'mm4-up', 'mm4-dn', 'items.map(mm4Chip)'):
        assert w in f, w
    assert 'mm4Render(d.mm4)' in APP
    # Колір: напрямок стоїть ПІД підписом — більша частина чипа праворуч.
    assert 'var split = 50 - pct / 2;' in APP
    assert ('.mm4-chip.up{background:linear-gradient(90deg,#a92b3c 0%,#a92b3c var(--split),'
            '#08775b var(--split),#08775b 100%)}') in CSS
    assert ('.mm4-chip.dn{background:linear-gradient(90deg,#08775b 0%,#08775b var(--split),'
            '#a92b3c var(--split),#a92b3c 100%)}') in CSS
    src = open(os.path.join(_HERE, 'web', 'info_portal.py'), encoding='utf-8').read()
    assert "'mm4': mm4" in src and 'mm4_view()' in src


def _tm():
    t = TM.__new__(TM)
    t._lock = threading.RLock()
    t._settings = {'telegram_alerts': True, 'test_telegram_alerts': True}
    t.owner_uid = None
    t.notifier = None
    return t


def test_real_trade_messages_go_privately_not_to_the_group():
    sent, group = [], []
    fake = types.ModuleType('web.tg_bot')
    fake.tg_send = lambda chat, text, buttons=None: sent.append((chat, text)) or True
    fake._admin_chat = lambda: '777'
    fake.notify_category = lambda cat, text: group.append((cat, text)) or True
    old = sys.modules.get('web.tg_bot')
    sys.modules['web.tg_bot'] = fake
    try:
        t = _tm()
        t._notify('▶️ ВІДКРИТО #BTCUSDT', category='trades')
        assert sent == [('777', '▶️ ВІДКРИТО #BTCUSDT')] and group == [], (sent, group)
        t._notify('▶️ ВІДКРИТО #ETHUSDT 🧪 ТЕСТ', is_test=True, category='trades')
        assert group == [('trades', '▶️ ВІДКРИТО #ETHUSDT 🧪 ТЕСТ')] and len(sent) == 1, 'paper — як і раніше в групу'
        # головний бот не прийняв → резервний приватний notifier, у групу — ні
        fake.tg_send = lambda chat, text, buttons=None: False
        got = []
        t.notifier = types.SimpleNamespace(send_message=lambda m: got.append(m) or True)
        t._notify('⏹ ЗАКРИТО #BTCUSDT', category='trades')
        assert got == ['⏹ ЗАКРИТО #BTCUSDT'] and len(group) == 1
    finally:
        if old is not None:
            sys.modules['web.tg_bot'] = old
        else:
            sys.modules.pop('web.tg_bot', None)
    # BE-move paper більше не тікає в приватний чат як «реальна»
    assert TMSRC.count("buf_note}\", is_test=is_shadow, category='trades')") == 2
    body = TMSRC[TMSRC.index('def _real_notify'):TMSRC.index('def _dir_dot')]
    assert 'notify_category' not in body, 'реальні — ніколи не в групу'


if __name__ == '__main__':
    n = 0
    for k, fn in list(globals().items()):
        if k.startswith('test_') and callable(fn):
            fn(); n += 1; print('ok', k)
    print(f'OK — {n} тестів')
