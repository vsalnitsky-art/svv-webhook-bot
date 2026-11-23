"""23.11: 💥 підтвердження розряду / 🪤 хибний старт · 🛑 «SL з» ШІ вимикається
(деф. ВИМК) · 🧩 фільтр і статистика за банером «МММ-Готовність» · ⏰ будильник
перемикає ШІ · рядок «ШІ вчиться» прибрано з таблиці."""
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
for _n in ('pybit', 'pybit.unified_trading'):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules['pybit.unified_trading'].HTTP = object
if 'detection' not in sys.modules:
    _pkg = types.ModuleType('detection')
    _pkg.__path__ = [os.path.join(ROOT, 'detection')]
    sys.modules['detection'] = _pkg
_lg = types.ModuleType('detection.activity_log')
_lg.log_activity = lambda *a, **k: None
sys.modules.setdefault('detection.activity_log', _lg)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load('detection.ai_signals', 'detection/ai_signals.py')
B = _load('detection.ai_brain', 'detection/ai_brain.py')
SB = _load('detection.sbr_signals', 'detection/sbr_signals.py')


def _bars(closes, hi_pad=0.2, t0=1_700_000_000):
    """1h-свічки: close з ряду, high/low ± pad."""
    return [{'t': (t0 + i * 3600) * 1000, 'p': c, 'h': c + hi_pad, 'l': c - hi_pad, 'o': c}
            for i, c in enumerate(closes)]


FLAT = [100.0] * 30            # коробка 99.8 … 100.2


def test_defaults():
    d = A.ai_settings_norm({})
    assert d['sl_enabled'] is False
    assert d['discharge_confirm'] is True and d['discharge_bars'] == 1
    assert d['false_start_exit'] is True and d['rdy_filter'] == 'off'
    assert A.ai_settings_norm({'rdy_filter': 'zzz'})['rdy_filter'] == 'off'
    assert A.ai_settings_norm({'discharge_bars': 9})['discharge_bars'] == 4


def test_discharge_states():
    # закритого пробою немає, жива ціна за межею → live
    r = A.discharge_state(_bars(FLAT), 'LONG', live_price=101)
    assert r['state'] == 'live'
    assert A.discharge_state(_bars(FLAT), 'LONG', live_price=100)['state'] == 'none'
    # пробій закрито, підтверджень ще немає → pending
    r = A.discharge_state(_bars(FLAT + [101.0]), 'LONG', live_price=101.2)
    assert r['state'] == 'pending' and abs(r['edge'] - 100.2) < 1e-9
    # наступна свічка втримала → confirmed
    r = A.discharge_state(_bars(FLAT + [101.0, 101.3]), 'LONG', live_price=101.4)
    assert r['state'] == 'confirmed', r
    # ...але жива ціна вже в коробці → ще не підтверджено
    assert A.discharge_state(_bars(FLAT + [101.0, 101.3]), 'LONG', live_price=100.0)['state'] == 'pending'
    # свічка закрилась назад у коробці → хибний старт
    r = A.discharge_state(_bars(FLAT + [101.0, 100.0]), 'LONG', live_price=100.0)
    assert r['state'] == 'false'
    # SHORT дзеркально
    r = A.discharge_state(_bars(FLAT + [99.0, 98.8]), 'SHORT', live_price=98.7)
    assert r['state'] == 'confirmed'
    # need=2 → одна свічка ще не підтвердження
    assert A.discharge_state(_bars(FLAT + [101.0, 101.3]), 'LONG', 101.4, need=2)['state'] == 'pending'


def test_false_start_after_entry():
    bars = _bars(FLAT + [101.0, 101.3, 100.1])
    oa = bars[-2]['t'] // 1000 + 10          # відкрито під час передостанньої свічки
    hit = A.false_start_hit('LONG', 100.2, bars, oa)
    assert hit and hit['close'] == 100.1
    assert A.false_start_hit('LONG', 100.2, bars[:-1], oa) is None
    # свічки, що закрились ДО входу, не рахуються
    assert A.false_start_hit('LONG', 100.2, _bars(FLAT + [100.0]), 1_800_000_000) is None


def test_entry_block_requires_confirmed_discharge():
    s = A.ai_settings_norm({})
    brain = {'ok': True, 'confidence': 70}
    why = A.ai_entry_block({}, 'LONG', s, brain, None, 0, [], None, 1e9,
                           discharge={'state': 'live', 'note': 'пробій лише на живій'}, scen_key='E2')
    assert why.startswith('💥 розряд не підтверджено')
    assert A.ai_entry_block({}, 'LONG', s, brain, None, 0, [], None, 1e9,
                            discharge={'state': 'confirmed'}, scen_key='E2') == ''
    # інші сценарії цієї перевірки не мають
    assert A.ai_entry_block({}, 'LONG', s, brain, None, 0, [], None, 1e9, scen_key='E1') == ''
    # вимкнено → не діє
    s2 = A.ai_settings_norm({'discharge_confirm': False})
    assert A.ai_entry_block({}, 'LONG', s2, brain, None, 0, [], None, 1e9,
                            discharge={'state': 'false'}, scen_key='E2') == ''


def test_rdy_filter():
    v = {'on': True, 'dir': 'LONG', 'coin': 'SHORT'}
    assert A.rdy_block('LONG', v, 'off') == ''
    assert A.rdy_block('LONG', v, 'banner') == ''
    assert 'вкладці SHORT' in A.rdy_block('LONG', v, 'coin')
    assert A.rdy_block('SHORT', v, 'banner').startswith('🧩 банер МММ-Готовність LONG')
    assert 'вимкнено' in A.rdy_block('LONG', {'on': False}, 'banner')
    assert '⚖' in A.rdy_block('LONG', {'on': True, 'dir': None}, 'banner')
    s = A.ai_settings_norm({'rdy_filter': 'banner'})
    why = A.ai_entry_block({}, 'SHORT', s, {'ok': True}, None, 0, [], None, 1e9, rdy=v)
    assert why.startswith('🧩')


def test_rdy_stats():
    rows = [{'side': 'LONG', 'pnl_pct': 2.0, 'rdy_open': {'on': True, 'dir': 'LONG'}, 'opened_by': 'ai'},
            {'side': 'LONG', 'pnl_pct': -1.0, 'rdy_open': {'on': True, 'dir': 'SHORT'}},
            {'side': 'SHORT', 'pnl_pct': 1.0, 'rdy_open': {'on': True, 'dir': 'SHORT'}},
            {'side': 'SHORT', 'pnl_pct': -0.5}]
    st = A.rdy_trade_stats(rows)
    g = st['groups']
    assert g['with']['n'] == 2 and g['with']['win_pct'] == 100.0 and g['with']['avg'] == 1.5
    assert g['against']['n'] == 1 and g['against']['win_pct'] == 0.0
    assert g['unknown']['n'] == 1
    assert st['ai_n'] == 1 and st['ai']['with']['n'] == 1


def test_exit_merge_stop_off_and_false_start():
    plan = {'stop': 99.0, 'target': 105.0, 'stop_off': True}
    m = B.exit_merge('LONG', {}, {}, 'confirmed', plan, 98.0)
    assert m['verdict'] != 'invalid' and not m['close']
    m = B.exit_merge('LONG', {}, {}, 'confirmed', dict(plan, stop_off=False), 98.0)
    assert m['verdict'] == 'invalid' and m['close']
    m = B.exit_merge('LONG', {}, {}, 'all', plan, 100.5,
                     charge={'pot': 90, 'min': 40},
                     false_start={'close': 100.1, 'edge': 100.2})
    assert m['verdict'] == 'false_start' and m['close']   # захисне, попри заряд і режим


class _Eng:
    def __init__(self, persist=True):
        self.s = {'enabled': False}
        self.db = self
        self.persist = persist
        self.stored = {'enabled': False}
        self._thread = None
        self.started = 0

    def ai_settings(self):
        return dict(self.s)

    def update_ai_settings(self, p):
        self.s.update(p)
        if self.persist:
            self.stored = dict(self.s)
        return dict(self.s)

    def get_setting(self, k, d=None):
        return self.stored

    def start(self):
        self.started += 1


def test_alarm_set_ai():
    e = _Eng()
    ok, note, ch = SB.alarm_set_ai(True, e)
    assert ok and ch and note == 'увімкнено' and e.started == 1
    assert SB.alarm_set_ai(True, e) == (True, 'вже увімкнено', False)
    bad = _Eng(persist=False)
    ok, note, ch = SB.alarm_set_ai(True, bad)
    assert not ok and 'НЕ збережено' in note
    assert SB.alarm_set_ai(True, None)[0] in (True, False)   # без рушія — не падає


def test_alarm_settings_and_text():
    s = dict(SB.DEFAULT_SETTINGS, alarm_ai='ON')
    SB.SbrEngine._alarm_clean(s)
    assert s['alarm_ai'] == 'on'
    s = dict(SB.DEFAULT_SETTINGS, alarm_ai='zz')
    SB.SbrEngine._alarm_clean(s)
    assert s['alarm_ai'] == ''
    txt = SB.alarm_tg_text('BTCUSDT', 'LONG', 'anom', {}, True, '', ai=(True, 'увімкнено', True), ai_act='on')
    assert '🤖 ШІ → <b>УВІМКНЕНО' in txt


def test_page():
    src = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
    assert 'вчиться ${e.n_good' not in src                  # рядок прибрано з таблиці
    for k in ('aip-sl-on', 'aip-dis-on', 'aip-dis-bars', 'aip-fs-exit', 'aip-rdy',
              'aip-rdy-stats', 'sbr-alarm-ai', 'alarm_ai:', 'sl_enabled:', 'rdy_filter:',
              '_aiDisLine(c)', "false_start: '🪤'"):
        assert k in src, k
    fl = open(os.path.join(ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    assert "'alarm_ai')" in fl


def test_trade_manager_wiring():
    tm = open(os.path.join(ROOT, 'detection', 'trade_manager.py'), encoding='utf-8').read()
    assert 'place_sl: bool = True' in tm and "pos['ai_sl_off'] = True" in tm
    assert "pos['ai_discharge']" in tm and 'false_start=fs' in tm
    assert tm.count("'rdy_open': pos.get('rdy_open')") >= 4
    assert tm.count('self._rdy_open_mark(symbol,') == 2
    ai = open(os.path.join(ROOT, 'detection', 'ai_signals.py'), encoding='utf-8').read()
    assert 'place_sl=sl_on' in ai and "plan0['stop_off'] = True" in ai


def _bars15(closes, t0, pad=0.05):
    return [{'t': (t0 + i * 900) * 1000, 'p': c, 'h': c + pad, 'l': c - pad, 'o': c}
            for i, c in enumerate(closes)]


def test_ltf_confirmation_uses_the_1h_box_level():
    h = _bars(FLAT)                                   # коробка 1h 99.8 … 100.2
    t0 = h[-1]['t'] // 1000 + 3600                    # 15m-свічки після останньої 1h
    s = A.ai_settings_norm({})
    assert s['discharge_tf'] == '1h' and s['discharge_bars_15m'] == 2
    assert A.ai_settings_norm({'discharge_tf': '1m'})['discharge_tf'] == '1h'
    # 15m: пробій 100.3 + дві свічки втримали → confirmed (на 1h ще нічого)
    r = A.discharge_state_ltf(h, _bars15([100.1, 100.3, 100.4, 100.5], t0), 'LONG', 100.5, need=2)
    assert r['state'] == 'confirmed' and r['tf'] == '15m' and abs(r['edge'] - 100.2) < 1e-9
    assert A.discharge_state(h, 'LONG', 100.5)['state'] == 'live'
    # одна свічка ще замало
    assert A.discharge_state_ltf(h, _bars15([100.3, 100.4], t0), 'LONG', 100.4, need=2)['state'] == 'pending'
    # повернулась у коробку → хибний старт
    assert A.discharge_state_ltf(h, _bars15([100.3, 100.1], t0), 'LONG', 100.1, need=2)['state'] == 'false'
    # SHORT дзеркально
    assert A.discharge_state_ltf(h, _bars15([99.7, 99.6, 99.5], t0), 'SHORT', 99.5, need=2)['state'] == 'confirmed'


def test_shadow_record_follow_and_stats():
    h = _bars(FLAT)
    rec = {'S': 'X', 'side': 'LONG', 'tf': '15m', 't': h[-1]['t'] // 1000 + 3600,
           'px': 100.0, 'edge': 100.2, 'last': 0.0, 'mfe': 0.0, 'mae': 0.0}
    A.dis_shadow_update(rec, 102.0, h, rec['t'] + 600)
    A.dis_shadow_update(rec, 99.0, h, rec['t'] + 1200)
    assert rec['mfe'] == 2.0 and rec['mae'] == -1.0 and rec['last'] == -1.0 and not rec.get('done')
    # закрита після підтвердження 1h-свічка назад у коробці → хибний старт
    h2 = h + [{'t': (rec['t'] + 60) * 1000, 'p': 100.0, 'h': 100.3, 'l': 99.9, 'o': 100.1}]
    A.dis_shadow_update(rec, 100.0, h2, rec['t'] + 4000)
    assert rec.get('false_t')
    A.dis_shadow_update(rec, 100.0, h2, rec['t'] + 25 * 3600)
    assert rec['done']
    st = A.dis_shadow_stats([rec, {'tf': '1h', 'last': 1.5, 'mfe': 2, 'mae': -0.5, 'lead_min': 40}])
    assert st['15m']['n'] == 1 and st['15m']['false_pct'] == 100.0
    assert st['1h']['win_pct'] == 100.0 and st['15m']['avg_lead_min'] == 40.0


def test_engine_shadow_and_selected_tf():
    NOW_E = 1.8e9
    eng = A.AiEngine(db=None, clock=lambda: NOW_E)
    eng._ai_s = A.ai_settings_norm({'discharge_tf': '15m'})
    eng._ai_s_at = 1e15
    h = _bars(FLAT, t0=int(NOW_E) - 40 * 3600)
    t0 = h[-1]['t'] // 1000 + 3600
    l15 = _bars15([100.3, 100.4, 100.5], t0)
    eng._pro_bars['X'] = {'1h': (NOW_E, h), '15m': (NOW_E, l15)}
    d = eng.discharge_for('X', 'LONG', NOW_E, 100.5)
    assert d['tf'] == '15m' and d['state'] == 'confirmed' and d['alts'][0]['tf'] == '1h'
    both = eng.discharge_both('X', 'LONG', NOW_E, 100.5)
    eng._dis_shadow_tick('X', 'LONG', both, 100.5, NOW_E)
    eng._dis_shadow_tick('X', 'LONG', both, 100.6, NOW_E + 30)        # той самий пробій — без дубля
    assert len(eng._dis_shadow) == 1 and eng._dis_shadow[0]['tf'] == '15m'
    eng._dis_shadow_follow({'X': 101.5}, NOW_E + 60)
    assert eng._dis_shadow[0]['mfe'] > 0
    assert eng.dis_shadow_state()['stats']['15m']['n'] == 1


def test_page_has_tf_choice_and_shadow():
    src = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
    for k in ('aip-dis-tf', 'aip-dis-bars15', 'aip-dis-stats', '_aipDisStatsHTML',
              'discharge_tf:', 'discharge_bars_15m:', 'ds.alt'):
        assert k in src, k


def _bars5(closes, t0, pad=0.03):
    return [{'t': (t0 + i * 300) * 1000, 'p': c, 'h': c + pad, 'l': c - pad, 'o': c}
            for i, c in enumerate(closes)]


def test_struct_5m_choch_bos_beyond_box():
    h = _bars(FLAT)                                       # коробка 1h 99.8 … 100.2
    t0 = h[-1]['t'] // 1000 + 3600
    m5 = _bars5([100.0, 100.1, 100.15, 100.1, 100.3, 100.35, 100.4, 100.45, 100.5, 100.5], t0)
    ev = lambda i, tag, d: {'to_t': m5[i]['t'], 'tag': tag, 'dir': d, 'level': 100}
    # CHoCH у коробці, BOS закрився за межею 100.2 → confirmed
    r = A.discharge_state_struct(h, m5, 'LONG', 100.5, events=[ev(2, 'CHoCH', 'bull'), ev(4, 'BOS', 'bull')])
    assert r['state'] == 'confirmed' and r['tf'] == '5m' and abs(r['edge'] - 100.2) < 1e-9, r
    # BOS усередині коробки — не сигнал
    r = A.discharge_state_struct(h, m5, 'LONG', 100.5, events=[ev(1, 'CHoCH', 'bull'), ev(2, 'BOS', 'bull')])
    assert r['state'] == 'pending'
    # BOS без CHoCH перед ним — не сигнал
    assert A.discharge_state_struct(h, m5, 'LONG', 100.5, events=[ev(4, 'BOS', 'bull')])['state'] == 'none'
    # після BOS зустрічний CHoCH → хибний старт
    r = A.discharge_state_struct(h, m5, 'LONG', 100.0, events=[ev(2, 'CHoCH', 'bull'), ev(4, 'BOS', 'bull'),
                                                               ev(7, 'CHoCH', 'bear')])
    assert r['state'] == 'false'
    # ціна повернулась у коробку → ще не підтверджено
    assert A.discharge_state_struct(h, m5, 'LONG', 100.1,
                                    events=[ev(2, 'CHoCH', 'bull'), ev(4, 'BOS', 'bull')])['state'] == 'pending'
    # зустрічна подія ДО CHoCH лише скидає ланцюг
    r = A.discharge_state_struct(h, m5, 'LONG', 100.5, events=[ev(0, 'BOS', 'bear'), ev(2, 'CHoCH', 'bull'),
                                                               ev(4, 'BOS', 'bull')])
    assert r['state'] == 'confirmed'
    # реальний детектор не падає на справжніх свічках
    assert A.discharge_state_struct(h, m5, 'LONG', 100.5)['state'] in ('none', 'pending', 'confirmed', 'false')


def test_pick_and_settings_5m():
    assert A.ai_settings_norm({'discharge_tf': '5m'})['discharge_tf'] == '5m'
    both = {'1h': {'state': 'live', 'tf': '1h'}, '15m': {'state': 'pending', 'tf': '15m'},
            '5m': {'state': 'confirmed', 'tf': '5m', 'edge': 1}}
    d = A.dis_pick(both, '5m')
    assert d['state'] == 'confirmed' and [a['tf'] for a in d['alts']] == ['1h', '15m']
    assert A.dis_pick(both, 'zz')['tf'] == '1h'
    st = A.dis_shadow_stats([{'tf': '1h', 'last': 1, 'lead': {'15m': 30, '5m': 50}},
                             {'tf': '5m', 'last': 2}, {'tf': '15m', 'last': 1}, {'tf': '1h', 'last': 1, 'lead_min': 20}])
    assert st['5m']['avg_lead_min'] == 50.0 and st['15m']['avg_lead_min'] == 25.0
    src = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
    assert 'value="5m">5m CHoCH+BOS' in src and "row('5m')" in src and '_AI_DIS_TF' in src


if __name__ == '__main__':
    n = 0
    for k, f in list(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
            n += 1
    print(f'OK {n}')
