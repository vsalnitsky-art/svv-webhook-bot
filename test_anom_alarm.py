"""⚠ «Аномалії обсягу» → особисті сповіщення · ⏰ будильник · ⚡ напруженість (18.11).

Дослівно: «оповіщення в телеграм по вибору монети, сигнал стосовно "Аномалії
обсягів"… максимально близько до моменту для входу в угоду» · «Додай до
головного банера поруч із кнопками "SHORT" загальний відсоток напруженості по
кожній із пружин» · «Додай цей самий сигнал до будильника, який вмикатиме
тумблер Trade Manager».

python test_anom_alarm.py
"""
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import test_ai_signals as L      # noqa: E402  (ai_signals + стаби)

A = L.A
NOW = L.NOW


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


SC = _load('detection.spring_coil', 'detection/spring_coil.py')
SB = _load('detection.spring_banner', 'detection/spring_banner.py')
V = _load('detection.vol_anomaly', 'detection/vol_anomaly.py')
S = _load('detection.sbr_signals', 'detection/sbr_signals.py')
IP = _load('info_portal_t', 'web/info_portal.py')


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _src(rel):
    return open(os.path.join(ROOT, rel), encoding='utf-8').read()


# ── ⚡ Напруженість пружини ───────────────────────────────────────────────
def test_tension_is_the_average_charge_of_all_coins():
    recs = {'A': {'charge': 80, 'state': 'charged', 'pot_long': 60, 'pot_short': 20},
            'B': {'charge': 40, 'state': 'flat'},
            'C': {'charge': 0, 'state': 'flat'}}
    agg = SC.aggregate(recs, total=3)
    _check(agg['tension'] == 40.0 and agg['charged_n'] == 1, agg)
    _check(SC.aggregate({}, total=0)['tension'] is None, 'немає монет → немає числа (а не 0)')
    v = SB.view('wl', 'Bybit', agg, None)
    _check(v['tension'] == 40.0 and v['charged_n'] == 1, v)


def test_both_banners_show_the_tension_next_to_the_status():
    app = _src('infosite/js/app.js')
    html = _src('templates/smart_money.html')
    _check('function spbTension' in app and "spbTension(b) + '<span class=\"spb-st" in app, 'інфосайт')
    _check('function _smcfTension' in html and '${_smcfTension(b)}<span class="smcf-st' in html, 'бот')
    _check('.spb-ten' in _src('infosite/css/style.css'), 'стиль')


# ── 👤 Особисті підписки кабінету ─────────────────────────────────────────
def test_personal_subscription_rules():
    wl = ['BTCUSDT', 'ETHUSDT']
    _check(IP.anom_norm(['btc', 'BTCUSDT', 'xrp'], wl) == ['BTCUSDT'], 'лише монети списку, без дублів')
    cur = {'wl': wl, 'sel': 'BTCUSDT', 'sbr': {}, 'anom': []}
    out, err = IP.apply_change(cur, {'anom': {'symbol': 'ETH', 'on': True}}, [])
    _check(not err and out['anom'] == ['ETHUSDT'], out)
    out2, _ = IP.apply_change(out, {'remove': 'ETHUSDT'}, [])
    _check(out2['anom'] == [], 'прибрана монета → і підписка')
    subs = [('1', ['BTCUSDT']), ('2', ['ETHUSDT', 'BTCUSDT'])]
    _check(IP.anom_recipients(subs, 'btcusdt') == ['1', '2'], 'отримувачі')
    _check(IP.anom_symbols(subs) == ['BTCUSDT', 'ETHUSDT'], 'монети підписок')


def test_subscribed_coins_stay_out_of_research_stats():
    syms, roles, _ = V.research_universe({}, {}, {}, NOW, subs=['AUSDT'])
    _check(roles == {'AUSDT': 'sub'}, roles)
    _check(V.is_dir_only({'ctx': {'role': 'sub'}}), 'роль sub — не дослідження')


# ── 🧩 Вибір сигналу ──────────────────────────────────────────────────────
def _r(side, ready=True, key='E1', last=NOW - 60):
    return {'side': side, 'veto': '', 'score': 50,
            'scenario': {'ready': ready, 'broken': '', 'key': key, 'last_t': last,
                         'name': '🔁 Розворот після виснаження',
                         'stages': [{'label': 'етап', 'done': True}]}}


def test_pick_one_side_and_respect_the_anomaly_direction():
    side, r = A.anom_pick(_r('LONG'), _r('SHORT', ready=False))
    _check(side == 'LONG', (side, r))
    _check(A.anom_pick(_r('LONG'), _r('SHORT'))[0] is None, 'обидва боки → тиша')
    _check(A.anom_pick(_r('LONG'), _r('SHORT', False), {'state': 'dir', 'dir': 'SHORT'})[0] is None,
           'аномалії 6 год проти → тиша')
    t = A.anom_tg_text('BTCUSDT', 'LONG', _r('LONG'), 64672.6, {'state': 'dir', 'dir': 'LONG'})
    _check('Аномалії обсягу' in t and '#BTCUSDT' in t and 'Розворот' in t and '64' in t, t)


def test_engine_sends_once_and_wakes_the_alarm():
    sent, alarm = [], []
    e = A.AiEngine(db=None, scanner=L._Sc(True), tm=L._Tm(), clock=lambda: NOW,
                   anom_subs=lambda: ['BTCUSDT'],
                   anom_send=lambda S, t: (sent.append((S, t)) or (1, 0)),
                   anom_alarm=lambda S, side, sc, px, now: alarm.append((S, side, sc)))
    sc = {'BTCUSDT': (_r('LONG'), _r('SHORT', False), {'price': 100.0}, None)}
    _check(e._anom_personal(['BTCUSDT'], sc, {}, {}, NOW) == 1 and len(sent) == 1, sent)
    _check(alarm == [('BTCUSDT', 'LONG', '🔁 Розворот після виснаження')], alarm)
    e._anom_personal(['BTCUSDT'], sc, {}, {}, NOW + 60)
    _check(len(sent) == 1 and len(alarm) == 1, 'той самий сценарій не повторюється')
    sc2 = {'BTCUSDT': (_r('LONG', last=NOW + 900), _r('SHORT', False), {}, None)}
    e._anom_personal(['BTCUSDT'], sc2, {}, {}, NOW + 900)
    _check(len(alarm) == 2 and len(sent) == 1, 'новий сценарій будить будильник, TG — не частіше раз на годину')


def test_alarm_only_coin_sends_nothing_personal():
    logs = []
    e = A.AiEngine(db=None, scanner=L._Sc(True), tm=L._Tm(), clock=lambda: NOW,
                   anom_subs=lambda: ['BTCUSDT'], anom_send=lambda S, t: (0, 0),
                   anom_alarm=lambda *a: logs.append(a))
    sc = {'BTCUSDT': (_r('LONG'), _r('SHORT', False), {}, None)}
    _check(e._anom_personal(['BTCUSDT'], sc, {}, {}, NOW) == 0 and logs, 'будильник — так, TG — ні')
    _check(not e._anom_last, 'без підписників охолодження не витрачається')


# ── ⏰ Будильник ──────────────────────────────────────────────────────────
class _MemDB:
    def __init__(self): self.d = {}
    def get_setting(self, k, d=None): return self.d.get(k, d)
    def set_setting(self, k, v): self.d[k] = v


class _FakeTM:
    """Імітує TradeManager: налаштування в памʼяті + персист у БД, як справжній
    `update_settings` (саме його читають сканер, черги FF і `_side_allowed`)."""
    def __init__(self, on=False, long_on=True, short_on=True, persist=True, accept=True):
        self.db = _MemDB()
        self._running = False
        self.persist, self.accept = persist, accept
        self._settings = {'enabled': on, 'allow_long_entries': long_on, 'allow_short_entries': short_on}
        self.db.set_setting('tm_settings', dict(self._settings))

    @property
    def on(self):
        return bool(self._settings['enabled'])

    def is_enabled(self):
        return bool(self._settings['enabled'])

    def get_settings(self):
        return dict(self._settings)

    def start(self):
        self._running = True

    def update_settings(self, patch):
        if self.accept:
            self._settings.update(patch)
            if self._settings['enabled'] and not self._running:
                self.start()
        if self.persist:
            self.db.set_setting('tm_settings', dict(self._settings))
        return dict(self._settings)


DG = _load('detection.direction_gate', 'detection/direction_gate.py')


def test_alarm_source_rules():
    st = {'alarm_on': True, 'alarm_symbol': 'btc', 'alarm_tfs': ['1h'], 'alarm_side': 'LONG'}
    sig = [('1h', {'side': 'LONG', 'score': 70})]
    _check(S.alarm_match(st, 'BTCUSDT', sig), 'деф. джерело = SBR (стара поведінка)')
    _check(not S.alarm_match({**st, 'alarm_src': 'anom'}, 'BTCUSDT', sig), '⚠ лише аномалії → SBR не будить')
    _check(S.alarm_anom_symbol({**st, 'alarm_src': 'both'}) == 'BTCUSDT', 'монета для рушія ШІ')
    _check(S.alarm_anom_symbol(st) is None, 'джерело SBR → аномалії не ведемо')
    _check(S.alarm_anom_match({**st, 'alarm_src': 'anom'}, 'BTCUSDT', 'LONG'), 'збіг')
    _check(S.alarm_anom_match({**st, 'alarm_src': 'anom'}, 'BTCUSDT', 'SHORT'),
           'бік НЕ обирається (18.11): будить будь-який бік')
    _check(not S.alarm_anom_match({**st, 'alarm_src': 'anom'}, 'ETHUSDT', 'SHORT'), 'інша монета')
    _check(S.alarm_src_of({'alarm_src': 'xxx'}) == 'sbr', 'невідоме → sbr')


def _engine_with(tm):
    """Рушій SBR, у якому кнопки й TM перемикає СПРАВЖНІЙ код
    (`direction_gate.live_set_only`, `alarm_set_tm`) над фейковим TM."""
    dg = types.ModuleType('detection.direction_gate')
    dg.live_set_only = lambda side, source='': DG.live_set_only(side, source, tm=tm)
    dg.live_enable = lambda side, source='': (True, 'увімкнено')
    pkg = sys.modules.setdefault('detection', types.ModuleType('detection'))
    sys.modules['detection.direction_gate'] = dg
    pkg.direction_gate = dg
    sent = []
    e = S.SbrEngine(_MemDB(), lambda: [], None, None)
    e._admin_notify = lambda t: sent.append(t) or True
    e._tm_switch = lambda on: S.alarm_set_tm(on, tm)
    e.update_settings({'alarm_on': True, 'alarm_symbol': 'BTC', 'alarm_side': 'LONG',
                       'alarm_tm': 'on', 'alarm_src': 'anom'})
    return e, sent


def _restore_dg():
    sys.modules['detection.direction_gate'] = DG
    sys.modules['detection'].direction_gate = DG


def test_anomaly_alarm_leaves_only_the_signal_side_and_switches_tm():
    """Вимога 18.11: бік не обирається; SHORT-сигнал при будь-якому стані кнопок
    → увімкнено ЛИШЕ SHORT, LONG вимкнено; TM увімкнено. Перевіряємо ЖИВИЙ стан
    (його читає весь алгоритм), запис у БД і запущений монітор, а не екран."""
    try:
        for before in ((True, True), (False, False), (True, False), (False, True)):
            tm = _FakeTM(on=False, long_on=before[0], short_on=before[1])
            e, sent = _engine_with(tm)
            _check(e.alarm_from_anom('BTCUSDT', 'SHORT', '🔁 Розворот', 64000.0, 100.0), 'будить SHORT')
            live = tm.get_settings()
            _check(DG.gates_from(live) == (False, True), (before, live))
            _check(DG.gates_from(tm.db.get_setting('tm_settings')) == (False, True), 'записано в БД')
            _check(tm.is_enabled() and tm.db.get_setting('tm_settings')['enabled'] and tm._running,
                   'TM увімкнено, збережено, монітор працює')
            L_ = e.get_settings()['alarm_last']
            _check(L_['src'] == 'anom' and L_['ok'] and L_['dir_ok'] and L_['tm_ok'], L_)
            _check(sent and 'Будильник ⚠ Аномалії' in sent[-1], sent)
            _check(('лише <b>SHORT</b>' in sent[-1]) == (before != (False, True)),
                   ('рядок про кнопки — лише коли їх реально змінено', before, sent[-1]))
        e.alarm_from_anom('BTCUSDT', 'LONG', 'x', 1.0, 200.0)
        _check(DG.gates_from(tm.get_settings()) == (True, False), 'наступний LONG → лише LONG')
        n = len(sent)
        e.alarm_from_anom('BTCUSDT', 'LONG', 'x', 1.0, 300.0)
        _check(len(sent) == n, 'стан уже такий → тиша')
        e._on_signals('BTCUSDT', [('1h', {'side': 'SHORT', 'score': 70, 'trigger': 1, 'sl': 1})], 400.0)
        _check(DG.gates_from(tm.get_settings()) == (True, False), 'джерело ⚠ → сигнал SBR не будить')
    finally:
        _restore_dg()


def test_alarm_reports_failure_when_state_did_not_really_change():
    """«Лише відображення» неможливе: TM не прийняв зміну / не записав у БД →
    будильник пише НЕВДАЧУ, а не «увімкнено»."""
    try:
        tm = _FakeTM(on=False, long_on=True, short_on=True, accept=False)
        e, sent = _engine_with(tm)
        e.alarm_from_anom('BTCUSDT', 'LONG', 'x', 1.0, 100.0)
        L_ = e.get_settings()['alarm_last']
        _check(not L_['ok'] and not L_['dir_ok'] and L_['tm_ok'] is False, L_)
        _check('не прийняв' in L_['note'] and 'НЕ' in sent[-1], (L_, sent))
        tm2 = _FakeTM(on=False, long_on=True, short_on=True, persist=False)
        ok, why, ch = DG.live_set_only('SHORT', tm=tm2)
        _check(not ok and 'БД' in why, why)
        ok, why, ch = S.alarm_set_tm(True, tm2)
        _check(not ok and 'БД' in why, why)
    finally:
        _restore_dg()


def test_wiring_and_ui():
    html = _src('templates/smart_money.html')
    app = _src('web/flask_app.py')
    js = _src('infosite/js/app.js')
    _check('id="sbr-alarm-src"' in html and "alarm_src: (document.getElementById('sbr-alarm-src')" in html, 'бот UI')
    _check("'alarm_tm', 'alarm_src'," in app and "'alarm_ai')" in app, 'маршрут приймає джерело')
    _check('data-bellanom' in js and 'saveMine({ anom: { symbol: sym, on: on } })' in js, 'кабінет')
    ai = _src('detection/ai_signals.py')
    _check('self._anom_alarm(S, side' in ai and 'self._anom_personal(subs' in ai, 'рушій ШІ')


if __name__ == '__main__':
    fails = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]:
        try:
            fn()
            print('✓', name)
        except Exception as e:
            fails += 1
            import traceback
            traceback.print_exc()
            print('✗', name, '—', repr(e))
    print('OK' if not fails else f'FAIL {fails}')
    sys.exit(1 if fails else 0)
