"""🔎 АУДИТ АЛГОРИТМУ ВСІХ ШЕСТИ МММ-БАНЕРІВ (24.10).

Дослівно: «Проаналізуй всіх банерів алгоритм на помилки. Щоб все було
професійно і ідеально.» Кожен тест — замок на конкретний знайдений дефект.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon   # noqa: E402  (ізольований fuel_filter + стаби)

_m = _mon._m
_HTML = _mon._HTML


def _check(c, m):
    if not c:
        raise AssertionError(m)


# ═══ 1. ПОРОЖНІЙ ЗНІМОК ≠ РІВНОВАГА ════════════════════════════════════════
def test_empty_snapshot_keeps_the_banner_status():
    """Після рестарту скан 🧭 Pull ще біжить → знімок порожній. Раніше net=0
    давав кандидата ⏸ FLAT, банер «падав у рівновагу» і слав Telegram."""
    snap = {'A': {'status': 'LONG', 'strength': 60}, 'B': {'status': 'LONG', 'strength': 40}}
    b, since, cand = _m.mm_bias_step(snap, 1000.0, 120, {}, 0.0, {})
    _check(b['dir'] == 'LONG' and since == 1000.0, b)
    for t in (1030.0, 1200.0, 1500.0):          # довше за вікно підтвердження
        b, since, cand = _m.mm_bias_step({}, t, 120, b, since, cand)
        _check(b['dir'] == 'LONG', f'порожній знімок не має перемикати статус: {b}')
        _check(since == 1000.0, 'таймер стану не скидається')
        _check(not cand and b.get('cand_dir') is None, 'кандидат не росте')
        _check(b.get('empty') is True, 'стан «немає даних» позначено')
    b, since, cand = _m.mm_bias_step(snap, 1530.0, 120, b, since, cand)
    _check(b['dir'] == 'LONG' and since == 1000.0 and not b.get('empty'),
           'дані повернулись — той самий статус і той самий таймер')
    print('✓ порожній знімок тримає статус банера (немає даних ≠ рівновага)')


def test_first_real_state_after_empty_start_is_immediate():
    """Блок щойно ввімкнули, даних ще немає → перший реальний стан одразу,
    без 120-секундного вікна і без таймера на «порожнечі»."""
    b, since, cand = _m.mm_bias_step({}, 1000.0, 120, {}, 0.0, {})
    _check(b['since'] == 0 and since == 0.0 and b['dir'] is None, b)
    _check(b['avg_str'] == 0.0, 'форма відповіді та сама')
    snap = {'A': {'status': 'SHORT', 'strength': 70}}
    b, since, cand = _m.mm_bias_step(snap, 1030.0, 120, b, since, cand)
    _check(b['dir'] == 'SHORT' and since == 1030.0, f'перший стан — одразу: {b}')
    print('✓ перший реальний стан після порожнього старту — без вікна підтвердження')


# ═══ 2. TELEGRAM: ПОЗНАЧКА ВЕДЕТЬСЯ Й ПРИ ВИМКНЕНОМУ ТУМБЛЕРІ ══════════════
def test_tg_mark_is_tracked_while_toggle_is_off():
    ff = _mon._mk()
    sent = []
    ff._broadcast_users = lambda *a, **k: (sent.append(a) or (True, ''))
    on, off = {'mm_bias_tg': True}, {'mm_bias_tg': False}

    def _tick(side, t, s):
        ff._mm_bias_since = t            # банер щойно перейшов у `side`
        ff._mm_bias_alert(side, t, s)
    _tick('LONG', 0.0, on)               # перший такт — мовчки
    _tick('SHORT', 10.0, off)
    _tick('LONG', 20.0, off)
    _check(not sent, 'вимкнений тумблер не шле')
    _check(ff._mm_bias_tg_last == 'LONG' and ff._mm_bias_tg_at == 20.0,
           'позначка ведеться і при вимкненому тумблері')
    _tick('SHORT', 50.0, on)
    _check(len(sent) == 1, sent)
    _check('30с' in sent[0][2] or '30 с' in sent[0][2],
           f'«тримався» — від реальної зміни (20с), а не від старої позначки: {sent[0][2]}')
    print('✓ Telegram банера: позначка ведеться при вимкненому тумблері')


def test_tg_held_comes_from_the_banner_timer_0911():
    """09.11: «тримався» = початок нового статусу банера − початок старого.
    Рестарт (сповіщувач бачить статус уперше) і вікно підтвердження не
    псують тривалість."""
    ff = _mon._mk()
    sent = []
    ff._broadcast_users = lambda *a, **k: (sent.append(a) or (True, ''))
    on = {'mm_bias_tg': True}
    # Після рестарту: банер відновив LONG, що триває з t=1000 (5 год тому).
    ff._mm_bias_since = 1000.0
    ff._mm_bias_alert('LONG', 19000.0, on)          # перший такт — мовчки
    # Кандидат SHORT зʼявився на 19500, підтвердився на 19620 → since=19500.
    ff._mm_bias_since = 19500.0
    ff._mm_bias_alert('SHORT', 19620.0, on)
    _check(len(sent) == 1 and '5г 08хв' in sent[0][2] and '↻' not in sent[0][2],
           f'тривалість із таймера банера, без ↻ і без вікна: {sent and sent[0][2]}')
    # Доба й більше — як у банері.
    ff._mm_bias_since = 19500.0 + 2 * 86400 + 3600
    ff._mm_bias_alert('LONG', ff._mm_bias_since + 120, on)
    _check('2д 01г 00хв' in sent[1][2], sent[1][2])
    print('✓ «тримався» — з таймера банера, переживає рестарт, показує доби')


def test_confirmed_state_starts_when_candidate_appeared():
    snap_l = {'A': {'status': 'LONG', 'strength': 70}}
    snap_s = {'A': {'status': 'SHORT', 'strength': 70}}
    b, since, cand = _m.mm_bias_step(snap_l, 100.0, 120, {}, 0.0, {})
    b, since, cand = _m.mm_bias_step(snap_s, 200.0, 120, b, since, cand)
    _check(b['dir'] == 'LONG' and since == 100.0, 'ще кандидат')
    b, since, cand = _m.mm_bias_step(snap_s, 330.0, 120, b, since, cand)
    _check(b['dir'] == 'SHORT' and since == 200.0 and b['since'] == 200,
           f'новий статус — з моменту появи кандидата: {since}')
    print('✓ підтверджений статус стартує з появи кандидата')


# ═══ 3. ⚡ CTR: НЕВДАХИ НЕ ЗʼЇДАЮТЬ БЮДЖЕТ ════════════════════════════════
class _FE:
    def __init__(self, good, bad):
        self.good, self.bad, self.calls = good, bad, []

    def peek_ctr_tf(self, S, tf):
        if S in self.good:
            return (999.0, {'stc': 40.0}, 300.0)      # застаріле значення
        return None

    def get_ctr_tf(self, S, tf):
        self.calls.append(S)
        return {'stc': 30.0} if S in self.good else None


def test_ctr_failures_do_not_starve_the_refresh():
    good = [f'G{i}USDT' for i in range(3)]
    bad = [f'B{i}USDT' for i in range(12)]
    fe = _FE(set(good), set(bad))
    ff = _mon._mk()
    ff._mmc_rd_last, ff._mmc_rd_fail = {}, {}
    ff._ctr_mm_readings(good + bad, fe)
    _check(len(fe.calls) == _m.CTR_MM_FETCH_PER_TICK and set(fe.calls) <= set(bad),
           'перший такт: невідомі першими')
    fe.calls.clear()
    for _ in range(3):
        ff._ctr_mm_readings(good + bad, fe)
    _check(set(good) <= set(fe.calls),
           f'застарілі STC добрих монет мусять оновитись попри невдах: {fe.calls}')
    _check(len(fe.calls) <= 3 * _m.CTR_MM_FETCH_PER_TICK, 'бюджет такту не перевищено')
    _check(_m.CTR_MM_FAIL_RETRY > 0, 'повтор невдачі — не щотакту')
    print('✓ ⚡ CTR: невдалі STC не блокують оновлення решти')


# ═══ 4. 🧭 PULL: НЕВДАЛИЙ СКАН ПОВТОРЮЄТЬСЯ ШВИДШЕ ═════════════════════════
def test_failed_pull_scan_retries_sooner():
    ff = _mon._mk()
    launched = []
    ff._mmp_scan_launch = lambda s, cfg: launched.append(cfg)
    ff._mmp_scan_running = False
    s = {'mmp_interval_min': 15}
    cfg = ff._mmp_cfg(s)
    now = time.time()
    ff._mmp_scan, ff._mmp_scan_at = {'ok': True, 'cfg': cfg}, now - 400
    _check(ff._mmp_scan_tick(s, now) is False, 'успішний скан чекає повний інтервал')
    ff._mmp_scan = {'ok': False, 'cfg': cfg, 'reason': 'біржа лягла'}
    _check(ff._mmp_scan_tick(s, now) is True and launched, 'невдалий — повтор через 5 хв')
    ff._mmp_scan_running = False
    info = ff._mmp_scan_info(s)
    _check(abs(info['next_at'] - (now - 400 + _m.MMP_FAIL_RETRY)) < 1,
           'відлік на сторінці — за тим самим правилом')
    print('✓ 🧭 Pull: невдалий скан повторюється через MMP_FAIL_RETRY')


# ═══ 5. СИГНАТУРА ТАБЛИЦЬ БАЧИТЬ ВЛАСНІ КОЛОНКИ КНИГ ═══════════════════════
def test_table_signature_covers_book_specific_columns():
    a = _HTML.index('function mmRender(')
    seg = _HTML[a:_HTML.index('function mmRenderBias(', a)]
    sig = seg[seg.index('const sig'):seg.index('if (tb.dataset.sig')]
    for f in ('r.f1.pct', 'r.f1.conf', 'r.stc_1h', 'r.stc_4h', 'r.above_pct',
              'r.magnet_price', 'r.magnet_dist', 'r.dec_verdict', 'r.dec_prob'):
        _check(f in sig, f'сигнатура таблиці мусить містити {f}')
    print('✓ сигнатура таблиць бачить прогноз, STC, масу, магніт і вердикт')


if __name__ == '__main__':
    n = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn(); n += 1
    print(f'\nOK — {n} тестів')
