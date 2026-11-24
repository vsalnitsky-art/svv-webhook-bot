"""🎓 САМОНАВЧАННЯ ШІ ПО ВСІХ НАПРЯМКАХ (вимога 24.11). ЧИСТІ функції, без I/O.

Дослівно: «Бот має самостійно тренувати і вдосконалювати свій мозок по всіх
напрямках, з якими він працює.»

До цього мозок учився лише на ВЛАСНИХ закритих угодах (еталон / антибаза), а
угод мало — вчився б місяцями. Тепер кожне рішення ШІ має ТІНЬОВИЙ результат,
незалежно від того, чи стало воно угодою:

• ВХІД — кожна готова точка входу (сценарії E1–E4, і ті, що йдуть ⚠ сигналом у
  Telegram / на графік) пишеться в журнал і ведеться 8 год закритими 1h-свічками:
  що сталося раніше — ціна пройшла +WIN_R × ризик (✅) чи −1 × ризик (❌).
  Обидва в одній свічці — рахуємо ❌ (консервативно, як у бектесті).
• ВИХІД — кожне закриття угоди ШІ: чи продовжила ціна рух у бік угоди на ≥1R
  раніше, ніж відкотилась на ½R (вихід «зарано») — 4 год.
• ПІДТВЕРДЖЕННЯ РОЗРЯДУ — тінь 1H / 15m / 5m (уже ведеться) → найкращий TF.

Що мозок робить із цим (кожна вага має стелю і поріг вибірки):
1. сценарій × бік → множник впевненості 0.6…1.4 (очікуваний R, усушений до
   нуля на малій вибірці) — гірші сценарії отримують нижчу впевненість і
   відсікаються порогом «🎯 впевненість ≥», кращі — вищу;
2. тіньові ✅ / ❌ — ВІРТУАЛЬНИЙ досвід для 🏆🧠 мозку еталонів (вага ½ від
   реальної угоди) — мозок дозріває на ринку, а не лише на власних угодах;
3. причина виходу, яка частіше за все виявлялась «зарано», перестає закривати
   сама (потрібне ще одне свідчення). Захисні виходи (стоп, хибний старт,
   сценарій, 🔥→🧽, усі групи) НЕ чіпаються ніколи;
4. TF підтвердження розряду «авто» → TF з найкращим тіньовим результатом.
Аномалії обсягу (ваги видів) калібруються як і раніше — `calibration()`.
"""
from typing import Dict, List, Optional

HORIZON_SEC = 8 * 3600          # тіньовий вхід ведемо 8 год
EXIT_HORIZON_SEC = 4 * 3600     # тінь після виходу — 4 год
WIN_R = 1.5                     # ✅ = +1.5R раніше за −1R
EXIT_LATE_R = 1.0               # «зарано» = ще +1R у бік угоди після виходу …
EXIT_BACK_R = 0.5               # … раніше, ніж ½R проти
DEF_RISK_PCT = 1.0              # ризик тіні, коли плану немає (% від ціни)
RISK_MIN, RISK_MAX = 0.3, 6.0
MIN_N = 15                      # менше випадків — вага 1.0 (не вигадуємо)
SHRINK = 20                     # усушка: вага × n / (n + SHRINK)
FACT_LO, FACT_HI = 0.6, 1.4
TRUST_MIN = 0.4                 # частка «вихід був вчасно» нижче → вихід сам не закриває
ENTRY_MAX = 3000
EXIT_MAX = 600
VIRTUAL_W = 0.5                 # вага віртуального досвіду в мозку еталонів
DIS_MIN_DONE = 20               # тінь TF розряду: стільки завершених, щоб обирати «авто»
PROTECTIVE = ('invalid', 'false_start', 'scenario', 'combo', 'all')
ADJUSTABLE = ('close', 'reversal', 'giveback', 'target', 'time')
SCEN_UA = {'E1': '🔁 Розворот після виснаження', 'E2': '💥 Старт розряду',
           'E3': '↗ Продовження після відкату', 'E4': '🪤 Спринг'}


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _sg(side) -> float:
    return 1.0 if str(side).upper() == 'LONG' else -1.0


def _bar(b) -> Optional[tuple]:
    """(відкриття, сек), high, low, close з бару будь-якого формату бота."""
    if not isinstance(b, dict):
        return None
    t = b.get('t', b.get('time', b.get('open_time')))
    t = _f(t)
    if t is None:
        return None
    if t > 1e11:
        t /= 1000.0
    h = _f(b.get('h', b.get('high')))
    lo = _f(b.get('l', b.get('low')))
    c = _f(b.get('p', b.get('close', b.get('c'))))
    if h is None or lo is None or c is None:
        return None
    return t, h, lo, c


def risk_of(plan_risk_pct) -> float:
    r = _f(plan_risk_pct)
    if r is None or r <= 0:
        return DEF_RISK_PCT
    return max(RISK_MIN, min(RISK_MAX, r))


def entry_new(S, side, scen, t, px, risk_pct=None, feat=None, conf=None,
              src='ai', dis=None, block='') -> Optional[Dict]:
    """Тіньовий запис готової точки входу. ЧИСТА."""
    p = _f(px)
    if not p or p <= 0 or str(side).upper() not in ('LONG', 'SHORT'):
        return None
    d = dis or {}
    v1 = d.get('vob1h') or {}
    return {'k': f'{S}|{str(side).upper()}|{scen}|{int(t)}', 'S': S, 'side': str(side).upper(),
            'scen': scen or '', 'src': src, 't': int(t), 'px': p, 'risk': round(risk_of(risk_pct), 3),
            'feat': dict(feat or {}), 'conf': _f(conf), 'dis_tf': d.get('tf'),
            'dis_state': d.get('state'), 'vob1h': v1.get('ok') if v1 else None,
            'block': (block or '')[:120], 'mfe': 0.0, 'mae': 0.0, 'res': None, 'done': False}


def _walk(rec, bars, start, horizon, up_lim, dn_lim, price=None, now=None):
    """Пройти бари після `start` до `horizon`: (mfe%, mae%, перший дотик 'up'|'dn'|None,
    кінцевий рух %). Дотик в одній свічці обох меж → 'dn' (консервативно)."""
    sg, p0 = _sg(rec['side']), float(rec['px'])
    mfe, mae, first, last = float(rec.get('mfe') or 0), float(rec.get('mae') or 0), None, None
    for b in bars or []:
        x = _bar(b)
        if x is None:
            continue
        t, h, lo, c = x
        if t < start or t >= start + horizon:
            continue
        fav = (h - p0) / p0 * 100.0 if sg > 0 else (p0 - lo) / p0 * 100.0
        adv = (lo - p0) / p0 * 100.0 if sg > 0 else (p0 - h) / p0 * 100.0
        mfe, mae = max(mfe, fav), min(mae, adv)
        if first is None:
            if adv <= -dn_lim:
                first = 'dn'
            elif fav >= up_lim:
                first = 'up'
        last = sg * (c - p0) / p0 * 100.0
    if price is not None and _f(price):
        mv = sg * (float(price) - p0) / p0 * 100.0
        mfe, mae = max(mfe, mv), min(mae, mv)
        last = mv
        if first is None:
            if mv <= -dn_lim:
                first = 'dn'
            elif mv >= up_lim:
                first = 'up'
    return round(mfe, 3), round(mae, 3), first, last


def entry_update(rec: Dict, bars: List[Dict], now: float, price=None) -> bool:
    """Ведемо тіньовий вхід 1h-свічками ПІСЛЯ нього (+ поточна ціна). ЧИСТА.
    → True, якщо запис змінився. res: 'win' (+WIN_R раніше −1R) · 'loss' ·
    'flat' (горизонт минув без дотику, r_end = кінцевий рух / ризик)."""
    if rec.get('done'):
        return False
    risk = float(rec.get('risk') or DEF_RISK_PCT)
    start = float(rec['t'])
    mfe, mae, first, last = _walk(rec, bars, start, HORIZON_SEC, WIN_R * risk, risk, price, now)
    ch = (mfe, mae) != (rec.get('mfe'), rec.get('mae'))
    rec['mfe'], rec['mae'] = mfe, mae
    if last is not None:
        rec['last'] = round(last, 3)
    if first == 'up':
        rec.update(res='win', r=WIN_R, done=True, end_t=int(now))
        return True
    if first == 'dn':
        rec.update(res='loss', r=-1.0, done=True, end_t=int(now))
        return True
    if now - start >= HORIZON_SEC:
        r = round(max(-1.0, min(WIN_R, float(rec.get('last') or 0) / risk)), 3)
        rec.update(res='flat', r=r, done=True, end_t=int(now))
        return True
    return ch


def exit_new(S, side, verdict, px, t, risk_pct=None, reason='', pnl=None) -> Optional[Dict]:
    """Тіньовий запис виходу угоди ШІ. ЧИСТА."""
    p = _f(px)
    if not p or p <= 0 or str(side).upper() not in ('LONG', 'SHORT'):
        return None
    return {'k': f'{S}|{int(t)}', 'S': S, 'side': str(side).upper(), 'verdict': verdict or '',
            'reason': reason or '', 'px': p, 't': int(t), 'risk': round(risk_of(risk_pct), 3),
            'pnl': _f(pnl), 'mfe': 0.0, 'mae': 0.0, 'res': None, 'done': False}


def exit_update(rec: Dict, bars: List[Dict], now: float, price=None) -> bool:
    """Чи вийшли вчасно: після виходу ціна пройшла ще +1R у бік угоди раніше
    за ½R проти → 'early' (зарано), інакше 'right'. ЧИСТА."""
    if rec.get('done'):
        return False
    risk = float(rec.get('risk') or DEF_RISK_PCT)
    mfe, mae, first, last = _walk(rec, bars, float(rec['t']), EXIT_HORIZON_SEC,
                                  EXIT_LATE_R * risk, EXIT_BACK_R * risk, price, now)
    ch = (mfe, mae) != (rec.get('mfe'), rec.get('mae'))
    rec['mfe'], rec['mae'] = mfe, mae
    if first == 'up':
        rec.update(res='early', done=True, end_t=int(now))
        return True
    if first == 'dn' or now - float(rec['t']) >= EXIT_HORIZON_SEC:
        rec.update(res='right', done=True, end_t=int(now))
        return True
    return ch


def shrunk(mean: float, n: int) -> float:
    return mean * n / (n + SHRINK)


def scen_factor(rs: List[float]) -> float:
    """Множник впевненості з очікуваного R сценарію. ЧИСТА."""
    n = len(rs)
    if n < MIN_N:
        return 1.0
    m = sum(rs) / n
    return round(max(FACT_LO, min(FACT_HI, 1.0 + 0.4 * shrunk(m, n))), 3)


def learn(entries: List[Dict], exits: List[Dict], dis_stats: Optional[Dict] = None) -> Dict:
    """🎓 Усе, чого мозок навчився з тіньових результатів. ЧИСТА.
    → {'scen': {E2|LONG: {n, win, loss, flat, wr, exp_r, factor}}, 'conf': [...],
       'exit': {verdict: {n, early, right, trust, ok}}, 'dis': {'best', 'by'},
       'entries': {n, done, open}, 'exits': {n, done}}."""
    done = [r for r in entries or [] if r.get('done') and r.get('res')]
    scen: Dict[str, Dict] = {}
    grp: Dict[str, List[Dict]] = {}
    for r in done:
        grp.setdefault(f"{r.get('scen') or '?'}|{r.get('side')}", []).append(r)
    for k, rs in sorted(grp.items()):
        rr = [float(r.get('r') or 0) for r in rs]
        n = len(rs)
        w = sum(1 for r in rs if r['res'] == 'win')
        lo = sum(1 for r in rs if r['res'] == 'loss')
        scen[k] = {'n': n, 'win': w, 'loss': lo, 'flat': n - w - lo,
                   'wr': round(100.0 * w / n, 1), 'exp_r': round(sum(rr) / n, 3),
                   'factor': scen_factor(rr)}
    conf = []
    for lo_b, hi_b in ((0, 40), (40, 55), (55, 70), (70, 101)):
        rs = [r for r in done if r.get('conf') is not None and lo_b <= float(r['conf']) < hi_b]
        if rs:
            conf.append({'lo': lo_b, 'hi': min(100, hi_b), 'n': len(rs),
                         'wr': round(100.0 * sum(1 for r in rs if r['res'] == 'win') / len(rs), 1),
                         'exp_r': round(sum(float(r.get('r') or 0) for r in rs) / len(rs), 3)})
    v1 = {}
    for key, cond in (('ok', True), ('no', False)):
        rs = [r for r in done if r.get('scen') == 'E2' and r.get('vob1h') is cond]
        if rs:
            v1[key] = {'n': len(rs), 'wr': round(100.0 * sum(1 for r in rs if r['res'] == 'win') / len(rs), 1),
                       'exp_r': round(sum(float(r.get('r') or 0) for r in rs) / len(rs), 3)}
    ex: Dict[str, Dict] = {}
    xd = [r for r in exits or [] if r.get('done') and r.get('res')]
    for v in sorted({r.get('verdict') or '?' for r in xd}):
        rs = [r for r in xd if (r.get('verdict') or '?') == v]
        n = len(rs)
        e = sum(1 for r in rs if r['res'] == 'early')
        # частка «вчасно», усушена до ½ на малій вибірці
        trust = round((n - e + 0.5 * SHRINK * 0.5) / (n + SHRINK * 0.5), 3)
        ok = not (v in ADJUSTABLE and n >= MIN_N and trust < TRUST_MIN)
        ex[v] = {'n': n, 'early': e, 'right': n - e, 'trust': trust, 'ok': ok}
    best, by = None, {}
    for tf, st in (dis_stats or {}).items():
        dn = int((st or {}).get('done') or 0)
        al = _f((st or {}).get('avg_last'))
        by[tf] = {'done': dn, 'avg_last': al}
    cands = [(v['avg_last'], tf) for tf, v in by.items()
             if tf in ('1h', '15m', '5m', 'v1h') and v['done'] >= DIS_MIN_DONE and v['avg_last'] is not None]
    if cands:
        best = max(cands)[1]
    return {'scen': scen, 'conf': conf, 'vob1h': v1, 'exit': ex, 'dis': {'best': best, 'by': by},
            'entries': {'n': len(entries or []), 'done': len(done),
                        'open': len([r for r in entries or [] if not r.get('done')])},
            'exits': {'n': len(exits or []), 'done': len(xd)}}


def conf_adjust(conf, scen: str, side: str, learned: Optional[Dict]) -> tuple:
    """Впевненість мозку × множник сценарію → (нова, множник). ЧИСТА."""
    c = _f(conf)
    k = f'{scen}|{str(side).upper()}'
    fac = float((((learned or {}).get('scen') or {}).get(k) or {}).get('factor') or 1.0)
    if c is None or fac == 1.0:
        return conf, 1.0
    return round(max(0.0, min(100.0, c * fac)), 1), fac


def exit_allowed(verdict: str, learned: Optional[Dict]) -> tuple:
    """Чи може ця причина виходу закрити угоду САМА. Захисні — завжди. ЧИСТА.
    → (ok, пояснення)."""
    if verdict in PROTECTIVE or verdict not in ADJUSTABLE:
        return True, ''
    st = ((learned or {}).get('exit') or {}).get(verdict) or {}
    if st and not st.get('ok', True):
        return False, (f"🎓 досвід: вихід «{verdict}» у {st['early']} з {st['n']} випадків був "
                       f"зарано (вчасно {round(st['trust'] * 100)}%) — чекаємо ще одного свідчення")
    return True, ''


def virtual_etalons(entries: List[Dict]) -> tuple:
    """✅ / ❌ тіньові входи → віртуальний досвід мозку еталонів. ЧИСТА."""
    good, anti = [], []
    for r in entries or []:
        if not r.get('done') or not r.get('feat') or len(r['feat']) < 4:
            continue
        base = {'sym': r.get('S'), 'side': r.get('side'), 't': r.get('t'), 'feat': r['feat'],
                'scen': r.get('scen'), 'virtual': True,
                'peak_r': round(float(r.get('mfe') or 0) / float(r.get('risk') or DEF_RISK_PCT), 2)}
        if r.get('res') == 'win':
            good.append(dict(base, r=WIN_R))
        elif r.get('res') == 'loss':
            anti.append(dict(base, r=-1.0))
    return good, anti


def keep_tail(rows: List[Dict], cap: int) -> List[Dict]:
    """Стеля журналу: спершу геть найстаріші ЗАВЕРШЕНІ, відкриті лишаються. ЧИСТА."""
    rows = list(rows or [])
    if len(rows) <= cap:
        return rows
    extra = len(rows) - cap
    out = []
    for r in rows:
        if extra > 0 and r.get('done'):
            extra -= 1
            continue
        out.append(r)
    return out[-cap:]
