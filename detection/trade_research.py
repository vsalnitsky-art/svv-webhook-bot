"""🔬 ДОСЛІДЖЕННЯ УГОДИ — запис і підсумок (16.11). ЧИСТІ функції, без I/O.

Вимога: «веди всю інформацію для дослідження "Пружина вистрілила" і "Вихід на
кульмінації (Виснаження)" для аналізу і розуміння, які ідеальні показники
використовувати» + «архівуй всю потрібну інформацію… як навчити бота
приймати ідеальні рішення».

Стан угоди — `pos['research']` = {ev: [...], s: [...], pk, tr, seen}:
  • `s`  — зрізи раз на SAMPLE_SEC: t, ціна, PnL, 🎯 прогрес до цілі Пружини,
           💥 готовність вистрілу (діапазон ×ATR, обсяг ×, коробка, стиснення),
           сплеск-кульмінація, стопи трейлів;
  • `ev` — ПОДІЇ в момент появи: 🎯 кульмінація (за / проти угоди), 💥 вистріл,
           рубежі прогресу до цілі 25/50/75/100 %, рубежі готовності вистрілу,
           озброєння й рух трейлу «Досягнення вершини», трейл 🔔 SBR;
  • `pk` / `tr` — ПІК і ДНО угоди з УСІМА показниками в ту мить: саме вони
           відповідають на «які показники були в ідеальну точку виходу».
Записи прибиті до часу; PnL — % руху ціни від входу (без плеча).
"""
from typing import Dict, List, Optional

SAMPLE_SEC = 300            # зріз показників раз на 5 хв
SAMPLES_MAX = 288           # стеля зрізів у позиції (доба по 5 хв; далі проріджуємо вдвічі)
EVENTS_MAX = 150
PROG_MILESTONES = (25, 50, 75, 100)
FIRE_MILESTONES = (50, 75)  # готовність вистрілу живого бару, %
TRAIL_MOVE_GAP = 900        # рух стопу трейлу — не частіше ніж раз на 15 хв
SNAP_KEYS = ('px', 'pnl', 'prog', 'exh', 'fp', 'rx', 'vx', 'bx', 'sq', 'clx',
             'fired', 'tstop', 'sstop')


def _r(v, n=3):
    try:
        return None if v is None else round(float(v), n)
    except (TypeError, ValueError):
        return None


def pnl_of(side, entry, price) -> Optional[float]:
    try:
        en, px = float(entry), float(price)
    except (TypeError, ValueError):
        return None
    if en <= 0 or px <= 0:
        return None
    sd = str(side or '').upper()
    if sd not in ('LONG', 'SHORT'):
        return None
    return (px / en - 1) * 100 * (1 if sd == 'LONG' else -1)


def snap_of(pos: Dict, price, vw: Optional[Dict], ex: Optional[Dict]) -> Dict:
    """Зріз показників угоди в цю мить."""
    vw = vw if isinstance(vw, dict) else {}
    ex = ex if isinstance(ex, dict) else {}
    parts = vw.get('parts') or {}
    f = vw.get('fired')
    tt = pos.get('exh_trail') if isinstance(pos.get('exh_trail'), dict) else {}
    st = pos.get('sbr_trail') if isinstance(pos.get('sbr_trail'), dict) else {}
    return {
        'px': _r(price, 10), 'pnl': _r(pnl_of(pos.get('side'), pos.get('entry_price'), price)),
        'prog': _r(ex.get('prog'), 1), 'exh': _r(ex.get('pct'), 1),
        'fp': _r(vw.get('pct'), 1), 'rx': _r(vw.get('range_x'), 2),
        'vx': _r(vw.get('vol_x'), 2), 'bx': _r(parts.get('box'), 1),
        'sq': vw.get('pre_sq'), 'clx': vw.get('climax'),
        'fired': (f.get('side') if isinstance(f, dict) else None),
        'tstop': _r(tt.get('stop_px'), 10), 'sstop': _r(st.get('stop_px'), 10),
    }


def _ev(rs: Dict, t: float, kind: str, snap: Dict, **kw) -> None:
    e = {'t': int(t), 'k': kind}
    e.update({k: snap.get(k) for k in SNAP_KEYS if snap.get(k) is not None})
    e.update({k: v for k, v in kw.items() if v is not None})
    ev = rs.setdefault('ev', [])
    ev.append(e)
    if len(ev) > EVENTS_MAX:
        del ev[:len(ev) - EVENTS_MAX]


def step(rs: Dict, side: str, now: float, snap: Dict, bar_t=None) -> List[Dict]:
    """Оновити стан дослідження зрізом `snap`. → нові події (для логу/тестів).
    `bar_t` — відкриття живого 1h-бару (ключ «одна подія на бар»)."""
    sd = str(side or '').upper()
    seen = rs.setdefault('seen', {})
    n0 = len(rs.get('ev') or [])
    pnl = snap.get('pnl')
    # пік / дно з показниками в ту мить
    if pnl is not None:
        pk = rs.get('pk')
        if pk is None or pnl > float(pk.get('pnl') or -1e9):
            rs['pk'] = dict(snap, t=int(now))
        tr = rs.get('tr')
        if tr is None or pnl < float(tr.get('pnl') or 1e9):
            rs['tr'] = dict(snap, t=int(now))
    bk = int(bar_t or 0)
    # 🎯 кульмінація (сплеск живого бару за межу коробки)
    clx = snap.get('clx')
    if clx in ('LONG', 'SHORT'):
        key = f'clx|{clx}|{bk}'
        if key not in seen:
            seen[key] = 1
            _ev(rs, now, 'climax', snap, rel='with' if clx == sd else 'against', dir=clx)
    # 💥 вистріл
    fd = snap.get('fired')
    if fd in ('LONG', 'SHORT'):
        key = f'fire|{fd}|{bk}'
        if key not in seen:
            seen[key] = 1
            _ev(rs, now, 'fire', snap, rel='with' if fd == sd else 'against', dir=fd)
    # рубежі прогресу до цілі Пружини
    pg = snap.get('prog')
    if pg is not None:
        for m in PROG_MILESTONES:
            key = f'prog|{m}'
            if pg >= m and key not in seen:
                seen[key] = 1
                _ev(rs, now, 'prog', snap, m=m)
    # рубежі готовності вистрілу (раз на бар)
    fp = snap.get('fp')
    if fp is not None:
        for m in FIRE_MILESTONES:
            key = f'fp|{m}|{bk}'
            if fp >= m and key not in seen and not fd:
                seen[key] = 1
                _ev(rs, now, 'fp', snap, m=m)
    # трейл «Досягнення вершини»
    ts = snap.get('tstop')
    if ts is not None:
        if 'trail' not in seen:
            seen['trail'] = int(now)
            seen['trail_px'] = ts
            _ev(rs, now, 'trail_arm', snap)
        elif ts != seen.get('trail_px') and now - float(seen.get('trail') or 0) >= TRAIL_MOVE_GAP:
            seen['trail'] = int(now)
            seen['trail_px'] = ts
            _ev(rs, now, 'trail_move', snap)
    ss = snap.get('sstop')
    if ss is not None and 'sbr_trail' not in seen:
        seen['sbr_trail'] = 1
        _ev(rs, now, 'sbr_trail', snap)
    # зріз раз на SAMPLE_SEC
    s = rs.setdefault('s', [])
    if not s or now - float(s[-1][0]) >= SAMPLE_SEC:
        s.append([int(now)] + [snap.get(k) for k in SNAP_KEYS])
        if len(s) > SAMPLES_MAX:
            rs['s'] = s[::2]
    # щоб `seen` не ріс безмежно (ключі з баром) — лишаємо останні 400
    if len(seen) > 400:
        for k in list(seen.keys())[:len(seen) - 400]:
            seen.pop(k, None)
    return (rs.get('ev') or [])[n0:]


def samples_dicts(rs: Dict) -> List[Dict]:
    return [dict(zip(('t',) + SNAP_KEYS, row)) for row in (rs or {}).get('s') or []]


def _first(ev, kind, rel=None):
    for e in ev:
        if e.get('k') == kind and (rel is None or e.get('rel') == rel):
            return e
    return None


def summary(side: str, entry, exit_price, pnl_pct, rs: Dict,
            anomalies: Optional[List[Dict]] = None) -> Dict:
    """Підсумок для калібровки: пік / дно з показниками, «що дав би вихід» на
    першій події кожного виду, і скільки від піку взяв фактичний вихід."""
    rs = rs or {}
    ev = rs.get('ev') or []
    pk, tr = rs.get('pk') or {}, rs.get('tr') or {}
    out = {'pnl': _r(pnl_pct), 'peak': pk or None, 'trough': tr or None}
    try:
        pv = float(pk.get('pnl'))
        out['peak_share'] = round(float(pnl_pct) / pv * 100, 1) if pv > 0 else None
        out['giveback'] = round(pv - float(pnl_pct), 3)
    except (TypeError, ValueError):
        pass
    alt = {}
    for name, kind, rel in (('climax_with', 'climax', 'with'),
                            ('climax_against', 'climax', 'against'),
                            ('fire_with', 'fire', 'with'),
                            ('fire_against', 'fire', 'against'),
                            ('prog50', 'prog', None), ('prog75', 'prog', None),
                            ('prog100', 'prog', None), ('trail_arm', 'trail_arm', None),
                            ('fp75', 'fp', None)):
        if name.startswith('prog'):
            e = next((x for x in ev if x.get('k') == 'prog' and x.get('m') == int(name[4:])), None)
        elif name == 'fp75':
            e = next((x for x in ev if x.get('k') == 'fp' and x.get('m') == 75), None)
        else:
            e = _first(ev, kind, rel)
        if e and e.get('pnl') is not None:
            alt[name] = {'t': e['t'], 'pnl': e['pnl'],
                         'vs_exit': _r(float(e['pnl']) - float(pnl_pct or 0))}
    sd = str(side or '').upper()
    for a in anomalies or []:
        h = a.get('hint')
        if not h or sd not in ('LONG', 'SHORT'):
            continue
        rel = 'with' if (h == 'up') == (sd == 'LONG') else 'against'
        name = f"anom_{a.get('kind')}_{rel}"
        if name in alt:
            continue
        p = pnl_of(sd, entry, a.get('price'))
        if p is not None:
            alt[name] = {'t': int(float(a.get('ts') or 0)), 'pnl': _r(p),
                         'vs_exit': _r(p - float(pnl_pct or 0)), 'tf': a.get('tf')}
    out['if_exit_at'] = alt
    out['events'] = len(ev)
    out['samples'] = len(rs.get('s') or [])
    out['anomalies'] = len(anomalies or [])
    return out


EV_UA = {
    'climax': '🎯 Кульмінація',
    'fire': '💥 Пружина вистрілила',
    'prog': '🎯 Прогрес до цілі',
    'fp': '💥 Готовність вистрілу',
    'trail_arm': '🎯 Трейл «вершина» озброєно',
    'trail_move': '🎯 Трейл «вершина» — стоп зсунуто',
    'sbr_trail': '🔔⏳ Трейл SBR озброєно',
}


def journey_events(rs: Dict) -> List[Dict]:
    """Події дослідження → вузли «Шляху монети»."""
    out = []
    for e in (rs or {}).get('ev') or []:
        k = e.get('k')
        nm = EV_UA.get(k, k)
        if k in ('climax', 'fire'):
            nm += ' ' + ('▲' if e.get('dir') == 'LONG' else '▼') + \
                (' за угодою' if e.get('rel') == 'with' else ' ПРОТИ угоди')
        elif k in ('prog', 'fp'):
            nm += f" {e.get('m')}%"
        out.append({'t': e.get('t'), 'event': 'research', 'detail': nm,
                    'side': e.get('dir'), 'source': 'R&D', 'x': dict(e)})
    return out
