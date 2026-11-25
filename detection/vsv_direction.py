"""🧭 VSV SIGNALS — ЙМОВІРНИЙ НАПРЯМОК + ФАКТИЧНА ТОЧНІСТЬ (вимога 25.11).

Дослівно: «додай поле, яке буде інформувати на ймовірний напрямок сигналу
(… чому цей напрямок не є 100% вірним і кінцевим), і додай на скільки відсотків
він все ж таки відповідає дійсності».

Тут ЛИШЕ ЧИСТІ функції (жодного I/O):

1. `direction(rec, adir, ai)` — ЗВЕДЕННЯ ДОКАЗІВ, які бот уже рахує, в один бік:
   • 🌀 натяг Пружини (`prob_long`) або бік імпульсу, що вже почався;
   • ⚠ order flow 6 год — `ai_signals.anomaly_direction` (поглинання, кульмінації,
     ініціатива);
   • 🤖 бік сетапу VSV·AI (лише коли сетап готовий);
   • 📈 рух ціни за 24 год (слабкий контекст).
   net = Σ вага·голос / Σ ваг ПРИСУТНІХ джерел → бік (|net| ≥ `FLAT`), інакше ⚖.
   `prob` — ВЛАСНА впевненість моделі 50…`P_MAX` (не 100: не буває).

2. ЖУРНАЛ (`journal_open` / `journal_update`) — кожен напрямок, що зʼявився в
   таблиці, перевіряється ринком: ціна пройшла +`th`% у бік прогнозу РАНІШЕ,
   ніж −`th`% проти → ✅; навпаки → ❌; за `EVAL_SEC` (24 год) нічого → «—»
   (у точність не входить). Поріг однаковий в обидва боки, тож випадкове
   вгадування дає ~50% — саме з цим числом треба порівнювати.

3. `accuracy(journal)` → фактична точність (усього і за кошиками впевненості);
   `row_accuracy` — число для рядка: кошик, якщо в ньому ≥ `MIN_BUCKET`
   перевірених, інакше загальна (≥ `MIN_N`), інакше None («збираємо дані»).
"""
from typing import Dict, List, Optional

W = {'spring': 0.35, 'flow': 0.35, 'ai': 0.20, 'trend': 0.10}
SRC_UA = {'spring': 'Потенціал «Пружини»', 'flow': 'Order flow · 6 год',
          'ai': 'Сценарій VSV·AI', 'trend': 'Контекст руху · 24 год'}
FLAT = 0.15            # |net| нижче — ⚖ напрямок не визначено
MIN_COVER = 0.35       # мінімум ваги присутніх джерел (Пружина або order flow)
P_MAX = 92.0           # стеля впевненості моделі — 100% не буває
FLOW_DIV = 0.8         # перевага сили боків аномалій, що дає повний голос
TREND_DIV = 5.0        # % руху за 24 год, що дає повний голос
EVAL_SEC = 24 * 3600   # горизонт перевірки прогнозу
TH_MIN, TH_MAX = 1.0, 4.0   # поріг перевірки, % (½ амплітуди в цих межах)
REOPEN_SEC = 6 * 3600  # той самий бік тієї самої монети — не частіше
JOURNAL_CAP = 2000
MIN_N = 20             # загальна точність показується від N перевірених
MIN_BUCKET = 10        # точність кошика — від N перевірених у ньому
BUCKETS = ((50, 60), (60, 70), (70, 80), (80, 101))


def _f(x, d=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return d
    return v if v == v else d


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _side_of(s: float) -> Optional[str]:
    return 'LONG' if s > 0 else ('SHORT' if s < 0 else None)


def evidence(rec: Dict, adir: Optional[Dict] = None, ai: Optional[Dict] = None) -> List[Dict]:
    """Голоси джерел: [{k, label, s (−1…+1, + = LONG), w, txt}]. Відсутнє
    джерело голосу не дає (а не голосує «за нуль»). Формулювання описують
    ринкові докази, а не видають імовірнісну оцінку за торгову рекомендацію."""
    rec = rec if isinstance(rec, dict) else {}
    out = []
    fired = rec.get('fired') if isinstance(rec.get('fired'), dict) else None
    fs = (fired or {}).get('side')
    if rec.get('state') == 'fired' and fs in ('LONG', 'SHORT'):
        s = 1.0 if fs == 'LONG' else -1.0
        out.append({'k': 'spring', 's': s, 'txt': f'імпульс підтверджує {fs}'})
    else:
        pl = _f(rec.get('prob_long'))
        if pl is not None and rec.get('state') not in (None, 'flat') and rec.get('lean'):
            s = _clamp((pl - 50.0) / 50.0)
            side = 'LONG' if s > 0 else 'SHORT'
            out.append({'k': 'spring', 's': s,
                        'txt': f'оцінка {side}: {max(pl, 100 - pl):.0f}%'})
    a = adir if isinstance(adir, dict) else None
    if a and a.get('state') in ('dir', 'weak', 'mixed'):
        up, dn = _f(a.get('up'), 0.0), _f(a.get('dn'), 0.0)
        s = _clamp((up - dn) / FLOW_DIV)
        if a.get('state') == 'weak':
            s *= 0.5
        if a.get('state') == 'mixed':
            s = 0.0
            txt = 'конфлікт агресії: обидва боки активні, переваги немає'
        else:
            txt = ('ініціатива ' + ('покупців' if s > 0 else 'продавців')
                   + (' · слабке підтвердження' if a.get('state') == 'weak' else '')) if s else 'паритет агресії'
        out.append({'k': 'flow', 's': s, 'txt': txt})
    x = ai if isinstance(ai, dict) else None
    if x and x.get('ready') and x.get('side') in ('LONG', 'SHORT'):
        conf = _f(x.get('confidence'))
        k = _clamp((conf / 100.0) if conf is not None else 0.7, 0.3, 1.0)
        s = k if x['side'] == 'LONG' else -k
        out.append({'k': 'ai', 's': s,
                    'txt': f'підтверджений сценарій {x["side"]}' + (f' · {x["scenario_name"]}' if x.get('scenario_name') else '')})
    m24 = _f(rec.get('move24_pct'))
    if m24 is not None:
        s = _clamp(m24 / TREND_DIV)
        out.append({'k': 'trend', 's': s, 'txt': f'зміна {m24:+.2f}%'})
    for e in out:
        e['w'] = W[e['k']]
        e['label'] = SRC_UA[e['k']]
        e['s'] = round(e['s'], 3)
    return out


def direction(rec: Dict, adir: Optional[Dict] = None, ai: Optional[Dict] = None) -> Dict:
    """→ {side, prob, net, agree, against, total, cover, parts, why}.
    side None — напрямок не визначено (⚖ або замало даних)."""
    ev = evidence(rec, adir, ai)
    cover = sum(e['w'] for e in ev)
    base = {'side': None, 'prob': None, 'net': 0.0, 'agree': 0, 'against': 0,
            'total': len(ev), 'cover': round(cover, 2), 'parts': ev}
    if cover < MIN_COVER or not any(e['k'] in ('spring', 'flow') for e in ev):
        return dict(base, why='недостатньо підтверджень: потрібен потенціал «Пружини» або order flow')
    net = sum(e['w'] * e['s'] for e in ev) / cover
    side = _side_of(net) if abs(net) >= FLAT else None
    sign = 1 if net > 0 else -1
    agree = sum(1 for e in ev if e['s'] * sign >= 0.1)
    against = sum(1 for e in ev if e['s'] * sign <= -0.1)
    cov_k = min(1.0, cover / sum(W.values()))
    prob = round(min(P_MAX, 50.0 + 42.0 * abs(net) * (0.6 + 0.4 * cov_k)), 0)
    out = dict(base, net=round(net, 3), agree=agree, against=against)
    if side is None:
        out['why'] = 'докази компенсують одне одного — ⚖ ринкової переваги немає'
        return out
    out.update(side=side, prob=prob,
               why=f'{agree} з {len(ev)} джерел підтверджують {side}' + (f', {against} спростовують' if against else ''))
    return out


def threshold(amp_pct) -> float:
    a = _f(amp_pct)
    return round(_clamp(0.5 * a, TH_MIN, TH_MAX), 2) if a else TH_MIN


# ── 📒 Журнал перевірки ────────────────────────────────────────────────────
def journal_open(journal: List[Dict], rows: List[Dict], now: float,
                 prices: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Новий запис для кожної монети таблиці з визначеним напрямком, якщо по
    цій монеті й БОЦІ немає відкритого запису і останній не новіший за
    `REOPEN_SEC`. Ціна входу — жива (`prices`); без живої ціни запис не
    відкривається (`prices=None` — лише тести: ціна рядка)."""
    j = list(journal or [])
    last = {}
    for e in j:
        k = (e.get('S'), e.get('side'))
        if e.get('st') == 'open' or (now - _f(e.get('t'), 0.0)) < REOPEN_SEC:
            last[k] = True
    for r in rows or []:
        h = (r or {}).get('hint') or {}
        side, S = h.get('side'), str((r or {}).get('symbol') or '').upper()
        if side not in ('LONG', 'SHORT') or not S or last.get((S, side)):
            continue
        # жива ціна обовʼязкова: ціна рядка = закритий 1h-бар, вхід зміщувався б
        px = _f(prices.get(S)) if prices is not None else _f(r.get('price'))
        if not px or px <= 0:
            continue
        j.append({'S': S, 'side': side, 't': now, 'px': px, 'th': threshold(r.get('amp_pct')),
                  'prob': h.get('prob'), 'st': 'open', 'best': 0.0, 'worst': 0.0})
        last[(S, side)] = True
    if len(j) > JOURNAL_CAP:
        done = [e for e in j if e.get('st') != 'open']
        drop = len(j) - JOURNAL_CAP
        gone = {id(e) for e in sorted(done, key=lambda e: _f(e.get('t'), 0.0))[:drop]}
        j = [e for e in j if id(e) not in gone]
    return j


def journal_update(journal: List[Dict], prices: Dict[str, float], now: float) -> List[Dict]:
    """Оновлює відкриті записи ціною моменту; закриває ✅ / ❌ / за часом «—»."""
    out = []
    for e in journal or []:
        e = dict(e)
        if e.get('st') == 'open':
            px = _f((prices or {}).get(e.get('S')))
            p0, th = _f(e.get('px')), _f(e.get('th'), TH_MIN)
            if px and p0:
                mv = (px - p0) / p0 * 100.0 * (1 if e.get('side') == 'LONG' else -1)
                e['best'] = round(max(_f(e.get('best'), 0.0), mv), 3)
                e['worst'] = round(min(_f(e.get('worst'), 0.0), mv), 3)
                e['last'] = round(mv, 3)
                if e['worst'] <= -th:
                    e.update(st='loss', done_t=now)
                elif e['best'] >= th:
                    e.update(st='win', done_t=now)
            if e.get('st') == 'open' and now - _f(e.get('t'), now) >= EVAL_SEC:
                e.update(st='flat', done_t=now)
        out.append(e)
    return out


def accuracy(journal: List[Dict]) -> Dict:
    """Фактична точність: {n, win, loss, flat, open, acc, buckets:{'60-70': {...}}}."""
    tot = {'win': 0, 'loss': 0, 'flat': 0, 'open': 0}
    bk = {f'{a}-{min(b, 100)}': {'win': 0, 'loss': 0} for a, b in BUCKETS}
    for e in journal or []:
        st = e.get('st')
        tot[st if st in tot else 'open'] += 1
        if st in ('win', 'loss'):
            p = _f(e.get('prob'), 50.0)
            for a, b in BUCKETS:
                if a <= p < b:
                    bk[f'{a}-{min(b, 100)}'][st] += 1
                    break
    n = tot['win'] + tot['loss']
    for v in bk.values():
        m = v['win'] + v['loss']
        v['n'] = m
        v['acc'] = round(100.0 * v['win'] / m, 1) if m else None
    return {'n': n, 'win': tot['win'], 'loss': tot['loss'], 'flat': tot['flat'],
            'open': tot['open'], 'acc': round(100.0 * tot['win'] / n, 1) if n else None,
            'min_n': MIN_N, 'buckets': bk}


def row_accuracy(acc: Optional[Dict], prob) -> Dict:
    """Точність для рядка: {acc, n, scope: bucket|all|None, need}."""
    acc = acc or {}
    p = _f(prob)
    if p is not None:
        for a, b in BUCKETS:
            if a <= p < b:
                v = (acc.get('buckets') or {}).get(f'{a}-{min(b, 100)}') or {}
                if (v.get('n') or 0) >= MIN_BUCKET:
                    return {'acc': v.get('acc'), 'n': v['n'], 'scope': 'bucket',
                            'range': f'{a}-{min(b, 100)}'}
                break
    n = acc.get('n') or 0
    if n >= MIN_N:
        return {'acc': acc.get('acc'), 'n': n, 'scope': 'all'}
    return {'acc': None, 'n': n, 'scope': None, 'need': MIN_N}


# ── 🔄 Зміна напрямку (вимога 25.11: «цей сигнал може змінити напрямок») ───
FLIP_SHOW_SEC = 6 * 3600   # скільки показувати позначку «змінився»


def track_flip(state: Dict, sym: str, side: Optional[str], now: float) -> Dict:
    """Памʼять напрямку монети: {side, since, prev, flip_t}. Напрямок
    перераховується на кожному оновленні таблиці і може розвернутись — тоді
    `prev` = попередній бік, `flip_t` = момент розвороту. ⚖ (None) теж стан:
    LONG → ⚖ → SHORT — це розворот (prev = LONG). Повертає НОВИЙ запис."""
    cur = dict((state or {}).get(sym) or {})
    if not cur:
        return {'side': side, 'since': now, 'prev': None, 'flip_t': None, 'last_dir': side}
    if side == cur.get('side'):
        return cur
    last_dir = cur.get('last_dir')
    nxt = {'side': side, 'since': now, 'prev': cur.get('prev'), 'flip_t': cur.get('flip_t'),
           'last_dir': side or last_dir}
    if side in ('LONG', 'SHORT') and last_dir in ('LONG', 'SHORT') and side != last_dir:
        nxt.update(prev=last_dir, flip_t=now)
    return nxt


def flip_view(rec: Dict, now: float) -> Optional[Dict]:
    """Позначка для рядка: {prev, t, ago_min} — лише свіжий розворот."""
    if not rec or not rec.get('flip_t') or not rec.get('prev'):
        return None
    ago = now - float(rec['flip_t'])
    if ago < 0 or ago > FLIP_SHOW_SEC:
        return None
    return {'prev': rec['prev'], 't': rec['flip_t'], 'ago_min': int(ago // 60)}
