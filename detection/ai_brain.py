"""🤖🧠 ЄДИНИЙ МОЗОК ШІ (вимога 17.11) — ЧИСТІ функції, без I/O.

Дослівно: «адаптуй це все в єдиний мозок і реалізуй головну ідею — професійно
визначити напрямок і вхід в угоду, а потім чіткий і правильний вихід із угоди».

Два джерела, які раніше працювали окремо, тут зведено в ОДНЕ рішення:
• ⚠ аномалії обсягу + 🌀 Пружина (`ai_signals.entry_score` / `exit_score`) —
  НАПРЯМОК і СТАРТ (комплекс реакцій, сценарії E1–E4 / X1–X5);
• графік закритих свічок 15m / 1h / 4h (`professional_ai`) — ПЛАН угоди
  (стоп від структури / ATR, ціль, R після комісій) і ДИСЦИПЛІНА супроводу
  (стоп плану, підтверджений розворот, віддача прибутку, часовий стоп).

ВХІД (`entry_check`) — три питання ПО ЧЕРЗІ, кожне має відповідь у чек-листі:
 1. 🧭 НАПРЯМОК — немає вето (коробка пробита в протилежний бік, зустрічний
    🔥→🧽, сильний зустрічний потік);
 2. 🚀 СТАРТ — сценарій входу пройдено повністю, точка свіжа (≤2 свічки);
 3. 📋 САНІТАРНІ перевірки графіка: дані свіжі й повні · волатильність у межах ·
    старт не спростовано імпульсом 15m · ЗАПАС до цілі (R після комісій ≥
    `MIN_NET_RR`) · навчена модель (коли вже готова) не проти.
 ⚠️ Трендові вето старого «🧠 Професійного фільтра» (тренд на 3 TF, конфлікт
 1h/4h, якість < 62) вхід НЕ блокують: заряджена пружина стоїть у стисненні, і
 на СТАРТІ розряду тренд ще плаский — ті вето різали б саме найкращу точку.
 Вони живуть у «тіньовому вердикті» (`strict`) — для досліджень, а старші TF
 дають КОНТЕКСТ (впевненість), а не заборону.
 ПЛАН — стоп від структури / ATR на вході (той самий `_risk_plan` мозку графіка),
 ціль — «🎯 Ймовірний хід» Пружини в бік угоди, якщо вона попереду, інакше
 `FALLBACK_TARGET_R`.

ВИХІД (`exit_merge`) — вердикт аномалій + супровід за планом, три групи
свідчень (📖 аномалії · 📉 ціна · 💰 прибуток) тепер бачать і графік:
 • ціна — ще й підтверджений розворот 15m/1h (дві закриті свічки);
 • прибуток — ще й ціль плану досягнута / віддано ≥ ½ піку (пік ≥ 1R).
 Захисне (обидва режими): 🛑 стоп плану зламано · 🧩 сценарій виходу ·
 🔥→🧽 підтверджений · ✅ усі групи. Режим «confirmed» ще: аномалії +
 підтвердження · 🔀 розворот · 💸 віддача прибутку · 🎯 ціль · ⏳ часовий стоп.
 Свіже ⚡/📈 продовження ЗА угодою тримає все, крім захисного.
"""
from __future__ import annotations

from typing import Dict, List, Optional

MIN_NET_RR = 1.3            # запас до цілі: R після комісій і проковзування
FALLBACK_TARGET_R = 2.2     # ціль, коли Пружина цілі попереду не дає
MOMENTUM_AGAINST = -0.25    # імпульс 15m проти входу — старт спростовано
ATR_MIN_PCT = 0.02          # волатильність 15m (ATR % ціни) — межі мозку графіка
ATR_MAX_PCT = 3.0
MODEL_MIN_P = 0.50          # навчена модель (OOS ≥ min) — не проти
HOLD_BLOCK = 0.6            # свіже продовження за угодою — тримаємо
ANOM_GATE = 0.3             # «аномалії кажуть закривати» (як у ai_signals)
GIVEBACK_PEAK_R = 1.0       # віддача прибутку рахується від піку ≥ 1R
GIVEBACK_SHARE = 0.5        # … і поточний R ≤ половини піку

CHECK_UA = {
    'direction': '🧭 напрямок',
    'start': '🚀 старт (сценарій входу)',
    'data': '📊 дані графіка 15m/1h/4h',
    'volatility': '🌡 волатильність 15m',
    'impulse': '⚡ імпульс 15m не проти',
    'room': '🎯 запас до цілі (R після комісій)',
    'model': '🎓 навчена модель',
}

EXIT_VERDICT_UA = {
    'invalid': '🛑 стоп плану зламано — сценарій угоди скасовано',
    'false_start': '🪤 хибний старт: свічка закрилась назад у коробці',
    'scenario': '🧩 сценарій виходу завершено',
    'combo': '🔥→🧽 підтверджений розворот проти угоди',
    'all': '✅ усі групи за вихід (аномалії · ціна · прибуток)',
    'close': '⚠ аномалії + підтвердження ціною / прибутком',
    'reversal': '🔀 підтверджений розворот 15m/1h (дві закриті свічки)',
    'giveback': '💸 віддача прибутку: від піку ≥1R лишилось ≤ половини',
    'target': '🎯 ціль плану досягнуто, продовження не тримає',
    'time': '⏳ сценарій не реалізувався за 8 год — сильніший протилежний бік',
    'hold': '⚡ продовження за угодою — тримаємо',
    'charged': '🌀 заряд у бік угоди ще є — вихід зарано',
    'watch': '👁 аномалії проти — чекаємо підтвердження',
    'wait': '⏳ ціна / прибуток без аномалій — чекаємо',
    'none': '· свідчень за вихід немає',
}
PROTECTIVE = ('invalid', 'false_start', 'scenario', 'combo', 'all')
# 🌀 Заряд у бік угоди (22.11) стримує СЛАБКІ причини виходу; сильні —
# стоп зламано, 🔥→🧽, усі групи, ціль плану, віддача від піку — діють завжди.
CHARGE_HOLDABLE = ('scenario', 'close', 'reversal', 'time')
CLOSE_CONFIRMED = PROTECTIVE + ('close', 'reversal', 'giveback', 'target', 'time')
PRO_REASON = {'professional_reversal': 'reversal', 'professional_profit_giveback': 'giveback',
              'professional_time_stop': 'time', 'professional_invalidation': 'invalid',
              'professional_target': 'target'}


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float('inf') else None


def _sg(side) -> float:
    return 1.0 if str(side).upper() == 'LONG' else -1.0


def _c(x, lo=0.0, hi=1.0) -> float:
    x = _f(x)
    return lo if x is None else max(lo, min(hi, x))


# ─────────────────────────── ПЛАН УГОДИ ───────────────────────────
def spring_target(side: str, rec: Optional[Dict], price: Optional[float]) -> Optional[float]:
    """«🎯 Ймовірний хід» Пружини в бік угоди, лише якщо ціль ПОПЕРЕДУ ціни."""
    rec = rec or {}
    tg = _f(rec.get('tgt_long' if str(side).upper() == 'LONG' else 'tgt_short'))
    px = _f(price)
    if tg is None or px is None or px <= 0 or tg <= 0:
        return None
    return tg if _sg(side) * (tg - px) > 0 else None


def plan_for(side: str, pro: Optional[Dict], rec: Optional[Dict]) -> Optional[Dict]:
    """План угоди з ризик-плану мозку графіка (стоп від структури / ATR на
    останніх закритих 15m) + ціль Пружини. Без аналізу графіка — None."""
    sd = str(side).upper()
    rp = (((pro or {}).get('sides') or {}).get(sd) or {}).get('risk_plan') or {}
    entry, stop = _f(rp.get('entry')), _f(rp.get('stop'))
    dist = _f(rp.get('risk_distance'))
    if not entry or entry <= 0 or stop is None or not dist or dist <= 0:
        return None
    cost = _f(rp.get('assumed_roundtrip_cost_pct')) or 0.0
    tgt = spring_target(sd, rec, entry)
    src = 'spring'
    if tgt is None:
        tgt, src = entry + _sg(sd) * FALLBACK_TARGET_R * dist, 'r'
    risk_pct = 100.0 * dist / entry
    reward_pct = 100.0 * abs(tgt - entry) / entry
    net_rr = (reward_pct - cost) / (risk_pct + cost) if risk_pct > 0 else 0.0
    return {'entry': round(entry, 10), 'stop': round(stop, 10), 'target': round(tgt, 10),
            'target_src': src, 'risk_pct': round(risk_pct, 3), 'reward_pct': round(reward_pct, 3),
            'cost_pct': round(cost, 4), 'net_rr': round(net_rr, 2)}


# ─────────────────────────────── ВХІД ───────────────────────────────
def entry_check(side: str, er: Optional[Dict], pro: Optional[Dict], rec: Optional[Dict]) -> Dict:
    """🤖 Рішення про вхід ОДНИМ мозком: напрямок → старт → санітарні
    перевірки графіка → план. `ok` — усі обовʼязкові пункти пройдено."""
    sd = str(side).upper()
    er, rec = er or {}, rec or {}
    checks: List[Dict] = []

    def add(k, ok, note, hard=True):
        checks.append({'k': k, 'label': CHECK_UA[k], 'ok': ok, 'note': str(note or ''), 'hard': hard})

    veto = er.get('veto') or ''
    add('direction', not veto, veto or f"натяг / потік у бік {sd}")
    sc = er.get('scenario') or {}
    if sc.get('ready') and not sc.get('broken'):
        add('start', True, f"{sc.get('name', '')} {sc.get('done')}/{sc.get('total')} — точка свіжа")
    elif sc.get('broken'):
        add('start', False, f"злам: {sc.get('broken')}")
    elif sc.get('stale'):
        add('start', False, 'сценарій завершено давно — точку входу пропущено')
    elif sc.get('done'):
        add('start', False, f"{sc.get('name', '')} формується {sc.get('done')}/{sc.get('total')}")
    else:
        add('start', False, 'реакцій сценарію ще немає')

    plan, strict, ctx = None, None, None
    pro = pro or {}
    if not pro.get('ok'):
        why = '; '.join((pro.get('data_issues') or [])[:3]) or 'немає закритих барів 15m/1h/4h'
        add('data', False, why)
    else:
        add('data', True, 'закриті свічки свіжі й повні')
        ps = (pro.get('sides') or {}).get(sd) or {}
        f = pro.get('features') or {}
        f15, f1h, f4h = f.get('15m') or {}, f.get('1h') or {}, f.get('4h') or {}
        atr = _f(f15.get('atr_pct'))
        if atr is None:
            add('volatility', False, 'немає ATR 15m')
        else:
            add('volatility', ATR_MIN_PCT <= atr <= ATR_MAX_PCT,
                f"ATR {atr:.2f}% (межі {ATR_MIN_PCT}–{ATR_MAX_PCT}%)")
        mom = _f(ps.get('momentum_signed'))
        if mom is not None:
            add('impulse', mom >= MOMENTUM_AGAINST,
                f"імпульс у бік входу {mom:+.2f}" + ('' if mom >= MOMENTUM_AGAINST else ' — старт спростовано'))
        plan = plan_for(sd, pro, rec)
        if plan is None:
            add('room', False, 'немає ризик-плану (стоп / ATR)')
        else:
            add('room', plan['net_rr'] >= MIN_NET_RR,
                f"R {plan['net_rr']} після комісій (мін {MIN_NET_RR}) · ціль "
                + ('Пружини' if plan['target_src'] == 'spring' else f'{FALLBACK_TARGET_R}R')
                + f" +{plan['reward_pct']:.2f}% · стоп −{plan['risk_pct']:.2f}%")
        model = pro.get('model') or {}
        p = _f(ps.get('model_probability'))
        if model.get('ready') and p is not None:
            add('model', p >= MODEL_MIN_P, f"score {p:.2f} (мін {MODEL_MIN_P})")
        else:
            add('model', None, f"вчиться: OOS {model.get('oos_samples') or 0}/{model.get('min_oos') or 200}"
                ' — на рішення не впливає', hard=False)
        # 🕶 ТІНЬОВИЙ вердикт — що сказав би строгий трендовий фільтр (дослідження).
        strict = {'ready': bool(ps.get('ready')), 'quality': ps.get('quality'),
                  'vetoes': list(ps.get('vetoes') or [])[:6]}
        # 🔭 КОНТЕКСТ старших TF — вага впевненості, не заборона.
        sg = _sg(sd)
        htf = (sg * (0.35 * (_f(f1h.get('trend')) or 0.0) + 0.65 * (_f(f4h.get('trend')) or 0.0)) + 1.0) / 2.0
        part = _c(((ps.get('components') or {}).get('participation')))
        ctx = {'htf': round(_c(htf), 3), 'participation': round(part, 3),
               'note': ('старші TF ПРОТИ — вхід проти течії, обережно'
                        if htf < 0.35 else ('старші TF за угодою' if htf > 0.6 else 'старші TF нейтральні'))}
    ok = all(c['ok'] for c in checks if c['hard'])
    try:
        pl = float(rec.get('prob_long'))
        lean = pl / 100.0 if sd == 'LONG' else 1.0 - pl / 100.0
    except (TypeError, ValueError):
        lean = 0.5
    score = _c((_f(er.get('score')) or 0.0) / 100.0)
    if ctx:
        conf = 0.40 * score + 0.20 * _c(lean) + 0.25 * ctx['htf'] + 0.15 * ctx['participation']
    else:
        conf = 0.65 * score + 0.35 * _c(lean)
    failed = [c for c in checks if c['hard'] and not c['ok']]
    return {'side': sd, 'ok': ok, 'checks': checks, 'plan': plan, 'strict': strict,
            'context': ctx, 'confidence': int(round(100 * conf)),
            'reason': '; '.join(f"{c['label']}: {c['note']}" for c in failed),
            'passed': sum(1 for c in checks if c['hard'] and c['ok']),
            'total': sum(1 for c in checks if c['hard'])}


def checks_text(r: Optional[Dict]) -> str:
    """«✓ напрямок · ✓ старт · ✗ запас …» — для рядка 🧾 Логу."""
    return ' · '.join(('✓ ' if c['ok'] else ('· ' if c['ok'] is None else '✗ ')) + c['label']
                      for c in (r or {}).get('checks') or [])


# ─────────────────────────────── ВИХІД ───────────────────────────────
def plan_from_pro(pro_d: Optional[Dict]) -> Optional[Dict]:
    """Запасний план для угоди, відкритої НЕ сигналом ШІ: ризик-план мозку
    графіка від ATR / структури на вході (ціль — FALLBACK_TARGET_R)."""
    er = (pro_d or {}).get('entry_risk') or {}
    stop, tgt = _f(er.get('stop')), _f(er.get('tp2'))
    if stop is None or tgt is None:
        return None
    return {'entry': _f(er.get('entry')), 'stop': stop, 'target': tgt, 'target_src': 'r',
            'risk_pct': _f(er.get('risk_pct')), 'net_rr': _f(er.get('net_rr_after_costs'))}


def exit_merge(side: str, anom: Optional[Dict], pro_d: Optional[Dict], mode: str = 'confirmed',
               plan: Optional[Dict] = None, price=None, charge: Optional[Dict] = None,
               false_start: Optional[Dict] = None) -> Dict:
    """🤖 Рішення про вихід ОДНИМ мозком: вердикт ⚠ аномалій (`exit_score`) +
    супровід за планом (`manage_position`). Повертає вердикт, групи, чек-лист."""
    sd = str(side).upper()
    sg = _sg(sd)
    anom, pro_d = anom or {}, pro_d or {}
    mode = mode if mode in ('confirmed', 'all') else 'confirmed'
    px = _f(price) or _f(pro_d.get('current_price'))
    own_plan = plan is not None          # план, зафіксований на вході сигналом ШІ
    plan = plan or plan_from_pro(pro_d)
    pr = PRO_REASON.get(pro_d.get('reason')) if pro_d.get('close') else None
    mon = pro_d.get('monitor') or {}
    r_now, peak_r = _f(pro_d.get('r_multiple')), _f(pro_d.get('peak_r'))
    g = dict(anom.get('groups') or {})
    hold = _f(anom.get('hold')) or 0.0
    held = hold >= HOLD_BLOCK
    notes: List[str] = []

    stop = _f((plan or {}).get('stop'))
    # Стоп / ціль судимо за ПЛАНОМ угоди; власні стоп і 2.2R мозку графіка —
    # лише коли угоду відкрито не сигналом ШІ (тоді план і є його план).
    # 🛑 «SL з» ШІ вимкнено (23.11): стоп плану — лише орієнтир ризику, угоду
    # він не закриває (і стоп мозку графіка теж — у угоди ШІ свій план).
    stop_off = bool((plan or {}).get('stop_off'))
    invalid = not stop_off and ((stop is not None and px is not None and sg * (px - stop) <= 0) or (
        pr == 'invalid' and not own_plan))
    tgt = _f((plan or {}).get('target'))
    tgt_hit = (tgt is not None and px is not None and sg * (px - tgt) >= 0) or (
        pr == 'target' and not own_plan)
    reversal = pr == 'reversal'
    trend_turn = bool(mon.get('trend_against') and mon.get('momentum_against'))
    giveback = pr == 'giveback' or (peak_r is not None and r_now is not None and peak_r >= GIVEBACK_PEAK_R
                                    and r_now <= GIVEBACK_SHARE * peak_r)
    price_g = bool(g.get('price')) or reversal or trend_turn
    profit_g = bool(g.get('profit')) or tgt_hit or giveback
    anom_g = bool(g.get('anom'))
    groups = {'anom': anom_g, 'price': price_g, 'profit': profit_g}
    if reversal:
        notes.append('🔀 розворот 15m/1h підтверджено двома свічками')
    elif trend_turn:
        notes.append('📉 тренд і імпульс 15m/1h повернули проти угоди')
    if tgt_hit:
        notes.append('🎯 ціль плану досягнуто')
    if giveback:
        notes.append(f'💸 пік {peak_r}R → зараз {r_now}R')

    av = anom.get('verdict') if anom.get('ready') else None
    if invalid:
        verdict = 'invalid'
    elif false_start:
        verdict = 'false_start'
        notes.append(f"🪤 1h-свічка закрилась {false_start.get('close')} за межею пробою "
                     f"{false_start.get('edge')} — розряд не відбувся")
    elif av in ('scenario', 'combo'):
        verdict = av
    elif anom_g and price_g and profit_g and hold < ANOM_GATE:
        verdict = 'all'
    elif held:
        verdict = 'hold'
    elif anom_g and (price_g or profit_g):
        verdict = 'close'
    elif reversal:
        verdict = 'reversal'
    elif (giveback and trend_turn) or pr == 'giveback':
        verdict = 'giveback'
    elif tgt_hit:
        verdict = 'target'
    elif pr == 'time':
        verdict = 'time'
    elif anom_g:
        verdict = 'watch'
    elif price_g or profit_g:
        verdict = 'wait'
    else:
        verdict = 'none'
    ch = None
    pot, need = _f((charge or {}).get('pot')), _f((charge or {}).get('min'))
    if pot is not None and need:
        ch = {'pot': round(pot, 1), 'min': need, 'hold': pot >= need}
        if ch['hold'] and verdict in CHARGE_HOLDABLE:
            notes.append(f'🌀 заряд у бік угоди {pot:.0f} ≥ {need:.0f} — «{EXIT_VERDICT_UA[verdict]}» ще зарано')
            verdict = 'charged'
    close = verdict in (PROTECTIVE if mode == 'all' else CLOSE_CONFIRMED)
    if verdict == 'charged':
        close = False
    checklist = list(anom.get('checklist') or [])
    checklist += [
        {'k': 'invalid', 'label': '🛑 стоп плану зламано' + (' (SL вимкнено — не діє)' if stop_off else ''),
         'v': 1.0 if invalid else 0.0, 'say': 'close' if invalid else ''},
        {'k': 'false_start', 'label': '🪤 хибний старт розряду', 'v': 1.0 if false_start else 0.0,
         'say': 'close' if false_start else ''},
        {'k': 'reversal', 'label': '🔀 розворот 15m/1h (2 свічки)',
         'v': 1.0 if reversal else (0.5 if trend_turn else 0.0), 'say': 'close' if reversal or trend_turn else ''},
        {'k': 'tgt', 'label': '🎯 ціль плану', 'v': 1.0 if tgt_hit else 0.0, 'say': 'close' if tgt_hit else ''},
        {'k': 'giveback', 'label': '💸 віддача від піку ≥1R', 'v': 1.0 if giveback else 0.0,
         'say': 'close' if giveback else ''},
    ]
    return {'side': sd, 'verdict': verdict, 'verdict_ua': EXIT_VERDICT_UA[verdict], 'close': close,
            'mode': mode, 'groups': groups, 'checklist': checklist, 'notes': notes,
            'plan': plan, 'r': r_now, 'peak_r': peak_r, 'hold': round(hold, 3),
            'data_ok': (pro_d.get('analysis') or {}).get('ok') if pro_d.get('analysis') else None,
            'pro_reason': pro_d.get('reason'), 'charge': ch}


def exit_groups_text(m: Optional[Dict]) -> str:
    g = (m or {}).get('groups') or {}
    return ' · '.join(f"{'✓' if g.get(k) else '✗'} {nm}" for k, nm in
                      (('anom', 'аномалії'), ('price', 'ціна'), ('profit', 'прибуток')))


RULES = {
    'entry': [
        '🧭 Напрямок — натяг Пружини і комплекс реакцій ⚠ аномалій; вето: коробку пробито в протилежний бік, зустрічний 🔥→🧽, сильний зустрічний потік.',
        '💥 Розряд (E2) — лише ПІДТВЕРДЖЕНИЙ: пробій коробки закритою 1h-свічкою і ще N закритих свічок утримали ціну за межею; повернення в коробку = хибний старт (вхід не беремо). 📦 Друге підтвердження — самий останній VOB 1H на графіку в бік розряду і не зламаний (немає / проти / breaker → вхід не беремо).',
        '🚀 Старт — сценарій входу пройдено повністю, остання реакція ≤ 2 свічки тому; дозволені сценарії — у налаштуваннях ШІ (деф. лише E2 «💥 Старт розряду»: інші спрацьовують запізно).',
        f'📋 Санітарні перевірки графіка (закриті 15m/1h/4h): дані свіжі · ATR 15m у межах {ATR_MIN_PCT}–{ATR_MAX_PCT}% · імпульс 15m не проти (≥ {MOMENTUM_AGAINST}) · запас до цілі: R після комісій ≥ {MIN_NET_RR} · навчена модель (коли готова) не проти.',
        f'🗺 План — стоп від структури / ATR останніх закритих 15m, ціль — «Ймовірний хід» Пружини (немає попереду → {FALLBACK_TARGET_R}R).',
        '🕶 Трендові вето (тренд на 3 TF, конфлікт 1h/4h, якість) НЕ блокують: на старті розряду тренд ще плаский. Вони пишуться «тіньовим вердиктом» для досліджень, а старші TF — у впевненість.',
    ],
    'exit': [
        'Захисне (обидва режими): 🛑 ціна зламала стоп плану (лише коли «SL з» ШІ увімкнено) · 🪤 хибний старт (E2: закрита 1h-свічка повернулась у коробку) · 🧩 сценарій виходу (X1–X5) · підтверджений 🔥→🧽 · ✅ усі три групи (аномалії · ціна · прибуток).',
        '«Аномалії + підтвердження» ще закриває: аномалії + ціна або прибуток · 🔀 підтверджений розворот 15m/1h (дві закриті свічки) · 💸 віддача ≥ ½ піку (пік ≥ 1R) · 🎯 ціль плану · ⏳ 8 год без результату, сильніший протилежний бік.',
        'Свіже ⚡/📈 продовження ЗА угодою тримає все, крім захисного. Немає даних графіка — працюють аномалії, стоп плану і далі діє.',
        '🌀 Поки заряд Пружини в бік угоди ≥ порогу (налаштування ШІ), слабкі причини (сценарій виходу, аномалії + підтвердження, розворот 15m/1h, час) — «ще зарано»; стоп плану, 🔥→🧽, усі групи, ціль і віддача від піку діють завжди.',
        '🎯 Вихід на кульмінації (налаштування ШІ, деф. увімкнено): «Виснаження» угоди ≥ порогу або сплеск у бік угоди, поки угода в плюсі → стоп ведеться від вершини на N·ATR(1h), закриття на першому відкаті (пороги — з Exit Rules Trade Manager). Діє окремо від вердикту ШІ і попри заряд Пружини.',
    ],
}
