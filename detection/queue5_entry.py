"""
🎯 Черга-5 — SMC-модель входу «максимально чіткий варіант» (5 кроків).

Реалізує алгоритм користувача, крок за кроком, поверх наявних компонентів бота:

  1) HTF-контекст  — Forecast 1H : Forecast 4H (get_forecast_engine).
                     bias LONG, якщо обидва >0; SHORT, якщо обидва <0.
                     HTF POI — 1H Order Block/FVG у бік bias (ціна в зоні).
  2) Тест зони + зняття ліквідності (M15) — ціна в HTF POI + свіп ліквідності:
     локальний swing-екстремум АБО PDH/PDL (Previous Day High/Low).
  3) CHoCH на M15 із закриттям тіла — smc_analyzer.structure_signal у бік bias.
  4) Entry Matrix (M15) — ціна в 15m FVG або OB у бік bias + Premium/Discount
     (LONG лише в Discount, SHORT лише в Premium).
  5) SL/TP/R:R — SL за екстремум зняття ліквідності + буфер; TP на протилежний
     пул ліквідності; R:R ≥ queue5_min_rr (дефолт 3), інакше — пропуск.

Модуль САМОДОСТАТНІЙ (сам тягне klines/forecast/smc), повертає рішення-словник.
Виконання (Market-вхід + SL/TP) робить двигун Черги-5 у fuel_filter.
"""
from typing import Dict, List, Optional, Any

# TF-и алгоритму (M15 робочий, 1H — HTF-структура/POI, D — PDH/PDL).
_TF_ENTRY = '15'
_TF_HTF = '1h'
_TF_DAY = 'D'


def _kv(k: Dict, *names, default=0.0) -> float:
    for n in names:
        if n in k and k[n] is not None:
            try:
                return float(k[n])
            except (TypeError, ValueError):
                pass
    return default


def _highs(kl):  return [_kv(k, 'h', 'high') for k in kl]
def _lows(kl):   return [_kv(k, 'l', 'low') for k in kl]
def _closes(kl): return [_kv(k, 'c', 'p', 'close') for k in kl]


def _fetch(md, sym, tf, limit):
    try:
        if hasattr(md, 'fetch_klines') and 'interval' in md.fetch_klines.__code__.co_varnames:
            return md.fetch_klines(sym, limit=limit, interval=tf)
        return md.fetch_klines(sym, limit=limit)
    except Exception:
        return None


def _step(n, name, ok, detail):
    return {'n': n, 'name': name, 'ok': bool(ok), 'detail': detail}


def evaluate(symbol: str, s: Dict, live_price: Optional[float] = None) -> Dict[str, Any]:
    """Оцінити всі 5 кроків Черги-5 для монети. Повертає:
      {ok, side, entry, sl, tp, rr, count, steps[], reason}.
    ok=True → усі 5 кроків зійшлись і угоду можна відкривати (Market)."""
    out = {'ok': False, 'side': None, 'entry': None, 'sl': None, 'tp': None,
           'rr': None, 'count': 0, 'steps': [], 'reason': ''}

    def done(reason=''):
        out['count'] = sum(1 for st in out['steps'] if st['ok'])
        if reason and not out['reason']:
            out['reason'] = reason
        return out

    try:
        min_rr = float(s.get('queue5_min_rr', 3.0) or 3.0)
    except (TypeError, ValueError):
        min_rr = 3.0
    try:
        buf = max(0.0, float(s.get('queue5_sl_buffer_pct', 0.10) or 0)) / 100.0
    except (TypeError, ValueError):
        buf = 0.001
    try:
        sweep_lb = int(s.get('queue5_sweep_lookback', 12) or 12)
    except (TypeError, ValueError):
        sweep_lb = 12

    # ── КРОК 1: HTF-контекст (Forecast 1H + 4H) ──────────────────────────
    bias = None
    try:
        from detection.forecast_engine import get_forecast_engine
        fe = get_forecast_engine()
        fc = fe.get(symbol) if fe else None
        f1 = (fc or {}).get('forecast_1h') or {}
        f4 = (fc or {}).get('forecast_4h') or {}
        s1 = float(f1.get('side', 0) or 0)
        s4 = float(f4.get('side', 0) or 0)
        if s1 > 0 and s4 > 0:
            bias = 'LONG'
        elif s1 < 0 and s4 < 0:
            bias = 'SHORT'
        out['steps'].append(_step(1, 'HTF контекст (Forecast 1H:4H)', bias is not None,
                                  f"1H {s1:+.0f} · 4H {s4:+.0f} → {bias or 'нема збігу'}"))
    except Exception as e:
        out['steps'].append(_step(1, 'HTF контекст (Forecast 1H:4H)', False, f'forecast err: {e}'))
    if bias is None:
        return done('HTF: 1H/4H не збіглись')
    out['side'] = bias

    # ── Дані ────────────────────────────────────────────────────────────
    try:
        from detection.market_data import get_market_data
        md = get_market_data()
    except Exception:
        md = None
    kl15 = _fetch(md, symbol, _TF_ENTRY, 320) if md else None
    kl1h = _fetch(md, symbol, _TF_HTF, 320) if md else None
    klD = _fetch(md, symbol, _TF_DAY, 6) if md else None
    if not kl15 or len(kl15) < 60 or not kl1h or len(kl1h) < 60:
        out['steps'].append(_step(2, 'Дані', False, 'недостатньо свічок'))
        return done('немає даних')

    entry = float(live_price) if live_price else _closes(kl15)[-1]
    out['entry'] = round(entry, 8)

    # PDH/PDL — попередній ЗАКРИТИЙ день.
    pdh = pdl = None
    if klD and len(klD) >= 2:
        pdh = _kv(klD[-2], 'h', 'high')
        pdl = _kv(klD[-2], 'l', 'low')

    # SMC-аналіз M15 (HTF-контекст = 1H) і 1H (для HTF POI).
    try:
        from detection.smc_analyzer import get_smc_analyzer, StructureSignal, PriceZone
        an = get_smc_analyzer()
        res15 = an.analyze(kl15, htf_klines=kl1h)
        res1h = an.analyze(kl1h)
    except Exception as e:
        out['steps'].append(_step(2, 'SMC-аналіз', False, f'smc err: {e}'))
        return done('smc помилка')

    # ── КРОК 2: HTF POI + зняття ліквідності ─────────────────────────────
    at_poi = (res1h.price_at_bullish_ob if bias == 'LONG' else res1h.price_at_bearish_ob)
    # 1H FVG у бік як альтернатива POI.
    if not at_poi:
        for fvg in (res1h.active_fvgs or []):
            if fvg.is_filled:
                continue
            if bias == 'LONG' and fvg.is_bullish and fvg.low <= entry <= fvg.high:
                at_poi = True; break
            if bias == 'SHORT' and (not fvg.is_bullish) and fvg.low <= entry <= fvg.high:
                at_poi = True; break

    # Свіп ліквідності на M15: для LONG знято sell-side (мінімум), для SHORT — buy-side.
    lows15 = _lows(kl15); highs15 = _highs(kl15)
    recent_low = min(lows15[-sweep_lb:]) if len(lows15) >= sweep_lb else min(lows15)
    recent_high = max(highs15[-sweep_lb:]) if len(highs15) >= sweep_lb else max(highs15)
    swing_low = res15.last_ll.price if res15.last_ll else (res15.last_hl.price if res15.last_hl else None)
    swing_high = res15.last_hh.price if res15.last_hh else (res15.last_lh.price if res15.last_lh else None)
    swept = False
    sweep_ref = None   # рівень, під/над яким знято ліквідність (для SL)
    if bias == 'LONG':
        for lvl in (swing_low, pdl):
            if lvl and recent_low < lvl <= entry:
                swept = True
                sweep_ref = lvl if sweep_ref is None else min(sweep_ref, lvl)
        if sweep_ref is None:
            sweep_ref = recent_low
    else:
        for lvl in (swing_high, pdh):
            if lvl and recent_high > lvl >= entry:
                swept = True
                sweep_ref = lvl if sweep_ref is None else max(sweep_ref, lvl)
        if sweep_ref is None:
            sweep_ref = recent_high
    out['steps'].append(_step(2, 'Тест зони + зняття ліквідності',
                              bool(at_poi and swept),
                              f"POI:{'✓' if at_poi else '—'} · свіп:{'✓' if swept else '—'}"))
    if not (at_poi and swept):
        return done('немає тесту POI / зняття ліквідності')

    # ── КРОК 3: CHoCH на M15 (закриття тіла) ─────────────────────────────
    ss = res15.structure_signal
    choch_ok = ((bias == 'LONG' and ss == StructureSignal.BULLISH_CHOCH)
                or (bias == 'SHORT' and ss == StructureSignal.BEARISH_CHOCH))
    out['steps'].append(_step(3, 'CHoCH на M15 (закриття тіла)', choch_ok,
                              f"{getattr(ss, 'value', ss)}"))
    if not choch_ok:
        return done('немає CHoCH M15 у бік')

    # ── КРОК 4: Entry Matrix (FVG/OB) + Premium/Discount ─────────────────
    in_ob = (res15.price_at_bullish_ob if bias == 'LONG' else res15.price_at_bearish_ob)
    in_fvg = False
    for fvg in (res15.active_fvgs or []):
        if fvg.is_filled:
            continue
        if bias == 'LONG' and fvg.is_bullish and fvg.low <= entry <= fvg.high:
            in_fvg = True; break
        if bias == 'SHORT' and (not fvg.is_bullish) and fvg.low <= entry <= fvg.high:
            in_fvg = True; break
    zone_ok = ((bias == 'LONG' and res15.price_zone == PriceZone.DISCOUNT)
               or (bias == 'SHORT' and res15.price_zone == PriceZone.PREMIUM))
    entry_ok = (in_ob or in_fvg) and zone_ok
    out['steps'].append(_step(4, 'Entry Matrix (FVG/OB + Prem/Disc)', entry_ok,
                              f"OB:{'✓' if in_ob else '—'} FVG:{'✓' if in_fvg else '—'} "
                              f"зона:{getattr(res15.price_zone, 'value', '')}"))
    if not entry_ok:
        return done('ціна не в FVG/OB або не в Discount/Premium')

    # ── КРОК 5: SL / TP / R:R ────────────────────────────────────────────
    # TP — протилежний пул ліквідності (найближчий у бік прибутку).
    tp = None
    if bias == 'LONG':
        cands = [x for x in (swing_high, pdh, recent_high) if x and x > entry]
        tp = min(cands) if cands else None
        sl = round(sweep_ref * (1.0 - buf), 8)
    else:
        cands = [x for x in (swing_low, pdl, recent_low) if x and x < entry]
        tp = max(cands) if cands else None
        sl = round(sweep_ref * (1.0 + buf), 8)
    rr = None
    ok5 = False
    if tp and sl:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk > 0:
            rr = round(reward / risk, 2)
            ok5 = rr >= min_rr
    out['sl'] = round(sl, 8) if sl else None
    out['tp'] = round(tp, 8) if tp else None
    out['rr'] = rr
    out['steps'].append(_step(5, f'SL/TP/R:R ≥ {min_rr:g}', ok5,
                              f"SL {out['sl']} · TP {out['tp']} · R:R {rr}"))
    if not ok5:
        return done(f'R:R {rr} < {min_rr:g}' if rr is not None else 'немає TP-пулу')

    out['ok'] = True
    return done()
