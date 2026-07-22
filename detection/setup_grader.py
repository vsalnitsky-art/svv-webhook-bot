"""
🎯 SMC Setup Grader — оцінка «найкращого моменту для входу» на 1H таймфреймі
за принципами Smart Money Concepts (ICT).

Філософія: угода трактується як свінг на 1H (4H — старший контекст / HTF-bias).
Оцінюємо 0..100 «готовність сетапу» + чек-лист конфлюенсів. Це РАДНИК, а не
команда: сам вхід ухвалює двигун/Черга-2. Класичний ланцюг «розумного гроша»:

    зламалась структура (CHoCH/BOS) у наш бік
      → ціна прийшла в POI (Order Block / FVG)
        → у правильній зоні (LONG у Discount, SHORT у Premium)
          → після зняття ліквідності (sweep) і з ціллю попереду
            → з підтримкою потоку ММ (бабло)
              → у правильний момент по CTR (не перекуплено/перепродано)
                → у бік HTF-4H і сесії ₿ BTCUSDT

Модуль ЧИСТИЙ (без побічних ефектів і важких імпортів): приймає вже зібрані
сигнали одним словником `sig`, повертає структуру-оцінку. Викликач (FuelFilter)
збирає `sig` зі своїх уже наявних розрахунків + свіжого SMC-аналізу 1H/4H.
"""

from typing import Dict, List, Optional, Any


# ── Ваги блоків (сума = 100 до застосування «гейтів») ────────────────────────
_WEIGHTS = {
    'structure': 25,   # злам структури + bias + HTF
    'poi':       15,   # Order Block / FVG як точка входу
    'zone':      15,   # Premium/Discount (дилінг-рейндж)
    'liquidity': 15,   # sweep ліквідності + ціль-магніт
    'mm':        12,   # потік/тиск ММ (бабло)
    'timing':    10,   # CTR/STC таймінг + імпульс
    'context':    8,   # HTF-сесія ₿ + funding + обсяг
}

# ── Пороги якості (ті самі, що у SCORE) ──────────────────────────────────────
_GRADE = [
    (72, 'ВІДМІННИЙ', '#2dd4bf'),
    (55, 'ХОРОШИЙ',   '#4ade80'),
    (40, 'СЕРЕДНІЙ',  '#94a3b8'),
    (25, 'СЛАБКИЙ',   '#fb923c'),
    (0,  'ВИЧЕРПАНО', '#f87171'),
]

_DEFAULTS = {
    'hot_min':        72,     # 🎯 лише коли score ≥ цього
    'exh_late':       85,     # виснаженість, що обмежує оцінку
    'sweep_age_max':  120,    # хв — «свіжий» sweep ліквідності
    'ctr_ob':         75,     # STC ≥ → перекупленість (погано для LONG)
    'ctr_os':         25,     # STC ≤ → перепроданість (погано для SHORT)
    'strict':         'strict',  # 'strict' | 'moderate' | 'soft'
}


def _clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def _grade_for(score: float):
    for thr, word, col in _GRADE:
        if score >= thr:
            return word, col
    return 'ВИЧЕРПАНО', '#f87171'


def _g(obj, name, default=None):
    """Безпечне читання поля (obj може бути dataclass-об'єктом або dict)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _enum_val(v):
    """Значення Enum або сам рядок."""
    return getattr(v, 'value', v)


# ─────────────────────────────────────────────────────────────────────────────
#  Блоки конфлюенсу — кожен повертає (frac 0..1, state, detail)
#  state: 'ok' (зелений) | 'warn' (жовтий) | 'miss' (сірий/червоний)
# ─────────────────────────────────────────────────────────────────────────────
def _state(frac):
    return 'ok' if frac >= 0.6 else 'warn' if frac >= 0.3 else 'miss'


def _block_structure(is_long, smc, htf_bias):
    sig = _enum_val(_g(smc, 'structure_signal')) or 'NONE'
    bias = _enum_val(_g(smc, 'market_bias')) or 'NEUTRAL'
    want = 'BULLISH' if is_long else 'BEARISH'
    frac = 0.0
    parts = []
    # 1) Злам структури у наш бік (CHoCH сильніший за BOS — це розворот).
    if want in sig:
        if 'CHOCH' in sig:
            frac += 0.60; parts.append('CHoCH у бік')
        elif 'BOS' in sig:
            frac += 0.50; parts.append('BOS у бік')
    elif sig != 'NONE':
        parts.append('структура проти')
    else:
        frac += 0.15; parts.append('структура нейтральна')
    # 2) Market bias 1H.
    if bias == want:
        frac += 0.25; parts.append('bias 1H збігається')
    elif bias == 'NEUTRAL':
        frac += 0.10
    # 3) HTF-4H bias.
    hb = _enum_val(htf_bias)
    if hb == want:
        frac += 0.15; parts.append('HTF-4H у бік')
    elif hb and hb != 'NEUTRAL':
        parts.append('HTF-4H проти')
    frac = _clamp(frac)
    return frac, _state(frac), (' · '.join(parts) or 'немає структури')


def _block_poi(is_long, smc):
    at_ob = _g(smc, 'price_at_bullish_ob' if is_long else 'price_at_bearish_ob', False)
    ob_count = _g(smc, 'active_bullish_obs' if is_long else 'active_bearish_obs', 0) or 0
    fvgs = _g(smc, 'active_fvgs', []) or []
    fvg_dir = sum(1 for f in fvgs if bool(_g(f, 'is_bullish', True)) == is_long
                  and not _g(f, 'is_filled', False))
    frac = 0.0
    parts = []
    if at_ob:
        frac += 0.60; parts.append('ціна на Order Block')
    if ob_count:
        frac += 0.20; parts.append(f'OB поруч ({ob_count})')
    if fvg_dir:
        frac += 0.20; parts.append(f'FVG у бік ({fvg_dir})')
    frac = _clamp(frac)
    return frac, _state(frac), (' · '.join(parts) or 'немає POI')


def _block_zone(is_long, smc):
    zone = _enum_val(_g(smc, 'price_zone')) or 'EQUILIBRIUM'
    lvl = _g(smc, 'zone_level', 0.5)
    try:
        lvl = float(lvl)
    except (TypeError, ValueError):
        lvl = 0.5
    # LONG вигідно в Discount (низько), SHORT — у Premium (високо).
    frac = (1.0 - lvl) if is_long else lvl
    frac = _clamp(frac)
    if is_long:
        word = 'Discount' if zone == 'DISCOUNT' else ('Equilibrium' if zone == 'EQUILIBRIUM' else 'Premium (дорого)')
    else:
        word = 'Premium' if zone == 'PREMIUM' else ('Equilibrium' if zone == 'EQUILIBRIUM' else 'Discount (дешево)')
    return frac, _state(frac), f'{word} ({lvl:.2f})'


def _block_liquidity(is_long, liq_levels, mark_price, mm_runway, cfg):
    """Sweep ліквідності + ціль-магніт попереду.

    Мапа: перед LONG-розворотом ціна знімає ліквідність ПІД собою — свіжий
    кластер long-ліквідацій нижче mark (лонги вибило) = sweep. Ціль — кластер
    ВИЩЕ (магніт). Для SHORT — дзеркально (short-ліквідації вище = sweep, ціль
    нижче)."""
    parts = []
    frac = 0.0
    lv = liq_levels or []
    mp = mark_price or 0
    age_max = cfg['sweep_age_max']
    sweep = False
    target = False
    if mp > 0 and lv:
        for r in lv:
            try:
                price = float(r.get('price'))
                side = r.get('side')
                age = float(r.get('age_min') or 9999)
            except (TypeError, ValueError, AttributeError):
                continue
            if is_long:
                if side == 'long' and price < mp and age <= age_max:
                    sweep = True
                if price > mp:
                    target = True
            else:
                if side == 'short' and price > mp and age <= age_max:
                    sweep = True
                if price < mp:
                    target = True
    if sweep:
        frac += 0.50; parts.append('свіжий sweep ліквідності')
    # Ціль-магніт: спершу з runway ММ (запас ходу), інакше — кластер попереду.
    room = _g(mm_runway, 'room_pct')
    if isinstance(room, (int, float)) and room > 0:
        frac += _clamp(room / 3.0) * 0.5   # ~3% ходу = повний бал за «є куди йти»
        parts.append(f'запас {room:.1f}%')
    elif target:
        frac += 0.30; parts.append('ціль-магніт попереду')
    frac = _clamp(frac)
    return frac, _state(frac), (' · '.join(parts) or 'ліквідність не підтверджує')


def _block_mm(is_long, mm_dir, mm_strength, mm_conflict):
    try:
        d = float(mm_dir)
    except (TypeError, ValueError):
        d = 0.0
    aligned = (d > 0.05 and is_long) or (d < -0.05 and not is_long)
    strg = float(mm_strength or 0)
    if mm_conflict:
        return 0.0, 'miss', 'конфлікт ціна↔ММ'
    if not aligned:
        return 0.0, 'miss', 'ММ не в бік'
    frac = _clamp(strg / 100.0)
    return frac, _state(frac), f'ММ у бік ({int(strg)}%)'


def _block_timing(is_long, ctr1h, cfg):
    stc = _g(ctr1h, 'stc')
    last = _g(ctr1h, 'last_dir')
    age = _g(ctr1h, 'age')
    if stc is None:
        return 0.3, 'warn', 'CTR ще формується'
    try:
        stc = float(stc)
    except (TypeError, ValueError):
        return 0.3, 'warn', 'CTR ще формується'
    frac = 0.0
    parts = []
    up = (last == 'up' or last == 'LONG')
    dn = (last == 'down' or last == 'SHORT')
    if is_long:
        if stc >= cfg['ctr_ob']:
            return 0.0, 'miss', f'CTR перекуплено ({stc:.0f})'
        frac += _clamp((cfg['ctr_ob'] - stc) / cfg['ctr_ob']) * 0.6
        if up:
            frac += 0.25; parts.append('CTR кросовер вгору')
    else:
        if stc <= cfg['ctr_os']:
            return 0.0, 'miss', f'CTR перепродано ({stc:.0f})'
        frac += _clamp((stc - cfg['ctr_os']) / (100 - cfg['ctr_os'])) * 0.6
        if dn:
            frac += 0.25; parts.append('CTR кросовер вниз')
    if isinstance(age, (int, float)) and age <= 5:
        frac += 0.15; parts.append('свіжий кросовер')
    frac = _clamp(frac)
    parts.insert(0, f'STC {stc:.0f}')
    return frac, _state(frac), ' · '.join(parts)


def _block_context(is_long, btc_dir, btc_start, funding_rate, funding_trend, vol_up, spike):
    frac = 0.0
    parts = []
    want = 'LONG' if is_long else 'SHORT'
    if btc_start and btc_dir == want:
        frac += 0.40; parts.append('₿ сесія у бік')
    elif btc_start and btc_dir in ('LONG', 'SHORT'):
        parts.append('₿ сесія проти')
    # Funding контраріан: LONG любить від'ємний funding (шорти платять), SHORT — навпаки.
    fr = funding_rate
    if isinstance(fr, (int, float)):
        fav = (fr < 0) if is_long else (fr > 0)
        if fav:
            frac += 0.30; parts.append(f'funding {fr:+.3f}%')
    if funding_trend is not None and funding_trend < 0:
        frac += 0.10; parts.append('funding поглиблюється')
    if vol_up or spike:
        frac += 0.20; parts.append('обсяг зростає' + (' · 🚀' if spike else ''))
    frac = _clamp(frac)
    return frac, _state(frac), (' · '.join(parts) or 'контекст нейтральний')


# ─────────────────────────────────────────────────────────────────────────────
#  Головна функція
# ─────────────────────────────────────────────────────────────────────────────
def grade_setup(direction: str, sig: Dict[str, Any], cfg: Optional[Dict] = None) -> Dict:
    """Оцінює готовність SMC-сетапу для `direction` ('LONG'/'SHORT').

    sig (усе опційне, None-безпечно):
      smc          — SMCAnalysisResult 1H (з уже врахованим 4H HTF)
      htf_bias     — MarketBias 4H ('BULLISH'/'BEARISH'/'NEUTRAL') [опц.]
      mp           — analyze_move_potential dict (runway_pct, exhaustion)
      mm_dir       — -1..1, mm_strength — 0..100, mm_conflict — bool
      mm_runway    — compute_mm['runway'] ({room_pct,...})
      ctr1h        — {stc,last_dir,age}
      liq_levels   — [{price,side,age_min,usd},...], mark_price — float
      btc_dir/btc_start/funding_rate/funding_trend/vol_up/spike — контекст
    """
    c = dict(_DEFAULTS)
    if cfg:
        c.update(cfg)
    d = (direction or '').upper()
    if d not in ('LONG', 'SHORT'):
        return {'ok': False, 'dir': d or None, 'score': 0, 'grade': '—',
                'hot': False, 'checks': [], 'vetoes': ['напрямок не визначено']}
    is_long = (d == 'LONG')
    smc = sig.get('smc')
    mp = sig.get('mp') or {}

    blocks = {}
    checks = []

    def add(key, label, res):
        frac, state, detail = res
        blocks[key] = frac
        checks.append({'key': key, 'label': label, 'state': state, 'detail': detail})
        return frac

    f_struct = add('structure', 'Структура', _block_structure(is_long, smc, sig.get('htf_bias')))
    f_poi    = add('poi', 'POI (OB/FVG)', _block_poi(is_long, smc))
    f_zone   = add('zone', 'Зона', _block_zone(is_long, smc))
    f_liq    = add('liquidity', 'Ліквідність',
                   _block_liquidity(is_long, sig.get('liq_levels'), sig.get('mark_price'),
                                    sig.get('mm_runway'), c))
    f_mm     = add('mm', 'Потік ММ',
                   _block_mm(is_long, sig.get('mm_dir'), sig.get('mm_strength'),
                             sig.get('mm_conflict')))
    f_time   = add('timing', 'Таймінг CTR', _block_timing(is_long, sig.get('ctr1h'), c))
    f_ctx    = add('context', 'Контекст',
                   _block_context(is_long, sig.get('btc_dir'), sig.get('btc_start'),
                                  sig.get('funding_rate'), sig.get('funding_trend'),
                                  sig.get('vol_up'), sig.get('spike')))

    # Зважена сума → 0..100.
    total_w = sum(_WEIGHTS.values())
    raw = sum(blocks[k] * _WEIGHTS[k] for k in _WEIGHTS) / total_w * 100.0

    # ── Гейти (обмежують оцінку, хай навіть решта зелена) ────────────────────
    vetoes = []
    exh = mp.get('exhaustion')
    if isinstance(exh, (int, float)) and exh >= c['exh_late']:
        vetoes.append(f'хід виснажено ({int(exh)}%)')
        raw = min(raw, 54)          # не вище «СЕРЕДНІЙ»
    if sig.get('mm_conflict'):
        vetoes.append('конфлікт ціна↔ММ')
        raw = min(raw, 54)
    # CTR жорстко проти — таймінг-блок віддав 'miss' через екстремум.
    if f_time == 0.0 and blocks.get('timing', 0) == 0.0:
        _ct = next((x for x in checks if x['key'] == 'timing'), None)
        if _ct and 'перекуплено' in _ct['detail'] or (_ct and 'перепродано' in _ct['detail']):
            vetoes.append('CTR проти входу')
            raw = min(raw, 39)      # не вище «СЛАБКИЙ»

    score = int(round(_clamp(raw, 0, 100)))
    grade, color = _grade_for(score)

    # ── 🎯 (суворо): усі ключові блоки мають зійтися ─────────────────────────
    strict = c.get('strict', 'strict')
    core_ok = (f_struct >= 0.5 and f_mm > 0 and (f_poi > 0 or f_liq >= 0.5)
               and f_time > 0 and not vetoes)
    if strict == 'soft':
        hot = (score >= c['hot_min'] - 8 and f_mm > 0 and not vetoes)
    elif strict == 'moderate':
        hot = (score >= c['hot_min'] and f_struct >= 0.4 and f_mm > 0 and not vetoes)
    else:  # strict
        hot = (score >= c['hot_min'] and core_ok)

    return {
        'ok': True,
        'dir': d,
        'score': score,
        'grade': grade,
        'color': color,
        'hot': bool(hot),
        'checks': checks,
        'vetoes': vetoes,
        'blocks': {k: round(v, 2) for k, v in blocks.items()},
    }


def checklist_text(res: Dict, sep: str = ' · ') -> str:
    """Короткий чек-лист рядком (для Telegram / tooltip)."""
    if not res or not res.get('ok'):
        return ''
    icon = {'ok': '✓', 'warn': '≈', 'miss': '·'}
    out = []
    for ch in res.get('checks', []):
        out.append(f"{icon.get(ch['state'], '·')} {ch['label']}")
    if res.get('vetoes'):
        out.append('⚠ ' + '; '.join(res['vetoes']))
    return sep.join(out)
