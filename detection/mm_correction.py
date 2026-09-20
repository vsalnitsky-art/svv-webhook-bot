"""
mm_correction — 🔻 ДЕТЕКТОР КОРЕКЦІЇ для банера «🧮 МММ-монітор» (вимога 19.09).

**Дослівно:** «Потрібно щоб бот моніторив графіки і відслідковував саме корекцію
по монетах… (наприклад на молодших таймфреймах VOB, OB або FVG почали
зʼявлятись на графіках в протилежному напрямку від банера "МММ-монітор"). Бот
має професійно аналізувати графіки і візуалізувати загальний вердикт на банері
"МММ-монітор" стосовно чи почалась або закінчилась корекція.» Плюс: «В період
корекції потрібно обмежити відкриття угод.»

**ЩО ТАКЕ КОРЕКЦІЯ ТУТ.** Банер тримає напрямок (напр. 🟢 LONG), але ринок
ПРЯМО ЗАРАЗ іде проти нього — і це видно по МОЛОДШИХ таймфреймах ОДРАЗУ ПО
БАГАТЬОХ монетах. Тобто це не «одна монета впала», а **ширина ринку**: скільки
монет монітора зараз структурно й ціново дивляться ПРОТИ банера.

**ТРИ ОЗНАКИ (шари).** Кожна — своя природа, і кожна коштує НУЛЬ запитів до
біржі (усе вже пораховано іншими вузлами бота):

  1. **📦 VOB проти** — частка монет, чий ПОТОЧНИЙ Volumized OB на молодшому TF
     (`volumized_timeframe`, звичайно 5m — той самий блок, що малюється на
     графіку) дивиться ПРОТИ банера. Джерело — кеш сканера, який він і так
     оновлює щоциклу; ми лише читаємо.
  2. **💹 Ціна проти** — частка монет, чий свіжий рух (вікно `MM_PRICE_WINDOW`,
     15 хв) іде ПРОТИ банера. Джерело — той самий знімок монітора, що малює
     колонку «Рух».
  3. **📉 Важіль просів** — сам важіль банера У БІК СВОГО НАПРЯМКУ впав від
     свого ПІКУ за вікно на ≥ N п.п. Це «тиск слабшає», тобто корекція вже
     всередині самого показника.

⚠️ **ЧОМУ НЕ FVG І НЕ 1H-OB** (обидва згадані як приклад):
  • **FVG** живе в ОКРЕМОМУ рушії (`fvg_detector` + `ctr_fvg_*`) зі своїм
    скануванням: щоб мати FVG по 200 монетах, довелось би ганяти ще один скан
    біржі — а ми свідомо не додаємо навантаження там, де вже є рівноцінний
    сигнал (📦 VOB на тому самому молодшому TF, безкоштовно).
  • **1H-OB** — СТАРШИЙ таймфрейм. Його розворот означає зміну ТРЕНДУ, а не
    корекцію в тренді; змішати їх означало б плутати дві різні події.

⚠️ **ПОРІГ ВХОДУ І ПОРІГ ВИХОДУ РІЗНІ** (`EXIT_MARGIN_*`): увійти в корекцію
можна лише за суворими порогами, а вийти — коли ознаки впали НИЖЧЕ за
послаблені. Один поріг на обидві події дав би миготіння рівно там, де ринок
найчастіше й зависає (той самий прийом, що `MM_BIAS_EXIT` у банера).
⚠️ **ПІДТВЕРДЖЕННЯ ЧАСОМ** — і на початок, і на кінець (`confirm_sec`): «почалась
корекція» і «корекція завершилась» це ПОДІЇ, а не миготіння.
⚠️ **МАЛА ВИБІРКА — НЕ ОЗНАКА.** Шар, під яким менше ніж `MIN_SAMPLE` монет із
даними, вважається НЕВИЗНАЧЕНИМ і в підрахунок не йде. «Немає даних» ≠ «немає
корекції» і ≠ «корекція є» — той самий принцип, що всюди в проєкті.

Модуль — ЧИСТІ функції без I/O: контекст (знімок, тренди, важіль) збирає
`fuel_filter`, він же тримає стан. Так детектор тестується без бота.
"""

from typing import Dict, List, Optional

# Дефолти налаштувань (живуть у блобі `fuel_filter_settings`).
DEFAULTS = {
    # Сам детектор. Дефолт УВІМК: він нічого не відкриває — лише показує
    # вердикт; блокуванням керує ОКРЕМИЙ тумблер нижче.
    'mm_corr_enabled': True,
    # Скільки ознак із трьох мусить збігтись, щоб оголосити корекцію.
    'mm_corr_min_layers': 2,
    # 📦 Частка монет, чий Volumized OB (молодший TF) дивиться ПРОТИ банера, %.
    'mm_corr_vob_pct': 60.0,
    # 💹 Частка монет, чий свіжий рух ціни йде ПРОТИ банера, %.
    'mm_corr_price_pct': 60.0,
    # 📉 На скільки п.п. важіль банера просів від свого піку за вікно.
    'mm_corr_lever_drop': 15.0,
    # ⏱ Скільки стан мусить протриматись (і на вхід, і на вихід), секунд.
    'mm_corr_confirm_sec': 120,
    # 🚫 Не відкривати угоди, поки триває корекція (вимога 19.09).
    'mm_corr_block_open': True,
}

# Гістерезис: наскільки послаблюємо пороги, коли корекція ВЖЕ триває.
EXIT_MARGIN_PCT = 10.0      # для часток (📦 VOB, 💹 Ціна)
EXIT_MARGIN_PP = 5.0        # для просідання важеля
# Менше монет із даними — шар НЕ визначений (вибірка не репрезентативна).
MIN_SAMPLE = 5
# Вікно, у якому шукаємо ПІК важеля для шару «важіль просів», секунди.
WINDOW_SEC = 1800
# Скільки показуємо «✅ корекція завершилась», перш ніж повернутись у ТРЕНД.
ENDED_SHOW_SEC = 900

STATES = ('trend', 'pending', 'on', 'ending', 'ended')


def _num(v, dflt=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return dflt


def _share(against: int, forward: int):
    """Частка «проти» серед монет, які МАЮТЬ напрямок. None — якщо вибірки
    немає взагалі (ділити нема на що)."""
    tot = against + forward
    return (against / tot * 100.0) if tot > 0 else None


def _layer(key, icon, name, pct, need, n, against, forward, note=''):
    """Один шар у єдиній формі. `ok=False` — шар НЕ визначений (мала вибірка
    або немає даних): він не «за» і не «проти», його просто не рахуємо."""
    ok = pct is not None and n >= MIN_SAMPLE
    return {'key': key, 'icon': icon, 'name': name,
            'pct': None if pct is None else round(pct, 1),
            'need': round(float(need), 1), 'n': n,
            'against': against, 'forward': forward,
            'ok': bool(ok), 'lit': bool(ok and pct >= need), 'note': note}


def vob_layer(trends: Dict, symbols, bias: str, need_pct: float,
              hold: bool = False, tf: str = '') -> Dict:
    """📦 Частка монет монітора, чий Volumized OB (молодший TF) ПРОТИ банера.

    `trends` — {СИМВОЛ: 'LONG'|'SHORT'|None} із кешу сканера (той самий, що
    малює ▲/▼ у watchlist і бокс на графіку). Беремо ЛИШЕ монети, які є в
    моніторі: банер описує саме їх.
    """
    need = float(need_pct) - (EXIT_MARGIN_PCT if hold else 0.0)
    if not trends:
        return _layer('vob', '📦', f'VOB проти{(" " + tf.upper()) if tf else ""}',
                      None, need, 0, 0, 0,
                      note='немає даних Volumized OB (блок 📦 вимкнено або '
                           'сканер ще не порахував)')
    syms = {str(x).upper() for x in (symbols or [])}
    a = f = 0
    for sym, t in trends.items():
        if syms and str(sym).upper() not in syms:
            continue
        if t not in ('LONG', 'SHORT'):
            continue
        if t == bias:
            f += 1
        else:
            a += 1
    return _layer('vob', '📦', f'VOB проти{(" " + tf.upper()) if tf else ""}',
                  _share(a, f), need, a + f, a, f)


def price_layer(snap: Dict, bias: str, need_pct: float,
                hold: bool = False) -> Dict:
    """💹 Частка монет, чий СВІЖИЙ рух ціни йде ПРОТИ банера.

    Джерело — `price_dir` із того самого знімка монітора (колонка «Рух»), тож
    «ціна йде вниз» у банері й у таблиці — це одне й те саме число.
    """
    need = float(need_pct) - (EXIT_MARGIN_PCT if hold else 0.0)
    want = 'up' if bias == 'LONG' else 'down'
    a = f = 0
    for v in (snap or {}).values():
        d = (v or {}).get('price_dir')
        if d not in ('up', 'down'):
            continue          # 'flat' / немає історії — це не рух проти
        if d == want:
            f += 1
        else:
            a += 1
    return _layer('price', '💹', 'Ціна проти', _share(a, f), need, a + f, a, f)


def lever_layer(now_pct, peak_pct, need_drop: float,
                hold: bool = False) -> Dict:
    """📉 Наскільки важіль банера просів від свого ПІКУ за вікно, п.п.

    ⚠️ Міряємо важіль У БІК БАНЕРА (зі знаком): якщо він встиг перевернутись,
    просідання чесно виходить більшим за сам пік — саме так це й виглядає на
    ринку.
    """
    need = max(0.0, float(need_drop) - (EXIT_MARGIN_PP if hold else 0.0))
    n = _num(now_pct)
    p = _num(peak_pct)
    if n is None or p is None:
        return {'key': 'lever', 'icon': '📉', 'name': 'Важіль просів',
                'pct': None, 'need': round(need, 1), 'n': 0,
                'against': 0, 'forward': 0, 'ok': False, 'lit': False,
                'note': 'ще немає історії важеля'}
    drop = max(0.0, p - n)
    return {'key': 'lever', 'icon': '📉', 'name': 'Важіль просів',
            'pct': round(drop, 1), 'need': round(need, 1), 'n': 1,
            'against': 0, 'forward': 0, 'ok': True,
            'lit': bool(need > 0 and drop >= need),
            'note': f'пік {round(p, 1)} п.п. → зараз {round(n, 1)} п.п.'}


def evaluate(snap: Dict, trends: Dict, bias: str, lever_now, lever_peak,
             cfg: Dict, tf: str = '') -> Dict:
    """Усі три шари РАЗОМ — і за суворими порогами, і за послабленими.

    Повертає `{'layers', 'lit', 'lit_hold', 'need', 'determined'}`:
      • `lit`      — скільки ознак за СУВОРИМИ порогами (рішення ПОЧАТИ);
      • `lit_hold` — скільки за ПОСЛАБЛЕНИМИ (рішення ТРИМАТИ) — гістерезис.
    """
    c = dict(DEFAULTS)
    c.update({k: v for k, v in (cfg or {}).items() if k in DEFAULTS})
    need = max(1, int(_num(c['mm_corr_min_layers'], 2) or 2))
    syms = list((snap or {}).keys())
    strict = [vob_layer(trends, syms, bias, c['mm_corr_vob_pct'], False, tf),
              price_layer(snap, bias, c['mm_corr_price_pct'], False),
              lever_layer(lever_now, lever_peak, c['mm_corr_lever_drop'], False)]
    relax = [vob_layer(trends, syms, bias, c['mm_corr_vob_pct'], True, tf),
             price_layer(snap, bias, c['mm_corr_price_pct'], True),
             lever_layer(lever_now, lever_peak, c['mm_corr_lever_drop'], True)]
    return {
        'layers': strict,
        'lit': sum(1 for x in strict if x['lit']),
        'lit_hold': sum(1 for x in relax if x['lit']),
        'need': need,
        'determined': sum(1 for x in strict if x['ok']),
    }


def next_state(prev: Optional[Dict], lit: int, lit_hold: int, need: int,
               now: float, confirm_sec: float,
               ended_show_sec: float = ENDED_SHOW_SEC) -> Dict:
    """ЧИСТА машина станів вердикту: `trend → pending → on → ending → ended`.

    • **trend**   — корекції немає;
    • **pending** — ознаки зійшлись, іде відлік підтвердження (видимий!);
    • **on**      — 🔻 КОРЕКЦІЯ триває (`since` — відколи);
    • **ending**  — ознаки зникли, іде відлік підтвердження кінця;
    • **ended**   — ✅ корекція завершилась (показуємо `ended_show_sec`, далі
      сам собою `trend`).

    ⚠️ ПІДТВЕРДЖЕННЯ СИМЕТРИЧНЕ. «Корекція закінчилась» — така сама подія, як
    і «почалась»: оголосити кінець на одному спокійному такті означало б
    повернути бота в ринок рівно на відкаті всередині корекції.
    ⚠️ ВХІД за `lit` (суворі пороги), УТРИМАННЯ за `lit_hold` (послаблені) —
    саме тут працює гістерезис.
    """
    p = dict(prev or {})
    st = p.get('state') if p.get('state') in STATES else 'trend'
    since = _num(p.get('since'), 0.0) or 0.0
    cand = _num(p.get('cand_since'), 0.0) or 0.0
    ended_at = _num(p.get('ended_at'), 0.0) or 0.0
    lasted = _num(p.get('lasted'), 0.0) or 0.0
    conf = max(0.0, _num(confirm_sec, 0.0) or 0.0)
    want_on = lit >= need
    stay_on = lit_hold >= need

    if st == 'ended' and (now - ended_at) >= max(0.0, ended_show_sec):
        st, ended_at, lasted = 'trend', 0.0, 0.0
    if st in ('trend', 'ended'):
        if want_on:
            if conf <= 0:
                st, since, cand = 'on', now, 0.0
                ended_at = lasted = 0.0
            else:
                st, cand = 'pending', now
    elif st == 'pending':
        if not want_on:
            st, cand = ('ended' if ended_at else 'trend'), 0.0
        elif (now - cand) >= conf:
            st, since, cand = 'on', now, 0.0
            ended_at = lasted = 0.0
    elif st == 'on':
        if not stay_on:
            if conf <= 0:
                st, cand = 'ended', 0.0
                ended_at, lasted = now, max(0.0, now - since)
            else:
                st, cand = 'ending', now
    elif st == 'ending':
        if stay_on:
            st, cand = 'on', 0.0
        elif (now - cand) >= conf:
            st, cand = 'ended', 0.0
            ended_at, lasted = now, max(0.0, now - since)
    return {'state': st, 'since': since, 'cand_since': cand,
            'ended_at': ended_at, 'lasted': lasted}


def is_on(state: Optional[Dict]) -> bool:
    """Корекція ТРИВАЄ саме зараз (лише підтверджений стан).

    ⚠️ `pending`/`ending` — це ВІДЛІК, а не факт: блокувати відкриття на них
    означало б реагувати на шум, від якого підтвердження й захищає.
    """
    return bool((state or {}).get('state') == 'on')


def verdict_text(state: Optional[Dict], layers: Optional[List[Dict]] = None) -> str:
    """Короткий людський підпис вердикту (для логу/Telegram; UI малює сам)."""
    st = (state or {}).get('state') or 'trend'
    if st == 'on':
        return '🔻 КОРЕКЦІЯ триває'
    if st == 'pending':
        return '⏳ ознаки корекції — чекаємо підтвердження'
    if st == 'ending':
        return '⏳ ознаки корекції зникли — чекаємо підтвердження кінця'
    if st == 'ended':
        return '✅ корекція завершилась'
    return '▶️ тренд (корекції немає)'
