"""Канонічні мітки сигналів/двигунів — ЄДИНЕ ДЖЕРЕЛО ПРАВДИ.

Проблема, яку це вирішує: користувач не міг зрозуміти, ВІД ЯКОГО СИГНАЛУ пішла
угода — бо `opened_by` подекуди ніс лише назву двигуна (напр. «Fuel Auto-Filter»,
«🎯 Черга-4»), а оригінальний сигнал (Volumized OB / CHoCH …) губився, коли
сигнал проходив через чергу FF і двигун штампував СВОЮ мітку.

Рішення: `opened_by` тепер зберігається у форматі **"<signal_code> → <engine_code>"**
(сирі коди, машинно-порівнювані — щоб не ламати логіку, що робить substring-
перевірки на 'choch'/'funding'/'POC'…), а ДЛЯ ПОКАЗУ `pretty_opened_by()` мапить
кожну частину в охайний бейдж: «🟪 Volumized OB → 🎯 Черга-4».

Формат обрав користувач: «Сигнал → Двигун», і мітка має супроводжувати угоду
СКРІЗЬ (лог, таблиці, графік, Telegram, інфосайт) — тож і бекенд, і фронтенд
використовують цей самий мапінг (JS-дзеркало — у smart_money.html / infosite).
"""

SEP = ' → '

# --- Оригінальний СИГНАЛ (звідки прийшов) : сирий код → бейдж --------------
SIGNAL_BADGES = {
    'choch':      '🟦 CHoCH',
    'choch_bos':  '🟦 CHoCH+BOS',
    'vob_alert':  '🟪 Volumized OB',
    'vob':        '💰 Volumized OB',      # funding-VOB (kind='vob')
    'opp':        '🔄 Реверс',
    'external':   '🔌 Зовнішня',
    'manual':     '✋ Ручний',
}

# --- ДВИГУН, що фактично відкрив : сирий код → бейдж -----------------------
ENGINE_BADGES = {
    'Q1':               '🎯 Черга-1',
    'Q2':               '🎯 Черга-2',
    'Q3':               '🎯 Готовність',
    'Q4':               '🎯 Черга-4',
    'Q3-VOB(funding)':  '💰 VOB+Шари',    # містить 'funding' → детекція funding-угод
    'POC-сетап':        '🎯 POC-сетап',   # містить 'POC-сетап' → POC exit-логіка
    'EXH':              '🔥 Виснаженість',
    'FF':               '🔥 FF',
    'direct':           '⚡ Прямий',       # FF вимкнено → відкрито одразу з сигналу
}


def signal_badge(code):
    """Бейдж оригінального сигналу за сирим кодом (fallback — сам текст)."""
    if not code:
        return ''
    code = str(code).strip()
    return SIGNAL_BADGES.get(code, code)


def engine_badge(code):
    """Бейдж двигуна за сирим кодом (fallback — сам текст)."""
    if not code:
        return ''
    code = str(code).strip()
    return ENGINE_BADGES.get(code, code)


def compose(signal_code, engine_code=None):
    """Зібрати МАШИННИЙ `opened_by` = "<signal> → <engine>" (сирі коди).

    Якщо engine немає — лише сигнал. Якщо немає сигналу — лише двигун.
    Саме цей рядок кладемо в position['opened_by'] / архів (машинно-
    порівнюваний), а `pretty_opened_by` робить з нього людський вигляд.
    """
    s = (signal_code or '').strip()
    e = (engine_code or '').strip()
    if s and e:
        return f"{s}{SEP}{e}"
    return s or e


def pretty_opened_by(raw):
    """Людський вигляд «Сигнал → Двигун» з будь-якого збереженого `opened_by`.

    Приймає: "<signal> → <engine>", одиничний код, або legacy-рядок (напр.
    старі «🎯 Черга-4 (усі 4 шари)» чи «choch»). Кожну частину, розділену
    ' → ', намагається змапити спершу як сигнал, потім як двигун; якщо код
    невідомий — лишає як є (щоб нічого не «зникало»). Може містити хвіст
    " · <verdict>", доданий пізніше — його не чіпаємо.
    """
    if not raw:
        return ''
    raw = str(raw)
    # Відрізати контекстний хвіст " · ..." (verdict), відновимо в кінці.
    tail = ''
    if ' · ' in raw:
        head, tail = raw.split(' · ', 1)
        tail = ' · ' + tail
    else:
        head = raw
    parts = [p.strip() for p in head.split(SEP)]
    pretty = []
    for i, p in enumerate(parts):
        if i == 0:
            # перша частина — очікуємо сигнал; якщо це насправді двигун —
            # мапимо і як двигун (щоб legacy-двигунні мітки теж гарнішали).
            pretty.append(SIGNAL_BADGES.get(p) or ENGINE_BADGES.get(p) or p)
        else:
            pretty.append(ENGINE_BADGES.get(p) or SIGNAL_BADGES.get(p) or p)
    return SEP.join(pretty) + tail


def signal_code_of(raw):
    """Витягти сирий КОД оригінального сигналу з `opened_by` (для логіки).
    Повертає першу частину до ' → ' і до ' · '. Порожньо — якщо немає.
    """
    if not raw:
        return ''
    head = str(raw).split(' · ', 1)[0]
    return head.split(SEP, 1)[0].strip()
