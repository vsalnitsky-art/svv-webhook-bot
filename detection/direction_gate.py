"""🚦 ГОЛОВНІ КНОПКИ НАПРЯМКУ (Trade Direction) — ЄДИНЕ ДЖЕРЕЛО ПРАВДИ.

Вимога користувача дослівно: «Це головні кнопки бота. Тобто якщо вони
вимкнені, це має означати, що бот взагалі не сканує і не відправляє сигнали,
відповідно, якщо увімкнені обидві кнопки, бот працює у всіх напрямках LONG і
SHORT. Бо працює в тому напрямку, в якому є ввімкненою кнопка.»

**Що було не так (аудит 08.09).** Кнопки писали `allow_long_entries` /
`allow_short_entries` у налаштування TM, а перевірялись у ТРЬОХ різних місцях,
кожне зі своїм списком ключів — і жодне не стояло на головному шляху:

  • `smc_scanner`  — не перевіряв ВЗАГАЛІ (жодної згадки ключів);
  • `fuel_filter.intercept` — не перевіряв, тож сигнал БУДЬ-ЯКОГО напрямку
    спокійно ставав у чергу;
  • двигун **Черги-4** — свідомо не перевіряв («приймаємо всі сигнали»);
  • `trade_manager._open_position/_open_shadow` — перевіряли, але під умовою
    `if not bypass_gates`, а Fuel Filter кличе TM саме з `bypass_gates=True`.

Разом це означало, що на робочій установці (працює Черга-4) кнопки **не
впливали ні на що**: з вимкненим LONG бот однаково відкривав LONG.

**Тепер ворота ОДНІ й читаються звідси.** Ключі більше не набираються руками
в чотирьох файлах — тільки `KEY_LONG` / `KEY_SHORT` із цього модуля.

⚠️ **FAIL-OPEN, коли стан невідомий** (немає TM, помилка читання) — свідомо.
Ці ворота вміють ЗУПИНИТИ торгівлю повністю, тож збій читання не має
перетворюватись на тиху зупинку бота: краще пропустити, ніж мовчки стати.
Дефолт обох ключів — `True`, як і в `DEFAULT_SETTINGS` трейд-менеджера.

⚠️ Модуль НЕ імпортує `trade_manager` на рівні файлу (циклічний імпорт) —
лише всередині `live_gates()`.
"""

from typing import Dict, Optional, Tuple

# Ключі в налаштуваннях TradeManager. Історична назва — не міняти: вони вже
# збережені в БД (`tm_settings`), перейменування зламало б робочу установку.
KEY_LONG = 'allow_long_entries'
KEY_SHORT = 'allow_short_entries'

LONG = 'LONG'
SHORT = 'SHORT'


def gates_from(settings: Optional[Dict]) -> Tuple[bool, bool]:
    """(allow_long, allow_short) з довільного словника налаштувань. ЧИСТА."""
    s = settings or {}
    return (bool(s.get(KEY_LONG, True)), bool(s.get(KEY_SHORT, True)))


def allows(settings: Optional[Dict], side: Optional[str]) -> bool:
    """Чи дозволений цей напрямок. ЧИСТА функція.

    Невідомий бік (не LONG/SHORT) → True: ці ворота про напрямок, і вигадувати
    відмову для того, чого вони не стосуються, не можна.
    """
    al, ash = gates_from(settings)
    sd = (side or '').upper().strip()
    if sd == LONG:
        return al
    if sd == SHORT:
        return ash
    return True


def both_off(settings: Optional[Dict]) -> bool:
    """Обидві кнопки вимкнені = бот на ПАУЗІ (жодних нових входів). ЧИСТА."""
    al, ash = gates_from(settings)
    return (not al) and (not ash)


def reason(settings: Optional[Dict], side: Optional[str]) -> str:
    """Людська причина відмови (укр.) або '' коли напрямок дозволено. ЧИСТА.

    Розрізняє ДВА стани, бо для оператора це різні речі: «вимкнено один
    напрямок» (бот працює, але лише в інший бік) і «вимкнено обидва»
    (бот стоїть).
    """
    if allows(settings, side):
        return ''
    sd = (side or '').upper().strip()
    if both_off(settings):
        return ('🚦 Обидва напрямки вимкнені (Trade Direction) — '
                'бот не приймає жодних сигналів')
    return (f'🚦 Кнопка {sd} вимкнена (Trade Direction) — '
            f'сигнали цього напрямку не приймаються')


def chip(settings: Optional[Dict], side: Optional[str]) -> str:
    """Сегмент для РОЗКЛАДУ фільтрів у 🧾 Лозі, напр. `Напрямок[LONG]:✗`.

    Повертає '' коли ОБИДВІ кнопки увімкнені: у цьому стані головний вимикач
    нічого не робить, і писати його в кожен рядок логу — шум. Щойно хоч одна
    кнопка вимкнена — сегмент з'являється в КОЖНОМУ рядку, щоб «чому нема
    сигналів» ніколи не доводилось вгадувати.
    """
    al, ash = gates_from(settings)
    if al and ash:
        return ''
    sd = (side or '').upper().strip() or '?'
    return f"Напрямок[{sd}]:{'✓' if allows(settings, side) else '✗'}"


def state_label(settings: Optional[Dict]) -> str:
    """Короткий підпис стану для UI/статусу: 'LONG+SHORT' / 'лише SHORT' / 'ПАУЗА'."""
    al, ash = gates_from(settings)
    if al and ash:
        return 'LONG+SHORT'
    if al:
        return 'лише LONG'
    if ash:
        return 'лише SHORT'
    return 'ПАУЗА (обидва вимкнені)'


def live_settings() -> Optional[Dict]:
    """Налаштування TM через синглтон. None, якщо TM ще не піднявся."""
    try:
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if tm and hasattr(tm, 'get_settings'):
            return tm.get_settings()
    except Exception:
        pass
    return None


def live_gates() -> Tuple[bool, bool]:
    """(allow_long, allow_short) ЗАРАЗ. Стан невідомий → (True, True)."""
    s = live_settings()
    if s is None:
        return (True, True)
    return gates_from(s)


def live_allows(side: Optional[str]) -> bool:
    """Чи дозволений напрямок ЗАРАЗ (fail-open, якщо TM недоступний)."""
    al, ash = live_gates()
    sd = (side or '').upper().strip()
    if sd == LONG:
        return al
    if sd == SHORT:
        return ash
    return True


def live_reason(side: Optional[str]) -> str:
    """Причина відмови ЗАРАЗ або '' (fail-open, якщо TM недоступний)."""
    s = live_settings()
    if s is None:
        return ''
    return reason(s, side)
