"""🌀 / 🔭 ВОРОТА «ЧЕРЕЗ ПРУЖИНУ» для обох таблиць (вимога 10.11) — ЧИСТІ правила.

Дослівно: «Додай такі ж фільтри як "🔔 Через МММ-SBR" і для обох пружин окремо
кожен» + «Монети "Пружина" WATCHLIST, які проходять ворота (виділи ці монети в
таблиці)».

Той самий принцип, що в МММ-воротах:
- «Через …» — монета стоїть у ВКЛАДЦІ таблиці того самого боку, що й сигнал
  (вкладки 🟢 LONG / 🔴 SHORT рахуються ТИМ САМИМ правилом, що `_sprFilter` на
  сторінці: потенціал у бік більший за протилежний і стан не «без заряду»);
- «за банером …» — ще й банер «Куди накручено» (`spring_coil.aggregate`) має
  той самий напрямок. ⚖ / замало заряджених → відмова.
«Невідомо» ≠ «за напрямком»: монети немає в таблиці → відмова.
Жодного I/O — дані подає сканер.
"""
from typing import Dict, Optional, Tuple

SIDES = ('LONG', 'SHORT')
TAG_WL = '🌀Пружина-WL'
TAG_XS = '🔭Пружина-скан'
NAME = {TAG_WL: '🌀 Пружина (WATCHLIST)', TAG_XS: '🔭 Пружина — скан біржі'}


def _f(x) -> float:
    try:
        v = float(x)
        return v if v == v else 0.0
    except (TypeError, ValueError):
        return 0.0


def tab_side(rec: Optional[Dict]) -> Optional[str]:
    """Вкладка монети в таблиці (дзеркало JS `_sprFilter`): LONG / SHORT / None."""
    if not isinstance(rec, dict) or rec.get('ok') is False:
        return None
    if (rec.get('state') or 'flat') == 'flat':
        return None
    pl, ps = _f(rec.get('pot_long')), _f(rec.get('pot_short'))
    if pl > ps:
        return 'LONG'
    if ps > pl:
        return 'SHORT'
    return None


def banner_side(agg: Optional[Dict]) -> Optional[str]:
    """Напрям банера «Куди накручено»: лише коли заряджених достатньо."""
    if not isinstance(agg, dict) or agg.get('enough') is False:
        return None
    d = agg.get('lean')
    return d if d in SIDES else None


def passes_gates(rec: Optional[Dict], settings: Dict, potential=None) -> Tuple[bool, Optional[str], str]:
    """Монета 🌀 WATCHLIST «проходить ворота» → (ok, бік, причина).

    Бік = вкладка монети; далі КОЖНІ увімкнені ворота «Пружини» мусять
    пропустити сигнал саме в цей бік: потенціал ≥ порогу (`spring_gate_enabled`)
    і заряд ≥ порогу (`spring_charge_gate_enabled`). `potential(rec, side)` —
    `spring_coil.potential` (враховує вистріл); без нього — сирий pot_*."""
    s = settings or {}
    side = tab_side(rec)
    if side is None:
        return False, None, 'без напрямку'
    if s.get('spring_gate_enabled', True):
        need = _f(s.get('spring_min_pot', 40))
        pot = potential(rec, side) if potential else None
        if pot is None:
            pot = _f(rec.get('pot_long' if side == 'LONG' else 'pot_short'))
        if _f(pot) < need:
            return False, side, f'потенціал {side} {_f(pot):.0f} < {need:.0f}'
    if s.get('spring_charge_gate_enabled', False):
        need = _f(s.get('spring_min_charge', 50))
        if _f(rec.get('charge')) < need:
            return False, side, f'заряд {_f(rec.get("charge")):.0f} < {need:.0f}'
    return True, side, ''


STATE_UA = {'building': '⏳ заряджається', 'flat': '😴 без заряду',
            'fired': '💥 вистрілила'}


def scan_member(row: Optional[Dict], wl_rec: Optional[Dict], min_charge) -> Tuple[Optional[Dict], str]:
    """Запис, за яким судимо монету в 🔭 таблиці скану → (запис, джерело).

    Монети WATCHLIST у таблицю скану НЕ потрапляють (вони вже в 🌀 Пружині),
    тож для них «в таблиці скану» = пройшла б її відбір: 🌀 заряджена і заряд
    ≥ фільтра скану — за ТИМ САМИМ записом бота (ті самі закриті 1h-бари)."""
    if isinstance(row, dict):
        return row, 'скан'
    if isinstance(wl_rec, dict):
        st, ch = wl_rec.get('state'), _f(wl_rec.get('charge'))
        if st == 'charged' and ch >= _f(min_charge):
            return wl_rec, 'WATCHLIST'
        # ⚠️ Справжня причина, а не голе «немає в таблиці»: монета WATCHLIST
        # у таблицю скану не потрапляє ніколи, тож судимо за відбором скану.
        if st != 'charged':
            return None, f'WATCHLIST: стан {STATE_UA.get(st, st or "—")}, а не 🌀 заряджена'
        return None, f'WATCHLIST: заряд {ch:.0f} < {_f(min_charge):.0f} (фільтр скану)'
    return None, ''


def book_decide(tag: str, rec: Optional[Dict], banner: Optional[str], side: str,
                need_banner: bool = False, src: str = '') -> Tuple[bool, bool, str, str]:
    """(applies, ok, chip, reason) воріт однієї «Пружини»."""
    side = str(side or '').upper()
    name = NAME.get(tag, tag)
    if side not in SIDES:
        return (False, True, '', '')
    cs = tab_side(rec)
    bn = f' · банер {banner or "⚖"}' if need_banner else ''
    where = f' · {src}' if src and src != 'скан' else ''
    if rec is None:
        if src and src != 'скан':
            return (True, False, f'{tag}[{src}{bn}]:✗',
                    f'{name}: {src} — у таблицю скану не пройшла б (сигнал {side} відхилено)')
        return (True, False, f'{tag}[немає в таблиці{bn}]:✗',
                f'{name}: монети немає в таблиці (сигнал {side} відхилено)')
    if cs != side:
        return (True, False, f'{tag}[вкладка {cs or "⚖"}{bn}{where}]:✗',
                f'{name}: монета у вкладці {cs or "⚖ без напрямку"}, а сигнал {side}')
    if need_banner and banner != side:
        return (True, False, f'{tag}[вкладка {cs}{bn}{where}]:✗',
                f'{name}: банер «Куди накручено» {banner or "⚖"}, а сигнал {side}')
    return (True, True, f'{tag}[вкладка {cs}{bn}{where}]:✓', '')


def banner_decide(tag: str, banner: Optional[str], side: str) -> Tuple[bool, bool, str, str]:
    """⚖ Ворота за банером «SMC Forecast» (вимога 11.11): банер LONG → лише
    LONG-сигнали, SHORT → лише SHORT; ⚖ рівновага / замало заряджених / немає
    даних → відмова («невідомо» ≠ «за напрямком»)."""
    side = str(side or '').upper()
    if side not in SIDES:
        return (False, True, '', '')
    if banner == side:
        return (True, True, f'{tag}[{banner}]:✓', '')
    return (True, False, f'{tag}[{banner or "⚖"}]:✗',
            f'⚖ SMC Forecast {tag.lstrip("⚖")}: банер {banner or "⚖ без напрямку"}, '
            f'а сигнал {side} (відхилено)')
