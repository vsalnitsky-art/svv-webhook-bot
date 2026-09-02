"""📐 «1H OB ЛИШЕ З CHoCH» — окремий тумблер воріт входу.

Запит користувача: «Додай окремий тумблер "1H OB лише з CHoCH" до
налаштувань, за замовчуванням увімкнено».

⚠️ ЩО САМЕ ФІКСУЄМО. Детектор створює Order Block на ОБИДВІ події структури
(`storeOrdeBlock` фаєриться і на BOS, і на CHoCH — точна Pine-семантика), і
тег осідає в `created_by_tag`:
  • CHoCH-блок — точка РОЗВОРОТУ (перший блок нової структури);
  • BOS-блок   — ПРОДОВЖЕННЯ (тренд уже йде і пробиває черговий свінг).
Тумблер вирішує, чи приймають ворота входу BOS-блок. Дефолт УВІМК = лише
CHoCH. Перевіряється ПОТОЧНИЙ (останній немітигований) блок.

⚠️ Тумблер — це УТОЧНЕННЯ воріт `ob_filter_enabled`, а не окремий фільтр:
при вимкненому OB-фільтрі він не робить нічого. На малюнок блоку, Manual SL
і такт `vob_one_per_ob` НЕ впливає — вони беруть той самий рядок БД як був.
"""
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "smc_scanner_choch_test", os.path.join(_HERE, "detection", "smc_scanner.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
S = _m.SMCScanner


def _check(c, msg):
    if not c:
        raise AssertionError(msg)


def _fake_db(row):
    """Підміняємо `storage.db_operations.get_db` рядком стану OB."""
    st = sys.modules.get('storage') or types.ModuleType('storage')
    st.__path__ = getattr(st, '__path__', [])
    sys.modules['storage'] = st
    dbo = types.ModuleType('storage.db_operations')
    dbo.get_db = lambda: types.SimpleNamespace(
        get_smc_ob_state=lambda sym, tf: row)
    sys.modules['storage.db_operations'] = dbo
    st.db_operations = dbo


def _ns(choch_only=True, tf='1h'):
    ns = types.SimpleNamespace()
    ns._settings = {'ob_filter_timeframe': tf,
                    'ob_filter_choch_only': choch_only}
    ns._ob_filter_allows = S._ob_filter_allows.__get__(ns)
    ns._ob_state_label = S._ob_state_label.__get__(ns)
    return ns


def _row(bias='BEARISH', tag='CHoCH'):
    return {'bias': bias, 'created_by_tag': tag, 'bar_high': 10.0,
            'bar_low': 9.0, 'bar_time': 1_700_000_000_000}


# ── дефолт ────────────────────────────────────────────────────────────────
def test_default_is_on():
    """Дефолт УВІМК — пряма вимога користувача. І це НОВИЙ ключ, тож
    `merged.update(stored)` у `_load_settings` віддасть саме дефолт: міграція
    (як для `pilot_autofill_tp`) тут не потрібна, бо в збереженому блобі
    ключа ще немає."""
    _check(_m.DEFAULT_SETTINGS.get('ob_filter_choch_only') is True,
           f"дефолт мав бути True: {_m.DEFAULT_SETTINGS.get('ob_filter_choch_only')}")
    print('✓ дефолт УВІМК (новий ключ — міграція не потрібна)')


def test_key_is_whitelisted_and_coerced():
    """`update_settings` бере ЛИШЕ ключі з білого списку — без цього UI
    посилав би тумблер даремно, а він мовчки не зберігався б."""
    src = open(os.path.join(_HERE, 'detection', 'smc_scanner.py')).read()
    _check("'ob_filter_choch_only'," in src, 'ключа немає у білому списку')
    _check("self._settings['ob_filter_choch_only'] = bool(" in src,
           'значення мусить нормалізуватись до bool (UI може прислати рядок)')
    print('✓ ключ у білому списку update_settings і зводиться до bool')


# ── головний замок ────────────────────────────────────────────────────────
def test_bos_block_is_rejected_when_on_and_accepted_when_off():
    """ГОЛОВНИЙ ЗАМОК: той самий блок правильного напрямку, створений BOS,
    мусить блокуватись при УВІМК і проходити при ВИМК."""
    _fake_db(_row('BEARISH', 'BOS'))
    _check(_ns(choch_only=True)._ob_filter_allows('X', 'SHORT') is False,
           'BOS-блок мусив бути заблокований при увімкненому тумблері')
    _check(_ns(choch_only=False)._ob_filter_allows('X', 'SHORT') is True,
           'при ВИМК тумблері BOS-блок — валідне джерело (стара поведінка)')
    print('✓ BOS-блок: блок при УВІМК, пропуск при ВИМК')


def test_choch_block_passes_in_both_modes():
    """CHoCH-блок правильного напрямку проходить завжди — тумблер його не
    стосується, він лише відсікає BOS."""
    _fake_db(_row('BULLISH', 'CHoCH'))
    for on in (True, False):
        _check(_ns(choch_only=on)._ob_filter_allows('X', 'LONG') is True,
               f'CHoCH-блок мав пройти (choch_only={on})')
    print('✓ CHoCH-блок проходить в обох режимах')


def test_direction_still_wins_over_the_tag():
    """Напрямок перевіряється ПЕРШИМ: CHoCH-блок протилежного боку — все одно
    блок. Тумблер лише ЗВУЖУЄ ворота, він нічого не дозволяє понад те, що
    дозволяв напрямок."""
    _fake_db(_row('BULLISH', 'CHoCH'))
    for on in (True, False):
        _check(_ns(choch_only=on)._ob_filter_allows('X', 'SHORT') is False,
               f'бичачий блок не може пускати SHORT (choch_only={on})')
    print('✓ напрямок лишається головним; тумблер лише звужує')


def test_missing_row_or_bias_blocks_in_both_modes():
    """Жорстка семантика «немає блоку = блок» не змінилась."""
    for row in (None, {'bias': None, 'created_by_tag': 'CHoCH'}):
        _fake_db(row)
        for on in (True, False):
            _check(_ns(choch_only=on)._ob_filter_allows('X', 'LONG') is False,
                   f'немає валідного блоку → блок (row={row}, on={on})')
    print('✓ немає блоку — блок в обох режимах (семантика не змінилась)')


def test_unknown_tag_blocks_when_on():
    """⚠️ Рядок БЕЗ тега (стара БД-строка, ще не перерахована) при УВІМК
    блокується: «будь-яка невизначеність = блок» — задекларована політика цих
    воріт. Скан перераховує рядок на кожному тіку, тож це триває один цикл."""
    for tag in (None, '', '?'):
        _fake_db(_row('BEARISH', tag))
        _check(_ns(choch_only=True)._ob_filter_allows('X', 'SHORT') is False,
               f'невідомий тег ({tag!r}) мусить блокувати при УВІМК')
        _check(_ns(choch_only=False)._ob_filter_allows('X', 'SHORT') is True,
               f'при ВИМК тег не має значення ({tag!r})')
    print('✓ блок без тега: блокує при УВІМК, не заважає при ВИМК')


# ── прозорість у 🧾 Лозі ──────────────────────────────────────────────────
def test_state_label_explains_the_decision():
    """Голе «OB(1h):✗» нічого не пояснює. Розклад мусить нести стан блоку."""
    _fake_db(_row('BEARISH', 'BOS'))
    _check(_ns()._ob_state_label('X') == 'BEARISH/BOS', _ns()._ob_state_label('X'))
    _fake_db(_row('BULLISH', 'CHoCH'))
    _check(_ns()._ob_state_label('X') == 'BULLISH/CHoCH', _ns()._ob_state_label('X'))
    _fake_db(None)
    _check(_ns()._ob_state_label('X') == 'не рахувався', _ns()._ob_state_label('X'))
    _fake_db({'bias': None})
    _check(_ns()._ob_state_label('X') == 'нема блоку', _ns()._ob_state_label('X'))
    print('✓ стан блоку читається словами (BEARISH/BOS)')


def _gate_ns(row, choch_only=True):
    """Повний `self` для `_signal_allowed` з увімкненим ЛИШЕ OB-фільтром."""
    _fake_db(row)
    ns = types.SimpleNamespace()
    ns._settings = {'ob_filter_enabled': True, 'ob_filter_timeframe': '1h',
                    'ob_filter_choch_only': choch_only,
                    'use_pd_zone_filter': False}
    ns._ob_filter_allows = S._ob_filter_allows.__get__(ns)
    ns._ob_state_label = S._ob_state_label.__get__(ns)
    ns._forecast_pair = lambda sym: ('—', '—')
    ns.get_pd_pct = lambda sym: None
    ns._decision_gate = lambda sym, side, at_intake=False: (True, '')
    return ns


def test_reason_tells_bos_apart_from_wrong_direction():
    """ТРИ РІЗНІ відмови не можна зливати в одну: «немає блоку»,
    «проти напрямку» і «створено BOS». Інакше в 🧾 Лозі не видно, що саме
    зробив новий тумблер."""
    gate = S._signal_allowed

    ok, reason, detail = gate(_gate_ns(_row('BEARISH', 'BOS')), 'X', 'SHORT')
    _check(ok is False, 'BOS-блок мав заблокувати')
    _check('BOS' in reason and 'CHoCH' in reason, f'причина: {reason}')
    _check('BEARISH/BOS' in detail and 'лише CHoCH' in detail, f'розклад: {detail}')

    ok2, reason2, _ = gate(_gate_ns(_row('BULLISH', 'CHoCH')), 'X', 'SHORT')
    _check(ok2 is False and 'проти напрямку' in reason2, f'причина: {reason2}')

    ok3, reason3, _ = gate(_gate_ns(None), 'X', 'SHORT')
    _check(ok3 is False and 'не рахувався' in reason3, f'причина: {reason3}')

    ok4, _, detail4 = gate(_gate_ns(_row('BEARISH', 'CHoCH')), 'X', 'SHORT')
    _check(ok4 is True, 'CHoCH-блок мав пройти ворота')
    _check('BEARISH/CHoCH' in detail4, f'розклад: {detail4}')
    print('✓ у лозі видно ЯКА саме відмова і з якими значеннями')


def test_toggle_does_nothing_while_ob_filter_is_off():
    """Тумблер — уточнення воріт `ob_filter_enabled`, а не окремий фільтр.
    OB-фільтр вимкнено → BOS-блок нікого не блокує."""
    ns = _gate_ns(_row('BEARISH', 'BOS'))
    ns._settings['ob_filter_enabled'] = False
    ok, reason, detail = S._signal_allowed(ns, 'X', 'SHORT')
    _check(ok is True and reason == '', f'{reason} / {detail}')
    _check('OB(' not in detail, f'вимкнений фільтр не має бути в розкладі: {detail}')
    print('✓ при вимкненому OB-фільтрі тумблер не діє')


# ── UI ────────────────────────────────────────────────────────────────────
def test_ui_wires_the_toggle_end_to_end():
    """Замок від «поле є, а не працює»: чекбокс, обробник, читання стану і
    ключ у POST мусять існувати РАЗОМ."""
    html = open(os.path.join(_HERE, 'templates', 'smart_money.html')).read()
    for needle in ('id="sm-ob-choch-only"',
                   'onchange="updateObChochOnly()"',
                   'async function updateObChochOnly()',
                   'ob_filter_choch_only: enabled',
                   's.ob_filter_choch_only !== false'):
        _check(needle in html, f'бракує у smart_money.html: {needle}')
    # Чекбокс у розмітці мусить бути `checked` — інакше до першого
    # завантаження стану користувач бачив би ВИМК, а бот працював би УВІМК.
    i = html.index('id="sm-ob-choch-only"')
    _check('checked' in html[i:i + 120], 'чекбокс має бути checked у розмітці')
    print('✓ UI: чекбокс + обробник + читання стану на місці')


if __name__ == '__main__':
    test_default_is_on()
    test_key_is_whitelisted_and_coerced()
    test_bos_block_is_rejected_when_on_and_accepted_when_off()
    test_choch_block_passes_in_both_modes()
    test_direction_still_wins_over_the_tag()
    test_missing_row_or_bias_blocks_in_both_modes()
    test_unknown_tag_blocks_when_on()
    test_state_label_explains_the_decision()
    test_reason_tells_bos_apart_from_wrong_direction()
    test_toggle_does_nothing_while_ob_filter_is_off()
    test_ui_wires_the_toggle_end_to_end()
    print('\nУсі тести «1H OB лише з CHoCH» пройдено ✅')
