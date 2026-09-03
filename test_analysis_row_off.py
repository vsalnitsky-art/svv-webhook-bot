"""🚫 РЯДОК АНАЛІЗУ ВИМКНЕНО ДЛЯ ПОКАЗУ — але АЛГОРИТМИ ЖИВІ.

Вимога користувача дослівно: «Щоб не навантажувати сторінку, вимкни повністю
все, що зображено на скріні. Не потрібно робити розрахунок для цих банерів.
Розрахунок і алгоритм залиш, вимкни лише для сторінки відображення.»

На скріні три банери, які стоять в одному рядку `.sm-analysis-row`:
  • 🏦 настрій біржі Binance (L/S %)        → `/api/sentiment`
  • 💧 драбина ліквідності BTC              → `/api/liquidity-ladder`
  • 📈/📉 Потенціал LONG/SHORT + виснаженість (панелі `sm-move-panel-*`)

⚠️ ДВІ ПОЛОВИНИ ВИМОГИ, і тест стереже ОБИДВІ:
  1. **Показ вимкнено** — розмітка прихована, і сторінка більше НЕ робить двох
     запитів (це знімає і мережу, і серверний розрахунок за ними).
  2. **Алгоритм НЕ видалено** — ендпоінти на місці, `analyze_move_potential`
     і далі рахується в `compute_bias`, а `fuel_filter._exhaustion` і далі його
     читає. Це не косметика: «Виснаженість» живить ворота відкриття, тож
     вирізати розрахунок разом із банером означало б тихо змінити ТОРГІВЛЮ.

⚠️ Саме тому «Потенціал» економить ЛИШЕ малювання, а не розрахунок: його числа
приходять усередині `/api/smc/bias`, який сторінці потрібен для вердикту, і той
самий результат споживає торгова логіка.
"""
import os
import re

_ROOT = os.path.dirname(os.path.abspath(__file__))
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html')).read()
_FLASK = open(os.path.join(_ROOT, 'web', 'flask_app.py')).read()
_FF = open(os.path.join(_ROOT, 'detection', 'fuel_filter.py')).read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _body(name, span=900):
    """Тіло JS-функції від її оголошення (грубо, але для замка достатньо)."""
    i = _HTML.index(f'function {name}(')
    return _HTML[i:i + span]


# ═══════════ 1. ПОКАЗ ВИМКНЕНО ═════════════════════════════════════════════
def test_single_switch_exists_and_is_off():
    """Вимикач мусить бути ОДИН і явний — інакше «повернути як було» стане
    археологією по трьох функціях і розмітці."""
    _check('const SM_ANALYSIS_ROW_ON = false;' in _HTML,
           'бракує єдиної константи-вимикача (або вона не false)')
    _check(_HTML.count('SM_ANALYSIS_ROW_ON') >= 5,
           'константа мусить керувати розміткою і всіма трьома функціями')
    print('✓ один вимикач `SM_ANALYSIS_ROW_ON = false`')


def test_page_makes_no_request_for_the_two_banners():
    """ГОЛОВНЕ ПО НАВАНТАЖЕННЮ. Обидва запити мусять відпадати ДО `fetch`,
    інакше сервер усе одно рахує, а ми лише ховаємо результат."""
    for fn, url in (('loadSentiment', '/api/sentiment'),
                    ('loadLiquidityLadder', '/api/liquidity-ladder')):
        b = _body(fn)
        _check('if (!SM_ANALYSIS_ROW_ON) return;' in b,
               f'{fn} мусить виходити за константою')
        # Вихід стоїть ПЕРЕД запитом — саме це економить сервер.
        guard = b.index('if (!SM_ANALYSIS_ROW_ON) return;')
        call = b.find('fetch(')
        if fn == 'loadSentiment':
            call = b.find('loadSentimentFor(')   # запит робить вона
        _check(call > guard,
               f'{fn}: вихід мусить стояти ДО запиту {url}, а не після')
    print('✓ /api/sentiment і /api/liquidity-ladder сторінкою не смикаються')


def test_move_panels_are_not_rendered():
    b = _body('renderMovePanel', 500)
    _check('if (!SM_ANALYSIS_ROW_ON) return;' in b,
           'renderMovePanel мусить виходити за константою')
    _check(b.index('if (!SM_ANALYSIS_ROW_ON) return;') < b.index('renderMoveSide('),
           'вихід мусить стояти ДО малювання панелей')
    print('✓ панелі «Потенціал» не малюються')


def test_markup_is_hidden_but_intact():
    """Розмітку НЕ видаляємо: повернення має коштувати одну константу.
    Ховаємо саме ОБГОРТКУ — порожній flex-рядок лишав би відступ і gap."""
    m = re.search(r'<div class="sm-analysis-row"[^>]*>', _HTML)
    _check(m, 'рядок аналізу зник із розмітки — його треба ЛИШИТИ прихованим')
    tag = m.group(0)
    _check('id="sm-analysis-row"' in tag, f'потрібен id для повернення: {tag}')
    _check('display:none' in tag, f'обгортка мусить бути прихована: {tag}')
    # Усі id, з якими працюють рендери, мають лишитись на місці.
    for _id in ('sm-sentiment-val', 'sm-sentiment-long', 'sm-liq', 'sm-liq-rows',
                'sm-move-panel-long', 'sm-move-panel-short'):
        _check(f'id="{_id}"' in _HTML, f'зник id «{_id}» — повернути буде нічим')
    print('✓ розмітка на місці, прихована обгорткою')


def test_switching_back_needs_only_the_constant():
    """Якщо константу поставити в true, рядок мусить ЗʼЯВИТИСЬ — інакше
    «повернути» означало б ще й правити інлайновий стиль у HTML."""
    _check("_arow.style.display = 'flex'" in _HTML,
           'при увімкненій константі обгортку треба показати назад')
    i = _HTML.index("_arow.style.display = 'flex'")
    _check('if (SM_ANALYSIS_ROW_ON)' in _HTML[i - 300:i],
           'показ назад мусить бути прив\'язаний до тієї самої константи')
    print('✓ повернення = одна константа, без правок розмітки')


# ═══════════ 2. АЛГОРИТМ ЖИВИЙ (друга половина вимоги) ═════════════════════
def test_endpoints_are_not_removed():
    """«Розрахунок і алгоритм залиш» — ендпоінти мусять лишитись робочими
    (ними користуються інші сторінки/інструменти й ручна перевірка)."""
    for route in ("@app.route('/api/sentiment')",
                  "@app.route('/api/liquidity-ladder')"):
        _check(route in _FLASK, f'ендпоінт видалено, а мав лишитись: {route}')
    print('✓ обидва ендпоінти на місці — вимкнено лише виклик зі сторінки')


def test_move_potential_is_still_computed_for_the_bot():
    """⚠️ НАЙВАЖЛИВІШЕ. `move_long`/`move_short` — це не тільки банер: із них
    `fuel_filter._exhaustion` бере «Виснаженість», яка бере участь у воротах
    відкриття. Прибрати розрахунок разом із банером = тихо змінити торгівлю."""
    _check('analyze_move_potential' in _FLASK,
           'розрахунок потенціалу мусить лишитись у бекенді')
    _check("'move_long': move_long" in _FLASK,
           'payload мусить і далі нести move_long/move_short')
    _check('def _exhaustion' in _FF and "get('move_long')" in _FF,
           'fuel_filter мусить і далі читати ці числа')
    print('✓ розрахунок потенціалу живий і далі живить ворота відкриття')


def test_liquidity_and_sentiment_modules_untouched():
    """Драбина й настрій — окремі модулі; вимикали ПОКАЗ, не їх."""
    _check(os.path.exists(os.path.join(_ROOT, 'detection', 'liquidation_map',
                                       'ladder.py')),
           'модуль драбини ліквідності мусить лишитись')
    _check('build_ladder' in _FLASK, 'ендпоінт драбини мусить і далі її будувати')
    print('✓ модулі розрахунку не чіпали')


if __name__ == '__main__':
    test_single_switch_exists_and_is_off()
    test_page_makes_no_request_for_the_two_banners()
    test_move_panels_are_not_rendered()
    test_markup_is_hidden_but_intact()
    test_switching_back_needs_only_the_constant()
    test_endpoints_are_not_removed()
    test_move_potential_is_still_computed_for_the_bot()
    test_liquidity_and_sentiment_modules_untouched()
    print('\nУсі тести «рядок аналізу вимкнено для показу» пройдено ✅')
