"""🛡 ПРАВА КОРИСТУВАЧА НА ІНФОСАЙТІ (вимога 24.11).

Дослівно: «В боті "🛡 Керування користувачами" мають бути по кожному
користувачу всі варіанти що можна робити а що не можна, що відображати а не
відображати. У адміністратора бота має бути доступ до всього, а користувачам
все налаштовує адміністратор з бота» + «Мається на увазі стосовно інфосайту,
які дані показувати користувачу а які ні».

• Зберігається в `prefs['perms']` ({ключ: bool}); пише ЛИШЕ адміністратор
  (`POST /api/admin/users/<id>` {action: 'set_perms'}). Сам користувач цей ключ
  змінити не може (`/api/me/prefs` його не чіпає).
• `effective(is_admin, raw)` — ЄДИНЕ правило: адміністратор → усе дозволено;
  користувач → збережене значення, інакше дефолт ключа; ключ із залежністю
  (`needs`) вимкнений, коли вимкнено його «батька» (шари графіка, сповіщення
  й торгівля живуть у Терміналі — без Термінала їх немає).
• Сервер ПЕРЕВІРЯЄ кожне право сам (маршрути `/api/info/*`, `coin_payload`
  без заборонених шарів, підписки Telegram, торгівля) — приховування на
  сторінці лише прибирає порожні кнопки.
"""

# (ключ, група, підпис, дефолт для користувача, залежність, підказка)
PERMS = (
    ('sec_terminal', 'sections', 'Термінал: WATCHLIST + графік', True, None,
     'Список монет користувача і графік з усіма шарами нижче.'),
    ('sec_signals', 'sections', 'VSV SIGNALS + банер VSV Forecast', True, None,
     'Таблиця сигналів ШІ (точки входу) і банер «куди накручено пружину».'),
    ('sec_paper', 'sections', 'P&L тестових угод бота', True, None,
     'Віртуальний рахунок, статистика й таблиці тестових (paper) угод бота.'),

    ('ly_struct', 'layers', 'CHoCH / BOS', True, 'sec_terminal', 'Події структури на графіку.'),
    ('ly_ob', 'layers', 'OB — Order Blocks', True, 'sec_terminal', ''),
    ('ly_vob', 'layers', 'VOB — Volumized OB', True, 'sec_terminal', ''),
    ('ly_imb', 'layers', 'IMB — імбаланси', True, 'sec_terminal', ''),
    ('ly_sbr', 'layers', 'SBR — сигнали S/B на графіку', True, 'sec_terminal',
     'Позначки S/B і бейджі останнього сигналу 1H/4H.'),
    ('ly_anom', 'layers', '⚠ Аномалії обсягу — сигнали входу на графіку', True, 'sec_terminal',
     'Позначки B/S там, де склався сценарій входу (той самий сигнал, що в Telegram).'),
    ('ly_hl', 'layers', 'HH / HL / LH / LL', True, 'sec_terminal', ''),
    ('ly_poc', 'layers', 'POC', True, 'sec_terminal', ''),
    ('ly_shl', 'layers', 'High / Low (Strong/Weak)', True, 'sec_terminal', ''),
    ('ly_pd', 'layers', 'Premium / Discount', True, 'sec_terminal', ''),
    ('ly_vol', 'layers', 'Обсяг: покупці / продавці + аномалії', True, 'sec_terminal', ''),
    ('ly_liq', 'layers', '💧 Ліквідність (профіль)', True, 'sec_terminal',
     'Профіль ліквідності за кнопкою (ходить на біржу).'),

    ('ntf_sbr', 'notify', '🔔 SBR-сигнали в Telegram', True, 'sec_terminal',
     'Особисті підписки на сигнали SBR по монетах WATCHLIST.'),
    ('ntf_anom', 'notify', '⚠ Аномалії обсягу в Telegram', True, 'sec_terminal',
     'Особисті сповіщення «сценарій входу».'),

    ('trade_paper', 'trade', '🧪 Торгівля за сигналом ⚠ — paper', False, 'sec_terminal',
     'Угоди йдуть у тестову книгу бота.'),
    ('trade_real', 'trade', '🏦 Торгівля за сигналом ⚠ — РЕАЛЬНА', False, 'sec_terminal',
     '⚠️ Реальні угоди відкриває Trade Manager БОТА — з ключами і депозитом бота.'),
)

GROUPS = (
    ('sections', '📺 Розділи інфосайту'),
    ('layers', '📈 Шари графіка'),
    ('notify', '📨 Сповіщення в Telegram'),
    ('trade', '🤖 Торгівля з кабінету'),
)

KEYS = tuple(p[0] for p in PERMS)
_DEF = {p[0]: p[3] for p in PERMS}
_NEED = {p[0]: p[4] for p in PERMS}

# які поля відповіді /api/info/coin належать шару (порожнє значення — без шару)
COIN_FIELDS = {
    'ly_struct': {'events': []},
    'ly_ob': {'ob': [], 'ob_wait': False},
    'ly_vob': {'vob': [], 'vob_wait': False},
    'ly_imb': {'imb': []},
    'ly_sbr': {'sbr': [], 'sbr_wait': False, 'sbr_last': {}},
    'ly_hl': {'hl': [], 'hl_wait': False},
    'ly_anom': {'anom_sig': []},
    'ly_poc': {'poc': None, 'poc_state': 'off'},
    'ly_shl': {'shl': None},
    'ly_pd': {'pd': None},
    'ly_vol': {'vsplit': None, 'vol_wait': False},
}


def norm(raw):
    """ЧИСТА: лише відомі ключі, значення bool (решту відкидаємо)."""
    raw = raw if isinstance(raw, dict) else {}
    return {k: bool(raw[k]) for k in KEYS if k in raw}


def effective(is_admin, raw):
    """ЧИСТА: {ключ: bool} — що користувачу РЕАЛЬНО дозволено."""
    if is_admin:
        return {k: True for k in KEYS}
    r = norm(raw)
    out = {k: r.get(k, _DEF[k]) for k in KEYS}
    for k in KEYS:
        n = _NEED.get(k)
        if n and not out.get(n):
            out[k] = False
    return out


def of_user(u):
    """Права живого користувача (обʼєкт User). Немає користувача → нічого."""
    if u is None:
        return {k: False for k in KEYS}
    try:
        import json
        prefs = json.loads(getattr(u, 'prefs', None) or '{}')
    except Exception:
        prefs = {}
    return effective(bool(getattr(u, 'is_admin', False)), prefs.get('perms'))


def apply(cur, body):
    """ЧИСТА: зміна прав з тіла адмін-запиту {perms: {ключ: bool}} або
    {key, on}; повертає нормалізований повний набір збережених значень."""
    out = {k: _DEF[k] for k in KEYS}
    out.update(norm(cur))
    if isinstance(body.get('perms'), dict):
        out.update(norm(body['perms']))
    k = body.get('key')
    if k in KEYS and 'on' in body:
        out[k] = bool(body.get('on'))
    return out


def books(perms):
    """ЧИСТА: книги торгівлі з кабінету, дозволені правами."""
    return tuple(b for b, k in (('paper', 'trade_paper'), ('real', 'trade_real'))
                 if (perms or {}).get(k))


def strip_coin(payload, perms):
    """ЧИСТА: копія відповіді графіка без заборонених шарів (кеш не чіпаємо)."""
    if not isinstance(payload, dict):
        return payload
    off = [k for k in COIN_FIELDS if not (perms or {}).get(k)]
    if not off:
        return payload
    out = dict(payload)
    for k in off:
        for f, v in COIN_FIELDS[k].items():
            if f in out:
                out[f] = v.copy() if isinstance(v, (list, dict)) else v
    out['perm_off'] = off
    return out


def sig(perms):
    """ЧИСТА: короткий підпис набору прав (у сигнатуру кешу сторінки)."""
    return ''.join('1' if (perms or {}).get(k) else '0' for k in KEYS)


def defs():
    """Опис прав для сторінок (адмін-панель, інфосайт)."""
    return {'groups': [{'key': g, 'label': l} for g, l in GROUPS],
            'perms': [{'key': k, 'group': g, 'label': l, 'default': d, 'needs': n, 'hint': h}
                      for k, g, l, d, n, h in PERMS]}
