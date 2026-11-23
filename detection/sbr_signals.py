"""
🔔 Session Band Reversal (SBR) — рушій сигналів 1H + 4H (вимога 29.10).

Дослівно: «Візьми скрипт для TradingView "Session Band Reversal Signals" на
100% адаптуй його для роботи бота … Сигнал має відслідковуватись як на 1Н так
і на 4Н таймфреймах. Відтвори сигнали один в один … як у TradingView.»

ДЖЕРЕЛО ПРАВДИ — `tools/tradingview_session_band_signals.pine` (v4). Тут той
самий розрахунок, бар у бар:

    mid   = ta.ema(hlc3, 5)        atr = ta.atr(14)   (RMA від true range)
    upper = mid + 0.5·atr          lower = mid − 0.5·atr
    upP, loP, atrP = рівні ПОПЕРЕДНЬОГО закритого бару
    shortScore = clamp(47.8 + 41.2·(high − upP)/atrP)
    longScore  = clamp(47.8 + 41.2·(loP − low)/atrP)
    SHORT: high ≥ upP · score ≥ 55 · червона свічка · close > upP
    LONG : low  ≤ loP · score ≥ 55 · зелена свічка  · close < loP
    Trigger = high/low бару · SL = trigger ± 3·atrP · Risk за Score
    «#N» — нумерація в межах доби (Europe/Kyiv), окремо SHORT і LONG.

⚠️ ТОЧНІСТЬ «ОДИН В ОДИН». Pine-функції рекурсивні (EMA/RMA залежать від
усієї історії). Вплив стартової точки затухає як (13/14)^N для ATR і (2/3)^N
для EMA — на 1000 барах це ~1e-32, тобто числа збігаються з TradingView до
останнього знака, ЯКЩО свічки ті самі. Тому бари — Bybit (`fetch_klines`
ходить спершу в Bybit), рівно ті, що `BYBIT:<SYM>.P` на графіку TV.
⚠️ Сигнал рахується ЛИШЕ на ЗАКРИТОМУ барі (у Pine `onClose` = true за
замовчуванням: колір свічки остаточний тільки на закритті). Живий бар
відкидається — інакше мітка зʼявлялась би і зникала всередині години.

ШВИДКІСТЬ. Рушій не чекає загального скану: окремий потік щоразу через
`SETTLE_SEC` після закриття бару 1H (і 4H на межах 00/04/08… UTC) паралельно
(`WORKERS`) тягне `BARS` свічок КОЖНОЇ монети й одразу шле Telegram. Монета,
по якій біржа ще не віддала закритий бар, переопитується кожні `RETRY_SEC`.

ЛИШЕ ЧИСТІ ФУНКЦІЇ ВГОРІ (юніт-тести), I/O — у класі `SbrEngine`.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
    _KYIV = ZoneInfo('Europe/Kyiv')
except Exception:          # pragma: no cover — без tzdata
    _KYIV = None

# ── Параметри індикатора — ДЕФОЛТИ Pine-скрипта (не міняти без нього!) ──────
DEFAULTS = {
    'mid_len': 5, 'atr_len': 14, 'band_k': 0.5,
    'sc_base': 47.8, 'sc_slope': 41.2, 'min_score': 55.0,
    # SL = close ± 3.7·ATR14 ПОТОЧНОГО бару (калібровано 04.11 за 12 мітками
    # еталона BTC/ETH 4H: медіана 3.71, розкид 3.64-3.83).
    'sl_k': 3.7, 'th_add': 90.0, 'th_1r': 70.0, 'th_half': 60.0,
    'rev_candle': True, 'close_out': True,
}

TFS = ('1h', '4h')
# 📶 УСІ TF сигналів SBR (вимога 17.11): будильник, група Telegram і особисті
# підписки кабінету обирають будь-який набір із ALL_TFS.
ALL_TFS = ('15m', '30m', '45m', '1h', '4h', '1d')
# ⏰ ДОДАТКОВІ TF (усе, крім 1H/4H) рушій рахує ЛИШЕ ЗА ПОТРЕБОЮ: для монети
# будильника, для монет групи TG і особистих підписок, де цей TF обрано.
# Книга 🔔 МММ-SBR, правило виходу і 🔭 скан і далі живуть на 1H/4H.
EXTRA_TFS = ('15m', '30m', '45m', '1d')
ALARM_ONLY_TFS = EXTRA_TFS          # історична назва (16.11: лише будильник)
TF_SEC = {'15m': 900, '30m': 1800, '45m': 2700, '1h': 3600, '4h': 14400, '1d': 86400}
TF_LABEL = {'15m': '15M', '30m': '30M', '45m': '45M', '1h': '1H', '4h': '4H', '1d': '1D'}
# 45m біржа не віддає → збираємо з 15m (межі 45m від 00:00 UTC, як TradingView:
# 1440 / 45 = 32 бари на добу, тож епоха й доба вирівняні однаково).
AGG_FROM = {'45m': ('15m', 3)}
ALARM_BARS = 1000      # 15m / 30m — одна сторінка Bybit; 45m — 3000 барів 15m
BARS = 1000            # «без економії»: 1 сторінка Bybit = 1000 барів на TF
SETTLE_SEC = 2.0       # пауза після закриття бару — біржа встигає його закрити
RETRY_SEC = 5.0        # повтор для монет, по яких закритого бару ще немає
FAIL_BACKOFF_MAX = 300.0
WORKERS = 8
LOOP_SEC = 1.0
TAIL = 12              # скільки останніх сигналів TF тримаємо на монету
SEND_MAX_LAG_FRAC = 0.25   # сигнал старший за ¼ бару після закриття — тиха база
DB_KEY_STATE = 'sbr_state_v1'
DB_KEY_SETTINGS = 'sbr_settings'
DEFAULT_TG_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TG_CATEGORY = 'sbr'

DEFAULT_SETTINGS = {
    'enabled': True,          # рушій (≈ 2 запити на монету на годину)
    'tg_enabled': True,       # майстер-вимикач 📨 у тему «🔔 Сигнали»
    'tg_symbols': list(DEFAULT_TG_SYMBOLS),
    # 📶 TF сигналів у Telegram (вимога 29.10): 1H / 4H / обидва.
    # Обидва вимкнені = обидва (інакше тумблер мовчки гасив би всю тему).
    'tg_1h': True,
    'tg_4h': True,
    # 📶 Будь-який набір із ALL_TFS (вимога 17.11). None → з tg_1h / tg_4h.
    'tg_tfs': None,
    # ⏰ БУДИЛЬНИК (вимога 01.11): свіжий сигнал SBR обраної монети / TF / боку
    # вмикає відповідну ГОЛОВНУ кнопку напрямку (Trade Direction).
    # `alarm_on` = ГОЛОВНИЙ ТУМБЛЕР будильника (деф. ВИМК): поки увімкнений,
    # будильник працює ПОСТІЙНО — кожен збіг вмикає кнопку, якщо вона вимкнена
    # (кнопка вже увімкнена → нічого не робимо і не шумимо).
    'alarm_on': False,
    'alarm_symbol': 'BTCUSDT',
    'alarm_tf': 'any',        # ІСТОРИЧНИЙ ключ: '1h' | '4h' | 'any' (лише міграція)
    # ⏰ TF, сигнали яких будять (вимога 16.11): будь-який набір із ALARM_TF_ALL.
    'alarm_tfs': ['1h', '4h'],
    'alarm_side': 'LONG',     # 'LONG' | 'SHORT'
    # 🏦 Дія над головним тумблером Trade Manager (вимога 16.11): '' — не
    # чіпати · 'on' — увімкнути · 'off' — вимкнути. Деф. не чіпати.
    'alarm_tm': '',
    # 🤖 Дія над ГОЛОВНИМ ТУМБЛЕРОМ ШІ (вимога 23.11): '' — не чіпати · 'on' —
    # увімкнути · 'off' — вимкнути. Деф. не чіпати.
    'alarm_ai': '',
    # ⏰ ДЖЕРЕЛО будильника (вимога 18.11): 'sbr' — свіжий сигнал 🔔 SBR (як і
    # було) · 'anom' — ⚠ «Аномалії обсягу» (готовий сценарій входу ШІ по монеті
    # будильника, той самий сигнал, що й особисті сповіщення) · 'both'.
    'alarm_src': 'sbr',
    'alarm_last': None,       # {at, symbol, side, tf, ok, note} — останнє спрацювання
}

ALARM_TFS = ('any', '1h', '4h')                     # історичний `alarm_tf`
ALARM_TF_ALL = ALL_TFS                              # вибір у новому UI
ALARM_SIDES = ('LONG', 'SHORT')
ALARM_TM = ('', 'on', 'off')
ALARM_SRC = ('sbr', 'anom', 'both')


def alarm_src_of(settings) -> str:
    """⏰ Джерело будильника. ЧИСТА. Невідоме → 'sbr' (стара поведінка)."""
    v = str((settings or {}).get('alarm_src') or 'sbr').lower()
    return v if v in ALARM_SRC else 'sbr'


def alarm_anom_symbol(settings) -> Optional[str]:
    """⏰ Монета, яку будильник чекає від ⚠ «Аномалій обсягу» (None — ні). ЧИСТА."""
    s = settings or {}
    if not s.get('alarm_on') or alarm_src_of(s) not in ('anom', 'both'):
        return None
    return alarm_norm_symbol(s.get('alarm_symbol'))


def alarm_anom_match(settings, symbol, side) -> bool:
    """⏰ Чи будить ⚠ сигнал «Аномалії обсягу» цієї монети. ЧИСТА.
    Бік НЕ обирається (вимога 18.11): будить сигнал БУДЬ-ЯКОГО боку, а кнопки
    напрямку будильник ставить у бік самого сигналу (`live_set_only`)."""
    S = alarm_anom_symbol(settings)
    return bool(S) and str(symbol or '').upper() == S and \
        str(side or '').upper() in ALARM_SIDES


def alarm_tfs_of(settings) -> tuple:
    """⏰ TF будильника. ЧИСТА. `alarm_tfs` (список) — головне; немає /
    порожньо → історичний `alarm_tf` ('any' → 1H+4H). Порядок — від молодшого."""
    s = settings or {}
    raw = s.get('alarm_tfs')
    if isinstance(raw, (list, tuple)):
        got = {str(x).lower() for x in raw}
        out = tuple(tf for tf in ALARM_TF_ALL if tf in got)
        if out:
            return out
    legacy = str(s.get('alarm_tf') or 'any').lower()
    return (legacy,) if legacy in TFS else TFS


def aggregate_bars(bars, tf_sec: int, base_sec: int):
    """Бари `base_sec` → бари `tf_sec` (45m із 15m). ЧИСТА. Лише ПОВНІ групи
    (неповна — живий бар або діра в історії біржі — відкидається)."""
    k = max(1, int(tf_sec // base_sec))
    groups: Dict[int, List[Dict]] = {}
    for b in sorted(bars or [], key=lambda x: int(x.get('t') or 0)):
        try:
            g = int(b['t']) // 1000 // tf_sec * tf_sec
        except Exception:
            continue
        groups.setdefault(g, []).append(b)
    out = []
    for g in sorted(groups):
        sub = groups[g]
        if len(sub) != k:
            continue
        cl = lambda x: float(x.get('p', x.get('c', x.get('close'))))
        out.append({'t': g * 1000, 'o': float(sub[0]['o']),
                    'h': max(float(x['h']) for x in sub), 'l': min(float(x['l']) for x in sub),
                    'p': cl(sub[-1]),
                    'v': sum(float(x.get('v') or 0) for x in sub),
                    'q': sum(float(x.get('q') or 0) for x in sub)})
    return out


def alarm_set_tm(on: bool, tm=None):
    """🏦 Будильник перемикає ГОЛОВНИЙ тумблер Trade Manager. → (ok, причина,
    змінено?). Пише ТИМ САМИМ шляхом, що й тумблер на сторінці
    (`update_settings({'enabled': …})` — з персистом), і ПЕРЕВІРЯЄ результат
    живим `is_enabled()`: успіх — лише коли TM справді в потрібному стані,
    а не «ми відправили запит»."""
    want = bool(on)
    if tm is None:
        try:
            from detection.trade_manager import get_trade_manager
            tm = get_trade_manager()
        except Exception as e:
            return False, f'Trade Manager недоступний: {e}', False
    if tm is None or not hasattr(tm, 'update_settings'):
        return False, 'Trade Manager ще не піднявся', False
    word = 'увімкнено' if want else 'вимкнено'
    try:
        if bool(tm.is_enabled()) == want:
            return True, 'вже ' + word, False
        tm.update_settings({'enabled': want})
        if bool(tm.is_enabled()) != want:
            return False, 'Trade Manager не прийняв зміну (стан не змінився)', False
        # 💾 Перевірка ЗАПИСУ (вимога 18.11): тумблер мусить пережити рестарт.
        db = getattr(tm, 'db', None)
        if db is not None and hasattr(db, 'get_setting'):
            try:
                from detection.trade_manager import DB_KEY_TM_SETTINGS as _K
            except Exception:
                _K = 'tm_settings'
            stored = db.get_setting(_K, None)
            if isinstance(stored, dict) and bool(stored.get('enabled')) != want:
                return False, word + ', але НЕ збережено в БД (після рестарту повернеться)', True
        # Увімкнено → монітор TM мусить працювати (`update_settings` сам його
        # запускає; страхуємо — без двигуна «увімкнений» TM нічого б не робив).
        if want and hasattr(tm, '_running') and not getattr(tm, '_running') and hasattr(tm, 'start'):
            try:
                tm.start()
            except Exception as e:
                return False, f'увімкнено, але монітор не запустився: {e}', True
        return True, word, True
    except Exception as e:
        return False, f'не вдалося змінити: {e}', False


def alarm_set_ai(on: bool, eng=None):
    """🤖 Будильник перемикає ГОЛОВНИЙ тумблер ШІ (вимога 23.11). → (ok,
    причина, змінено?). Пише тим самим шляхом, що тумблер на сторінці
    (`AiEngine.update_ai_settings({'enabled': …})`), і ПЕРЕВІРЯЄ: налаштування,
    які читає такт ШІ, і запис у БД (переживе рестарт); увімкнено → потік ШІ
    мусить працювати."""
    want = bool(on)
    if eng is None:
        try:
            from detection.ai_signals import get_ai_engine
            eng = get_ai_engine()
        except Exception as e:
            return False, f'ШІ недоступний: {e}', False
    if eng is None or not hasattr(eng, 'update_ai_settings'):
        return False, 'рушій ШІ ще не піднявся', False
    word = 'увімкнено' if want else 'вимкнено'
    try:
        if bool(eng.ai_settings().get('enabled')) == want:
            return True, 'вже ' + word, False
        eng.update_ai_settings({'enabled': want})
        if bool(eng.ai_settings().get('enabled')) != want:
            return False, 'ШІ не прийняв зміну (стан не змінився)', False
        db = getattr(eng, 'db', None)
        if db is not None and hasattr(db, 'get_setting'):
            try:
                from detection.ai_signals import AI_DB_SETTINGS as _K
            except Exception:
                _K = 'ai_settings_v1'
            stored = db.get_setting(_K, None)
            if isinstance(stored, str):
                try:
                    stored = json.loads(stored)
                except Exception:
                    stored = None
            if isinstance(stored, dict) and bool(stored.get('enabled')) != want:
                return False, word + ', але НЕ збережено в БД (після рестарту повернеться)', True
        if want and hasattr(eng, 'start'):
            th = getattr(eng, '_thread', None)
            if not (th and th.is_alive()):
                try:
                    eng.start()
                except Exception as e:
                    return False, f'увімкнено, але потік ШІ не запустився: {e}', True
        return True, word, True
    except Exception as e:
        return False, f'не вдалося змінити: {e}', False


def alarm_norm_symbol(sym) -> str:
    """'btc' / 'BTC' / 'btcusdt' → 'BTCUSDT'. Порожньо → BTCUSDT. ЧИСТА."""
    s = ''.join(ch for ch in str(sym or '').upper() if ch.isalnum())
    if not s:
        return 'BTCUSDT'
    return s if s.endswith('USDT') else s + 'USDT'


def alarm_match(settings, symbol, items):
    """⏰ Чи будить цей набір СВІЖИХ сигналів будильник. ЧИСТА.
    `items` = [(tf, сигнал)]. → (tf, сигнал) першого збігу (молодший TF
    раніше — реакція швидша) або None. Будильник вимкнений / інша монета /
    інший бік / інший TF → None."""
    s = settings or {}
    if not s.get('alarm_on') or alarm_src_of(s) not in ('sbr', 'both'):
        return None
    if str(symbol or '').upper() != alarm_norm_symbol(s.get('alarm_symbol')):
        return None
    side = str(s.get('alarm_side') or 'LONG').upper()
    tfs = alarm_tfs_of(s)
    for tf, sig in sorted(items or [], key=lambda x: TF_SEC.get(x[0], 0)):
        if not isinstance(sig, dict) or str(sig.get('side') or '').upper() != side:
            continue
        if tf not in tfs:
            continue
        return tf, sig
    return None


def tfs_norm(raw) -> tuple:
    """Набір TF із ALL_TFS у порядку від молодшого (невідомі відкидаються). ЧИСТА."""
    got = {str(x).lower() for x in (raw or [])} if isinstance(raw, (list, tuple, set)) else set()
    return tuple(tf for tf in ALL_TFS if tf in got)


def tg_tfs_of(settings) -> tuple:
    """TF, сигнали яких ідуть у Telegram (група). ЧИСТА. `tg_tfs` (список із
    ALL_TFS) — головне; немає → історичні `tg_1h` / `tg_4h`. Порожньо → 1H+4H
    (інакше вибір мовчки гасив би всю тему)."""
    s = settings or {}
    out = tfs_norm(s.get('tg_tfs'))
    if out:
        return out
    out = tuple(tf for tf, k in (('1h', 'tg_1h'), ('4h', 'tg_4h')) if bool(s.get(k, True)))
    return out or TFS


# ═══════════════════════════ ЧИСТІ ФУНКЦІЇ ═══════════════════════════════════

def _clamp(x: float) -> float:
    return max(0.0, min(100.0, x))


def pine_ema(src: List[float], length: int) -> List[Optional[float]]:
    """`ta.ema`: alpha = 2/(len+1), старт — SMA перших `len` значень."""
    out: List[Optional[float]] = [None] * len(src)
    if length <= 0 or len(src) < length:
        return out
    a = 2.0 / (length + 1)
    v = sum(src[:length]) / length
    out[length - 1] = v
    for i in range(length, len(src)):
        v = a * src[i] + (1 - a) * v
        out[i] = v
    return out


def pine_rma(src: List[float], length: int) -> List[Optional[float]]:
    """`ta.rma`: alpha = 1/len, старт — SMA перших `len` значень."""
    out: List[Optional[float]] = [None] * len(src)
    if length <= 0 or len(src) < length:
        return out
    a = 1.0 / length
    v = sum(src[:length]) / length
    out[length - 1] = v
    for i in range(length, len(src)):
        v = a * src[i] + (1 - a) * v
        out[i] = v
    return out


def pine_atr(h: List[float], l: List[float], c: List[float],
             length: int) -> List[Optional[float]]:
    """`ta.atr(len)` = `ta.rma(ta.tr(true), len)`; перший TR = high − low."""
    tr = []
    for i in range(len(h)):
        if i == 0:
            tr.append(h[i] - l[i])
        else:
            tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return pine_rma(tr, length)


def risk_text(score: float, cfg: Dict = None) -> str:
    c = cfg or DEFAULTS
    if score >= c['th_add']:
        return '1R + Add'
    if score >= c['th_1r']:
        return '1R'
    if score >= c['th_half']:
        return '0.5R'
    return '0.25R'


def kyiv_dt(ts_sec: float) -> datetime:
    if _KYIV is not None:
        return datetime.fromtimestamp(ts_sec, _KYIV)
    return datetime.utcfromtimestamp(ts_sec + 2 * 3600)   # грубий фолбек


def closed_bars(bars: List[Dict], tf_sec: int, now: float) -> List[Dict]:
    """Лише ЗАКРИТІ бари (час відкриття + тривалість ≤ зараз)."""
    out = []
    for b in bars or []:
        try:
            t = float(b['t']) / 1000.0
        except Exception:
            continue
        if t + tf_sec <= now + 0.5:
            out.append(b)
    return out


def compute_signals(bars: List[Dict], tf_sec: int, cfg: Dict = None) -> List[Dict]:
    """ВСІ сигнали SBR на ЗАКРИТИХ барах `bars` (найстаріший першим).

    Бар: {t (мс, відкриття), o, h, l, p (close)} — формат `fetch_klines`;
    `c`/`close` теж приймаються. Повертає [{side, t, close_t, trigger, score,
    risk, sl, n, time_kyiv}] у хронологічному порядку.
    """
    c = dict(DEFAULTS)
    c.update(cfg or {})
    n = len(bars or [])
    if n < max(c['mid_len'], c['atr_len']) + 2:
        return []
    o, h, l, cl, t = [], [], [], [], []
    for b in bars:
        o.append(float(b['o']))
        h.append(float(b['h']))
        l.append(float(b['l']))
        cl.append(float(b.get('p', b.get('c', b.get('close')))))
        t.append(int(b['t']))
    hlc3 = [(h[i] + l[i] + cl[i]) / 3.0 for i in range(n)]
    mid = pine_ema(hlc3, int(c['mid_len']))
    atr = pine_atr(h, l, cl, int(c['atr_len']))
    out: List[Dict] = []
    s_cnt = l_cnt = 0
    last_day = None
    for i in range(n):
        ts = t[i] / 1000.0
        d = kyiv_dt(ts).day
        if last_day is None or d != last_day:
            s_cnt = l_cnt = 0
            last_day = d
        if i == 0 or mid[i - 1] is None or atr[i - 1] is None:
            continue
        atr_p = atr[i - 1]
        if not atr_p or atr_p <= 0:
            continue
        atr_c = atr[i] or atr_p          # SL — від ATR поточного (закритого) бару
        up_p = mid[i - 1] + c['band_k'] * atr_p
        lo_p = mid[i - 1] - c['band_k'] * atr_p
        s_sc = _clamp(c['sc_base'] + c['sc_slope'] * (h[i] - up_p) / atr_p)
        l_sc = _clamp(c['sc_base'] + c['sc_slope'] * (lo_p - l[i]) / atr_p)
        short_sig = (h[i] >= up_p and s_sc >= c['min_score']
                     and (not c['rev_candle'] or cl[i] < o[i])
                     and (not c['close_out'] or cl[i] > up_p))
        long_sig = (l[i] <= lo_p and l_sc >= c['min_score']
                    and (not c['rev_candle'] or cl[i] > o[i])
                    and (not c['close_out'] or cl[i] < lo_p))
        tk = kyiv_dt(ts).strftime('%H:%M')
        if short_sig:
            s_cnt += 1
            out.append({'side': 'SHORT', 't': t[i], 'close_t': ts + tf_sec,
                        'trigger': h[i], 'score': round(s_sc, 2),
                        'risk': risk_text(s_sc, c), 'sl': cl[i] + c['sl_k'] * atr_c,
                        'n': s_cnt, 'time_kyiv': tk})
        if long_sig:
            l_cnt += 1
            out.append({'side': 'LONG', 't': t[i], 'close_t': ts + tf_sec,
                        'trigger': l[i], 'score': round(l_sc, 2),
                        'risk': risk_text(l_sc, c), 'sl': cl[i] - c['sl_k'] * atr_c,
                        'n': l_cnt, 'time_kyiv': tk})
    return out


def tf_snapshot(bars: List[Dict], tf: str, now: float, cfg: Dict = None) -> Optional[Dict]:
    """Стан одного TF по монеті: останній закритий бар + хвіст сигналів."""
    tf_sec = TF_SEC[tf]
    cb = closed_bars(bars, tf_sec, now)
    if not cb:
        return None
    sigs = compute_signals(cb, tf_sec, cfg)
    last_t = int(cb[-1]['t'])
    tail = sigs[-TAIL:]
    return {'closed_t': last_t, 'bars': len(cb),
            'price': float(cb[-1].get('p', cb[-1].get('c', 0)) or 0),
            'last': tail[-1] if tail else None, 'tail': tail}


def combine(s1: Optional[Dict], s4: Optional[Dict]) -> Optional[Dict]:
    """🔀 ОДИН рядок на монету з двох TF (вимога 6: «якщо сигнал є на 1Н і
    потім повторюється на 4Н — додатковий запис не потрібен, колонка показує,
    який сигнал прийшов, або обидва»). ЧИСТА.

    • Напрямок = бік НАЙСВІЖІШОГО сигналу (за моментом закриття бару).
    • Колонка TF = усі TF, чий ОСТАННІЙ сигнал у тому самому боці
      (`'1H'` / `'4H'` / `'1H+4H'`).
    • Одночасне закриття з протилежними боками → вирішує 4H (старший TF).
    • Сила = Score вирішального сигналу (при збігу двох — більший).
    Немає жодного сигналу → None.
    """
    c = [(sig['close_t'], tf, sig) for tf, sig in (('1h', s1), ('4h', s4)) if sig]
    if not c:
        return None
    c.sort(key=lambda x: (x[0], 1 if x[1] == '4h' else 0))
    _, tf_d, sig_d = c[-1]
    side = sig_d['side']
    agree = [(tf, sig) for _, tf, sig in c if sig['side'] == side]
    tfs = [tf for tf in TFS if any(a[0] == tf for a in agree)]
    return {'side': side, 'tfs': tfs, 'tf_label': '+'.join(TF_LABEL[x] for x in tfs),
            'strength': int(round(max(sig['score'] for _, sig in agree))),
            'decider': tf_d, 'signal': sig_d,
            'by_tf': {tf: sig for tf, sig in agree}}


def is_fresh(snap: Optional[Dict]) -> bool:
    """Сигнал стоїть САМЕ на останньому закритому барі TF (щойно зʼявився)."""
    if not snap or not snap.get('last'):
        return False
    return int(snap['last']['t']) == int(snap.get('closed_t') or -1)


def fmt_price(p) -> str:
    try:
        from detection.ob_alert import fmt_price as _fp
        return _fp(p)
    except Exception:
        try:
            v = float(p)
        except Exception:
            return '—'
        return f'{v:,.2f}' if v >= 100 else (f'{v:.4f}' if v >= 1 else f'{v:.6f}')


def tg_text(symbol: str, items: List[tuple]) -> str:
    """Текст Telegram (формат користувача 30.10):

        🔔 SBR · 🟢 LONG #2
        #BTCUSDT · 1H · 76.27%
        🎯 Trigger: 78,540.20
        🛑 SL: 77,912.40

    `items` = [(tf, signal)] одного боку — 1H і 4H, що зійшлись на одному
    закритті, ідуть ОДНИМ повідомленням (`1H+4H`; рівні й номер — старшого TF,
    Score — більший). Telegram не фарбує текст, тож «кольори» дають емодзі-
    маркери, **жирний** для напрямку й Score і `моноширинний` для цін."""
    tf_lbl = '+'.join(TF_LABEL[tf] for tf, _ in sorted(items, key=lambda x: TF_SEC[x[0]]))
    side = items[0][1]['side']
    head_sig = max(items, key=lambda x: TF_SEC[x[0]])[1]    # старший TF — головний
    score = max(float(sig['score']) for _, sig in items)
    emo = '🟢' if side == 'LONG' else '🔴'
    return '\n'.join([
        f"🔔 <b>SBR</b> · {emo} <b>{side} #{head_sig['n']}</b>",
        f"<b>#{symbol}</b> · <code>{tf_lbl}</code> · 📊 <b>{score:.2f}%</b>",
        f"🎯 Trigger: <code>{fmt_price(head_sig['trigger'])}</code>",
        f"🛑 SL: <code>{fmt_price(head_sig['sl'])}</code>",
    ])


# ═══════════════════════════ РУШІЙ (I/O) ═════════════════════════════════════

class SbrEngine:
    """Фоновий потік: тягне бари 1H/4H одразу після закриття, рахує сигнали,
    шле Telegram по монетах, відмічених кнопкою 🔔 у WATCHLIST."""

    def __init__(self, db=None, get_watchlist=None, fetch=None, notify=None,
                 clock=None, get_extra=None, personal=None, spring=None,
                 get_extra_tfs=None):
        self.db = db
        # 📶 {tf: [монети]} особистих підписок на ДОДАТКОВІ TF (17.11).
        self._get_extra_tfs = get_extra_tfs or (lambda: {})
        # 👤 особисті підписки з кабінету (вимога 09.11): монети, які треба
        # вести навіть поза WATCHLIST, і доставка сигналу в особистий чат.
        self._get_extra = get_extra or (lambda: [])
        self._personal = personal
        self._spring = spring          # 🔭 Пружина — скан біржі (попередження TG)
        self._get_watchlist = get_watchlist or (lambda: [])
        self._fetch = fetch            # (sym, tf, limit) → bars | None
        self._notify = notify          # (text) → (ok, why)
        self._admin_notify = None      # ⏰ будильник → особисто адміну (None = за замовч.)
        self._tm_switch = None         # ⏰ (on) → (ok, причина, змінено) — для тестів
        self._ai_switch = None         # ⏰ 🤖 те саме для головного тумблера ШІ
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._state: Dict[str, Dict] = {}      # {SYM: {'1h': snap, '4h': snap}}
        self._sent: Dict[str, int] = {}        # {'SYM|tf': t бару, про який уже сказано}
        self._fail: Dict[str, tuple] = {}      # {'SYM|tf': (k, наступна спроба)}
        self._attempt: Dict[str, float] = {}   # {tf: остання спроба}
        self._events: List[Dict] = []          # останні сигнали (для UI)
        self._ts = 0.0
        self._last_scan: Dict[str, Dict] = {}
        self._thread = None
        self._running = False
        self._settings = self._load_settings()
        self._restore()

    # ── налаштування ──────────────────────────────────────────────────────
    def _load_settings(self) -> Dict:
        s = dict(DEFAULT_SETTINGS)
        s['tg_symbols'] = list(DEFAULT_TG_SYMBOLS)
        try:
            raw = self.db.get_setting(DB_KEY_SETTINGS, None) if self.db else None
            if isinstance(raw, dict):
                s.update({k: v for k, v in raw.items() if k in DEFAULT_SETTINGS})
        except Exception:
            pass
        s['tg_symbols'] = sorted({str(x).upper() for x in (s.get('tg_symbols') or []) if x})
        s['enabled'] = bool(s.get('enabled', True))
        s['tg_enabled'] = bool(s.get('tg_enabled', True))
        self._tg_clean(s)
        self._alarm_clean(s)
        return s

    @staticmethod
    def _tg_clean(s: Dict) -> None:
        """`tg_tfs` — головне; `tg_1h`/`tg_4h` лишаються дзеркалом (старі UI)."""
        tfs = list(tg_tfs_of(s))
        s['tg_tfs'] = tfs
        s['tg_1h'] = '1h' in tfs
        s['tg_4h'] = '4h' in tfs

    @staticmethod
    def _alarm_clean(s: Dict) -> None:
        s['alarm_on'] = bool(s.get('alarm_on', False))
        s['alarm_symbol'] = alarm_norm_symbol(s.get('alarm_symbol'))
        tf = str(s.get('alarm_tf') or 'any').lower()
        s['alarm_tf'] = tf if tf in ALARM_TFS else 'any'
        s['alarm_tfs'] = list(alarm_tfs_of(s))
        sd = str(s.get('alarm_side') or 'LONG').upper()
        s['alarm_side'] = sd if sd in ALARM_SIDES else 'LONG'
        tm = str(s.get('alarm_tm') or '').lower()
        s['alarm_tm'] = tm if tm in ALARM_TM else ''
        ai = str(s.get('alarm_ai') or '').lower()
        s['alarm_ai'] = ai if ai in ALARM_TM else ''
        s['alarm_src'] = alarm_src_of(s)
        if not isinstance(s.get('alarm_last'), dict):
            s['alarm_last'] = None

    def get_settings(self) -> Dict:
        with self._lock:
            return {**self._settings, 'tg_symbols': list(self._settings['tg_symbols']),
                    'alarm_tfs': list(self._settings.get('alarm_tfs') or []),
                    'tg_tfs': list(self._settings.get('tg_tfs') or []),
                    'all_tfs': list(ALL_TFS)}

    def _save_settings(self):
        try:
            if self.db:
                self.db.set_setting(DB_KEY_SETTINGS, self.get_settings())
        except Exception as e:
            print(f"[SBR] settings save error: {e}")

    def update_settings(self, patch: Dict) -> Dict:
        with self._lock:
            if 'enabled' in patch:
                self._settings['enabled'] = bool(patch['enabled'])
            if 'tg_enabled' in patch:
                self._settings['tg_enabled'] = bool(patch['tg_enabled'])
            if 'tg_tfs' in patch:
                self._settings['tg_tfs'] = list(tfs_norm(patch['tg_tfs'])) or None
            elif 'tg_1h' in patch or 'tg_4h' in patch:     # старий UI: лише 1H/4H
                keep = [tf for tf in tg_tfs_of(self._settings) if tf not in TFS]
                for tf, k in (('1h', 'tg_1h'), ('4h', 'tg_4h')):
                    if bool(patch.get(k, self._settings.get(k, True))):
                        keep.append(tf)
                self._settings['tg_tfs'] = list(tfs_norm(keep)) or None
            self._tg_clean(self._settings)
            if 'tg_symbols' in patch and isinstance(patch['tg_symbols'], list):
                self._settings['tg_symbols'] = sorted(
                    {str(x).upper() for x in patch['tg_symbols'] if x})
            for k in ('alarm_on', 'alarm_symbol', 'alarm_tf', 'alarm_tfs', 'alarm_side',
                      'alarm_tm', 'alarm_src', 'alarm_ai'):
                if k in patch:
                    self._settings[k] = patch[k]
            if 'alarm_tf' in patch and 'alarm_tfs' not in patch:
                self._settings['alarm_tfs'] = None     # старий UI → з `alarm_tf`
            self._alarm_clean(self._settings)
        self._save_settings()
        if self._settings['enabled']:
            self.start()
        return self.get_settings()

    def set_tg_symbol(self, symbol: str, on: bool) -> Dict:
        S = str(symbol or '').upper()
        if not S:
            return self.get_settings()
        with self._lock:
            cur = set(self._settings['tg_symbols'])
            (cur.add if on else cur.discard)(S)
            self._settings['tg_symbols'] = sorted(cur)
        self._save_settings()
        return self.get_settings()

    def is_enabled(self) -> bool:
        return bool(self._settings.get('enabled', True))

    # ── збереження стану ─────────────────────────────────────────────────
    def _restore(self):
        try:
            blob = self.db.get_setting(DB_KEY_STATE, None) if self.db else None
        except Exception:
            blob = None
        if not isinstance(blob, dict):
            return
        st = blob.get('state') or {}
        if isinstance(st, dict):
            self._state = {str(k).upper(): v for k, v in st.items() if isinstance(v, dict)}
        sent = blob.get('sent') or {}
        if isinstance(sent, dict):
            self._sent = {str(k): int(v) for k, v in sent.items()
                          if isinstance(v, (int, float))}
        ev = blob.get('events') or []
        if isinstance(ev, list):
            self._events = [e for e in ev if isinstance(e, dict)][-50:]
        self._ts = float(blob.get('ts') or 0.0)

    def _persist(self):
        try:
            if not self.db:
                return
            with self._lock:
                blob = {'ts': self._ts, 'state': dict(self._state),
                        'sent': dict(self._sent), 'events': list(self._events[-50:])}
            self.db.set_setting(DB_KEY_STATE, blob)
        except Exception as e:
            print(f"[SBR] persist error: {e}")

    # ── читання ──────────────────────────────────────────────────────────
    def rows(self) -> Dict[str, Dict]:
        """{SYM: combine(...) + snaps} — читає книга 🔔 МММ-SBR."""
        with self._lock:
            st = {k: dict(v) for k, v in self._state.items()}
        out = {}
        for S, v in st.items():
            s1, s4 = v.get('1h'), v.get('4h')
            cmb = combine((s1 or {}).get('last'), (s4 or {}).get('last'))
            out[S] = {'combo': cmb, '1h': s1, '4h': s4,
                      'fresh': {tf: is_fresh(v.get(tf)) for tf in TFS},
                      'price': (s1 or s4 or {}).get('price')}
        return out

    def snaps(self, symbol: str) -> Dict[str, Dict]:
        """{'1h': snap, '4h': snap} однієї монети (лише памʼять, без мережі).
        Читає 🚪 правило виходу «протилежний сигнал 🔔 МММ-SBR» у TM."""
        S = str(symbol or '').upper()
        with self._lock:
            v = dict(self._state.get(S) or {})
        return {tf: v.get(tf) for tf in TFS if v.get(tf)}

    def state(self) -> Dict:
        now = self._clock()
        with self._lock:
            ls = {k: dict(v) for k, v in self._last_scan.items()}
            ev = list(self._events[-30:])
            n = len(self._state)
        nxt = {tf: (now // TF_SEC[tf] + 1) * TF_SEC[tf] + SETTLE_SEC for tf in TFS}
        return {'settings': self.get_settings(), 'running': self._running,
                'coins': n, 'ts': self._ts, 'last_scan': ls, 'next_at': nxt,
                'events': ev, 'server_now': now}

    # ── потік ────────────────────────────────────────────────────────────
    def start(self):
        if self._running or not self.is_enabled():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name='sbr-engine', daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                if self.is_enabled():
                    self.tick()
            except Exception as e:
                print(f"[SBR] tick error: {e}")
            time.sleep(LOOP_SEC)

    def _universe(self) -> List[str]:
        try:
            wl = [str(x).upper() for x in (self._get_watchlist() or []) if x]
        except Exception:
            wl = []
        s = self.get_settings()
        tg = s['tg_symbols']
        # ⏰ Монета будильника рахується ЗАВЖДИ, навіть поза watchlist.
        al = [s['alarm_symbol']] if s.get('alarm_on') else []
        try:
            ex = [str(x).upper() for x in (self._get_extra() or []) if x]
        except Exception:
            ex = []
        return list(dict.fromkeys(wl + tg + al + ex))

    def _due(self, syms: List[str], tf: str, now: float) -> List[str]:
        """Монети, яким бракує ОСТАННЬОГО закритого бару TF."""
        tf_sec = TF_SEC[tf]
        bound = int(now // tf_sec) * tf_sec           # відкриття живого бару
        if now < bound + SETTLE_SEC:
            bound -= tf_sec                           # ще чекаємо закриття
        need_t = (bound - tf_sec) * 1000              # відкриття останнього закритого
        out = []
        with self._lock:
            for S in syms:
                snap = (self._state.get(S) or {}).get(tf)
                if snap and int(snap.get('closed_t') or 0) >= need_t:
                    continue
                f = self._fail.get(f'{S}|{tf}')
                if f and now < f[1]:
                    continue
                out.append(S)
        return out

    def tick(self) -> Dict:
        """Один прохід: по кожному TF — дотягнути монети без останнього
        закритого бару. Повертає {tf: [символи, що оновились]}."""
        now = self._clock()
        syms = self._universe()
        if not syms or self._fetch is None:
            return {}
        done: Dict[str, List[str]] = {}
        jobs = []
        for tf, pool in self._plan(syms):
            due = self._due(pool, tf, now)
            if not due:
                continue
            if now - self._attempt.get(tf, 0.0) < RETRY_SEC and \
                    any((self._state.get(S) or {}).get(tf) for S in due):
                continue
            self._attempt[tf] = now
            jobs += [(S, tf) for S in due]
        if not jobs:
            return {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(lambda j: (j, self._safe_fetch(*j)), jobs))
        now = self._clock()
        fresh: Dict[str, List[tuple]] = {}
        for (S, tf), bars in res:
            ok, item = self._absorb(S, tf, bars, now)
            if not ok:
                continue
            done.setdefault(tf, []).append(S)
            if item:
                fresh.setdefault(S, []).append(item)
        with self._lock:
            self._ts = now
            for tf in TF_SEC:
                if tf in done or any(j[1] == tf for j in jobs):
                    self._last_scan[tf] = {'at': now, 'took': round(time.time() - t0, 2),
                                           'asked': sum(1 for j in jobs if j[1] == tf),
                                           'ok': len(done.get(tf, []))}
        for S, items in fresh.items():
            self._on_signals(S, items, now)
        if done:
            self._kick_book()
            self._persist()
        return done

    def _extra_plan(self, s: Dict) -> Dict[str, List[str]]:
        """{tf: [монети]} ДОДАТКОВИХ TF (EXTRA_TFS): ⏰ будильник · 📨 група TG
        (лише монети 🔔 і лише обрані TF) · 👤 особисті підписки кабінету."""
        out: Dict[str, List[str]] = {}

        def add(tf, sym):
            if tf in EXTRA_TFS and sym:
                lst = out.setdefault(tf, [])
                if sym not in lst:
                    lst.append(str(sym).upper())
        if s.get('alarm_on') and alarm_src_of(s) in ('sbr', 'both'):
            for tf in alarm_tfs_of(s):
                add(tf, s['alarm_symbol'])
        if s.get('tg_enabled'):
            for tf in tg_tfs_of(s):
                for S in s.get('tg_symbols') or []:
                    add(tf, S)
        try:
            for tf, syms in (self._get_extra_tfs() or {}).items():
                for S in syms or []:
                    add(str(tf).lower(), S)
        except Exception:
            pass
        return out

    def _plan(self, syms: List[str]) -> List[tuple]:
        """[(tf, монети)] на прохід: 1H/4H — увесь всесвіт; додаткові TF
        (15m/30m/45m/1d) — ЛИШЕ монети, де цей TF обрано (`_extra_plan`)."""
        plan = [(tf, syms) for tf in TFS]
        ex = self._extra_plan(self.get_settings())
        for tf in EXTRA_TFS:
            if ex.get(tf):
                plan.append((tf, ex[tf]))
        return plan

    @staticmethod
    def _need_t(tf: str, now: float) -> int:
        """Час відкриття (мс) ОСТАННЬОГО закритого бару TF на момент `now`."""
        tf_sec = TF_SEC[tf]
        bound = int(now // tf_sec) * tf_sec
        if now < bound + SETTLE_SEC:
            bound -= tf_sec
        return (bound - tf_sec) * 1000

    def _absorb(self, S: str, tf: str, bars, now: float):
        """Прийняти бари монети: оновити стан і вирішити, чи це свіжий сигнал
        для Telegram. → (стан оновлено?, (tf, сигнал) | None). ЄДИНЕ місце —
        і для проходу `tick`, і для точкового `ensure_current`, тож TG-сигнал
        не губиться, ким би бари не були дотягнуті."""
        key = f'{S}|{tf}'
        snap = tf_snapshot(bars, tf, now) if bars else None
        if not snap:
            k = (self._fail.get(key) or (0, 0))[0] + 1
            self._fail[key] = (k, now + min(FAIL_BACKOFF_MAX, RETRY_SEC * 2 ** k))
            return False, None
        self._fail.pop(key, None)
        with self._lock:
            self._state.setdefault(S, {})[tf] = snap
        if snap['closed_t'] < self._need_t(tf, now):
            return True, None     # біржа ще не закрила бар → повтор за RETRY_SEC
        last = snap.get('last')
        if not is_fresh(snap):
            return True, None
        with self._lock:
            if self._sent.get(key) == int(last['t']):
                return True, None
            self._sent[key] = int(last['t'])
        lag = now - float(last['close_t'])
        if lag > TF_SEC[tf] * SEND_MAX_LAG_FRAC:
            return True, None     # бот лежав на закритті — тиха база, не спам
        return True, (tf, last)

    def is_current(self, symbol: str, tfs=TFS) -> bool:
        """Чи має монета в памʼяті ОСТАННІЙ закритий бар кожного з `tfs`
        (лише памʼять). False одразу після закриття бару, поки прохід
        `tick` до монети ще не дійшов."""
        S = str(symbol or '').upper()
        now = self._clock()
        with self._lock:
            v = dict(self._state.get(S) or {})
        for tf in tfs:
            snap = v.get(tf)
            if not snap or int(snap.get('closed_t') or 0) < self._need_t(tf, now):
                return False
        return True

    def ensure_current(self, symbol: str, tfs=TFS) -> bool:
        """🔒 Ворота «Через МММ-SBR» (кейс OKBUSDT 31.10): сигнал на закритті
        бару міг пройти ворота ДО того, як прохід `tick` дістався монети, —
        тобто за старим станом SBR. Тут ДОТЯГУЄМО відсутній закритий бар САМЕ
        цієї монети (1-2 запити, лише у вікні після закриття бару). Свіжий
        сигнал іде в Telegram тим самим шляхом, що й з `tick`.
        → True, якщо стан монети оновлено."""
        S = str(symbol or '').upper()
        if not S or self._fetch is None:
            return False
        now = self._clock()
        with self._lock:
            v = dict(self._state.get(S) or {})
        miss = [tf for tf in tfs
                if not v.get(tf) or int(v[tf].get('closed_t') or 0) < self._need_t(tf, now)]
        if not miss:
            return False
        upd, fresh = False, []
        for tf in miss:
            ok, item = self._absorb(S, tf, self._safe_fetch(S, tf), self._clock())
            upd = upd or ok
            if item:
                fresh.append(item)
        if fresh:
            self._on_signals(S, fresh, self._clock())
        if upd:
            self._persist()
        return upd

    def _kick_book(self):
        """Нові бари → книга 🔔 МММ-SBR перебудовується ОДРАЗУ (не чекаючи
        такту двигуна): вкладки LONG/SHORT змінюються в мить закриття бару."""
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
            if ff is not None and hasattr(ff, 'on_sbr_update'):
                ff.on_sbr_update()
        except Exception:
            pass

    def _safe_fetch(self, S: str, tf: str):
        try:
            if tf in AGG_FROM:
                base, k = AGG_FROM[tf]
                return aggregate_bars(self._fetch(S, base, ALARM_BARS * k),
                                      TF_SEC[tf], TF_SEC[base])
            if tf in ALARM_ONLY_TFS:
                return self._fetch(S, tf, ALARM_BARS)
            return self._fetch(S, tf, BARS)
        except Exception as e:
            print(f"[SBR] fetch {S} {tf} error: {e}")
            return None

    def _on_signals(self, S: str, items: List[tuple], now: float):
        """Свіжі сигнали монети: подія в журнал + Telegram (якщо відмічена)."""
        s = self.get_settings()
        try:
            self._alarm_check(S, items, now, s)
        except Exception as e:
            print(f"[SBR] alarm error {S}: {e}")
        # 📶 Додаткові TF (15m/30m/45m/1d) — лише туди, де їх обрано: група
        # (tg_tfs) і особисті підписки. 🔭 скан і журнал книги — 1H/4H.
        ex = self._extra_plan(s)
        grp_extra = set(tg_tfs_of(s)) if (s.get('tg_enabled') and S in s.get('tg_symbols', [])) else set()
        pers_extra = {tf for tf, syms in ex.items() if S in syms}
        all_items = [it for it in items if it[0] in TFS or it[0] in grp_extra or it[0] in pers_extra]
        base = [it for it in items if it[0] in TFS]
        items = all_items
        if not items:
            return
        by_side: Dict[str, List[tuple]] = {}
        for tf, sig in items:
            by_side.setdefault(sig['side'], []).append((tf, sig))
        # 👤 Особисті сповіщення (кабінет) — НЕЗАЛЕЖНО від групових налаштувань:
        # користувач сам обрав монету й TF. Лише активні акаунти (фільтр у БД).
        pers_ok = pers_bad = 0
        if self._personal:
            try:
                pers_ok, pers_bad = self._personal(S, items) or (0, 0)
            except Exception as e:
                print(f"[SBR] personal send error {S}: {e}")
        # 🔭 Монети з таблиці «Пружина — скан біржі» (кнопки 1H/4H): незалежно
        # від групових тумблерів SBR — у таблиці людина обрала монету сама.
        if self._spring and base:
            try:
                self._spring(S, base)
            except Exception as e:
                print(f"[SBR] spring-scan hook error {S}: {e}")
        for side, grp in by_side.items():
            tf_lbl = '+'.join(TF_LABEL[tf] for tf, _ in sorted(grp, key=lambda x: TF_SEC[x[0]]))
            ev = {'symbol': S, 'side': side, 'tfs': tf_lbl, 'at': now,
                  'score': max(g[1]['score'] for g in grp),
                  'trigger': grp[-1][1]['trigger'], 'tg': None}
            _tg_grp = [g for g in grp if g[0] in tg_tfs_of(s)]
            if s['tg_enabled'] and S in s['tg_symbols'] and self._notify and _tg_grp:
                try:
                    ok, why = self._notify(tg_text(S, _tg_grp))
                except Exception as e:
                    ok, why = False, str(e)
                ev['tg'] = bool(ok)
                if not ok:
                    ev['tg_err'] = str(why)[:160]
            with self._lock:
                self._events.append(ev)
                self._events = self._events[-50:]
            try:
                from detection.activity_log import log_activity
                txt = (f"🔔 SBR {tf_lbl} {side} · Score {ev['score']:.2f} · "
                       f"Trigger {fmt_price(ev['trigger'])}")
                if ev['tg'] is True:
                    txt += ' · 📨 надіслано в «🔔 Сигнали»'
                elif ev['tg'] is False:
                    txt += f" · 📨 Telegram НЕ прийняв: {ev.get('tg_err')}"
                if pers_ok or pers_bad:
                    txt += f' · 👤 особисто: {pers_ok}' + (f' (не прийнято {pers_bad})' if pers_bad else '')
                    ev['tg'] = ev['tg'] if ev['tg'] is not None else bool(pers_ok)
                if ev['tg'] is not None:
                    log_activity(S, 'signal', txt, side=side, source='SBR',
                                 extra={'tfs': tf_lbl, 'score': ev['score']})
            except Exception:
                pass


    def alarm_anom_symbol(self) -> Optional[str]:
        """⏰ Монета, яку будильник чекає від ⚠ «Аномалій обсягу» (лише памʼять)."""
        with self._lock:
            return alarm_anom_symbol(self._settings)

    def alarm_from_anom(self, S: str, side: str, scenario: str = '', price=None,
                        now: Optional[float] = None) -> bool:
        """⏰ ⚠ «Аномалії обсягу»: готовий сценарій входу по монеті будильника
        (БУДЬ-ЯКИЙ бік) → лишити увімкненою лише кнопку боку сигналу + дія над
        Trade Manager, обидва — з перевіркою реального стану. → чи спрацював."""
        now = time.time() if now is None else now
        with self._lock:
            s = dict(self._settings)
        if not alarm_anom_match(s, S, side):
            return False
        sig = {'side': str(side).upper(), 'scenario': scenario, 'price': price}
        self._alarm_fire(str(S).upper(), 'anom', sig, now, s)
        return True

    def _alarm_check(self, S: str, items: List[tuple], now: float, s: Dict) -> None:
        """⏰ Збіг → увімкнути головну кнопку напрямку + (за налаштуванням)
        увімкнути / вимкнути Trade Manager → 🧾 Лог + TG особисто адміну."""
        hit = alarm_match(s, S, items)
        if not hit:
            return
        tf, sig = hit
        self._alarm_fire(S, tf, sig, now, s)

    def _alarm_fire(self, S: str, tf: str, sig: Dict, now: float, s: Dict) -> None:
        """Спільна дія будильника для обох джерел. `tf` = TF сигналу SBR або
        'anom' (⚠ «Аномалії обсягу»)."""
        anom = tf == 'anom'
        tf_lbl = '⚠ Аномалії' if anom else TF_LABEL.get(tf, tf)
        side = str(sig.get('side')).upper()
        try:
            from detection import direction_gate as _dg
        except Exception:
            import importlib.util as _ilu, os as _os
            _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'direction_gate.py')
            _sp = _ilu.spec_from_file_location('_sbr_dg', _p)
            _dg = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_dg)
        # ⚠ Аномалії: лишити увімкненою РІВНО кнопку боку сигналу, другу —
        # вимкнути (незалежно від стану до будильника), з перевіркою живого
        # стану й запису в БД. SBR — як і було: увімкнути обраний бік.
        if anom:
            ok, note, _dch = _dg.live_set_only(side, source='Alarm anomalies')
        else:
            ok, note = _dg.live_enable(side, source='SBR alarm')
            _dch = bool(ok) and note != 'вже увімкнено'
        act = str(s.get('alarm_tm') or '')
        tm_res = None
        if act in ('on', 'off'):
            tm_res = (self._tm_switch or alarm_set_tm)(act == 'on')
        act_ai = str(s.get('alarm_ai') or '')
        ai_res = None
        if act_ai in ('on', 'off'):
            ai_res = (self._ai_switch or alarm_set_ai)(act_ai == 'on')
        last = {'at': now, 'symbol': S, 'side': side, 'tf': tf_lbl, 'src': 'anom' if anom else 'sbr',
                'scenario': sig.get('scenario') if anom else None,
                'ok': bool(ok) and (tm_res is None or bool(tm_res[0])), 'note': note,
                'dir_ok': bool(ok), 'score': sig.get('score'), 'trigger': sig.get('trigger'),
                'tm': act, 'tm_ok': None if tm_res is None else bool(tm_res[0]),
                'tm_note': None if tm_res is None else tm_res[1],
                'ai': act_ai, 'ai_ok': None if ai_res is None else bool(ai_res[0]),
                'ai_note': None if ai_res is None else ai_res[1]}
        if ai_res is not None and not ai_res[0]:
            last['ok'] = False
        with self._lock:
            self._settings['alarm_last'] = last
        self._save_settings()
        dir_changed = bool(_dch)
        tm_changed = bool(tm_res and tm_res[2])
        tm_failed = bool(tm_res is not None and not tm_res[0])
        ai_changed = bool(ai_res and ai_res[2])
        ai_failed = bool(ai_res is not None and not ai_res[0])
        if ok and not dir_changed and not tm_changed and not tm_failed \
                and not ai_changed and not ai_failed:
            return      # усе й так у потрібному стані — без рядка в лозі й без TG
        what = (f"{sig.get('scenario') or 'сценарій входу'}" if anom
                else f"Score {float(sig.get('score') or 0):.2f}")
        txt = (f"⏰ Будильник {'⚠ Аномалії' if anom else 'SBR'}: {S} {side} {'' if anom else tf_lbl + ' '}"
               f"({what}) → "
               + ((f"головні кнопки: {note}" if anom else f"головну кнопку {side} {note}") if ok
                  else (f"головні кнопки НЕ змінено: {note}" if anom
                        else f"кнопку {side} НЕ увімкнено: {note}")))
        if tm_res is not None:
            txt += (f" · Trade Manager {tm_res[1]}" if tm_res[0]
                    else f" · Trade Manager НЕ змінено: {tm_res[1]}")
        if ai_res is not None:
            txt += (f" · 🤖 ШІ {ai_res[1]}" if ai_res[0]
                    else f" · 🤖 ШІ НЕ змінено: {ai_res[1]}")
        print(f"[SBR] {txt}")
        try:
            from detection.activity_log import log_activity
            log_activity(S, 'signal', txt, side=side, source='SBR',
                         extra={'alarm': True, 'ok': bool(last['ok']), 'tm': act,
                                'tm_ok': last['tm_ok'], 'ai': act_ai, 'ai_ok': last['ai_ok']})
        except Exception:
            pass
        # 📨 Будильник — ОСОБИСТО адміну в головний бот, а НЕ в групу (вимога
        # 08.11) і незалежно від групового тумблера `tg_enabled`. Рядка про
        # увімкнену головну кнопку немає (на прохання); лише відмова. Trade
        # Manager — інша річ (реальна торгівля), тож його зміна пишеться завжди.
        notify = self._admin_notify or _default_admin_notify
        try:
            notify(alarm_tg_text(S, side, tf, sig, ok, note, tm=tm_res, tm_act=act,
                                 dir_changed=dir_changed, ai=ai_res, ai_act=act_ai))
        except Exception:
            pass


_instance: Optional[SbrEngine] = None


def _default_fetch(sym: str, tf: str, limit: int):
    from detection.market_data import get_market_data
    return get_market_data().fetch_klines(sym, limit=limit, interval=tf)


def alarm_tg_text(S, side, tf, sig, ok=True, note='', tm=None, tm_act='',
                  dir_changed=False, ai=None, ai_act='') -> str:
    """⏰ Текст будильника для особистого чату: сигнал без рядка про кнопку."""
    sd = str(side or '').upper()
    if tf == 'anom':          # ⚠ «Аномалії обсягу» (вимога 18.11)
        lines = [f"⏰ <b>Будильник ⚠ Аномалії</b> · #{S} "
                 f"{'🟢' if sd == 'LONG' else '🔴'} <b>{sd}</b>"]
        if sig.get('scenario'):
            lines.append(f"🧩 {sig.get('scenario')}")
        if sig.get('price'):
            lines.append(f"💵 Ціна: <code>{fmt_price(sig.get('price'))}</code>")
    else:
        lines = [f"⏰ <b>Будильник SBR</b> · #{S} "
                 f"{'🟢' if sd == 'LONG' else '🔴'} <b>{sd}</b> "
                 f"<code>{TF_LABEL.get(tf, tf)}</code>"]
        try:
            lines.append(f"📊 Score <b>{float(sig.get('score') or 0):.2f}%</b>")
        except Exception:
            pass
    if sig.get('trigger') is not None:
        lines.append(f"🎯 Trigger: <code>{fmt_price(sig.get('trigger'))}</code>")
    if sig.get('sl') is not None:
        lines.append(f"🛑 SL: <code>{fmt_price(sig.get('sl'))}</code>")
    if not ok:
        lines.append(f"⚠️ Головні кнопки НЕ змінено: {note}" if tf == 'anom'
                     else f"⚠️ Кнопку {sd} НЕ увімкнено: {note}")
    elif tf == 'anom' and dir_changed:
        lines.append(f"🚦 Головні кнопки → лише <b>{sd}</b>")
    if tm is not None:
        if not tm[0]:
            lines.append(f"⚠️ Trade Manager НЕ змінено: {tm[1]}")
        elif tm[2]:
            lines.append("🏦 Trade Manager → <b>" + ('УВІМКНЕНО' if tm_act == 'on' else 'ВИМКНЕНО')
                         + "</b>")
    if ai is not None:
        if not ai[0]:
            lines.append(f"⚠️ ШІ НЕ змінено: {ai[1]}")
        elif ai[2]:
            lines.append("🤖 ШІ → <b>" + ('УВІМКНЕНО' if ai_act == 'on' else 'ВИМКНЕНО') + "</b>")
    return "\n".join(lines)


def _default_admin_notify(text: str):
    """Особисто адміну (TELEGRAM_CHAT_ID) з головного бота, повз групу."""
    from web.tg_bot import tg_send, _admin_chat
    chat = _admin_chat()
    return bool(chat) and bool(tg_send(chat, text))


def _default_notify(text: str):
    from web.tg_bot import notify_category, last_send_error
    ok = bool(notify_category(TG_CATEGORY, text))
    return ok, ('' if ok else (last_send_error() or 'не надіслано'))


def _default_extra():
    out = []
    try:
        from web.info_portal import sbr_extra_symbols
        out += list(sbr_extra_symbols() or [])
    except Exception:
        pass
    try:
        from detection.spring_scan import sbr_symbols
        out += list(sbr_symbols() or [])
    except Exception:
        pass
    out += _open_trade_symbols()
    return out


def _open_trade_symbols() -> List[str]:
    """💼 Монети ВІДКРИТИХ угод (real + paper) — рушій веде їх навіть поза
    WATCHLIST (вимога 11.11): інакше колонка 🚨, правило виходу «протилежний
    SBR» і 🧱-автомат по угоді з 🔭 скану / ✋ ручного відкриття не мали
    б даних узагалі. Лише памʼять TM."""
    try:
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if tm is None:
            return []
        with tm._lock:
            keys = list((getattr(tm, '_positions', {}) or {}).keys()) + \
                list((getattr(tm, '_shadow_positions', {}) or {}).keys())
        return [str(k).upper() for k in keys if k]
    except Exception:
        return []


def _default_spring(symbol, items):
    from detection.spring_scan import on_sbr_signals
    return on_sbr_signals(symbol, items)


def _default_personal(symbol, items):
    from web.info_portal import sbr_personal_send
    return sbr_personal_send(symbol, items)


def _default_extra_tfs():
    try:
        from web.info_portal import sbr_extra_tfs
        return sbr_extra_tfs() or {}
    except Exception:
        return {}


def init_sbr_engine(db=None, get_watchlist=None, fetch=None, notify=None) -> SbrEngine:
    global _instance
    if _instance is None:
        _instance = SbrEngine(db, get_watchlist, fetch or _default_fetch,
                              notify or _default_notify,
                              get_extra=_default_extra, personal=_default_personal,
                              spring=_default_spring, get_extra_tfs=_default_extra_tfs)
        if _instance.is_enabled():
            _instance.start()
    return _instance


def get_sbr_engine() -> Optional[SbrEngine]:
    return _instance
