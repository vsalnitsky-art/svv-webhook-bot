"""⚠ АНОМАЛІЇ ОБСЯГУ — серверний детектор + журнал із наслідками (16.11).

Вимога: «Пиши в "Шлях монети" все, що стосується "Аномалій", щоб мати повне
розуміння і можливість розробити алгоритм самостійно ботом приймати рішення
на основі даних "Аномалій" — відкриття угоди і вихід з угоди».

Правило — ТЕ САМЕ, що малює графік кабінету (`volAnomalyAt` у
infosite/js/app.js, тест звіряє пороги): база = МЕДІАНА обсягу 20 попередніх
свічок, пріоритет 🧽 absorb → 🔥 climax → ⚡ side → 📈 spike. Розклад
покупці / продавці — за ПІДСВІЧКАМИ Bybit TF/10 (той самий метод, що
відсотки IMB і колонка «Обсяг»).

Що робить рушій:
  • на закритті КОЖНОГО бару 15m і 1h — по монетах «🌀 заряджені» обох Пружин (+24 год розряду) і відкритих угод
    (один запит свічок; підсвічки — лише коли обсяг ≥ ×1.5, тобто рідко);
  • аномалію пише в таблицю `sob_vol_anomaly` з КОНТЕКСТОМ: тренд перед
    свічкою, рух самої свічки, відкриті угоди по монеті (бік, PnL у ту мить);
  • на наступних закриттях дописує НАСЛІДОК: рух ціни через 1 / 4 / 8 свічок,
    максимум ↑ / ↓ за 8 свічок → «розворот чи продовження» стає ЦИФРОЮ.

У 🧾 Лог роботи бота аномалії НЕ пишуться (флуд) — їх показує «Шлях монети»
в хронології угоди і статистика `stats()`. Торгівлі цей модуль НЕ стосується:
нічого не відкриває і не закриває (тест-замок AST).
"""
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

# Дзеркало VOL_AN з infosite/js/app.js (тест звіряє).
AN = {'base': 20, 'minBase': 10, 'rv': 1.5, 'spike': 2, 'climax': 3,
      'side': 0.75, 'absorb': 0.65}
TFS = ('15m', '1h')
TF_SEC = {'15m': 900, '1h': 3600}
TF_LABEL = {'15m': '15M', '1h': '1H'}
OUT_N = (1, 4, 8)            # наслідок: рух через N закритих свічок
PRE_N = 8                    # «тренд перед» — зміна за N свічок до аномалії
FETCH_BARS = 40              # 20 база + 8 наслідок + запас
SETTLE_SEC = 4               # біржа віддає закритий бар не миттєво
RETRY_SEC = 8
WORKERS = 4
KEEP_DAYS = 120              # історія для статистики (дефолт налаштування keep_days)
KEEP_DAYS_MIN, KEEP_DAYS_MAX = 7, 730
KINDS = ('absorb', 'climax', 'side', 'spike')
KIND_UA = {'absorb': '🧽 Поглинання', 'climax': '🔥 Кульмінація',
           'side': '⚡ Перекіс', 'spike': '📈 Сплеск'}
DB_SETTINGS = 'vol_anomaly_settings'
DEFAULT_SETTINGS = {'enabled': True, 'keep_days': KEEP_DAYS}
# 🔬 Хто бере участь (вимога 17.11): монети зі статусом «🌀 заряджена» в
# 🌀 Пружині WATCHLIST і 🔭 Пружині — скан біржі (дослідження ВІДКРИТТЯ) +
# монети відкритих угод (дослідження ВИХОДУ). Заряд зник → монету ведемо ще
# AFTER_CHARGE_SEC: саме тоді відбувається розряд, який і досліджуємо.
AFTER_CHARGE_SEC = 24 * 3600
DB_CHARGED = 'vol_anomaly_charged_v1'
ROLE_UA = {'entry': 'заряджена (вхід)', 'after': 'після заряду (розряд)',
           'exit': 'відкрита угода (вихід)'}
BB_KLINE_URL = 'https://api.bybit.com/v5/market/kline'


# ───────────────────────── ЧИСТІ ФУНКЦІЇ ─────────────────────────
def keep_days_norm(v) -> int:
    """Скільки днів тримати аномалії (автоочистка рушія раз на добу)."""
    try:
        return max(KEEP_DAYS_MIN, min(KEEP_DAYS_MAX, int(float(v))))
    except (TypeError, ValueError):
        return KEEP_DAYS


def median(a: List[float]) -> float:
    b = sorted(a)
    n = len(b)
    if not n:
        return 0.0
    return b[(n - 1) // 2] if n % 2 else (b[n // 2 - 1] + b[n // 2]) / 2.0


def anomaly_at(rows: List[Dict], i: int) -> Optional[Dict]:
    """1:1 з `volAnomalyAt`: rows[i] = {v, buy (частка 0..1 | None), up (bool)}."""
    try:
        r = rows[i]
        v = float(r.get('v') or 0)
    except (IndexError, TypeError, ValueError, AttributeError):
        return None
    if not v > 0:
        return None
    prev = []
    j = i - 1
    while j >= 0 and len(prev) < AN['base']:
        try:
            pv = float(rows[j].get('v') or 0)
        except (TypeError, ValueError):
            pv = 0.0
        if pv > 0:
            prev.append(pv)
        j -= 1
    if len(prev) < AN['minBase']:
        return None
    med = median(prev)
    if not med > 0:
        return None
    rv = v / med
    sh, up = r.get('buy'), r.get('up')
    if sh is not None and rv >= AN['rv'] and up is not None:
        if sh >= AN['absorb'] and not up:
            return {'k': 'absorb', 'rv': rv, 'sh': sh, 'by': 'sell', 'med': med}
        if sh <= 1 - AN['absorb'] and up:
            return {'k': 'absorb', 'rv': rv, 'sh': sh, 'by': 'buy', 'med': med}
    if rv >= AN['climax']:
        return {'k': 'climax', 'rv': rv, 'sh': sh, 'med': med}
    if sh is not None and rv >= AN['rv'] and (sh >= AN['side'] or sh <= 1 - AN['side']):
        return {'k': 'side', 'rv': rv, 'sh': sh, 'dir': 'up' if sh >= 0.5 else 'dn', 'med': med}
    if rv >= AN['spike']:
        return {'k': 'spike', 'rv': rv, 'sh': sh, 'med': med}
    return None


def hint_of(kind: str, by: Optional[str], d: Optional[str]) -> Optional[str]:
    """Куди «вказує» аномалія за своєю природою: 🧽 — бік РОЗВОРОТУ (купували,
    а свічка червона → вниз), ⚡ — бік ТИСКУ. 🔥 / 📈 — без напрямку (їх зміст
    залежить від тренду перед ними → `pre_chg` у записі)."""
    if kind == 'absorb':
        return 'dn' if by == 'sell' else 'up'
    if kind == 'side':
        return d
    return None


def text_of(rec: Dict) -> str:
    """Людський опис запису (дзеркало підказок графіка `volAnText`)."""
    k = rec.get('kind')
    x = '×%.1f' % float(rec.get('rv') or 0)
    sh = rec.get('buy_share')
    bp = None if sh is None else int(round(float(sh) * 100))
    if k == 'absorb':
        if rec.get('hint') == 'dn':
            return f'покупці агресивні ({bp}%), обсяг {x}, але свічка ЧЕРВОНА — тиск поглинули продавці → ризик розвороту ВНИЗ'
        return f'продавці агресивні ({100 - bp}%), обсяг {x}, але свічка ЗЕЛЕНА — тиск поглинули покупці → можливий розворот ВГОРУ'
    if k == 'climax':
        return f'обсяг {x} від медіани 20 свічок — вершина / дно руху, ліквідації або новина' + \
            (f' · покупці {bp}%' if bp is not None else '')
    if k == 'side':
        if rec.get('hint') == 'up':
            return f'агресивні покупці {bp}% при обсязі {x} — односторонній тиск ВГОРУ'
        return f'агресивні продавці {100 - bp}% при обсязі {x} — односторонній тиск ВНИЗ'
    return f'сплеск обсягу {x} без явної переваги боку' + \
        (f' · покупці {bp}%' if bp is not None else '')


def bar_rows(bars: List[Dict]) -> List[Dict]:
    """Бари біржі → рядки детектора {t (сек), v (монети), up, o, h, l, c}."""
    out = []
    for b in bars or []:
        try:
            c = float(b.get('p', b.get('close')))
            o = float(b['o'])
            q = float(b.get('q') or 0)
            v = q if q > 0 else (float(b.get('v') or 0) / c if c > 0 else 0.0)
            out.append({'t': int(b['t']) // 1000 if int(b['t']) > 10 ** 11 else int(b['t']),
                        'v': v, 'up': c > o if c != o else None, 'buy': None,
                        'o': o, 'h': float(b['h']), 'l': float(b['l']), 'c': c})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def outcome_of(rows: List[Dict], bar_t: int, side_hint: Optional[str] = None) -> Dict:
    """Наслідок аномалії зі свічки `bar_t` по ЗАКРИТИХ рядках: рух закриття
    через 1/4/8 свічок (%), максимум угору / вниз за 8 свічок (%), `done` —
    уже є 8 свічок після неї."""
    idx = next((k for k, r in enumerate(rows) if r['t'] == int(bar_t)), None)
    if idx is None:
        return {}
    base = rows[idx]['c']
    if not base > 0:
        return {}
    out = {}
    for n in OUT_N:
        if idx + n < len(rows):
            out[f'o{n}'] = round((rows[idx + n]['c'] / base - 1) * 100, 3)
    nxt = rows[idx + 1: idx + 1 + max(OUT_N)]
    if nxt:
        out['hi'] = round((max(r['h'] for r in nxt) / base - 1) * 100, 3)
        out['lo'] = round((min(r['l'] for r in nxt) / base - 1) * 100, 3)
    out['done'] = idx + max(OUT_N) < len(rows)
    return out


def verdict_of(rec: Dict) -> Optional[str]:
    """«розворот» / «продовження» / None (ще рано / рух малий) — за рухом через
    4 свічки проти ТРЕНДУ ПЕРЕД аномалією. Поріг шуму — 0.2%."""
    o4, pre = rec.get('o4'), rec.get('pre_chg')
    if o4 is None or pre is None or abs(float(pre)) < 0.2 or abs(float(o4)) < 0.2:
        return None
    return 'reversal' if (float(o4) > 0) != (float(pre) > 0) else 'continuation'


def stats(rows: List[Dict]) -> List[Dict]:
    """Зведення для калібровки: на кожну пару (вид, TF) — скільки аномалій,
    з них із наслідком, частка «розворот» проти тренду перед нею, частка, де
    ціна пішла в бік підказки (для 🧽 / ⚡), середній рух через 4 / 8 свічок."""
    acc: Dict = {}
    for r in rows or []:
        if is_dir_only(r):
            continue                       # 🧭 лише для напрямку воріт — не дослідження
        key = (r.get('kind'), r.get('tf'))
        a = acc.setdefault(key, {'kind': key[0], 'tf': key[1], 'n': 0, 'done': 0,
                                 'rev': 0, 'cont': 0, 'hint_ok': 0, 'hint_n': 0,
                                 's4': 0.0, 's8': 0.0, 'n4': 0, 'n8': 0})
        a['n'] += 1
        if r.get('done'):
            a['done'] += 1
        v = verdict_of(r)
        if v == 'reversal':
            a['rev'] += 1
        elif v == 'continuation':
            a['cont'] += 1
        h, o4 = r.get('hint'), r.get('o4')
        if h and o4 is not None and abs(float(o4)) >= 0.2:
            a['hint_n'] += 1
            if (float(o4) > 0) == (h == 'up'):
                a['hint_ok'] += 1
        sg = 1.0
        if h == 'dn':
            sg = -1.0
        if o4 is not None:
            a['s4'] += sg * float(o4)
            a['n4'] += 1
        if r.get('o8') is not None:
            a['s8'] += sg * float(r['o8'])
            a['n8'] += 1
    out = []
    for a in acc.values():
        dec = a['rev'] + a['cont']
        out.append({'kind': a['kind'], 'tf': a['tf'], 'n': a['n'], 'done': a['done'],
                    'reversal_pct': round(a['rev'] / dec * 100, 1) if dec else None,
                    'decided': dec,
                    'hint_pct': round(a['hint_ok'] / a['hint_n'] * 100, 1) if a['hint_n'] else None,
                    'hint_n': a['hint_n'],
                    'avg4': round(a['s4'] / a['n4'], 3) if a['n4'] else None,
                    'avg8': round(a['s8'] / a['n8'], 3) if a['n8'] else None})
    out.sort(key=lambda x: (KINDS.index(x['kind']) if x['kind'] in KINDS else 9, x['tf'] or ''))
    return out


def relation(hint: Optional[str], side: str) -> Optional[str]:
    """Аномалія відносно угоди: 'with' / 'against' / None."""
    sd = str(side or '').upper()
    if not hint or sd not in ('LONG', 'SHORT'):
        return None
    return 'with' if (hint == 'up') == (sd == 'LONG') else 'against'


def journey_events(rows: List[Dict], side: str = None, entry: float = None) -> List[Dict]:
    """Записи аномалій → вузли «Шляху монети» (той самий формат, що
    `_trade_journey`: {t, event, detail, side, source, x})."""
    out = []
    sd = str(side or '').upper()
    for r in rows or []:
        try:
            t = float(r.get('ts') or 0)
        except (TypeError, ValueError):
            continue
        x = {k: r.get(k) for k in ('kind', 'tf', 'rv', 'buy_share', 'hint', 'price',
                                   'bar_chg', 'pre_chg', 'o1', 'o4', 'o8', 'hi', 'lo',
                                   'done', 'bar_t')}
        x['verdict'] = verdict_of(r)
        rel = relation(r.get('hint'), sd)
        x['rel'] = rel
        try:
            en = float(entry or 0)
            px = float(r.get('price') or 0)
            if en > 0 and px > 0 and sd in ('LONG', 'SHORT'):
                x['pnl_then'] = round((px / en - 1) * 100 * (1 if sd == 'LONG' else -1), 3)
        except (TypeError, ValueError):
            pass
        h = r.get('hint')
        out.append({'t': t, 'event': 'anomaly', 'detail': text_of(r),
                    'side': 'LONG' if h == 'up' else ('SHORT' if h == 'dn' else None),
                    'source': 'VOL ' + TF_LABEL.get(r.get('tf'), str(r.get('tf') or '')),
                    'x': x})
    return out


def is_dir_only(row: Dict) -> bool:
    """Аномалія монети, яку вели ЛИШЕ заради 🧭 напрямку воріт «Пружини» (роль
    `dir`): у статистику / калібровку дослідження вона не йде (вимога 17.11 —
    дослідження = заряджені монети + угоди)."""
    ctx = (row or {}).get('ctx')
    return isinstance(ctx, dict) and ctx.get('role') == 'dir'


def research_universe(trades, charged, seen, now, keep_sec=AFTER_CHARGE_SEC, extra=None):
    """🔬 Монети дослідження. ЧИСТА.
    trades  — {SYM: …} відкритих угод (вихід);
    charged — {SYM: 'wl'|'scan'} монет зі статусом «🌀 заряджена» ЗАРАЗ (вхід);
    seen    — {SYM: [src, коли востаннє заряджена]} (памʼять розряду).
    → (список монет, {SYM: ролі через «+»}, оновлений seen). Заряд зник, а
    угоди немає → монета лишається `keep_sec` (роль after), далі забувається."""
    out_seen = {}
    for S, v in (seen or {}).items():
        try:
            src, t = (v[0], float(v[1])) if isinstance(v, (list, tuple)) else ('wl', float(v))
        except (TypeError, ValueError, IndexError):
            continue
        if now - t <= keep_sec:
            out_seen[str(S).upper()] = [src, t]
    for S, src in (charged or {}).items():
        out_seen[str(S).upper()] = [src or 'wl', float(now)]
    roles: Dict = {}
    for S in (trades or {}):
        roles.setdefault(str(S).upper(), []).append('exit')
    cnow = {str(S).upper() for S in (charged or {})}
    for S in out_seen:
        roles.setdefault(S, []).append('entry' if S in cnow else 'after')
    # 🧭 монети, по яких потрібен лише НАПРЯМОК воріт «Пружини» (вимога 18.11):
    # роль `dir`, якщо в дослідженні їх ще немає.
    for S in (extra or ()):
        roles.setdefault(str(S).upper(), ['dir'])
    syms = list(roles.keys())
    return syms, {S: '+'.join(r) for S, r in roles.items()}, out_seen


# ───────────────────────── РУШІЙ ─────────────────────────
class VolAnomalyMonitor:
    def __init__(self, db=None, md=None, get_watchlist=None, get_trades=None,
                 get_charged=None, get_dir_syms=None):
        self.db = db
        self._md = md
        self._get_watchlist = get_watchlist
        self._get_trades = get_trades
        self._get_charged = get_charged
        self._get_dir_syms = get_dir_syms
        self._seen: Dict = {}            # SYM → [src, коли востаннє заряджена]
        self._roles: Dict = {}           # SYM → роль у дослідженні (останній прохід)
        self._seen_saved = 0.0
        self._seen_keys = ()
        self._done: Dict = {}            # (SYM, tf) → bar_t, оброблений останнім
        self._tried: Dict = {}           # (SYM, tf) → час останньої спроби
        self._lock = threading.Lock()
        self._settings = dict(DEFAULT_SETTINGS)
        self._last_purge = 0.0
        self._stat = {'last_at': 0, 'found': 0, 'checked': 0, 'errors': 0, 'last_err': ''}
        self._stop = False
        self._thread = None
        try:
            st = (db.get_setting(DB_SETTINGS, None) if db else None) or {}
            if isinstance(st, str):
                st = json.loads(st)
            self._settings.update({k: st[k] for k in DEFAULT_SETTINGS if k in st})
            self._settings['keep_days'] = keep_days_norm(self._settings.get('keep_days'))
        except Exception:
            pass
        try:
            sv = (db.get_setting(DB_CHARGED, None) if db else None) or {}
            if isinstance(sv, str):
                sv = json.loads(sv)
            self._seen = dict(sv.get('seen') or {}) if isinstance(sv, dict) else {}
        except Exception:
            self._seen = {}

    # налаштування
    def get_settings(self) -> Dict:
        return dict(self._settings)

    def update_settings(self, patch: Dict) -> Dict:
        if 'enabled' in patch:
            self._settings['enabled'] = bool(patch['enabled'])
        if 'keep_days' in patch:
            self._settings['keep_days'] = keep_days_norm(patch.get('keep_days'))
        try:
            if self.db:
                self.db.set_setting(DB_SETTINGS, dict(self._settings))
        except Exception:
            pass
        return self.get_settings()

    def state(self) -> Dict:
        rl = dict(self._roles)
        uni = {'entry': sum(1 for r in rl.values() if 'entry' in r),
               'after': sum(1 for r in rl.values() if 'after' in r),
               'exit': sum(1 for r in rl.values() if 'exit' in r),
               'dir': sum(1 for r in rl.values() if r == 'dir'), 'total': len(rl)}
        return {'settings': self.get_settings(), 'stat': dict(self._stat),
                'tfs': list(TFS), 'an': dict(AN), 'universe': uni,
                'after_sec': AFTER_CHARGE_SEC}

    # джерела
    def _md_get(self):
        if self._md is None:
            from detection.market_data import get_market_data
            self._md = get_market_data()
        return self._md

    def _trades(self) -> Dict:
        """{SYM: [{book, side, entry}]} — лише памʼять TM."""
        try:
            if self._get_trades:
                return self._get_trades() or {}
            from detection.trade_manager import get_trade_manager
            tm = get_trade_manager()
            if tm is None:
                return {}
            out: Dict = {}
            with tm._lock:
                items = [('R', k, v) for k, v in (tm._positions or {}).items()] + \
                    [('S', k, v) for k, v in (tm._shadow_positions or {}).items()]
            for bk, k, v in items:
                out.setdefault(str(k).upper(), []).append(
                    {'book': bk, 'side': v.get('side'), 'entry': v.get('entry_price')})
            return out
        except Exception:
            return {}

    def _charged_now(self) -> Dict:
        """{SYM: 'wl'|'scan'} — монети зі статусом «🌀 заряджена» обох Пружин
        (лише памʼять сканера і рушія 🔭; без мережі)."""
        if self._get_charged:
            try:
                return dict(self._get_charged() or {})
            except Exception:
                return {}
        out: Dict = {}
        try:
            from detection.smc_scanner import get_smc_scanner
            sc = get_smc_scanner()
            if sc is not None:
                wl = list(self._get_watchlist() if self._get_watchlist else sc.get_watchlist() or [])
                for S, r in (sc.spring_for(wl) or {}).items():
                    if (r or {}).get('state') == 'charged':
                        out[str(S).upper()] = 'wl'
        except Exception:
            pass
        try:
            import sys as _sys
            m = _sys.modules.get('detection.spring_scan')
            eng = m.get_spring_scan() if m and hasattr(m, 'get_spring_scan') else None
            for r in (eng.rows() if eng else []):
                if r.get('state') == 'charged' and r.get('bybit') is not False and r.get('symbol'):
                    out.setdefault(str(r['symbol']).upper(), 'scan')
        except Exception:
            pass
        return out

    def _dir_syms(self) -> List[str]:
        """🧭 Монети WATCHLIST, по яких ворота «Через Пружину-WL» судять напрямок
        за аномаліями — лише коли ці ворота УВІМКНЕНІ (інакше зайвих запитів
        до біржі не робимо)."""
        if self._get_dir_syms:
            try:
                return list(self._get_dir_syms() or [])
            except Exception:
                return []
        try:
            from detection.smc_scanner import get_smc_scanner
            sc = get_smc_scanner()
            st = (sc.get_settings() or {}) if sc is not None else {}
            if st.get('sprw_gate_enabled') and st.get('spring_enabled', True):
                return list(self._get_watchlist() if self._get_watchlist else sc.get_watchlist() or [])
        except Exception:
            pass
        return []

    def universe(self, now: Optional[float] = None) -> List[str]:
        """🔬 Заряджені монети обох Пружин (вхід) + ще AFTER_CHARGE_SEC після
        втрати заряду (розряд) + відкриті угоди (вихід)."""
        now = time.time() if now is None else now
        syms, roles, seen = research_universe(self._trades(), self._charged_now(),
                                              self._seen, now, extra=self._dir_syms())
        self._seen, self._roles = seen, roles
        keys = tuple(sorted(seen))
        if self.db and (keys != self._seen_keys or now - self._seen_saved >= 600):
            try:
                self.db.set_setting(DB_CHARGED, {'seen': seen})
                self._seen_keys, self._seen_saved = keys, now
            except Exception:
                pass
        return syms

    def _split(self, sym: str, tf: str, t_sec: int) -> Optional[float]:
        """Частка покупців свічки за підсвічками Bybit TF/10 або None."""
        try:
            from detection.imbalance import (pine_sub_minutes, split_from_subbars,
                                             sub_base_interval, group_subbars)
            bs = TF_SEC[tf]
            sub = pine_sub_minutes(bs)
            base = sub_base_interval(sub)
            t_ms = int(t_sec) * 1000
            md = self._md_get()
            r = md._session.get(BB_KLINE_URL, params={
                'category': 'linear', 'symbol': sym, 'interval': str(base),
                'start': t_ms, 'end': t_ms + bs * 1000 - 1, 'limit': 1000}, timeout=10)
            rows = ((r.json().get('result') or {}).get('list') or []) if r.status_code == 200 else []
            bars = [{'t': int(x[0]), 'o': float(x[1]), 'p': float(x[4]), 'v': float(x[5])}
                    for x in rows if t_ms <= int(x[0]) < t_ms + bs * 1000]
            if len(bars) < 2:
                return None
            b, s = split_from_subbars(group_subbars(bars, base, sub, t_ms))
            return (b / (b + s)) if (b + s) > 0 else None
        except Exception:
            return None

    def process(self, sym: str, tf: str, now: float) -> Optional[bool]:
        """Обробити останній закритий бар. → True (готово) / None (бар ще не
        віддано, повтор). Пише аномалію і наслідки попередніх."""
        bs = TF_SEC[tf]
        live = int(now // bs) * bs
        target = live - bs
        kl = self._md_get().fetch_klines(sym, limit=FETCH_BARS + 1, interval=tf) or []
        rows = [r for r in bar_rows(kl) if r['t'] < live]
        if not rows or rows[-1]['t'] < target:
            return None
        i = len(rows) - 1
        r = rows[i]
        # Підсвічки — лише коли обсяг уже ≥ ×1.5 (інакше 🧽 / ⚡ неможливі).
        prev = [x['v'] for x in rows[max(0, i - AN['base']):i] if x['v'] > 0]
        med = median(prev) if len(prev) >= AN['minBase'] else 0
        if med > 0 and r['v'] / med >= AN['rv'] and r['up'] is not None:
            r['buy'] = self._split(sym, tf, r['t'])
        a = anomaly_at(rows, i)
        self._stat['checked'] += 1
        if a:
            self._store(sym, tf, rows, i, a, now)
        self._update_outcomes(sym, tf, rows)
        return True

    def _store(self, sym, tf, rows, i, a, now):
        r = rows[i]
        pre = None
        if i - PRE_N >= 0 and rows[i - PRE_N]['c'] > 0:
            pre = round((r['c'] / rows[i - PRE_N]['c'] - 1) * 100, 3)
        rec = {
            'symbol': sym, 'tf': tf, 'bar_t': int(r['t']),
            'ts': float(r['t'] + TF_SEC[tf]), 'kind': a['k'],
            'hint': hint_of(a['k'], a.get('by'), a.get('dir')),
            'bar_dir': 'up' if r['up'] else ('dn' if r['up'] is False else None),
            'rv': round(a['rv'], 3),
            'buy_share': None if a.get('sh') is None else round(float(a['sh']), 4),
            'vol': r['v'], 'med': a.get('med'), 'price': r['c'],
            'bar_chg': round((r['c'] / r['o'] - 1) * 100, 3) if r['o'] > 0 else None,
            'pre_chg': pre,
            'ctx': {'trades': self._trades().get(sym) or [],
                    'role': self._roles.get(sym) or ('exit' if self._trades().get(sym) else ''),
                    'src': (self._seen.get(sym) or [None])[0]},
        }
        try:
            if self.db and hasattr(self.db, 'add_vol_anomaly'):
                self.db.add_vol_anomaly(rec)
                self._stat['found'] += 1
        except Exception as e:
            self._stat['errors'] += 1
            self._stat['last_err'] = str(e)[:200]

    def _update_outcomes(self, sym, tf, rows):
        if not self.db or not hasattr(self.db, 'get_vol_anomalies'):
            return
        try:
            since = rows[0]['t']
            pend = self.db.get_vol_anomalies(symbol=sym, tf=tf, undone=True,
                                             since=since, limit=50) or []
        except Exception:
            return
        for p in pend:
            o = outcome_of(rows, p.get('bar_t'))
            if not o:
                continue
            try:
                self.db.update_vol_anomaly(p['id'], o)
            except Exception:
                pass

    # цикл
    def tick(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        if not self._settings.get('enabled', True):
            return 0
        syms = self.universe(now)
        jobs = []
        for tf in TFS:
            bs = TF_SEC[tf]
            live = int(now // bs) * bs
            if now < live + SETTLE_SEC:
                continue
            target = live - bs
            for s in syms:
                k = (s, tf)
                if self._done.get(k) == target:
                    continue
                if now - self._tried.get(k, 0) < RETRY_SEC:
                    continue
                self._tried[k] = now
                jobs.append((s, tf, target))
        if not jobs:
            return 0

        def run(j):
            s, tf, target = j
            try:
                if self.process(s, tf, now):
                    self._done[(s, tf)] = target
            except Exception as e:
                self._stat['errors'] += 1
                self._stat['last_err'] = f'{s} {tf}: {e}'[:200]

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(run, jobs))
        self._stat['last_at'] = now
        if now - self._last_purge > 86400 and self.db and hasattr(self.db, 'purge_vol_anomalies'):
            self._last_purge = now
            try:
                self.db.purge_vol_anomalies(keep_days_norm(self._settings.get('keep_days')))
            except Exception:
                pass
        return len(jobs)

    def _loop(self):
        time.sleep(30)
        while not self._stop:
            try:
                self.tick()
            except Exception as e:
                print(f'[VOL-AN] tick error: {e}')
            time.sleep(2)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name='vol-anomaly', daemon=True)
        self._thread.start()


_instance: Optional[VolAnomalyMonitor] = None


def init_vol_anomaly(db=None, start_loop: bool = True, **kw) -> VolAnomalyMonitor:
    global _instance
    if _instance is None:
        _instance = VolAnomalyMonitor(db=db, **kw)
        if start_loop:
            _instance.start()
    return _instance


def get_vol_anomaly() -> Optional[VolAnomalyMonitor]:
    return _instance
