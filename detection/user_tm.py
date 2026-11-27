"""👤 ОСОБИСТИЙ TRADE MANAGER КОРИСТУВАЧА — дзеркало бота (вимога 24.11).

Дослівно: «Trade Manager — має працювати окремо під кожного користувача… має
бути у кожного користувача свій і відображатись на інфосайті і мати чітко свій
алгоритм роботи… Тобто він має бути дзеркалом бота, але виконувати торгівлю саме
під конкретного користувача.»

• Це ТОЙ САМИЙ клас `TradeManager` (той самий алгоритм: розмір, авто-SL,
  правила виходу, автопілот, трейли, беззбиток, звірка з біржею), але окремий
  ЕКЗЕМПЛЯР на кожного користувача:
  – 🔐 ключі — ЛИШЕ власника (`web.user_keys`), конектор `BybitConnector.for_keys`
    (ключів бота не читає ніколи); немає ключів → реальна книга недоступна;
  – 💾 дані — `ScopedDb`: усі ключі БД TM з префіксом `u<uid>:` (налаштування,
    угоди real + paper, закриті); архів бота (`sob_trade_archive`, 🔬 дослідження)
    НЕ наповнює — статистика бота не змішується з чужою;
  – 🧾 лог — власний (`owner_scope`), у спільний 🧾 Лог бота не пише;
  – 📨 Telegram — ЛИШЕ особисто власнику, ні в групу, ні адміну;
  – ❤️ черги Fuel Filter, 🤖 ШІ, 🎓 еталони бота — не чіпає.
• 🪞 ДЗЕРКАЛО:
  – налаштування (`mirror_settings`, деф. УВІМК) — алгоритм бота копіюється в TM
    користувача, крім ОСОБИСТИХ ключів (`PERSONAL_KEYS`: увімкнення, Test Mode,
    розмір угоди, плече, ліміт, paper-депозит, Telegram);
  – угоди (`mirror_signals`, деф. ВИМК) — бот відкрив / закрив угоду → TM
    користувача повторює її на своєму рахунку в обраній книзі (`mirror_book`).
• Сигнали ⚠ «Аномалії обсягу» з кабінету йдуть сюди ж (`execute`), якщо
  користувач — не адміністратор (адміністратор = бот = ключі бота).
"""
import threading
import time
from collections import deque
from typing import Dict, Optional

from detection.trade_manager import TradeManager, DEFAULT_SETTINGS

PERSONAL_KEYS = ('enabled', 'test_mode', 'sizing_mode', 'fixed_usd_amount',
                 'fixed_pct_balance', 'risk_pct_balance', 'leverage',
                 'max_open_positions', 'paper_deposit', 'paper_sizing_mode',
                 'paper_margin_pct', 'paper_margin_usd', 'paper_leverage',
                 'paper_risk_pct', 'paper_fee_pct', 'telegram_alerts',
                 'test_telegram_alerts')
META_DEFAULTS = {'mirror_settings': True, 'mirror_signals': False, 'mirror_book': 'paper'}
META_KEY = 'utm_meta'
LOG_KEY = 'tm_user_log'
LOG_MAX = 300
LOG_FLUSH_SEC = 10.0
SYNC_SEC = 30.0
CHAT_TTL = 300.0
BOOKS_TTL = 30.0

_reg: Dict[int, 'UserTradeManager'] = {}
_reg_lock = threading.RLock()


class ScopedDb:
    """Обгортка БД: ключі налаштувань TM — свої на кожного користувача; архів
    бота (угоди, дослідження) TM користувача не наповнює."""
    NOOP = ('archive_trade', 'archive_research', 'update_archive_trade')

    def __init__(self, base, uid):
        self._b = base
        self._p = f'u{int(uid)}:'

    def key(self, k):
        return self._p + str(k)

    def get_setting(self, k, default=None):
        if self._b is None:
            return default
        return self._b.get_setting(self.key(k), default)

    def set_setting(self, k, v):
        if self._b is None:
            return None
        return self._b.set_setting(self.key(k), v)

    def __getattr__(self, n):
        if n in ScopedDb.NOOP:
            return lambda *a, **k: None
        return getattr(self._b, n)


def personal_only(s: Dict) -> Dict:
    """ЧИСТА: лише особисті ключі налаштувань."""
    return {k: s[k] for k in PERSONAL_KEYS if k in (s or {})}


def mirror_merge(own: Dict, bot: Dict) -> Dict:
    """ЧИСТА: алгоритм бота + особисті ключі користувача (вони сильніші)."""
    out = dict(own or {})
    for k, v in (bot or {}).items():
        if k in DEFAULT_SETTINGS and k not in PERSONAL_KEYS:
            out[k] = v
    return out


def meta_norm(m) -> Dict:
    m = m if isinstance(m, dict) else {}
    out = dict(META_DEFAULTS)
    for k in ('mirror_settings', 'mirror_signals'):
        if k in m:
            out[k] = bool(m[k])
    if m.get('mirror_book') in ('paper', 'real'):
        out['mirror_book'] = m['mirror_book']
    return out


def _bot_tm():
    try:
        from detection.trade_manager import get_trade_manager
        return get_trade_manager()
    except Exception:
        return None


def _base_db():
    try:
        from storage.db_operations import get_db
        return get_db()
    except Exception:
        return None


def _scanner():
    bt = _bot_tm()
    return getattr(bt, 'scanner', None) if bt is not None else None


def _connector_for(uid):
    try:
        from web import user_keys
        k, s = user_keys.load(uid)
    except Exception:
        k, s = None, None
    if not (k and s):
        return None
    try:
        from core.bybit_connector import BybitConnector
        return BybitConnector.for_keys(k, s)
    except Exception as e:
        print(f'[UTM] u{uid} connector: {e}')
        return None


class UserTradeManager(TradeManager):
    """Trade Manager конкретного користувача (див. докстрінг модуля)."""

    def __init__(self, uid: int, email: str = '', db=None, scanner=None, bybit=None):
        self.owner_uid = int(uid)
        self.owner_email = email or ''
        self._owner_tag = f'#u{int(uid)}'
        self._ulog = deque(maxlen=LOG_MAX)
        self._ulog_dirty = False
        self._ulog_flush_at = 0.0
        self._log_sink = self._log_ev
        self._personal_notify = self._send_personal
        self._chat = (None, 0.0)
        self._books = ((), 0.0)
        self._sync_at = 0.0
        sdb = ScopedDb(db, uid) if db is not None else None
        self._sdb = sdb
        super().__init__(db=sdb, notifier=None, bybit=bybit, scanner=scanner)
        self._meta = meta_norm(sdb.get_setting(META_KEY, {}) if sdb else {})
        try:
            raw = sdb.get_setting(LOG_KEY, []) if sdb else []
            for ev in (raw or [])[-LOG_MAX:]:
                if isinstance(ev, dict):
                    self._ulog.append(ev)
        except Exception:
            pass
        self.mirror_sync(force=True)

    # ── ізоляція ──────────────────────────────────────────────────────────
    @staticmethod
    def _ai_owned(pos) -> bool:
        return False          # ШІ належить боту; угоди користувача ведуть правила TM

    def _is_tradeable(self, symbol: str) -> bool:
        return True           # TRD — список бота; користувач обирає монети сам

    def _fresh_settings(self) -> Dict:
        cfg = super()._fresh_settings()
        bt = _bot_tm()
        if bt is not None:
            try:
                cfg = mirror_merge(cfg, bt.get_settings())
            except Exception:
                pass
        cfg.update({'enabled': False, 'test_mode': False})
        return cfg

    # ── лог / Telegram власника ────────────────────────────────────────────
    def _log_ev(self, ev: Dict) -> None:
        self._ulog.append(ev)
        self._ulog_dirty = True
        now = time.time()
        if now - self._ulog_flush_at >= LOG_FLUSH_SEC:
            self.flush_log()

    def flush_log(self) -> None:
        if not self._ulog_dirty or self._sdb is None:
            return
        self._ulog_flush_at = time.time()
        self._ulog_dirty = False
        try:
            self._sdb.set_setting(LOG_KEY, list(self._ulog))
        except Exception:
            pass

    def user_log(self, limit: int = 100):
        return list(self._ulog)[-int(limit or 100):][::-1]

    def log(self, symbol, event, detail, side=None, source='UTM'):
        from detection.activity_log import owner_scope, log_activity
        with owner_scope(self._log_sink):
            log_activity(symbol, event, detail, side=side, source=source)

    def _chat_id(self):
        cid, at = self._chat
        if time.time() - at < CHAT_TTL:
            return cid
        cid = None
        try:
            from web.auth import get_user_by_id
            u = get_user_by_id(self.owner_uid)
            cid = getattr(u, 'telegram_chat_id', None) if u else None
        except Exception:
            cid = None
        self._chat = (cid, time.time())
        return cid

    def _send_personal(self, msg: str) -> None:
        cid = self._chat_id()
        if not cid:
            return
        try:
            from web.tg_bot import tg_send
            tg_send(cid, '👤 <b>Мій Trade Manager</b>\n' + str(msg))
        except Exception as e:
            print(f'[UTM{self._owner_tag}] tg: {e}')

    # ── дзеркало налаштувань бота ──────────────────────────────────────────
    def meta(self) -> Dict:
        return dict(self._meta)

    def set_meta(self, upd: Dict) -> Dict:
        self._meta = meta_norm({**self._meta, **(upd or {})})
        if self._sdb is not None:
            try:
                self._sdb.set_setting(META_KEY, self._meta)
            except Exception:
                pass
        self.mirror_sync(force=True)
        return self.meta()

    def mirror_sync(self, force: bool = False) -> bool:
        if not self._meta.get('mirror_settings'):
            return False
        now = time.time()
        if not force and now - self._sync_at < SYNC_SEC:
            return False
        self._sync_at = now
        bt = _bot_tm()
        if bt is None or bt is self:
            return False
        try:
            merged = mirror_merge(self._settings, bt.get_settings())
        except Exception:
            return False
        with self._lock:
            if merged == self._settings:
                return False
            self._settings = merged
        self._persist_settings()
        return True

    def update_personal(self, new: Dict) -> Dict:
        """Зміни з кабінету: при увімкненому дзеркалі — лише особисті ключі."""
        new = dict(new or {})
        if self._meta.get('mirror_settings'):
            new = personal_only(new)
        return self.update_settings(new)

    def _tick(self):
        try:
            self.mirror_sync()
        except Exception:
            pass
        super()._tick()
        self.flush_log()

    # ── права ──────────────────────────────────────────────────────────────
    def perms_reset(self):
        self._books = ((), 0.0)

    def _perms(self):
        bks, at = self._books
        if time.time() - at < BOOKS_TTL:
            return bks
        out = {'books': (), 'mirror': False}
        try:
            from web.auth import get_user_by_id
            from web.user_trading import _perms_mod
            u = get_user_by_id(self.owner_uid)
            if u is not None and not getattr(u, 'disabled', False):
                P = _perms_mod()
                pe = P.of_user(u)
                out = {'books': tuple(P.books(pe)), 'mirror': bool(P.mirror_ok(pe))}
        except Exception:
            pass
        self._books = (out, time.time())
        return out

    def books(self):
        """🛡 Книги, які адміністратор дозволив користувачу."""
        return self._perms()['books']

    def mirror_allowed(self) -> bool:
        """🛡 Адміністратор дозволив дублювання угод бота."""
        return self._perms()['mirror']

    def set_bybit(self, conn) -> None:
        with self._lock:
            self.bybit = conn
            self.__dict__.pop('_wallet_cache', None)
            self._live_positions = {}
            self._live_positions_at = 0.0


# ── реєстр ────────────────────────────────────────────────────────────────
def get(uid) -> Optional[UserTradeManager]:
    try:
        return _reg.get(int(uid))
    except (TypeError, ValueError):
        return None


def has_keys(uid) -> bool:
    """🔐 Користувач підключив СВОЇ ключі API (лише запис, без розшифрування)."""
    try:
        from web import user_keys
        return user_keys._record(uid) is not None
    except Exception:
        return False


def approved(uid) -> bool:
    """🛡 Адміністратор дозволив користувачу торгівлю (хоча б одну книгу)."""
    try:
        from web.auth import get_user_by_id
        from web.user_trading import user_books
        u = get_user_by_id(uid)
        return bool(u is not None and not getattr(u, 'is_admin', False)
                    and not getattr(u, 'disabled', False) and user_books(u))
    except Exception:
        return False


def has_book_data(uid) -> bool:
    """У БД лишились відкриті угоди цього TM (ключі могли прибрати)."""
    db = _base_db()
    if db is None:
        return False
    try:
        return bool(db.get_setting(f'u{int(uid)}:tm_positions', {}) or
                    db.get_setting(f'u{int(uid)}:tm_shadow_positions', {}))
    except Exception:
        return False


def ensure(uid, email: str = '') -> Optional[UserTradeManager]:
    """TM користувача. ЗʼЯВЛЯЄТЬСЯ ЛИШЕ ПІСЛЯ ПІДКЛЮЧЕННЯ КЛЮЧІВ API (вимога
    24.11); без ключів — лише якщо лишились угоди, які треба довести."""
    try:
        uid = int(uid)
    except (TypeError, ValueError):
        return None
    with _reg_lock:
        t = _reg.get(uid)
        if t is not None:
            return t
    # 🛡 Спершу СХВАЛЕННЯ адміністратора (право на книгу), потім ключі власника.
    # Без схвалення TM живе лише заради угод, що лишились (їх треба довести).
    if not ((approved(uid) and has_keys(uid)) or has_book_data(uid)):
        return None
    with _reg_lock:
        t = _reg.get(uid)
        if t is not None:
            return t
        t = UserTradeManager(uid, email, db=_base_db(), scanner=_scanner(),
                             bybit=_connector_for(uid))
        _reg[uid] = t
    try:
        t.start()
    except Exception as e:
        print(f'[UTM] u{uid} start: {e}')
    return t


def all_tms():
    with _reg_lock:
        return list(_reg.values())


def on_keys_changed(uid) -> None:
    t = get(uid)
    if t is not None:
        t.set_bybit(_connector_for(uid))
    elif has_keys(uid):
        try:
            from web.auth import get_user_by_id
            u = get_user_by_id(uid)
            ensure(uid, getattr(u, 'email', '') or '')
        except Exception as e:
            print(f'[UTM] u{uid} start after keys: {e}')


def drop(uid) -> None:
    with _reg_lock:
        t = _reg.pop(int(uid), None)
    if t is not None:
        try:
            t.stop()
            t.flush_log()
        except Exception:
            pass


def execute(uid, email, symbol, side, book, note='', opened_by='anom') -> Dict:
    """Сигнал користувача → ЙОГО Trade Manager (книга перевіряється правами)."""
    t = ensure(uid, email)
    if t is None:
        return {'ok': False, 'status': 'rejected', 'mode': book,
                'reason': 'підключіть свої ключі API — без них Trade Manager не працює'}
    if book not in t.books():
        return {'ok': False, 'status': 'rejected', 'mode': book,
                'reason': f'книга {book} не дозволена адміністратором'}
    if book == 'real' and t.bybit is None:
        return {'ok': False, 'status': 'rejected', 'mode': book,
                'reason': 'ваші ключі біржі не задано — реальну угоду не відкрити'}
    from detection.activity_log import owner_scope
    with owner_scope(t._log_sink):
        try:
            return t.cab_signal_trade(symbol, side, book, opened_by=opened_by, note=note)
        except Exception as e:
            return {'ok': False, 'status': 'rejected', 'mode': book, 'reason': str(e)[:200]}


def hand_open(uid, email, symbol, side, book) -> Dict:
    """✋ Користувач сам відкриває угоду у СВОЄМУ Trade Manager."""
    return execute(uid, email, symbol, side, book, note='✋ вручну з кабінету',
                   opened_by='manual')


def hand_add(uid, email, symbol, book) -> Dict:
    """➕ Користувач сам ДОДАЄ до відкритої угоди (розмір — як у нової угоди).
    Рішення людини — без вимоги «кращої ціни» сигнального усереднення."""
    t = ensure(uid, email)
    S = str(symbol or '').upper().strip()
    if t is None or book not in ('paper', 'real'):
        return {'ok': False, 'reason': 'невірні параметри'}
    if book not in t.books():
        return {'ok': False, 'reason': f'книга {book} не дозволена адміністратором'}
    is_shadow = book == 'paper'
    with t._lock:
        p = (t._shadow_positions if is_shadow else t._positions).get(S)
        sd = str((p or {}).get('side') or '').upper()
    if not p:
        return {'ok': False, 'reason': f'{S}: угоди в цій книзі немає'}
    if book == 'real' and t.bybit is None:
        return {'ok': False, 'reason': 'ваші ключі біржі не задано'}
    px = t._get_current_price(S)
    if not px:
        return {'ok': False, 'reason': f'{S}: немає ціни'}
    from detection.activity_log import owner_scope
    with owner_scope(t._log_sink):
        r = t._cab_average(S, sd, is_shadow, float(px))
        if r.get('ok'):
            n = int((r.get('pos') or {}).get('cab_avg_n') or 0)
            t.log(S, 'opened', f'✋ ➕ додано до угоди {sd} ({book}) по {t._fmt_price(px)} · '
                  f'додавання #{n} · середній вхід '
                  f'{t._fmt_price((r.get("pos") or {}).get("entry_price"))}', side=sd)
            return {'ok': True, 'avg_n': n}
        t.log(S, 'rejected', f'✋ ➕ додати не вдалось — {r.get("reason")}', side=sd)
        return {'ok': False, 'reason': r.get('reason') or 'не вдалось'}


def hand_close(uid, email, symbol, book) -> Dict:
    t = ensure(uid, email)
    S = str(symbol or '').upper().strip()
    if t is None:
        return {'ok': False, 'reason': 'TM недоступний'}
    from detection.activity_log import owner_scope
    with owner_scope(t._log_sink):
        try:
            return (t.manual_close_shadow(S) if book == 'paper' else t.manual_close(S)) or {}
        except Exception as e:
            return {'ok': False, 'reason': str(e)[:200]}


def mirror_plan(meta: Dict, books, kind: str, is_paper: bool,
                allowed: bool = True) -> Optional[str]:
    """ЧИСТА: у яку книгу TM користувача повторює подію угоди бота (або None).
    Потрібні: дозвіл адміністратора (`allowed`) + тумблер користувача + книга."""
    m = meta_norm(meta)
    if not allowed or not m['mirror_signals'] or kind not in ('open', 'close'):
        return None
    bk = m['mirror_book']
    return bk if bk in (books or ()) else None


def _mirror_one(t: UserTradeManager, kind: str, rec: Dict, is_paper: bool) -> None:
    bk = mirror_plan(t.meta(), t.books(), kind, is_paper, t.mirror_allowed())
    S = str(rec.get('symbol') or '').upper()
    sd = str(rec.get('side') or '').upper()
    if not bk or not S or sd not in ('LONG', 'SHORT'):
        return
    src = 'paper' if is_paper else 'real'
    from detection.activity_log import owner_scope
    with owner_scope(t._log_sink):
        if kind == 'open':
            if bk == 'real' and t.bybit is None:
                t.log(S, 'rejected', '🪞 Дзеркало: угоду бота не повторено — ваші ключі не задано',
                      side=sd)
                return
            try:
                from detection.signal_labels import compose, signal_code_of
                ob = compose(signal_code_of(rec.get('opened_by')) or 'manual', 'MIR')
            except Exception:
                ob = 'manual → MIR'
            r = t.cab_signal_trade(S, sd, bk, opened_by=ob, note='🪞 дзеркало бота')
            if r.get('status') in ('opened', 'reversed'):
                with t._lock:
                    book = t._shadow_positions if bk == 'paper' else t._positions
                    p = book.get(S)
                    if p is not None:
                        p['mirror_bot'] = {'src': src, 'at': time.time()}
                (t._persist_shadow_positions if bk == 'paper' else t._persist_positions)()
            return
        # close: закриваємо лише угоду, відкриту дзеркалом із цієї книги бота
        with t._lock:
            p = (t._shadow_positions if bk == 'paper' else t._positions).get(S)
            ok = bool(p and (p.get('mirror_bot') or {}).get('src') == src
                      and str(p.get('side')).upper() == sd)
        if not ok:
            return
        try:
            if bk == 'paper':
                t.manual_close_shadow(S, reason='mirror_close')
            else:
                t.manual_close(S, reason='mirror_close')
            t.log(S, 'closed', '🪞 Дзеркало: бот закрив угоду — закрито і вашу', side=sd)
        except Exception as e:
            t.log(S, 'rejected', f'🪞 Дзеркало: закриття не вдалось — {e}', side=sd)


def mirror_event(kind: str, rec: Dict, is_paper: bool) -> None:
    tms = [t for t in all_tms() if t.meta().get('mirror_signals')]
    if not tms:
        return

    def _run():
        for t in tms:
            try:
                _mirror_one(t, kind, rec, is_paper)
            except Exception as e:
                print(f'[UTM] mirror u{t.owner_uid}: {e}')
    threading.Thread(target=_run, daemon=True, name='utm-mirror').start()


def boot(delay: float = 20.0) -> None:
    """Підняти TM користувачів, яким адміністратор дав торгівлю, і тих, у кого
    лишились угоди (права могли зняти — угоди все одно треба вести)."""
    def _run():
        time.sleep(delay)
        try:
            from web.auth import get_session, User
            from web.user_trading import user_books
        except Exception as e:
            print(f'[UTM] boot: {e}')
            return
        s = get_session()
        try:
            users = [(u.id, u.email, bool(u.is_admin), tuple(user_books(u)))
                     for u in s.query(User).all()]
        except Exception as e:
            print(f'[UTM] boot users: {e}')
            users = []
        finally:
            s.close()
        n = 0
        for uid, email, adm, bks in users:
            if adm:
                continue
            if (bks and has_keys(uid)) or has_book_data(uid):
                if ensure(uid, email) is not None:
                    n += 1
        print(f'[UTM] 👤 особистих Trade Manager: {n}')
    threading.Thread(target=_run, daemon=True, name='utm-boot').start()
