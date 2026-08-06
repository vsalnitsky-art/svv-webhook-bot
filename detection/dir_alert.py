"""
dir_alert — ПЕР-МОНЕТНІ Telegram-оповіщення про ЗМІНУ НАПРЯМКУ (1H / 4H).

Живить панель «Smart Direction» на /smart-money: під кожною монетою оператор
може окремо УВІМКНУТИ оповіщення (за замовчуванням ВИМКНЕНО). Коли ввімкнено,
фоновий потік періодично рахує прогноз-напрямок для цієї монети на 1H і 4H
(той самий forecast-движок, що й бейджі «🔮 1H / 🔮 4H») і, щойно напрямок
ФЛІПНЕТЬСЯ (LONG↔SHORT) на 1H АБО 4H, шле ОДИН Telegram у топік 🎯 Напрямок.

Едж-тригер: алерт лише на ФАКТИЧНІЙ зміні збереженого напрямку ТФ на новий
НЕ-нейтральний бік (не щотіку). Кулдаун per-coin гасить дрижання.

Стан персиститься в БД (`sm_dir_alert_state`), тож перелік увімкнених монет і
останній зафіксований напрямок переживають рестарт бота.

Визначення напрямку/сили ТФ (ЄДИНЕ, використовується і тут, і в UI-підписі):
  side  = forecast_1h/4h['side'] (1=LONG, -1=SHORT, 0=немає)
  conf  = forecast_1h/4h['confidence'] (0..100)
  сила  : conf ≥ 66 → «сильний», 40..65 → «помірний», <40 → «слабкий».
"""

import time
import json
import threading
from typing import Optional, Dict, List

CYCLE_SECS = 90            # як часто перевіряти напрямок увімкнених монет
COOLDOWN_SECS = 300        # мін. пауза між алертами по одній монеті
_DB_STATE = 'sm_dir_alert_state'   # {SYM: {enabled, last_1h, last_4h, last_alert_ts}}


def strength_word(conf) -> str:
    """Підпис сили за впевненістю прогнозу (%). Спільне визначення з UI."""
    try:
        c = float(conf or 0)
    except (TypeError, ValueError):
        c = 0.0
    if c >= 66:
        return 'сильний'
    if c >= 40:
        return 'помірний'
    return 'слабкий'


def _side_word(side) -> str:
    return 'LONG' if side == 1 else ('SHORT' if side == -1 else '—')


class DirAlertDaemon:
    def __init__(self, db):
        self._db = db
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ── персист-стан ────────────────────────────────────────────────────
    def _load(self) -> Dict:
        try:
            raw = self._db.get_setting(_DB_STATE, {}) or {}
            if isinstance(raw, str):
                raw = json.loads(raw or '{}')
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _save(self, state: Dict):
        try:
            self._db.set_setting(_DB_STATE, state)
        except Exception as e:
            print(f"[DirAlert] save error: {e}")

    # ── публічне API ────────────────────────────────────────────────────
    def is_enabled(self, symbol: str) -> bool:
        sym = (symbol or '').upper().strip()
        with self._lock:
            return bool((self._load().get(sym) or {}).get('enabled'))

    def set_enabled(self, symbol: str, on: bool) -> Dict:
        """Увімкнути/вимкнути оповіщення для однієї монети. При ВВІМКНЕННІ
        одразу знімаємо поточний напрямок як базу (щоб перший же тік не дав
        фальшивий «фліп»)."""
        sym = (symbol or '').upper().strip()
        if sym.endswith('.P'):
            sym = sym[:-2]
        if not sym:
            return {'ok': False, 'reason': 'no symbol'}
        with self._lock:
            state = self._load()
            rec = state.get(sym) or {}
            rec['enabled'] = bool(on)
            if on:
                s1, s4 = self._current_sides(sym)
                rec['last_1h'] = s1
                rec['last_4h'] = s4
                rec.setdefault('last_alert_ts', 0)
            state[sym] = rec
            self._save(state)
        if on and (self._thread is None or not self._thread.is_alive()):
            self.start()
        return {'ok': True, 'symbol': sym, 'enabled': bool(on)}

    def list_enabled(self) -> List[str]:
        with self._lock:
            return sorted(s for s, r in self._load().items()
                          if isinstance(r, dict) and r.get('enabled'))

    def status(self) -> Dict:
        with self._lock:
            state = self._load()
        coins = {}
        for s, r in state.items():
            if isinstance(r, dict) and r.get('enabled'):
                coins[s] = {'last_1h': r.get('last_1h', 0),
                            'last_4h': r.get('last_4h', 0),
                            'last_alert_ts': r.get('last_alert_ts', 0)}
        return {'enabled_coins': sorted(coins.keys()), 'coins': coins,
                'running': bool(self._thread and self._thread.is_alive())}

    # ── обчислення напрямку ─────────────────────────────────────────────
    def _current_sides(self, symbol: str):
        """(side_1h, side_4h) з forecast-движка (1/-1/0). Свіже (on-demand)."""
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if not fe:
                return 0, 0
            fc = fe.ensure_fresh(symbol, max_age=CYCLE_SECS) or fe.get(symbol) or {}
            f1 = (fc.get('forecast_1h') or {})
            f4 = (fc.get('forecast_4h') or {})
            s1 = f1.get('side') if f1.get('side') in (1, -1) else 0
            s4 = f4.get('side') if f4.get('side') in (1, -1) else 0
            return s1, s4
        except Exception:
            return 0, 0

    def _forecast_pair(self, symbol: str):
        """Повні forecast-дані 1H/4H (для тексту алерта)."""
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if not fe:
                return {}, {}
            fc = fe.get(symbol) or {}
            return (fc.get('forecast_1h') or {}), (fc.get('forecast_4h') or {})
        except Exception:
            return {}, {}

    # ── петля ───────────────────────────────────────────────────────────
    def _run(self):
        self._stop.wait(10)      # дати синглтонам догрузитись
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[DirAlert] tick error: {e}")
            self._stop.wait(CYCLE_SECS)

    def _tick(self):
        coins = self.list_enabled()
        if not coins:
            return
        now = time.time()
        for sym in coins:
            try:
                s1, s4 = self._current_sides(sym)
            except Exception:
                continue
            with self._lock:
                state = self._load()
                rec = state.get(sym) or {}
                if not rec.get('enabled'):
                    continue
                prev1 = rec.get('last_1h', 0)
                prev4 = rec.get('last_4h', 0)
                # Едж: фліп ТФ на новий НЕ-нейтральний бік.
                flip1 = s1 in (1, -1) and s1 != prev1
                flip4 = s4 in (1, -1) and s4 != prev4
                changed = []
                if flip1:
                    changed.append(('1Н', _side_word(prev1), _side_word(s1)))
                if flip4:
                    changed.append(('4Н', _side_word(prev4), _side_word(s4)))
                # Завжди тримаємо останній НЕ-нейтральний бік як базу.
                if s1 in (1, -1):
                    rec['last_1h'] = s1
                if s4 in (1, -1):
                    rec['last_4h'] = s4
                cooldown_ok = (now - float(rec.get('last_alert_ts', 0) or 0)) >= COOLDOWN_SECS
                fire = bool(changed) and cooldown_ok
                if fire:
                    rec['last_alert_ts'] = now
                state[sym] = rec
                self._save(state)
            if fire:
                try:
                    self._send_alert(sym, s1, s4, changed)
                except Exception as e:
                    print(f"[DirAlert] send error {sym}: {e}")

    def _send_alert(self, sym: str, s1: int, s4: int, changed: list):
        f1, f4 = self._forecast_pair(sym)
        c1 = int(f1.get('confidence') or 0)
        c4 = int(f4.get('confidence') or 0)

        def _line(lbl, side, conf):
            if side == 1:
                dot, word = '🟢', 'LONG'
            elif side == -1:
                dot, word = '🔴', 'SHORT'
            else:
                return f"🔮 {lbl}: ⚪ немає напрямку"
            return f"🔮 {lbl}: {dot} {word} {conf}% ({strength_word(conf)})"

        chg = ' · '.join(f"{tf} {old}→{new}" for tf, old, new in changed)
        body = (f"🎯 <b>Зміна напрямку</b> · #{sym}\n"
                f"{_line('1Н', s1, c1)}\n"
                f"{_line('4Н', s4, c4)}\n"
                f"↻ Змінилось: {chg}")
        try:
            from web.tg_bot import notify_category
            notify_category('signal', body)
        except Exception as e:
            print(f"[DirAlert] tg error {sym}: {e}")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='dir-alert')
        self._thread.start()
        print("[DirAlert] daemon started")


_instance: Optional[DirAlertDaemon] = None


def init_dir_alert(db) -> DirAlertDaemon:
    """Синглтон + автостарт, якщо в БД уже є хоч одна ввімкнена монета
    (переживає рестарт бота)."""
    global _instance
    if _instance is None:
        _instance = DirAlertDaemon(db)
        try:
            if _instance.list_enabled():
                _instance.start()
                print("[DirAlert] restored enabled coins from DB — loop running")
        except Exception:
            pass
    return _instance


def get_dir_alert() -> Optional[DirAlertDaemon]:
    return _instance
