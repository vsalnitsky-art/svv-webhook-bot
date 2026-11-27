"""BotSetting-backed interface dictionaries with a versioned local seed."""
import hashlib
import json
import os
import threading
import time

from web.i18n import SUPPORTED_LANGUAGES, normalize_language

_SEED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'storage', 'i18n_seed.json')
_lock = threading.Lock()
_seed_cache = None


def _seed():
    global _seed_cache
    if _seed_cache is None:
        with _lock:
            if _seed_cache is None:
                with open(_SEED_PATH, 'r', encoding='utf-8') as handle:
                    _seed_cache = json.load(handle)
    return _seed_cache


def _key(lang):
    return 'i18n:dict:' + normalize_language(lang)


def _version(strings):
    raw = json.dumps(strings, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]


def _valid_strings(value):
    if not isinstance(value, dict):
        return None
    clean = {}
    for key, translated in value.items():
        if not isinstance(key, str) or len(key) > 4000:
            continue
        if isinstance(translated, str) and len(translated) <= 8000:
            clean[key] = translated
        elif key == '_phrases' and isinstance(translated, list):
            clean[key] = [x for x in translated if isinstance(x, str)][:4000]
    return clean


def get_dictionary(lang):
    lang = normalize_language(lang)
    if lang not in SUPPORTED_LANGUAGES:
        return None
    fallback = _valid_strings(_seed().get(lang)) or {}
    try:
        from storage.db_operations import get_db
        db = get_db()
        stored = db.get_setting(_key(lang))
        if stored is None:
            payload = {'strings': fallback, 'version': _version(fallback),
                       'updated_at': int(time.time()), 'source': 'seed'}
            db.set_setting(_key(lang), payload)
            stored = payload
        existing = _valid_strings(stored.get('strings') if isinstance(stored, dict) else None)
        if existing is None:
            existing = _valid_strings(stored) or {}
        # Seed updates add newly introduced interface strings without overwriting
        # translations already customised by an administrator in BotSetting.
        strings = dict(fallback)
        strings.update(existing)
        if strings != existing:
            payload = {'strings': strings, 'version': _version(strings),
                       'updated_at': int(time.time()), 'source': 'database+seed'}
            db.set_setting(_key(lang), payload)
        return {'ok': True, 'language': lang, 'strings': strings,
                'version': _version(strings), 'source': 'database'}
    except Exception as exc:
        return {'ok': True, 'language': lang, 'strings': fallback,
                'version': _version(fallback), 'source': 'local_fallback',
                'warning': str(exc)[:200]}


def save_dictionary(lang, strings):
    lang = normalize_language(lang)
    if lang not in SUPPORTED_LANGUAGES:
        return None
    clean = _valid_strings(strings)
    if clean is None:
        raise ValueError('dictionary_must_be_an_object')
    payload = {'strings': clean, 'version': _version(clean),
               'updated_at': int(time.time()), 'source': 'admin'}
    from storage.db_operations import get_db
    if not get_db().set_setting(_key(lang), payload):
        raise RuntimeError('dictionary_save_failed')
    return {'ok': True, 'language': lang, **payload}
