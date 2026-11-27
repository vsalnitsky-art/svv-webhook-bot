import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

# The managed test Python has no web packages. Stub only their import surface;
# language detection and contact persistence below execute the real bot code.
flask = types.ModuleType('flask')
flask.session = {}
flask.request = Mock()
for name in ('redirect', 'url_for', 'jsonify', 'g', 'render_template_string', 'abort'):
    setattr(flask, name, Mock())
sys.modules['flask'] = flask
werkzeug = types.ModuleType('werkzeug')
werkzeug.__path__ = []
security = types.ModuleType('werkzeug.security')
security.generate_password_hash = Mock()
security.check_password_hash = Mock()
sys.modules['werkzeug'] = werkzeug
sys.modules['werkzeug.security'] = security
itsdangerous = types.ModuleType('itsdangerous')
itsdangerous.BadSignature = type('BadSignature', (Exception,), {})
itsdangerous.SignatureExpired = type('SignatureExpired', (Exception,), {})


class DummySerializer:
    def __init__(self, *args, **kwargs):
        pass

    def dumps(self, value):
        return 'signed-test-token'


itsdangerous.URLSafeTimedSerializer = DummySerializer
sys.modules['itsdangerous'] = itsdangerous

# Isolate Telegram onboarding from Flask startup and the production database.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
web_pkg = types.ModuleType('web')
web_pkg.__path__ = [os.path.join(ROOT, 'web')]
sys.modules['web'] = web_pkg
storage_pkg = types.ModuleType('storage')
storage_pkg.__path__ = [os.path.join(ROOT, 'storage')]
sys.modules['storage'] = storage_pkg
db_models = types.ModuleType('storage.db_models')
db_models.get_session = Mock()
db_models.User = object
sys.modules['storage.db_models'] = db_models


class MemoryDB:
    def __init__(self):
        self.values = {}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)

    def set_setting(self, key, value):
        self.values[key] = value


DB = MemoryDB()
db_operations = types.ModuleType('storage.db_operations')
db_operations.get_db = lambda: DB
sys.modules['storage.db_operations'] = db_operations

from web import auth, telegram_i18n, tg_bot


class TelegramLanguageDetectionTests(unittest.TestCase):
    def setUp(self):
        DB.values.clear()

    @staticmethod
    def message(language='en', chat_id=777):
        return {
            'text': '/start',
            'chat': {'id': chat_id, 'type': 'private'},
            'from': {
                'id': chat_id,
                'username': 'trader',
                'first_name': 'Test',
                'language_code': language,
            },
        }

    def start(self, language='en', send=None):
        send = send or Mock(return_value=True)
        with patch.object(auth, 'get_user_by_chat', return_value=None), \
             patch.object(tg_bot, '_admin_chat', return_value=''), \
             patch.object(tg_bot, '_link_begin', return_value=False), \
             patch.object(tg_bot, 'base_url', return_value='https://bot.example'), \
             patch.object(tg_bot, 'tg_send', send):
            tg_bot._handle_message(self.message(language))
        return send

    def test_first_start_uses_and_saves_regional_telegram_language(self):
        send = self.start('de-DE')

        self.assertIn('Willkommen', send.call_args.args[1])
        button = send.call_args.kwargs['buttons'][0][0]
        self.assertEqual(button['text'], '📝 Auf der Website registrieren')
        contact = DB.values['tg_bot_contacts']['777']
        self.assertEqual(contact['language'], 'de')
        self.assertEqual(contact['language_source'], 'telegram')

    def test_repeated_start_does_not_replace_saved_contact_language(self):
        self.start('en-GB')
        send = self.start('de-DE')

        self.assertEqual(DB.values['tg_bot_contacts']['777']['language'], 'en')
        self.assertIn('Welcome to VSV Bot', send.call_args.args[1])
        self.assertNotIn('Willkommen', send.call_args.args[1])

    def test_unregistered_contact_language_is_used_for_personal_messages(self):
        auth.record_bot_contact('777', 'trader', 'Test', 'es')

        with patch.object(auth, 'get_user_by_chat', return_value=None):
            self.assertEqual(telegram_i18n.language_for_chat('777'), 'es')

    def test_registered_profile_language_has_priority_over_telegram_contact(self):
        auth.record_bot_contact('777', 'trader', 'Test', 'de')
        user = SimpleNamespace(id=7, prefs=json.dumps({'language': 'en'}))

        with patch.object(auth, 'get_user_by_chat', return_value=user):
            self.assertEqual(telegram_i18n.language_for_chat('777'), 'en')

    def test_unsupported_telegram_language_falls_back_to_ukrainian(self):
        send = self.start('fr-FR')

        self.assertEqual(DB.values['tg_bot_contacts']['777']['language'], 'uk')
        self.assertIn('Вітаю у VSV Bot', send.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
