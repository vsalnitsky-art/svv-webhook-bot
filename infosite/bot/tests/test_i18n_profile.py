import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from flask import Flask

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
        return True


DB = MemoryDB()
db_operations = types.ModuleType('storage.db_operations')
db_operations.get_db = lambda: DB
sys.modules['storage.db_operations'] = db_operations

from web import auth, info_portal
from web.i18n_store import get_dictionary


class ProfileLanguageTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config.update(TESTING=True, SECRET_KEY='test')
        self.user = SimpleNamespace(id=5, email='user', prefs=json.dumps({'language': 'de'}),
                                    is_admin=False, disabled=False)
        with patch.object(info_portal, 'vsv_tg_start', return_value=None):
            info_portal.register_info_portal(self.app)
        auth.register_auth_routes(self.app)
        self.client = self.app.test_client()

    def test_seed_is_saved_in_bot_setting(self):
        DB.values.clear()
        result = get_dictionary('en')
        self.assertTrue(result['ok'])
        self.assertEqual(result['source'], 'database')
        self.assertIn('i18n:dict:en', DB.values)
        self.assertIn('Terminal', result['strings'].values())

    def test_public_dictionary_endpoint(self):
        response = self.client.get('/api/i18n/en')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['language'], 'en')
        self.assertTrue(response.get_json()['strings'])

    def test_profile_language_is_read_and_updated(self):
        with patch.object(auth, 'current_user', return_value=self.user), \
             patch.object(auth, 'set_user_pref') as save:
            response = self.client.get('/api/me/language')
            self.assertEqual(response.get_json()['language'], 'de')
            response = self.client.post('/api/me/language', json={'language': 'es'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['language'], 'es')
            save.assert_called_once_with(5, 'language', 'es')

    def test_language_endpoint_requires_profile(self):
        with patch.object(auth, 'current_user', return_value=None):
            self.assertEqual(self.client.get('/api/me/language').status_code, 401)
            self.assertEqual(self.client.post('/api/me/language', json={'language': 'en'}).status_code, 401)

    def test_dictionary_write_is_admin_only(self):
        with patch.object(auth, 'current_user', return_value=self.user):
            response = self.client.post('/api/i18n/en', json={'strings': {'Термінал': 'Terminal'}})
            self.assertEqual(response.status_code, 403)
        self.user.is_admin = True
        with patch.object(auth, 'current_user', return_value=self.user):
            response = self.client.post('/api/i18n/en', json={'strings': {'Термінал': 'Terminal'}})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['strings']['Термінал'], 'Terminal')

    def test_auth_form_contains_selected_language(self):
        with self.app.test_request_context('/login?lang=de'):
            html = auth._page('Login', auth._login_form(lang='de'), lang='de')
        self.assertIn('html lang="de"', html)
        self.assertIn('value="de" selected', html)
        self.assertIn('Anmelden', html)
        self.assertIn('VSV_AUTH_STRINGS', html)

    def test_i18n_routes_are_public_for_reading(self):
        self.assertTrue(auth._is_public('/api/i18n/en'))
        self.assertFalse(auth._is_public('/api/me/language'))

    def test_registration_saves_explicit_choice_not_device_locale(self):
        new_user = SimpleNamespace(id=55, prefs='{}')
        tg = types.ModuleType('web.tg_bot')
        tg.tg_send = Mock(return_value=True)
        with patch.object(auth, '_read_tg_token', return_value={
                'c': '999', 'u': 'trader', 'n': 'Trader', 'l': 'en', 'p': False}), \
             patch.object(auth, 'create_user', return_value=55), \
             patch.object(auth, 'get_user_by_email', return_value=None), \
             patch.object(auth, 'get_user_by_id', return_value=new_user), \
             patch.object(auth, '_update_user') as update, \
             patch.object(auth, 'notify_new_user_to_admin'), \
             patch.dict(sys.modules, {'web.tg_bot': tg}):
            response = self.client.post('/register?tg=signed', data={
                'login': 'trader', 'password': 'password8', 'password2': 'password8',
                'language': 'pl'})
        self.assertEqual(response.status_code, 200)
        prefs = json.loads(update.call_args.kwargs['prefs'])
        self.assertEqual(prefs['language'], 'pl')
        self.assertEqual(prefs['tg']['lang'], 'en')
        self.assertIn('Konto', response.get_data(as_text=True))
        self.assertIn('Konto', tg.tg_send.call_args.args[1])

    def test_login_choice_becomes_profile_preference(self):
        from werkzeug.security import generate_password_hash
        account = SimpleNamespace(id=8, email='trader', is_admin=False, disabled=False,
                                  password_hash=generate_password_hash('password8'))
        with patch.object(auth, 'current_user', return_value=None), \
             patch.object(auth, 'get_user_by_email', return_value=account), \
             patch.object(auth, '_throttled', return_value=False), \
             patch.object(auth, '_is_active', return_value=True), \
             patch.object(auth, 'login_user'), \
             patch.object(auth, 'set_user_pref') as save:
            response = self.client.post('/login', data={
                'email': 'trader', 'password': 'password8', 'language': 'es', 'next': '/'})
        self.assertEqual(response.status_code, 302)
        save.assert_called_once_with(8, 'language', 'es')


if __name__ == '__main__':
    unittest.main()
