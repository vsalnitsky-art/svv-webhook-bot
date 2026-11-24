import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Isolate the authentication code from the production application and database.
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

from flask import Flask
from web import auth


class AuthNavigationTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config.update(TESTING=True, SECRET_KEY='navigation-test-secret')
        auth.install_auth_gate(self.app)
        auth.register_auth_routes(self.app)
        self.user = SimpleNamespace(
            id=7,
            email='trader@example.com',
            is_admin=True,
            disabled=False,
            password_hash='test-hash',
            login_log='[]',
        )

    def _login(self, destination):
        with patch.object(auth, 'get_user_by_email', return_value=self.user), \
             patch.object(auth, 'check_password_hash', return_value=True), \
             patch.object(auth, '_is_active', return_value=True), \
             patch.object(auth, '_update_user'):
            return self.app.test_client().post(
                '/login?next=' + destination,
                data={'email': self.user.email, 'password': 'test-password'},
                follow_redirects=False,
            )

    def test_every_bot_login_lands_on_smart_money(self):
        for destination in ('/', '/smart-money', '/dashboard', '/tickr',
                            '/settings', 'https://example.com/unsafe'):
            with self.subTest(destination=destination):
                response = self._login(destination)
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.headers['Location'], '/smart-money')

    def test_info_site_token_handoff_is_preserved(self):
        response = self._login('/info-login')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/info-login')

    def test_active_bot_session_cannot_choose_another_start_page(self):
        with patch.object(auth, 'current_user', return_value=self.user), \
             patch.object(auth, '_is_active', return_value=True):
            response = self.app.test_client().get('/login?next=/dashboard')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/smart-money')

    def test_protected_pages_require_login_before_any_navigation(self):
        for path in ('/', '/smart-money', '/dashboard', '/tickr', '/settings', '/cabinet'):
            with self.subTest(path=path):
                response = self.app.test_client().get(path, follow_redirects=False)
                self.assertEqual(response.status_code, 302)
                self.assertTrue(response.headers['Location'].startswith('/login?next='))

    def test_only_public_shell_and_auth_routes_bypass_the_gate(self):
        for path in ('/smart-money', '/dashboard', '/tickr', '/settings', '/admin/users'):
            with self.subTest(path=path):
                self.assertFalse(auth._is_public(path))
        for path in ('/login', '/register', '/static/info/img/logo.png', '/info/'):
            with self.subTest(path=path):
                self.assertTrue(auth._is_public(path))


if __name__ == '__main__':
    unittest.main()
