import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Load the target modules without web/__init__.py, whose production import boots
# Flask and the real database.  Tests must never touch live state.
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

from web import auth
from web import tg_bot


class AccessNotificationTests(unittest.TestCase):
    def setUp(self):
        self.user = SimpleNamespace(
            id=7, email='trader', telegram_chat_id='777', is_admin=False,
            access_until=None, prefs=json.dumps({'language': 'de'}))

    def test_rejection_uses_registration_language(self):
        delivered = []
        with patch.object(auth, 'get_user_by_id', return_value=self.user), \
             patch.object(auth, '_access_delivery', side_effect=lambda uid, decision, text, buttons=None: delivered.append((uid, decision, text)) or True):
            self.assertTrue(auth.notify_user_rejected(7))
        self.assertEqual(delivered[0][1], 'rejected')
        self.assertIn('abgelehnt', delivered[0][2])
        self.assertNotIn('відхилено', delivered[0][2].lower())

    def test_approval_uses_profile_language_and_localized_login_link(self):
        self.user.prefs = json.dumps({'language': 'en'})
        fake = types.ModuleType('web.tg_bot')
        fake.base_url = lambda: 'https://bot.example'
        fake.tg_send = Mock(return_value=True)
        fake.last_send_error = lambda: ''
        fake.notify_category = Mock()
        fake.invite_user_to_group = Mock()
        fake.group_unrestrict = Mock()
        with patch.object(auth, 'get_user_by_id', return_value=self.user), \
             patch.object(auth, 'set_user_pref'), \
             patch.dict(sys.modules, {'web.tg_bot': fake}):
            self.assertTrue(auth.notify_user_approved(7))
        text = fake.tg_send.call_args.args[1]
        buttons = fake.tg_send.call_args.kwargs['buttons']
        self.assertIn('Your account is active', text)
        self.assertEqual(buttons[0][0]['text'], '🔐 Sign in')
        self.assertTrue(buttons[0][0]['url'].endswith('/login?lang=en'))
        fake.invite_user_to_group.assert_called_once_with('777', back=False)

    def test_api_failure_is_not_reported_as_delivered(self):
        fake = types.ModuleType('web.tg_bot')
        fake.tg_send = Mock(return_value=False)
        fake.last_send_error = lambda: 'Forbidden: bot was blocked by the user'
        fake.notify_category = Mock()
        with patch.object(auth, 'get_user_by_id', return_value=self.user), \
             patch.object(auth, 'set_user_pref') as save_pref, \
             patch.dict(sys.modules, {'web.tg_bot': fake}):
            self.assertFalse(auth._access_delivery(7, 'rejected', 'test'))
        audit = save_pref.call_args.args[2]
        self.assertFalse(audit['sent'])
        self.assertIn('Forbidden', audit['error'])
        self.assertIn('НЕ доставлено', fake.notify_category.call_args.args[1])

    def test_missing_telegram_is_audited_without_api_call(self):
        self.user.telegram_chat_id = None
        with patch.object(auth, 'get_user_by_id', return_value=self.user), \
             patch.object(auth, 'set_user_pref') as save_pref:
            self.assertFalse(auth._access_delivery(7, 'approved', 'test'))
        self.assertFalse(save_pref.call_args.args[2]['sent'])
        self.assertIn('Telegram', save_pref.call_args.args[2]['error'])

    def test_callback_rejection_sends_and_reports_delivery(self):
        cb = {'id': 'cb1', 'data': 'rj:7', 'from': {'id': 1},
              'message': {'chat': {'id': 1}, 'message_id': 9}}
        with patch.object(tg_bot, '_admin_chat', return_value=''), \
             patch.object(auth, 'get_user_by_id', return_value=self.user), \
             patch.object(auth, '_update_user') as update, \
             patch.object(auth, 'notify_user_rejected', return_value=True), \
             patch.object(tg_bot, 'group_access_sync'), \
             patch.object(tg_bot, '_answer_cb') as answer, \
             patch.object(tg_bot, '_edit') as edit:
            tg_bot._handle_callback(cb)
        update.assert_called_once_with(7, approved=False, disabled=True)
        self.assertIn('доставлено', answer.call_args.args[1])
        self.assertIn('користувача повідомлено', edit.call_args.args[2])

    def test_callback_approval_reports_failed_delivery(self):
        cb = {'id': 'cb2', 'data': 'ap:7', 'from': {'id': 1},
              'message': {'chat': {'id': 1}, 'message_id': 10}}
        with patch.object(tg_bot, '_admin_chat', return_value=''), \
             patch.object(auth, 'get_user_by_id', return_value=self.user), \
             patch.object(auth, 'approve_user', return_value=False), \
             patch.object(tg_bot, '_answer_cb') as answer, \
             patch.object(tg_bot, '_edit') as edit:
            tg_bot._handle_callback(cb)
        self.assertIn('НЕ доставлено', answer.call_args.args[1])
        self.assertTrue(answer.call_args.args[2])
        self.assertIn('НЕ доставлено', edit.call_args.args[2])


if __name__ == '__main__':
    unittest.main()
