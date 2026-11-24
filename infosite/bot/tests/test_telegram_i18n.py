import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
web_pkg = types.ModuleType('web')
web_pkg.__path__ = [os.path.join(ROOT, 'web')]
sys.modules['web'] = web_pkg

from web import telegram_i18n, tg_bot


class TelegramLocalizationTests(unittest.TestCase):
    def test_complete_personal_flows_exist_in_every_language(self):
        keys = ('start_new', 'start_registered', 'status_active', 'status_pending',
                'link_failed', 'link_ok', 'link_active', 'admin_reply',
                'support_sent', 'support_unavailable', 'login_site', 'register_site')
        for lang in telegram_i18n._EXTRA:
            for key in keys:
                with self.subTest(language=lang, key=key):
                    translated = telegram_i18n.telegram_tr(lang, key)
                    self.assertTrue(translated)
                    self.assertNotEqual(translated, key)

    def test_non_slavic_languages_have_no_ukrainian_residue(self):
        for lang in ('en', 'pl', 'de', 'es'):
            for key in telegram_i18n._EXTRA[lang]:
                with self.subTest(language=lang, key=key):
                    translated = telegram_i18n.telegram_tr(lang, key)
                    self.assertNotRegex(translated, r'[іїєґІЇЄҐ]')

    def test_html_market_values_and_button_markup_are_preserved(self):
        translated = telegram_i18n.telegram_tr(
            'de', 'start_registered', email='trader', status='✅ BTCUSDT +1.25%')
        self.assertIn('<b>trader</b>', translated)
        self.assertIn('✅ BTCUSDT +1.25%', translated)
        self.assertNotIn('<b>{email}</b>', translated)

        with patch.object(telegram_i18n, 'language_for_chat', return_value='en'):
            text, buttons = telegram_i18n.translate_chat_message(
                777, telegram_i18n.telegram_tr('uk', 'support_sent'),
                [[{'text': telegram_i18n.telegram_tr('uk', 'login_site'),
                   'url': 'https://example.test'}]])
        self.assertIn('administrator', text)
        self.assertEqual(buttons[0][0]['text'], '🔓 Sign in to the website')
        self.assertEqual(buttons[0][0]['url'], 'https://example.test')

    def test_unlinked_and_group_chats_keep_original_message(self):
        source = telegram_i18n.telegram_tr('uk', 'support_sent')
        with patch.object(telegram_i18n, 'language_for_chat', return_value=None):
            self.assertEqual(telegram_i18n.translate_chat_message(-100777, source), (source, None))

    def test_tg_send_passes_localized_html_and_buttons_to_bot_api(self):
        with patch.object(tg_bot, '_api', return_value={'ok': True}) as api, \
             patch.object(telegram_i18n, 'language_for_chat', return_value='es'):
            sent = tg_bot.tg_send(
                777, telegram_i18n.telegram_tr('uk', 'support_sent'),
                [[{'text': telegram_i18n.telegram_tr('uk', 'login_site'), 'callback_data': 'login'}]])
        self.assertTrue(sent)
        payload = api.call_args.args[1]
        self.assertIn('administrador', payload['text'])
        self.assertEqual(payload['parse_mode'], 'HTML')
        self.assertEqual(payload['reply_markup']['inline_keyboard'][0][0]['text'],
                         '🔓 Iniciar sesión en el sitio')

    def test_localization_failure_does_not_block_delivery(self):
        with patch.object(tg_bot, '_api', return_value={'ok': True}) as api, \
             patch.object(telegram_i18n, 'translate_chat_message', side_effect=RuntimeError('test')):
            self.assertTrue(tg_bot.tg_send(777, 'BTCUSDT <b>LONG</b>'))
        self.assertEqual(api.call_args.args[1]['text'], 'BTCUSDT <b>LONG</b>')


if __name__ == '__main__':
    unittest.main()
