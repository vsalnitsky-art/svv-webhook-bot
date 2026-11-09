"""🔕 Зміна/продовження терміну доступу — БЕЗ повідомлення користувачу (09.11).

Дослівно: «Прибери ці повідомлення стосовно що продлили на 10 днів чи там на
30 днів. Саму таких повідомлень не потрібно.»
"""
import ast
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = open(os.path.join(_HERE, 'web', 'auth.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _set_access_branch():
    i = _SRC.index("elif action == 'set_access':")
    j = _SRC.index("elif action == 'kick':", i)
    return _SRC[i:j]


def test_set_access_sends_nothing_to_the_user():
    b = _set_access_branch()
    _check('Ваш доступ оновлено' not in b, 'повідомлення «Ваш доступ оновлено» прибрано')
    _check('notify_user_approved(uid, silent=True)' in b,
           '«Ваш акаунт активовано» на зміні терміну не шлеться')
    _check('_notify_change(admin_text=' in b, 'адміну аудит-рядок лишається')
    print('✓ зміна терміну доступу — без повідомлення користувачу')


def test_silent_still_restores_group_access():
    tree = ast.parse(_SRC)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == 'notify_user_approved')
    src = ast.unparse(fn)
    _check('silent' in [a.arg for a in fn.args.args], 'параметр silent')
    _check('if not silent' in src, 'повідомлення під умовою')
    _check('invite_user_to_group' in src and 'group_unrestrict' in src
           and "'ever_approved'" in src, 'група й позначка — як і раніше')
    print('✓ silent лише гасить текст, доступ до групи відновлюється')


def test_approve_still_notifies():
    i = _SRC.index('def approve_user(')
    j = _SRC.index('\ndef ', i + 5)
    _check('notify_user_approved(uid)' in _SRC[i:j], 'схвалення нового акаунта — як було')
    print('✓ схвалення нового акаунта повідомлення лишило')


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); n += 1
    print(f'\nOK — {n} тестів')
