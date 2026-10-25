"""Замок: `_update_btc_verdict` не читає змінних, яких не присвоює (кейс 25.10:
NameError `fdir` обривав такт двигуна на кожній події ₿-сеансу)."""
import ast, pathlib

SRC = pathlib.Path(__file__).with_name('detection').joinpath('fuel_filter.py').read_text(encoding='utf-8')


def _fn(name):
    for n in ast.walk(ast.parse(SRC)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(name)


def test_fdir_is_assigned_before_use():
    fn = _fn('_update_btc_verdict')
    stored = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert 'fdir' in loaded and 'fdir' in stored


def test_banner_refresh_keeps_raw_dir():
    assert 'self._btc_live_fdir' in ast.unparse(_fn('_btc_banner_live_refresh'))


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); print('✓', k)
