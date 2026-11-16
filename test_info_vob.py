"""📦 Volumized OB · інфосайт — окремі параметри детектора для графіка кабінету."""
import ast
from test_hl_peak_paper import _ip

IP = open('web/info_portal.py', encoding='utf-8').read()
SC = open('detection/smc_scanner.py', encoding='utf-8').read()
SM = open('templates/smart_money.html', encoding='utf-8').read()

BOT = {'volumized_swing_length': 8, 'volumized_ob_end_method': 'Wick',
       'volumized_max_atr_mult': 3.5, 'volumized_zone_count': 'One',
       'volumized_combine_obs': True}


def test_unset_keys_inherit_the_bot_block():
    ip = _ip()
    p = ip.info_vob_params(dict(BOT))
    assert (p['swing'], p['end'], p['atr'], p['zone'], p['combine']) == (8, 'Wick', 3.5, 'One', True)
    assert p['on'] is True and p['own'] is False
    assert ip.info_vob_params(None)['swing'] == 10                        # дефолти детектора


def test_own_params_are_separate_and_change_the_signature():
    ip = _ip()
    base = ip.info_vob_params(dict(BOT))
    p = ip.info_vob_params(dict(BOT, info_vob_swing_length=5, info_vob_zone_count='Medium',
                                info_vob_combine_obs=False))
    assert (p['swing'], p['zone'], p['combine']) == (5, 'Medium', False)
    assert p['own'] is True and p['pk'] != base['pk']
    assert ip.info_vob_params(dict(BOT, info_vob_swing_length=8))['own'] is False   # те саме число
    assert ip.info_vob_params(dict(BOT, info_vob_enabled=False))['on'] is False
    assert ip.info_vob_params(dict(BOT, info_vob_swing_length='x'))['swing'] == 10  # сміття → дефолт


def test_compute_and_payload_use_the_infosite_params():
    assert "swing_length=vp['swing'], ob_end_method=vp['end']" in IP
    assert "zone_count=vp['zone']" in IP and "combine_obs=vp['combine']" in IP
    assert "if vp['own']:" in IP                       # пам'ять сканера — лише за однакових параметрів
    assert "if vob_tf != OFF and vp['on']:" in IP      # вимкнено → VOB не рахується
    assert "d.get('pk') != vp['pk']" in IP             # чужий кеш не показуємо
    assert "_vp['on'], _vp['pk']" in IP                # і кеш відповіді теж


def test_bot_detector_does_not_read_infosite_keys():
    """Сигнали, ▲/▼ і бокс бота — лише volumized_*."""
    tree = ast.parse(SC)
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in ('_scan', '_latest_volumized_ob',
                                                            '_volumized_zones', 'get_chart_data'):
            assert 'info_vob_' not in ast.unparse(fn), fn.name


def test_settings_keys_whitelisted_and_ui_block():
    for k in ('info_vob_enabled', 'info_vob_swing_length', 'info_vob_ob_end_method',
              'info_vob_max_atr_mult', 'info_vob_zone_count', 'info_vob_combine_obs'):
        assert SC.count("'%s'" % k) >= 3, k            # дефолт · білий список · валідація
        assert k in SM
    assert SM.index('id="sm-volumized-summary"') < SM.index('📦 Volumized OB · інфосайт')
    for i in ('sm-ivob-enabled', 'sm-ivob-swing', 'sm-ivob-end-method', 'sm-ivob-atr-mult',
              'sm-ivob-zone-count', 'sm-ivob-combine'):
        assert 'id="%s"' % i in SM
    assert 'async function updateInfoVobSettings()' in SM


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); n += 1
    print('OK', n)
