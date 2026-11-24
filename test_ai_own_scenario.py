"""🤖 ШІ — ВЛАСНИЙ СЦЕНАРІЙ · 🏆🧠 ЕТАЛОН = ГОЛОВНИЙ МОЗОК · 🌀 заряд на виході ·
🧹 автоочистка з позначенням · 🎯 вибір для інфосайту · фірмова іконка (22.11).

Запуск: python3 test_ai_own_scenario.py
"""
import importlib.util
import os
import sys
import tempfile
import types

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
_DBF = os.path.join(tempfile.mkdtemp(), 'ai_clean.db')
os.environ['DATABASE_URL'] = 'sqlite:///' + _DBF
for n in ('pybit', 'pybit.unified_trading'):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules['pybit.unified_trading'].HTTP = object
if 'detection' not in sys.modules:
    _pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
    sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


_lg = types.ModuleType('detection.activity_log'); _lg.log_activity = lambda *a, **k: None
sys.modules['detection.activity_log'] = _lg
A = _load('detection.ai_signals', 'detection/ai_signals.py')
B = _load('detection.ai_brain', 'detection/ai_brain.py')
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
_TM = open(os.path.join(_ROOT, 'detection', 'trade_manager.py'), encoding='utf-8').read()
_JS = open(os.path.join(_ROOT, 'infosite', 'js', 'app.js'), encoding='utf-8').read()
_IP = open(os.path.join(_ROOT, 'web', 'info_portal.py'), encoding='utf-8').read()
NOW = 1_800_000_000.0


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _feat(x):
    return {k: x for k in A.FEAT_KEYS}


# ── налаштування ─────────────────────────────────────────────────────────
def test_settings_norm_and_migrate():
    s = A.ai_settings_norm({'scenarios': ['e9'], 'book': 'xx', 'max_open': 999, 'etalon_min_p': 5})
    _check(s['scenarios'] == ['E2'] and s['book'] == 'auto' and s['max_open'] == 50
           and s['etalon_min_p'] == 0.95, s)
    _check(A.AI_DEFAULTS['scenarios'] == ['E2'] and A.AI_DEFAULTS['enabled'] is False,
           'деф.: лише «💥 Старт розряду», ШІ вимкнено')
    m = A.ai_settings_migrate({'ai_signal_enabled': True, 'ai_min_charge': 70},
                              {'use_ai_exit': False, 'ai_exit_mode': 'all'})
    # вихід ШІ — для ВЛАСНИХ угод, тож старий тумблер TM на нього не переноситься
    _check(m['enabled'] and m['min_charge'] == 70 and m['exit_enabled'] and m['exit_mode'] == 'all', m)


def test_scen_allowed_and_limits():
    r = {'scenario': {'key': 'E2'}}
    _check(A.scen_allowed(r, ['E2']) and not A.scen_allowed(r, ['E1']), 'дозволені сценарії')
    s = A.ai_settings_norm({'allow_short': False, 'max_open': 2, 'max_per_30m': 2, 'flip_cooldown_min': 60})
    _check(A.ai_limits_block('SHORT', s, 0, [], None, NOW), 'SHORT вимкнено')
    _check(A.ai_limits_block('LONG', s, 2, [], None, NOW), 'стеля одночасних')
    _check(A.ai_limits_block('LONG', s, 0, [NOW - 10, NOW - 20], None, NOW), 'стеля за 30 хв')
    _check(A.ai_limits_block('LONG', s, 0, [], ['SHORT', NOW - 600], NOW), 'фліп-кулдаун')
    _check(not A.ai_limits_block('LONG', s, 0, [], ['LONG', NOW - 600], NOW), 'той самий бік — можна')


# ── 🏆🧠 ГОЛОВНИЙ МОЗОК ───────────────────────────────────────────────────
def test_etalon_brain_learns_then_decides():
    good = [{'feat': _feat(0.8), 'side': 'LONG', 'r': 2.0, 'peak_r': 3.0, 'sym': f'G{i}'} for i in range(10)]
    anti = [{'feat': _feat(0.2), 'side': 'LONG', 'r': -1.0, 'peak_r': 0.2, 'sym': f'B{i}'} for i in range(5)]
    eb = A.etalon_brain(_feat(0.8), good[:3], anti[:2], 'LONG', {})
    _check(not eb['mature'] and eb['verdict'] == 'learning', f'мало досвіду → вирішують правила: {eb}')
    eb = A.etalon_brain(_feat(0.8), good, anti, 'LONG', {'etalon_min_p': 0.6, 'etalon_min_sim': 0.3})
    _check(eb['mature'] and eb['verdict'] == 'take' and eb['p_good'] == 1.0 and eb['tp_r'] == 2.4, eb)
    eb = A.etalon_brain(_feat(0.2), good, anti, 'LONG', {'etalon_min_p': 0.6, 'etalon_min_sim': 0.3})
    _check(eb['mature'] and eb['verdict'] == 'skip' and 'невдалими' in eb['reason'], eb)


def test_etalon_plan_target_from_experience():
    plan = {'stop': 99.0, 'target': 102.0, 'risk_pct': 1.0, 'reward_pct': 2.0}
    p = A.etalon_plan(plan, 'LONG', {'mature': True, 'tp_r': 3.0})
    _check(abs(p['target'] - 103.0) < 1e-6 and p['target_src'] == 'etalon', p)
    _check(A.etalon_plan(plan, 'LONG', {'mature': False, 'tp_r': 3.0}) is plan, 'не зрілий — план правил')
    p = A.etalon_plan({'stop': 101.0, 'target': 98.0, 'risk_pct': 1.0}, 'SHORT', {'mature': True, 'tp_r': 2.0})
    _check(abs(p['target'] - 98.0) < 1e-6, p)


def test_rank_is_led_by_the_etalon_brain_when_mature():
    v = {'ready': False, 'pick': {}, 'brain': {}, 'etalon': {'mature': True, 'p_good': 1.0}}
    v2 = dict(v, etalon={'mature': True, 'p_good': 0.0})
    _check(A.ai_rank(v, None) > A.ai_rank(v2, None) + 40, 'мозок еталонів — головна складова рангу')


def test_record_carries_all_useful_info():
    closed = {'symbol': 'XUSDT', 'side': 'LONG', 'opened_at': NOW - 3600, 'closed_at': NOW,
              'pnl_pct': 3.0, 'peak_pnl_pct': 4.0, 'mae_pnl_pct': -0.5, 'reason': 'ai_exit',
              'entry_price': 100, 'exit_price': 103, 'ai_feat': _feat(0.7), 'ai_scenario': 'E2',
              'ai_plan': {'stop': 99, 'target': 104, 'risk_pct': 1.0, 'reward_pct': 4.0},
              'ai_entry': {'scenario': {'key': 'E2'}, 'spring': {'charge': 80}},
              'ai_exit': {'verdict': 'target', 'verdict_ua': '🎯 ціль', 'checklist': [{'label': 'x', 'v': 1}]},
              'research': {'ev': [{'t': NOW - 1800, 'k': 'fire', 'note': '💥'}]},
              'research_summary': {'first': {'fire': {'pnl': 3.5}}}}
    rec = A.etalon_record(closed)
    for k in ('feat', 'entry', 'plan', 'exit', 'events', 'lessons', 'peak_r', 'mae_r', 'capture', 'dur_sec'):
        _check(rec.get(k) not in (None, [], {}), f'запис еталона без «{k}»: {rec}')
    _check(rec['r'] == 3.0 and rec['peak_r'] == 4.0 and rec['capture'] == 0.75 and rec['dur_sec'] == 3600, rec)
    _check(rec['events'][0]['dt'] == 1800 and any('піку' in x for x in rec['lessons']), rec)


class _Db:
    def __init__(self):
        self.store, self.cleaned = {}, []

    def get_setting(self, k, d=None):
        return self.store.get(k, d)

    def set_setting(self, k, v):
        self.store[k] = v

    def get_vol_anomalies(self, **kw):
        return []

    def cleanup_ai_learning(self, now, dry=False, **kw):
        self.cleaned.append((dry, kw))
        return {'episodes': 1, 'anomalies': 2, 'research': 3}


def _eng():
    db = _Db()
    e = A.AiEngine(db=db, scanner=types.SimpleNamespace(get_settings=lambda: {}), tm=None, clock=lambda: NOW)
    e._ai_s = A.ai_settings_norm({'etalon_enabled': True}); e._ai_s_at = 1e15
    return e, db


def test_consider_fills_both_bases():
    e, db = _eng()
    base = {'symbol': 'XUSDT', 'side': 'LONG', 'opened_at': NOW - 600, 'ai_feat': _feat(0.5),
            'ai_plan': {'risk_pct': 1.0}}
    _check(e.etalon_consider(dict(base, pnl_pct=2.0, closed_at=NOW)), 'відмінна → еталон')
    _check(e.etalon_consider(dict(base, pnl_pct=-0.8, closed_at=NOW + 1)), 'невдала → антибаза')
    _check(not e.etalon_consider(dict(base, pnl_pct=0.3, closed_at=NOW + 2)), 'середня — нікуди')
    _check(len(db.store[A.ETALON_DB]) == 1 and len(db.store[A.ETALON_ANTI_DB]) == 1, db.store.keys())
    st = e.etalon_state()
    _check(st['n'] == 1 and st['anti_n'] == 1 and st['mature'] is False, st)
    _check(e.etalon_clear() == 2 and not e._etalon and not e._anti, 'очистка обох баз')


def test_cleanup_marks_then_deletes():
    e, db = _eng()
    e._cleanup_at = NOW - 60                      # щойно чистили → лише позначення
    e._etalon = [{'feat': _feat(0.5), 't': NOW - 200 * 86400}, {'feat': _feat(0.5), 't': NOW}]
    _check(e._cleanup(NOW) is None and db.cleaned and db.cleaned[-1][0] is True, 'позначення = dry')
    _check(e._cleanup_mark.get('research') == 3 and e._cleanup_mark.get('etalon') == 1
           and len(e._etalon) == 2, e._cleanup_mark)
    res = e._cleanup(NOW + A.CLEANUP_EVERY_SEC)
    _check(res and db.cleaned[-1][0] is False and res['etalon'] == 1 and len(e._etalon) == 1, res)
    _check(e.state()['cleanup_rules_ua'], 'правила видно в панелі')


def test_cleanup_on_sqlite():
    from storage.db_models import Base, engine, AiEpisode, VolAnomaly, TradeResearch
    Base.metadata.create_all(engine, tables=[AiEpisode.__table__, VolAnomaly.__table__, TradeResearch.__table__])
    from storage.db_operations import DBOperations, get_session
    se = get_session()
    d = 86400
    se.add_all([
        AiEpisode(symbol='A', ended_at=NOW - 10 * d, signal_side=None),        # без сигналу, 10 дн → геть
        AiEpisode(symbol='B', ended_at=NOW - 10 * d, signal_side='LONG'),      # з сигналом, 10 дн → лишається
        AiEpisode(symbol='C', ended_at=NOW - 50 * d, signal_side='LONG'),      # 50 дн → геть
        VolAnomaly(symbol='A', ts=NOW - 3 * d, ctx='{"role": "dir"}', done=False),
        VolAnomaly(symbol='B', ts=NOW - 3 * d, ctx='{"role": "entry"}', done=True),
        VolAnomaly(symbol='C', ts=NOW - 70 * d, ctx='{"role": "entry"}', done=True),
        TradeResearch(symbol='A', closed_at=NOW - 40 * d, opened_by='vob_alert → Q4'),
        TradeResearch(symbol='B', closed_at=NOW - 40 * d, opened_by='ai'),
        TradeResearch(symbol='C', closed_at=NOW - 70 * d, opened_by='ai'),
    ])
    se.commit(); se.close()
    db = DBOperations.__new__(DBOperations)
    dry = DBOperations.cleanup_ai_learning(db, NOW, dry=True, **A.CLEANUP_RULES)
    # 22.11: дата — лише для сміття (без сигналу / 🧭 dir / не-ШІ угоди / відпрацьовані
    # аномалії); сигнальні епізоди і дослідження угод ШІ чистяться за ЯКІСТЮ, не за датою.
    _check(dry == {'episodes': 1, 'anomalies': 2, 'research': 1}, f'позначено: {dry}')
    se = get_session()
    _check(se.query(AiEpisode).count() == 3, 'dry нічого не видаляє'); se.close()
    res = DBOperations.cleanup_ai_learning(db, NOW, **A.CLEANUP_RULES)
    _check(res == dry, f'видалено те, що позначено: {res}')
    se = get_session()
    _check({x.symbol for x in se.query(AiEpisode)} == {'B', 'C'}
           and {x.symbol for x in se.query(VolAnomaly)} == {'B'}
           and {x.symbol for x in se.query(TradeResearch)} == {'B', 'C'}, 'лишилось потрібне')
    se.close()


# ── 🌀 заряд на виході ───────────────────────────────────────────────────
def test_charge_holds_weak_exit_but_not_protective():
    anom = {'verdict': 'close', 'close': True, 'groups': {'anom': True, 'price': True}, 'checklist': []}
    try:
        m = B.exit_merge('LONG', dict(anom), None, 'confirmed', charge={'pot': 70, 'min': 40})
    except Exception as e:
        raise AssertionError(f'exit_merge: {e}')
    if m.get('verdict') in B.CHARGE_HOLDABLE + ('charged',):
        _check(m['verdict'] == 'charged' and not m.get('close'), f'заряд ще є → вихід зарано: {m}')
    _check('charged' in B.EXIT_VERDICT_UA and 'invalid' not in B.CHARGE_HOLDABLE
           and 'target' not in B.CHARGE_HOLDABLE, 'захисне закриває завжди')


# ── TM: власні угоди ШІ ──────────────────────────────────────────────────
def test_tm_ai_owned_and_stop():
    _check("def _ai_owned(pos)" in _TM and "def ai_open(" in _TM and "def _ai_exit_tick(" in _TM, 'TM')
    _check("pos['ai_entry'] = entry" in _TM, 'знімок входу лягає на позицію')
    _check("'ai_entry': pos.get('ai_entry')" in _TM and "'ai_exit': pos.get('ai_exit')" in _TM,
           'на закритті в еталон іде і вхід, і вихід')
    _check("AI_TG_MARK = '🤖 <b>VSV·AI</b>'" in _TM and _TM.count('ai_tg_line(') >= 4,
           'маркер ШІ в Telegram (відкриття + 2 закриття)')


# ── маркування фірмовою іконкою ─────────────────────────────────────────
def test_brand_icon_everywhere():
    _check("const AI_BRAND_IMG = '<img class=\"ai-brand\" src=\"/static/vsv-logo.png\"" in _HTML, 'бот: іконка')
    _check("signalCodeOf(raw) === 'ai'" in _HTML and '${AI_BRAND_IMG}${_AIX_ICON' in _HTML, 'угоди + 🚨')
    _check("x === 'AI' ? aiBrand(" in _HTML, '🧾 лог')
    _check('id="aip-body"' in _HTML and "togglePanel('aip')" in _HTML, 'панель ШІ')
    _check('AI_BRAND' in _JS and 'rec-ai' in _JS, 'інфосайт: рекомендовані')
    _check('def rec_ai_merge' in _IP, 'вибір ШІ для «Рекомендовані ботом»')
    _check("'🤖 <b>VSV·AI</b>'" in open(os.path.join(_ROOT, 'detection', 'ai_signals.py'), encoding='utf-8').read(),
           'TG ⚠ аномалій теж марковано')



# ── 22.11 (друга правка): очистка за ЯКІСТЮ, головний тумблер, VSV SIGNALS ──
def test_etalon_prune_drops_worst_not_oldest():
    old_good = {'sym': 'OLD', 'side': 'LONG', 'r': 3.0, 'capture': 0.8, 'feat': _feat(0.2), 't': NOW - 300 * 86400}
    new_bad = {'sym': 'NEW', 'side': 'LONG', 'r': 1.5, 'capture': 0.1, 'feat': _feat(0.9), 't': NOW}
    plan = A.etalon_prune_plan([old_good, new_bad], 'good', keep=1)
    _check([i for i, _ in plan] == [1], f'старий, але відмінний лишається; свіжий слабкий — геть: {plan}')
    twin = dict(old_good, sym='TWIN', r=2.0, t=NOW)
    plan = A.etalon_prune_plan([old_good, twin], 'good', keep=10)
    _check([i for i, _ in plan] == [1] and 'дубль' in plan[0][1], f'дубль гіршого входу: {plan}')
    _check(A.etalon_quality({'r': -2.0}, 'anti') > A.etalon_quality({'r': -0.6}, 'anti'),
           'антибаза: глибший збиток — цінніший урок')


def test_learning_prune_keeps_informative_and_protected():
    rows = [{'id': 1, 'sym': 'A', 't': NOW, 'pnl': 0.05, 'peak': 0.1, 'mae': 0.05},
            {'id': 2, 'sym': 'B', 't': NOW - 90 * 86400, 'pnl': 4.0, 'peak': 6.0, 'mae': 1.0},
            {'id': 3, 'sym': 'C', 't': NOW - 1, 'pnl': 0.0, 'peak': 0.0, 'mae': 0.0}]
    _check(A.learning_prune_plan(rows, 'research', keep=1) == [1, 3], 'нудні геть, показовий лишається')
    _check(sorted(A.learning_prune_plan(rows, 'research', keep=1, protected={('C', int(NOW - 1))})) == [1, 2],
           'запис еталона захищено')
    _check(A.learning_prune_plan(rows, 'research', keep=5) == [], 'у межах ліміту — нічого')


def test_heartbeat_tells_what_ai_is_doing():
    view = {'XUSDT': {'symbol': 'XUSDT', 'src': 'wl', 'ready': True, 'best': 'LONG',
                      'pick': {'side': 'LONG', 'scenario': {'key': 'E2', 'ready': True}}},
            'YUSDT': {'symbol': 'YUSDT', 'src': 'scan', 'ready': False, 'best': 'SHORT',
                      'pick': {'side': 'SHORT', 'scenario': {'key': 'E2', 'done': 2, 'total': 3}}}}
    txt, key = A.ai_heartbeat_text(view, {'enabled': True}, {}, 0, 0)
    _check('перебрано 2' in txt and 'X ▲ E2' in txt and 'Y ▼ E2 2/3' in txt, txt)
    _check(key == ('X ▲ E2',), f'ключ змін — набір готових точок: {key}')
    _check('ai_scan' in _HTML and "SOLO_EVENTS = new Set(['ob_new', 'ai_scan', 'xspr_in'])" in _HTML,
           'огляд ШІ — окремим рядком у 🧾 Лозі')
    _check("_SOLO = {'ob_new', 'ai_scan', 'xspr_in'}" in open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read(),
           'експорт логу розуміє те саме')


def test_readiness_names_the_blocker():
    s = dict(A.AI_DEFAULTS, enabled=True, book='auto')
    r = A.ai_readiness(s, spring_on=False, tm_on=True, test_mode=True)
    _check(r['state'] == 'blocked' and 'Пружина' in r['text'], r)
    r = A.ai_readiness(s, True, False, False)
    _check(r['state'] == 'blocked' and 'Test Mode' in r['text'], r)
    _check(A.ai_readiness(dict(s, book='paper'), True, True, False)['state'] == 'blocked', 'paper без Test Mode')
    tk = {'last_at': NOW}
    _check(A.ai_readiness(s, True, False, True, coins=7, tick=tk)['state'] == 'watching', 'auto → paper')
    _check(A.ai_readiness(s, True, True, True, coins=7, ready=2, tick=tk)['state'] == 'ready', 'є точки')
    _check(A.ai_readiness(dict(s, max_open=2), True, True, True, open_n=2)['state'] == 'full', 'ліміт')
    _check(A.ai_readiness(dict(s, enabled=False), True, True, True)['state'] == 'off', 'вимкнено')


def test_master_toggle_on_the_main_bar():
    _check('id="ai-master-box"' in _HTML and 'id="ai-master-on"' in _HTML, 'тумблер у рядку Trade Direction')
    _check(_HTML.index('id="ai-master-box"') < _HTML.index('id="sbr-alarm-box"'), 'поруч із будильником')
    _check("function aiMasterToggle(el)" in _HTML and "ck.checked = !!el.checked; aipSave()" in _HTML,
           'той самий шлях збереження, що й панель')
    _check("aiMasterRender(a, d.readiness)" in _HTML and "/api/ai/state?light=1" in _HTML, 'стан і перешкоди')
    fa = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    _check("request.args.get('light')" in fa and "'readiness': eng.readiness()" in fa, 'легкий стан без БД')


def test_vsv_signals_only_ai_with_exchange():
    import ast
    tree = ast.parse(_IP)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rec_ai_only')
    ns = {}
    exec(compile(ast.Module([fn], []), 'rec_ai_only', 'exec'), ns)
    rows = [{'symbol': 'A', 'side': 'LONG', 'src': 'wl', 'ai': {'ready': True}},
            {'symbol': 'B', 'side': 'SHORT', 'src': 'scan', 'ai': {'ready': True}},
            {'symbol': 'C', 'side': 'LONG', 'src': 'wl', 'ai': {'ready': False}},
            {'symbol': 'D', 'side': 'LONG', 'src': 'wl'},
            {'symbol': 'E', 'side': 'SHORT', 'src': 'ai', 'ai': {'ready': True, 'src': 'scan'}}]
    out = ns['rec_ai_only'](rows, 'binance')
    _check([(r['symbol'], r['ex']) for r in out] == [('A', 'bybit'), ('B', 'binance'), ('E', 'binance')], out)
    idx = open(os.path.join(_ROOT, 'infosite', 'index.html'), encoding='utf-8').read()
    _check('VSV SIGNALS' in idx and 'Рекомендовані ботом</span>' not in idx, 'у шапці лише VSV SIGNALS')
    head = idx[idx.index('id="rec-tbl"'):idx.index('</table>', idx.index('id="rec-tbl"'))]
    _check('colspan="8"' in head, '8 колонок')
    rr = _JS[_JS.index('function recRow(r)'):_JS.index('function recRender()')]
    _check('SPB_ICON[ex]' in rr and 'rec-ai-slot' in rr, 'біржа замість ШІ🌀, слот «ВХІД»')
    css = open(os.path.join(_ROOT, 'infosite', 'css', 'style.css'), encoding='utf-8').read().replace(' ', '')
    _check('#rec-tbl{table-layout:fixed;width:100%}' in css and '.rec-ai-slot{flex:0066px' in css, 'у ширину екрана')


def test_scenario_view_carries_signal_price_and_time():
    S = _load('detection.anomaly_scenarios', 'detection/anomaly_scenarios.py')
    src = open(os.path.join(_ROOT, 'detection', 'anomaly_scenarios.py'), encoding='utf-8').read()
    _check("'last_px'" in src and "'px': _f((it or {}).get('price'))" in src, 'ціна реакції на етапі')
    _check('sc.last_px' in _HTML and 'function _aiEventT(c)' in _HTML and 'table-layout:fixed' in _HTML,
           'таблиця ШІ: ціна і час сигналу, два рядки, сортування за появою')
    _check(hasattr(S, 'view'), 'view')



# ── 22.11 (третя правка): «SL з» ШІ · впевненість · чому точка не стала угодою ──
def test_stop_source_and_bounds():
    s = dict(A.AI_DEFAULTS)
    plan = {'stop': 99.86, 'target': 104.0, 'target_src': 'spring', 'cost_pct': 0.1}
    p = A.ai_stop_for('LONG', 100.0, plan, None, '', dict(s, sl_source='plan'))
    _check(abs(p['stop'] - 99.2) < 1e-9 and p['risk_pct'] == 0.8 and 'мінімуму' in p['stop_label'],
           f'тісний стоп 0.14% (кейс BZ) відсунуто до 0.8%: {p}')
    p = A.ai_stop_for('LONG', 100.0, plan, 98.5, '1H OB', dict(s, sl_source='ob'))
    _check(p['stop'] == 98.5 and p['stop_label'].startswith('1H OB') and p['target'] == 104.0, p)
    p = A.ai_stop_for('SHORT', 100.0, {'stop': 100.2}, 109.0, '1H OB', dict(s, sl_source='ob'))
    _check(abs(p['stop'] - 104.0) < 1e-9 and 'стелі' in p['stop_label'], f'задалекий OB → стеля 4%: {p}')
    _check(abs(p['target'] - (100 - 2.2 * 4)) < 1e-9, 'ціль 2.2R тягнеться за новим стопом')
    p = A.ai_stop_for('LONG', 100.0, plan, None, '', dict(s, sl_source='ob'))
    _check('OB не знайдено' in p['stop_label'], 'OB немає → структура / ATR')
    p = A.ai_stop_for('LONG', 100.0, plan, None, '', dict(s, sl_source='pct', sl_fallback_pct=2))
    _check(abs(p['stop'] - 98.0) < 1e-9, '% від входу')
    _check(A.ai_settings_norm({'sl_source': 'xx'})['sl_source'] == 'ob', 'невідоме джерело → OB')


def test_entry_block_reasons():
    s = dict(A.AI_DEFAULTS, min_conf=40)
    ok = {'ok': True, 'confidence': 55}
    _check(A.ai_entry_block({}, 'LONG', s, ok, None, 0, [], None, NOW) == '', 'відкриваємо')
    _check('впевненість 30 < 40' in A.ai_entry_block({}, 'LONG', s, dict(ok, confidence=30), None, 0, [], None, NOW),
           'мала впевненість')
    _check(A.ai_entry_block({'signal': {'side': 'LONG'}}, 'LONG', s, ok, None, 0, [], None, NOW) == '',
           'сигнал старої версії (без угоди ШІ) більше НЕ блокує мовчки')
    _check('вже була угода' in A.ai_entry_block({'signal': {'side': 'LONG', 'own': True}}, 'LONG', s, ok,
                                                  None, 0, [], None, NOW), 'одна угода на епізод')
    _check('ліміт' in A.ai_entry_block({}, 'LONG', s, ok, None, 5, [], None, NOW), 'ліміт угод')
    _check(A.ai_entry_block({'tries': {'LONG': [3, NOW]}}, 'LONG', s, ok, None, 0, [], None, NOW) == '',
           'старі спроби (відхилення воротами) не рахуються')


def test_block_is_visible_everywhere():
    view = {'CLUSDT': {'symbol': 'CLUSDT', 'src': 'scan', 'ready': True, 'best': 'LONG',
                       'block': 'ліміт', 'pick': {'side': 'LONG', 'scenario': {'key': 'E2', 'ready': True}}}}
    txt, key = A.ai_heartbeat_text(view, {'enabled': True}, {}, 0, 0)
    _check('⛔ не відкрито: CL ▲ E2 — ліміт' in txt and key == (), txt)
    _check('c.block' in _HTML and "rd.blocked" in _HTML and 'p.ai_conf' in _HTML, 'таблиця · тумблер · угоди')
    _check("pos['ai_conf']" in _TM and "впевненість {cf}%" in _TM, 'впевненість на позиції й у Telegram')
    _check('id="aip-sl-src"' in _HTML and 'id="aip-min-conf"' in _HTML, 'налаштування на панелі')


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith('test_') and callable(v)):
        try:
            fn()
            print('✓', name)
        except Exception as e:
            fails += 1
            print('✗', name, '—', repr(e))
    print('OK' if not fails else f'FAIL {fails}')
    sys.exit(1 if fails else 0)
