"""🛡 АДМІН-ПАНЕЛЬ У СТИЛІ ІНФОСАЙТУ (вимога 24.11).

Дослівно: «Сторінка "Керувати" — відкривай цілою сторінкою, а не вікном, і
зроби адекватне розташування. Щоб все було рівненько… Зроби все в стилі
інфосайту. 🛡 Керування користувачами — також все в стилі інфосайту.»

• `/admin/users` — список: KPI в один ряд, стан пошти, пошук + фільтр-вкладки,
  таблиця; рядок / «Керувати →» відкривають ОКРЕМУ сторінку користувача.
• `/admin/users/<id>` — сторінка користувача: шапка + сітка карток (ліворуч
  статус / термін / роль / пароль / безпека / видалення, праворуч права
  інфосайту перемикачами по групах).
• Дизайн-токени — ті самі, що в `infosite/css/style.css` (поверхні, лінії,
  акценти, Inter + IBM Plex Mono). Дані — ті самі API (`/api/admin/users`,
  `POST /api/admin/users/<id>`), логіки тут немає — лише показ.
• Без Jinja: сторінка збирається рядками, динамічні значення екрануються
  (`markupsafe.escape` на сервері, `esc()` у JS).
"""
from markupsafe import escape

FONTS = ('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700'
         '&family=Inter:wght@400;500;600;700;800&display=swap')

CSS = r"""
:root{
  --p-ink-0:#000;--p-ink-1:oklch(.225 .006 265);--p-ink-2:oklch(.26 .007 265);--p-ink-3:oklch(.30 .008 265);
  --txt:oklch(.93 .005 260);--txt-dim:oklch(.70 .007 260);--muted:oklch(.57 .007 260);
  --accent:oklch(.66 .12 262);--accent-2:oklch(.76 .09 215);--long:oklch(.72 .17 155);--short:oklch(.63 .22 25);
  --warn:oklch(.78 .16 75);--bybit:#f7a600;
  --bg:var(--p-ink-0);--surface:color-mix(in oklch,var(--p-ink-1) 42%,var(--p-ink-2));--elev:var(--p-ink-2);--elev-2:var(--p-ink-3);
  --line:oklch(.30 .007 265);--line-soft:oklch(.255 .006 265);
  --radius:10px;--font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,monospace;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;min-height:100dvh;background:var(--bg);color:var(--txt);font:14px/1.5 var(--font);
  -webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
a{color:inherit;text-decoration:none}button,input,select{font:inherit;color:inherit}
button{cursor:pointer}button:disabled{opacity:.4;cursor:not-allowed}
::-webkit-scrollbar{width:7px;height:7px}::-webkit-scrollbar-thumb{background:var(--elev-2);border-radius:6px}
.topbar{height:58px;position:sticky;top:0;z-index:50;display:flex;align-items:center;gap:22px;padding:0 18px;
  background:var(--p-ink-1);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:10px;font-weight:800;letter-spacing:.02em}
.brand img{width:30px;height:30px;border-radius:8px;object-fit:cover}
.brand small{display:block;font:600 10px var(--mono);letter-spacing:.16em;color:var(--accent-2)}
.nav{display:flex;gap:2px}
.nav a{padding:7px 11px;border-radius:6px;color:var(--txt-dim);font-weight:600;font-size:13px}
.nav a:hover{background:var(--elev);color:var(--txt)}.nav a.active{background:var(--elev);color:var(--txt);box-shadow:inset 0 -2px 0 var(--accent)}
.who{margin-left:auto;display:flex;align-items:center;gap:10px;font-size:12px;color:var(--txt-dim)}
.who b{color:var(--txt);font-weight:600}
.wrap{max-width:1280px;margin:0 auto;padding:20px 16px 48px;display:grid;gap:14px}
.wrap>*{min-width:0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:16px}
.card-h{display:flex;align-items:center;gap:8px;margin:0 0 12px;font-size:12px;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase;color:var(--txt-dim)}
.card-h .r{margin-left:auto;text-transform:none;letter-spacing:0;font-weight:500;color:var(--muted)}
.page-h{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.page-h h1{margin:0;font-size:22px;letter-spacing:-.02em}
.page-h p{margin:2px 0 0;color:var(--txt-dim);font-size:13px}
.page-h .act{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;min-height:34px;padding:0 14px;
  border:1px solid var(--line);border-radius:8px;background:var(--elev);font-weight:600;font-size:13px;
  white-space:nowrap;transition:border-color .15s,background .15s}
.btn:hover{border-color:var(--accent)}
.btn-primary{background:var(--accent);border-color:var(--accent);color:#fff}.btn-primary:hover{filter:brightness(1.08)}
.btn-ok{color:var(--long);border-color:color-mix(in srgb,var(--long) 35%,var(--line))}
.btn-warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 35%,var(--line))}
.btn-danger{color:var(--short);border-color:color-mix(in srgb,var(--short) 40%,var(--line));
  background:color-mix(in srgb,var(--short) 6%,var(--elev))}
.btn-sm{min-height:28px;padding:0 10px;font-size:12px;border-radius:6px}
.kpis{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:10px;overflow-x:auto}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:12px 14px}
.kpi .l{font:600 10px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.kpi .v{font:700 26px var(--mono);margin-top:4px;line-height:1.1}
.c-long{color:var(--long)}.c-short{color:var(--short)}.c-warn{color:var(--warn)}.c-acc{color:var(--accent-2)}.c-dim{color:var(--txt-dim)}
.badge{display:inline-flex;align-items:center;gap:4px;height:20px;padding:0 8px;border-radius:4px;font:600 10.5px var(--mono);
  letter-spacing:.04em;border:1px solid var(--line);background:var(--elev);color:var(--txt-dim);white-space:nowrap}
.b-on{color:var(--long);border-color:color-mix(in srgb,var(--long) 35%,var(--line));background:color-mix(in srgb,var(--long) 8%,transparent)}
.b-off{color:var(--short);border-color:color-mix(in srgb,var(--short) 35%,var(--line));background:color-mix(in srgb,var(--short) 8%,transparent)}
.b-warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 35%,var(--line));background:color-mix(in srgb,var(--warn) 8%,transparent)}
.b-adm{color:var(--bybit);border-color:color-mix(in srgb,var(--bybit) 40%,var(--line));background:color-mix(in srgb,var(--bybit) 9%,transparent)}
.b-acc{color:var(--accent-2);border-color:color-mix(in srgb,var(--accent-2) 35%,var(--line))}
.badges{display:flex;gap:6px;flex-wrap:wrap}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.inp{height:34px;padding:0 12px;border:1px solid var(--line);border-radius:8px;background:var(--p-ink-1);min-width:0}
.inp:focus{border-color:var(--accent);outline:none}
.tabs{display:flex;gap:2px;padding:3px;border:1px solid var(--line);border-radius:8px;background:var(--p-ink-1)}
.tabs button{border:0;background:transparent;padding:5px 11px;border-radius:5px;font-size:12px;font-weight:600;color:var(--txt-dim)}
.tabs button.on{background:var(--elev-2);color:var(--txt)}
.tabs button span{font-family:var(--mono);color:var(--muted);margin-left:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{font:600 10px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--muted);text-align:left;
  padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:10px;border-bottom:1px solid var(--line-soft);vertical-align:middle;white-space:nowrap}
tbody tr:nth-child(even) td{background:color-mix(in oklch,var(--p-ink-2) 35%,transparent)}
tbody tr.row{cursor:pointer}tbody tr.row:hover td{background:var(--elev)}
td.num,.mono{font-family:var(--mono)}
.u-cell{display:flex;align-items:center;gap:10px}
.ava{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;flex:none;font:700 12px var(--mono);
  background:color-mix(in srgb,var(--accent) 22%,var(--elev));color:var(--txt);border:1px solid var(--line)}
.ava.lg{width:54px;height:54px;font-size:20px}
.u-cell b{display:block;font-weight:600}.u-cell small{display:block;color:var(--muted);font-size:11px}
.tbl-wrap{overflow-x:auto}
.empty{padding:26px;text-align:center;color:var(--muted)}
.smtp{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.dot{width:9px;height:9px;border-radius:50%;flex:none}
details.more summary{cursor:pointer;color:var(--txt-dim);font-size:12px;margin-top:8px}
details.more .body{font-size:12px;color:var(--txt-dim);line-height:1.6;margin-top:8px}
.note{font-size:12px;color:var(--txt-dim);line-height:1.6}
.alert{padding:10px 12px;border-radius:8px;font-size:12px;border:1px solid var(--line)}
.alert.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 30%,var(--line));background:color-mix(in srgb,var(--warn) 6%,transparent)}
.alert.err{color:var(--short);border-color:color-mix(in srgb,var(--short) 30%,var(--line));background:color-mix(in srgb,var(--short) 6%,transparent)}
.form-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;align-items:end}
.fld label{display:block;font:600 10px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:5px}
.fld .inp{width:100%}
.chk{display:inline-flex;align-items:center;gap:8px;font-size:13px;color:var(--txt-dim);cursor:pointer}
.chk input{width:16px;height:16px;accent-color:var(--accent);margin:0}
.crumb{display:inline-flex;align-items:center;gap:6px;color:var(--txt-dim);font-size:13px;font-weight:600}
.crumb:hover{color:var(--txt)}
.hero{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.hero h1{margin:0;font-size:22px;letter-spacing:-.02em;word-break:break-all}
.hero .meta{color:var(--muted);font:500 12px var(--mono);margin-top:2px}
.hero .side{margin-left:auto;text-align:right}
.hero .side .big{font:700 22px var(--mono)}.hero .side .lbl{font:600 10px var(--mono);letter-spacing:.14em;color:var(--muted);text-transform:uppercase}
.grid2{display:grid;grid-template-columns:minmax(0,400px) minmax(0,1fr);gap:14px;align-items:start}
.col{display:grid;gap:14px;min-width:0}
.kv{display:grid;grid-template-columns:auto 1fr;gap:7px 14px;font-size:13px}
.kv .k{color:var(--muted)}.kv .v{text-align:right;font-weight:600;min-width:0;overflow-wrap:anywhere}
.btns{display:grid;gap:8px;margin-top:12px}
.btns.two{grid-template-columns:1fr 1fr}
.seg{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}
.seg .btn{padding:0 6px}
.big-val{font:700 20px var(--mono);margin-bottom:10px}
.row-inline{display:flex;gap:8px}.row-inline .inp{flex:1}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.perm-grp{border:1px solid var(--line-soft);border-radius:8px;overflow:hidden}
.perm-grp+.perm-grp{margin-top:12px}
.perm-grp .gh{display:flex;align-items:center;gap:8px;padding:9px 12px;background:var(--p-ink-1);
  font:600 10px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--txt-dim);border-bottom:1px solid var(--line-soft)}
.perm-grp .gh .r{margin-left:auto;letter-spacing:.04em;color:var(--muted)}
.prow{display:flex;align-items:center;gap:14px;padding:10px 12px}
.prow+.prow{border-top:1px solid var(--line-soft)}
.prow .t{flex:1;min-width:0}.prow .t b{display:block;font-weight:600;font-size:13px}
.prow .t small{display:block;color:var(--muted);font-size:11.5px;line-height:1.45;margin-top:1px}
.prow.dead .t{opacity:.5}.prow.dead small.why{color:var(--warn);opacity:1}
.prow.trd b{color:var(--warn)}
.sw{position:relative;width:42px;height:24px;flex:none;display:inline-block}
.sw input{position:absolute;inset:0;opacity:0;margin:0;cursor:pointer;z-index:2}
.sw .tr{position:absolute;inset:0;border-radius:999px;background:var(--p-ink-1);border:1px solid var(--line);transition:.18s}
.sw .tr:before{content:"";position:absolute;top:3px;left:3px;width:16px;height:16px;border-radius:50%;background:var(--muted);transition:.18s}
.sw input:checked~.tr{background:color-mix(in srgb,var(--long) 30%,var(--p-ink-1));border-color:var(--long)}
.sw input:checked~.tr:before{transform:translateX(18px);background:var(--long)}
.sw input:disabled~.tr{opacity:.4}
.presets{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.danger-card{border-color:color-mix(in srgb,var(--short) 35%,var(--line))}
.toast{position:fixed;right:16px;bottom:16px;z-index:999;padding:10px 14px;border-radius:8px;font-size:13px;font-weight:600;
  background:var(--elev);border:1px solid var(--line);box-shadow:0 18px 50px rgba(0,0,0,.45);transform:translateY(20px);opacity:0;transition:.2s}
.toast.show{transform:none;opacity:1}.toast.ok{border-color:var(--long);color:var(--long)}.toast.err{border-color:var(--short);color:var(--short)}
@media(max-width:980px){.grid2{grid-template-columns:1fr}.form-grid{grid-template-columns:1fr 1fr}}
@media(max-width:640px){.topbar{height:auto;flex-wrap:wrap;padding:8px 12px;gap:8px}.who span{display:none}
  .presets,.seg{grid-template-columns:1fr 1fr}.form-grid{grid-template-columns:1fr}.hero .side{margin-left:0;text-align:left}}
"""

# Базові хелпери — для ВСІХ сторінок; ADMIN_JS — лише сторінкам адмін-панелі
# (у кабінет користувача адмінський JS не потрапляє).
COMMON_JS = r"""
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmt(s){return s?String(s).slice(0,16).replace('T',' '):'—';}
function initial(e){return esc(String(e||'?').trim().charAt(0).toUpperCase()||'?');}
let _toastT=null;
function toast(msg,ok){let t=document.getElementById('toast');if(!t){t=document.createElement('div');t.id='toast';document.body.appendChild(t);}
  t.className='toast '+(ok===false?'err':'ok');t.textContent=msg;requestAnimationFrame(()=>t.classList.add('show'));
  clearTimeout(_toastT);_toastT=setTimeout(()=>t.classList.remove('show'),3200);}
"""

ADMIN_JS = r"""
async function act(id,a,extra){
  const r=await fetch('/api/admin/users/'+id,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(Object.assign({action:a},extra||{}))});
  const d=await r.json().catch(()=>({ok:false,error:'HTTP '+r.status}));
  if(!d.ok){toast(d.error||'Помилка',false);return null;}
  return d;}
function grpBadge(u){
  if(u.grp_state==='banned') return '<span class="badge b-off" title="Забанено в Telegram-групі'+(u.tg_banned_at?' адміністратором групи':' ботом')+'">🚫 TG</span>';
  if(u.grp_state==='restricted') return '<span class="badge b-warn" title="Обмежено в Telegram-групі (стара версія) — наступна синхронізація забанить">🔇 TG</span>';
  return '';}
function grpText(u){
  if(!u.tg_linked) return ['без Telegram','c-dim'];
  if(u.grp_state==='banned') return [u.tg_banned_at?'🚫 забанено адміністратором групи':'🚫 забанено ботом','c-short'];
  if(u.grp_state==='restricted') return ['🔇 обмежено (буде бан)','c-warn'];
  if(u.blocked) return ['⏳ бан чекає синхронізації','c-warn'];
  return [u.active?'✓ доступ відкрито':'—', u.active?'c-long':'c-dim'];}
function statusBadge(u){
  return statusBadge0(u)+grpBadge(u);}
function statusBadge0(u){
  if(u.disabled) return '<span class="badge b-off">вимкнено</span>';
  if(u.active) return '<span class="badge b-on">активний</span>';
  if(u.approved&&u.email_confirmed) return '<span class="badge">неактивний</span>';
  return '<span class="badge b-warn">очікує</span>';}
function accessBadge(u){
  if(u.days_left==null) return '<span class="badge b-acc">безлім</span>';
  return u.days_left>0?'<span class="badge b-on">'+u.days_left+' дн</span>':'<span class="badge b-off">завершено</span>';}
function roleBadge(u){return u.is_admin?'<span class="badge b-adm">адмін</span>':'<span class="badge">користувач</span>';}
"""


def _shell(title, active, me_email, info_url, body, script, nav=None, brand='ADMIN',
           home='/admin/users', admin_js=True):
    if nav is None:
        nav = [('/admin/users', 'Користувачі', 'users'), ('/cabinet', 'Кабінет', 'cab'),
               ('/smart-money', 'Smart Money', 'smart'), ('/', 'Dashboard', 'dash'),
               ('/database', 'База даних', 'db'), ((info_url or '/info/'), 'Інфосайт', 'info')]
    nav_html = ''.join(f'<a href="{escape(h)}" class="{"active" if k == active else ""}">{escape(t)}</a>'
                       for h, t, k in nav)
    return ('<!doctype html><html lang="uk"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)} · VSV</title>'
            '<link rel="icon" type="image/png" href="/favicon.ico">'
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            f'<link rel="stylesheet" href="{FONTS}">'
            f'<style>{CSS}</style>'
            '<link rel="stylesheet" href="/static/css/vsv-theme.css?v=300"></head>'
            '<body class="vsv-bot-body vsv-admin-body">'
            f'<header class="topbar"><a class="brand" href="{escape(home)}">'
            '<img src="/info/img/logo.png" alt="VSV">'
            f'<span>VSV<small>{escape(brand)}</small></span></a>'
            f'<nav class="nav" aria-label="Навігація кабінету">{nav_html}</nav>'
            f'<div class="who" id="nav-user-chip"><span><b>{escape(me_email or "")}</b></span>'
            '<a class="btn btn-sm btn-danger" href="/logout">Вийти</a></div></header>'
            f'<main class="wrap">{body}</main>'
            f'<script>{COMMON_JS}{ADMIN_JS if admin_js else ""}{script}</script>'
            '<script src="/static/js/vsv-theme.js?v=300"></script></body></html>')


# ── 1. список користувачів ───────────────────────────────────────────────
LIST_BODY = (
    '<div class="page-h"><div><h1>🛡 Керування користувачами</h1>'
    '<p>Акаунт активний лише після підтвердження email і схвалення адміністратора.</p></div>'
    '<div class="act"><button class="btn btn-primary" onclick="toggleCreate()">＋ Створити користувача</button></div></div>'
    '<div class="kpis" id="kpis"></div>'
    '<section class="card" id="createbox" style="display:none">'
    '<div class="card-h">＋ Новий користувач<span class="r">одразу активний · 30 днів</span></div>'
    '<div class="form-grid">'
    '<div class="fld"><label>Логін</label><input class="inp" id="c_login" autocomplete="off"></div>'
    '<div class="fld"><label>Пароль (мін. 8)</label><input class="inp" id="c_pw" autocomplete="off"></div>'
    '<div class="fld"><label>Telegram chat_id</label><input class="inp" id="c_tgc" placeholder="необовʼязково"></div>'
    '<div class="fld"><label>@Telegram</label><input class="inp" id="c_tgu" placeholder="необовʼязково"></div>'
    '</div><div class="toolbar" style="margin:14px 0 0">'
    '<label class="chk"><input id="c_adm" type="checkbox"> адміністратор</label>'
    '<label class="chk"><input id="c_bot" type="checkbox"> 🤖 доступ до бота</label>'
    '<button class="btn btn-primary" style="margin-left:auto" onclick="createUser()">Створити</button>'
    '</div><div id="c_msg" class="note c-short" style="margin-top:8px"></div></section>'
    '<section class="card" id="smtp"></section>'
    '<section class="card"><div class="toolbar">'
    '<input class="inp" id="q" placeholder="🔎 Пошук за логіном / email / Telegram" style="width:300px;max-width:100%" oninput="render()">'
    '<div class="tabs" id="tabs"></div></div>'
    '<div class="tbl-wrap"><table><thead><tr><th>#</th><th>Користувач</th><th>Роль</th><th>Статус</th>'
    '<th>Доступ</th><th>Сайти</th><th>Останній вхід</th><th>Входи</th><th></th></tr></thead>'
    '<tbody id="tbody"></tbody></table></div><div id="empty" class="empty" style="display:none">Нічого не знайдено</div>'
    '</section>')

LIST_JS = r"""
let USERS=[],SMTP=null,PREVPEND=null,FLT=(()=>{try{return localStorage.getItem('adm_flt')||'all';}catch(e){return 'all';}})();
const FILTERS=[['all','Усі',u=>true],['pending','Очікують',u=>!u.approved&&!u.is_admin],['active','Активні',u=>u.active],
  ['admin','Адміни',u=>u.is_admin],['disabled','Вимкнені',u=>u.disabled]];
async function load(){
  const d=await (await fetch('/api/admin/users')).json();
  USERS=d.users||[];SMTP=d.smtp_status||{};USERS.forEach((u,i)=>u.seq=i+1);
  if(PREVPEND!=null&&(d.pending||0)>PREVPEND) toast('🆕 Новий користувач очікує схвалення');
  PREVPEND=d.pending||0;kpis();renderSmtp();render();}
function kpis(){
  const n=f=>USERS.filter(f).length;
  const k=[['Усього',USERS.length,''],['Очікують схвалення',n(u=>!u.approved&&!u.is_admin),'c-warn'],
    ['Активні',n(u=>u.active),'c-long'],['Адміни',n(u=>u.is_admin),'c-acc'],['Вимкнені',n(u=>u.disabled),'c-dim']];
  document.getElementById('kpis').innerHTML=k.map(x=>'<div class="kpi"><div class="l">'+x[0]+'</div><div class="v '+x[2]+'">'+x[1]+'</div></div>').join('');
  document.getElementById('tabs').innerHTML=FILTERS.map(f=>'<button class="'+(f[0]===FLT?'on':'')+'" onclick="setFlt(\''+f[0]+'\')">'+f[1]+'<span>'+n(f[2])+'</span></button>').join('');}
function setFlt(f){FLT=f;try{localStorage.setItem('adm_flt',f);}catch(e){}kpis();render();}
function renderSmtp(){
  const s=SMTP||{},last=s.last||{},prov=(s.provider||'').toUpperCase(),el=document.getElementById('smtp');
  let col,txt;
  if(!s.configured){col='var(--warn)';txt='Пошта не налаштована — листи не йдуть (посилання підтвердження дублюються в Telegram адміна)';}
  else if(last.ok===false){col='var(--short)';txt=prov+': остання відправка не вдалась';}
  else if(last.ok===true){col='var(--long)';txt=prov+': працює';}
  else {col='var(--accent-2)';txt=prov+': налаштовано, відправок ще не було';}
  let det='';
  if(s.configured){det+= s.is_api?('Транспорт: <b>'+esc(prov)+' (HTTP API)</b> · from: <b>'+esc(s.from||'—')+'</b>')
     :('SMTP: <b>'+esc(s.host||'—')+':'+esc(s.port||'—')+'</b> · user: <b>'+esc(s.user||'—')+'</b> · from: <b>'+esc(s.from||'—')+'</b> · TLS: <b>'+esc(s.tls)+'</b>');
    if(last.ok===false) det+='<div class="alert err" style="margin-top:8px">Помилка ('+esc(last.stage||'?')+'): '+esc(last.error||'—')+'</div>';
    if(last.ok===true) det+='<br>Останній лист: '+esc(last.to||'')+(last.at?(' · '+new Date(last.at*1000).toLocaleString()):'');
  } else det='Render блокує SMTP-порти. Варіанти: HTTP-API ключ у env (<b>RESEND_API_KEY</b> / BREVO_API_KEY / SENDGRID_API_KEY / MAILJET_API_KEY + MAILJET_SECRET_KEY) і <b>MAIL_FROM</b>; або схвалюйте користувачів вручну тут.';
  el.innerHTML='<div class="card-h">✉ Пошта<span class="r" id="smtpmsg"></span></div>'+
    '<div class="smtp"><span class="dot" style="background:'+col+'"></span><span style="font-weight:600">'+esc(txt)+'</span>'+
    '<button class="btn btn-sm" style="margin-left:auto" onclick="smtpTest()">🔌 Тестовий лист собі</button></div>'+
    '<details class="more"><summary>Деталі</summary><div class="body">'+det+'</div></details>';}
async function smtpTest(){
  const m=document.getElementById('smtpmsg');m.textContent='надсилаю…';
  const d=await (await fetch('/api/admin/smtp-test',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).json();
  SMTP=d.status||SMTP;renderSmtp();
  toast(d.ok?('Лист надіслано на '+d.to+(d.tg?' · Telegram ✓':'')):('Пошта: помилка'+(d.tg?' · Telegram ✓':'')),!!(d.ok||d.tg));}
function render(){
  const q=(document.getElementById('q').value||'').toLowerCase(),f=(FILTERS.find(x=>x[0]===FLT)||FILTERS[0])[2];
  const list=USERS.filter(f).filter(u=>(u.email+' '+(u.tg_user||'')+' '+(u.tg_name||'')).toLowerCase().includes(q));
  document.getElementById('empty').style.display=list.length?'none':'block';
  document.getElementById('tbody').innerHTML=list.map(u=>{
    const tg=(u.tg_name||u.tg_user)?('📨 '+esc([u.tg_name,(u.tg_user?'@'+u.tg_user:'')].filter(Boolean).join(' '))):(u.tg_linked?'📨 Telegram':'без Telegram');
    const ip=u.distinct_ips>1?' <span class="badge b-warn" title="Входи з різних IP">⚠ '+u.distinct_ips+' IP</span>':'';
    return '<tr class="row" onclick="go('+u.id+')"><td class="num c-dim">'+u.seq+'</td>'+
      '<td><div class="u-cell"><span class="ava">'+initial(u.email)+'</span><div><b>'+esc(u.email)+'</b><small>'+tg+'</small></div></div></td>'+
      '<td>'+roleBadge(u)+'</td><td>'+statusBadge(u)+'</td><td>'+accessBadge(u)+'</td>'+
      '<td>'+(u.bot_access?'<span class="badge b-adm">🤖 бот</span>':'<span class="badge">інфосайт</span>')+'</td>'+
      '<td class="num c-dim">'+fmt(u.last_login)+'</td><td class="num">'+(u.login_count||0)+ip+'</td>'+
      '<td style="text-align:right"><a class="btn btn-sm" href="/admin/users/'+u.id+'" onclick="event.stopPropagation()">Керувати →</a></td></tr>';
  }).join('');}
function go(id){location.href='/admin/users/'+id;}
function toggleCreate(){const b=document.getElementById('createbox');b.style.display=b.style.display==='none'?'block':'none';}
async function createUser(){
  const v=id=>document.getElementById(id).value,m=document.getElementById('c_msg');
  const r=await fetch('/api/admin/users/create',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({login:v('c_login'),password:v('c_pw'),is_admin:document.getElementById('c_adm').checked,
      bot_access:document.getElementById('c_bot').checked,telegram_chat_id:v('c_tgc'),telegram_username:v('c_tgu')})});
  const d=await r.json().catch(()=>({}));
  if(!d.ok){m.textContent=d.error||'Помилка';return;}
  m.textContent='';['c_login','c_pw','c_tgc','c_tgu'].forEach(i=>document.getElementById(i).value='');
  toast('Користувача створено');toggleCreate();load();}
load();setInterval(()=>{if(!document.hidden)load();},15000);
"""


def render_list(me_email, info_url):
    return _shell('Користувачі', 'users', me_email, info_url, LIST_BODY, LIST_JS)


# ── 2. сторінка одного користувача ───────────────────────────────────────
USER_JS = r"""
const UID=Number(document.getElementById('upage').dataset.uid);
let U=null,PD=null,LINK=null;
async function load(){
  const d=await (await fetch('/api/admin/users')).json();
  const list=d.users||[];list.forEach((u,i)=>u.seq=i+1);
  U=list.find(x=>x.id===UID);PD=d.perm_defs||PD;LINK=(d.pending_links||{})[U?U.email:'']||null;
  if(!U){document.getElementById('upage').innerHTML='<div class="card empty">Користувача не знайдено (можливо, видалено). <a class="btn btn-sm" href="/admin/users">← До списку</a></div>';return;}
  draw();}
async function run(a,extra,msg){const d=await act(UID,a,extra);if(d){toast(msg||'Збережено');await load();}return d;}
function btn(label,a,extra,cls,msg,conf){
  const js=((conf?'if(!confirm('+JSON.stringify(conf)+'))return;':'')+'run('+JSON.stringify(a)+','+JSON.stringify(extra||{})+','+JSON.stringify(msg||'')+')').replace(/&/g,'&amp;').replace(/'/g,'&#39;');
  return '<button class="btn '+(cls||'')+'" onclick=\''+js+'\'>'+label+'</button>';}
function draw(){
  const u=U,tg=[u.tg_name,(u.tg_user?'@'+u.tg_user:'')].filter(Boolean).join(' ');
  const dl=u.days_left,acc=dl==null?'∞ безлімітний':(dl>0?(dl+' дн'):'завершено');
  let h='';
  h+='<a class="crumb" href="/admin/users">← Усі користувачі</a>';
  h+='<section class="card"><div class="hero"><span class="ava lg">'+initial(u.email)+'</span><div><h1>'+esc(u.email)+'</h1>'+
    '<div class="meta">№ '+u.seq+' · ID '+u.id+(tg?' · 📨 '+esc(tg):'')+(u.created_at?' · створено '+fmt(u.created_at):'')+'</div>'+
    '<div class="badges" style="margin-top:8px">'+roleBadge(u)+statusBadge(u)+
    (u.email_confirmed?'<span class="badge b-on">email ✓</span>':'<span class="badge b-warn">email ✗</span>')+
    (u.approved?'<span class="badge b-on">схвалено</span>':'<span class="badge b-warn">не схвалено</span>')+
    (u.tg_linked?'<span class="badge b-acc">📨 Telegram</span>':'<span class="badge">без Telegram</span>')+
    (u.bot_access?'<span class="badge b-adm">🤖 бот</span>':'<span class="badge">лише інфосайт</span>')+'</div></div>'+
    '<div class="side"><div class="lbl">Доступ</div><div class="big '+(dl!=null&&dl<=0?'c-short':'')+'">'+acc+'</div></div></div></section>';
  if(LINK) h+='<div class="alert warn">Посилання підтвердження (пошта не налаштована): <a class="mono" style="word-break:break-all" href="'+esc(LINK.link)+'">'+esc(LINK.link)+'</a></div>';
  h+='<div class="grid2"><div class="col">';
  // статус
  h+='<section class="card"><div class="card-h">✅ Статус і схвалення</div><div class="kv">'+
    '<span class="k">Email</span><span class="v '+(u.email_confirmed?'c-long':'c-warn')+'">'+(u.email_confirmed?'підтверджено':'не підтверджено')+'</span>'+
    '<span class="k">Схвалення</span><span class="v '+(u.approved?'c-long':'c-warn')+'">'+(u.approved?'схвалено':'очікує')+'</span>'+
    '<span class="k">Стан</span><span class="v '+(u.disabled?'c-short':(u.active?'c-long':''))+'">'+(u.disabled?'вимкнено':(u.active?'активний':'неактивний'))+'</span>'+
    '<span class="k">Telegram-група</span><span class="v '+grpText(u)[1]+'">'+grpText(u)[0]+'</span></div>'+
    '<div class="btns'+(u.email_confirmed?'':' two')+'">'+
    (u.approved?btn('✕ Відкликати схвалення','revoke',{},'btn-warn','Схвалення знято'):btn('✓ Схвалити (30 днів)','approve',{},'btn-ok','Схвалено'))+
    (u.email_confirmed?'':btn('✉ Підтвердити email','confirm',{},'','Email підтверджено'))+
    (u.email_confirmed?'':btn('↻ Надіслати лист','resend',{},'','Лист надіслано'))+'</div></section>';
  // Telegram-група (вимога 23.11): блок у боті = бан у групі, і навпаки
  if(u.tg_linked&&!u.is_admin) h+='<section class="card"><div class="card-h">📨 Telegram-група</div>'+
    '<div class="big-val '+grpText(u)[1]+'" style="font-size:15px">'+grpText(u)[0]+'</div>'+
    '<div class="note" style="margin:8px 0 10px">Акаунт заблоковано (вимкнено / строк минув / схвалення знято) → бан у кожному чаті бота. Доступ відновлено → бан знято + особисте запрошення. Бан / розбан адміністратором групи в Telegram вимикає / вмикає акаунт тут.'+
    (u.tg_banned_at?' Забанено в групі: '+fmt(u.tg_banned_at)+'.':'')+'</div>'+
    '<div class="btns">'+btn('🔄 Синхронізувати з групою','group_sync',{},'','Синхронізовано з групою')+'</div></section>';
  // термін
  h+='<section class="card"><div class="card-h">⏳ Термін доступу</div><div class="big-val '+(dl!=null&&dl<=0?'c-short':'')+'">'+acc+'</div>'+
    '<div class="seg">'+btn('10 дн','set_access',{days:10},'','Доступ 10 днів')+btn('15 дн','set_access',{days:15},'','Доступ 15 днів')+
    btn('30 дн','set_access',{days:30},'','Доступ 30 днів')+btn('∞','set_access',{days:0},'','Доступ без обмеження')+
    '<button class="btn" onclick="customDays()">Свій…</button></div></section>';
  // сайти (лише користувач)
  if(!u.is_admin) h+='<section class="card"><div class="card-h">🌐 Доступ до сайтів</div><div class="kv">'+
    '<span class="k">Інфосайт</span><span class="v c-long">завжди</span>'+
    '<span class="k">Бот (торгова панель)</span><span class="v '+(u.bot_access?'c-long':'c-dim')+'">'+(u.bot_access?'надано':'немає')+'</span></div>'+
    '<div class="btns">'+(u.bot_access?btn('🤖 Забрати доступ до бота','revoke_bot',{},'btn-warn','Доступ до бота знято'):btn('🤖 Надати доступ до бота','grant_bot',{},'btn-ok','Доступ до бота надано'))+'</div></section>';
  // роль і стан
  h+='<section class="card"><div class="card-h">👑 Роль і стан</div><div class="btns">'+
    (u.is_admin?btn('− Зняти права адміністратора','remove_admin',{},'btn-warn','Права адміна знято','Зняти права адміністратора?'):
                btn('+ Зробити адміністратором','make_admin',{},'','Тепер адміністратор','Надати права адміністратора? Адмін має доступ до всього.'))+
    (u.disabled?btn('▶ Увімкнути акаунт','enable',{},'btn-ok','Акаунт увімкнено'):btn('⏸ Вимкнути акаунт','disable',{},'btn-warn','Акаунт вимкнено','Вимкнути акаунт? Користувач втратить доступ.'))+
    btn('🚪 Розлогінити (скинути сесії)','kick',{},'','Сесії скинуто')+'</div></section>';
  // пароль
  h+='<section class="card"><div class="card-h">🔑 Пароль</div><div class="row-inline">'+
    '<input class="inp" id="np" type="text" autocomplete="off" placeholder="Новий пароль (мін. 8)">'+
    '<button class="btn btn-primary" onclick="setPw()">Задати</button></div></section>';
  // безпека
  const ips=(u.recent_ips||[]);
  h+='<section class="card"><div class="card-h">🔒 Безпека / входи</div>'+
    (u.distinct_ips>1?'<div class="alert warn" style="margin-bottom:10px">⚠ Входи з '+u.distinct_ips+' різних IP — можливе поширення пароля. Новий вхід вимикає попередню сесію.</div>':'')+
    '<div class="kv"><span class="k">Останній вхід</span><span class="v mono">'+fmt(u.last_login)+'</span>'+
    '<span class="k">Останній IP</span><span class="v mono">'+esc(u.last_ip||'—')+'</span>'+
    '<span class="k">Усього входів</span><span class="v mono">'+(u.login_count||0)+'</span></div>'+
    (ips.length?'<div class="chips">'+ips.map(ip=>'<span class="badge mono">'+esc(ip)+'</span>').join('')+'</div>':'')+'</section>';
  // небезпечно
  h+='<section class="card danger-card"><div class="card-h c-short">⚠ Небезпечно</div>'+
    '<div class="note" style="margin-bottom:10px">Акаунт буде видалено назавжди, користувача прибере з Telegram-групи.</div>'+
    '<button class="btn btn-danger" style="width:100%" onclick="delUser()">🗑 Видалити користувача</button></section>';
  h+='</div><div class="col">'+permsHTML(u)+'</div></div>';
  document.getElementById('upage').innerHTML=h;}
function permsHTML(u){
  const D=PD||{groups:[],perms:[]};
  let h='<section class="card"><div class="card-h">🛡 Інфосайт: що показувати і що дозволено</div>';
  if(u.is_admin) return h+'<div class="alert" style="color:var(--bybit)">Адміністратор — доступ до всього. Права налаштовуються лише для користувачів.</div></section>';
  const raw=u.perms||{},eff=u.perms_eff||{},val=p=>(p.key in raw)?!!raw[p.key]:!!p.default;
  const all={},view={},none={};D.perms.forEach(p=>{all[p.key]=true;none[p.key]=false;view[p.key]=p.group!=='trade';});
  h+='<p class="note" style="margin:-4px 0 12px">Сервер перевіряє кожне право сам: закритий розділ не віддає даних, заборонений шар графіка не надсилається, сповіщення й торгівля без права не виконуються.</p>';
  h+='<div class="presets">'+btn('✓ Усе дозволити','set_perms',{perms:all},'btn-ok','Усе дозволено')+
    btn('👁 Лише перегляд','set_perms',{perms:view},'','Лише перегляд (без торгівлі)')+
    btn('⛔ Усе заборонити','set_perms',{perms:none},'btn-danger','Усе заборонено')+'</div>';
  D.groups.forEach(g=>{
    const ps=D.perms.filter(p=>p.group===g.key);if(!ps.length)return;
    const on=ps.filter(p=>eff[p.key]).length;
    h+='<div class="perm-grp"><div class="gh">'+esc(g.label)+'<span class="r">'+on+' / '+ps.length+'</span></div>';
    ps.forEach(p=>{
      const v=val(p),dead=v&&!eff[p.key],need=(D.perms.find(x=>x.key===p.needs)||{}).label||p.needs;
      h+='<div class="prow'+(dead?' dead':'')+(p.group==='trade'?' trd':'')+'"><div class="t"><b>'+esc(p.label)+'</b>'+
        (p.hint?'<small>'+esc(p.hint)+'</small>':'')+(dead?'<small class="why">не діє: вимкнено «'+esc(need)+'»</small>':'')+'</div>'+
        '<label class="sw"><input type="checkbox" '+(v?'checked ':'')+'onchange="permSet(\''+p.key+'\',this)"><span class="tr"></span></label></div>';
    });
    h+='</div>';});
  return h+'</section>';}
async function permSet(key,el){
  if(el.checked&&key==='trade_real'&&!confirm('Дозволити РЕАЛЬНУ торгівлю?\n\nУгоди відкриває ОСОБИСТИЙ Trade Manager користувача — на ЙОГО рахунку Bybit, ЙОГО ключами (зʼявиться після підключення ключів).')){el.checked=false;return;}
  el.disabled=true;await run('set_perms',{key:key,on:el.checked},'Права оновлено');}
function customDays(){const n=prompt('На скільки днів надати доступ? (0 = без обмеження)');if(n===null)return;run('set_access',{days:parseInt(n)||0},'Термін оновлено');}
async function setPw(){const p=document.getElementById('np').value;if(!p)return;const d=await run('set_password',{password:p},'Пароль змінено');if(d)document.getElementById('np').value='';}
async function delUser(){if(!confirm('Видалити користувача '+U.email+' назавжди?'))return;const d=await act(UID,'delete');if(d)location.href='/admin/users';}
load();setInterval(()=>{if(!document.hidden&&!document.activeElement.matches('input'))load();},30000);
"""


def render_user(uid, me_email, info_url):
    body = f'<div id="upage" data-uid="{int(uid)}" style="display:grid;gap:14px"><div class="card empty">Завантаження…</div></div>'
    return _shell('Користувач', 'users', me_email, info_url, body, USER_JS)


# ── 3. 👤 ОСОБИСТИЙ КАБІНЕТ (вимога 24.11) ───────────────────────────────
# «Особистий кабінет користувача зроби повною сторінкою і з доступом лише із
# головної сторінки авторизованого користувача.» Повна сторінка в стилі
# інфосайту (ті самі токени й шапка, що адмін-панель); дані з сервера —
# екрановані, дії — ті самі API (`/api/me/*`). Гейт входу — `auth.cabinet_entry_ok`.
CAB_CSS = """
.cab-sect+.cab-sect{margin-top:14px}
.cab-row{display:flex;align-items:center;gap:14px;padding:10px 0}
.cab-row+.cab-row{border-top:1px solid var(--line-soft)}
.cab-row .t{flex:1;min-width:0}.cab-row .t b{display:block;font-weight:600;font-size:13px}
.cab-row .t small{display:block;color:var(--muted);font-size:11.5px;line-height:1.45;margin-top:1px}
.cab-msg{margin-top:8px;font-size:12px;min-height:0}.cab-msg.ok{color:var(--long)}.cab-msg.err{color:var(--short)}
.cab-stack{display:grid;gap:8px}
"""

CAB_JS = r"""
function msg(id,t,ok){const m=document.getElementById(id);if(!m)return;m.className='cab-msg '+(ok===false?'err':(ok?'ok':''));m.textContent=t||'';}
async function j(url,opt){const r=await fetch(url,opt);return r.json().catch(()=>({ok:false,error:'HTTP '+r.status}));}
function post(url,body){return j(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});}
async function chpw(){
  const p=document.getElementById('np'),p2=document.getElementById('np2');
  const d=await post('/api/me/password',{password:p.value,password2:p2.value});
  msg('pwm',d.ok?'✅ Пароль змінено':'⚠️ '+(d.error||'Помилка'),!!d.ok);
  if(d.ok){p.value='';p2.value='';toast('Пароль змінено');}}
async function tglink(){
  msg('tgm','⏳ …');
  const d=await j('/api/me/tg-link',{method:'POST'});
  if(!d.ok||!d.url){msg('tgm','⚠️ '+(d.error||'Помилка'),false);return;}
  document.getElementById('tgm').className='cab-msg ok';
  document.getElementById('tgm').innerHTML='Відкрийте бота і натисніть «Start» — Telegram привʼяжеться до цього акаунта. '
    +'<a href="'+esc(d.url)+'" target="_blank" rel="noopener"><b>Відкрити @'+esc(d.bot)+'</b></a> (посилання діє 30 хв, одноразове).';
  try{window.open(d.url,'_blank');}catch(e){}}
async function saventf(){
  const nb=document.getElementById('nbtc'),nf=document.getElementById('nfnd');if(!nb||!nf)return;
  const d=await post('/api/me/notify',{notify_btc:nb.checked,notify_funding:nf.checked});
  toast(d.ok?'Збережено':(d.error||'Помилка'),!!d.ok);}
const TRB={paper:'🧪 paper',real:'🏦 реальна'};
async function trload(){
  const el=document.getElementById('trs');if(!el)return;
  try{
    const d=await j('/api/me/trading');
    if(!d.allowed){el.innerHTML='<span class="c-dim">🔒 '+esc(d.reason||'немає доступу')+'</span>';return;}
    const cs=Object.keys(d.coins||{}),tm=d.tm||{};
    const kv=(k,v)=>'<span class="k">'+k+'</span><span class="v">'+v+'</span>';
    let h='<div class="kv">'+kv('Торгівля за сигналами',d.enabled?'<span class="c-long">увімкнено</span>':'<span class="c-short">вимкнено</span>')+
      kv('Монет',cs.length?cs.map(x=>esc(x.replace(/USDT$/,''))+(d.is_admin?' '+(TRB[d.coins[x]]||esc(d.coins[x])):'')).join(', '):'—');
    if(d.is_admin) h+=kv('Дозволені книги',(d.books||[]).map(b=>TRB[b]||esc(b)).join(', '))+
      kv('Trade Manager бота',tm.enabled?'<span class="c-long">увімкнено</span>':'<span class="c-short">вимкнено</span>')+
      kv('Test Mode',tm.test_mode?'<span class="c-long">увімкнено</span>':'<span class="c-dim">вимкнено</span>');
    else h+=kv('Ваші ключі',tm.need_keys?'<span class="c-warn">не підключено</span>':'<span class="c-long">підключено</span>')+
      kv('Ваш Trade Manager',tm.need_keys?'<span class="c-dim">зʼявиться після ключів</span>':(tm.enabled?'<span class="c-long">реальна торгівля увімкнена</span>':'<span class="c-dim">реальна торгівля вимкнена</span>'));
    el.innerHTML=h+'</div>';
  }catch(e){el.textContent='⚠️ не вдалося завантажити';}}
const SRC={db_encrypted:'🟢 збережено в боті (зашифровано)',env_plain:'🟡 змінна середовища (без шифрування)',
  env_encrypted:'🟢 змінна середовища (зашифровано)',none:'🔴 не задано'};
async function kload(){
  const el=document.getElementById('kst');if(!el)return;
  try{const k=await j('/api/me/keys');
    el.innerHTML='<div class="kv"><span class="k">Джерело</span><span class="v">'+esc(SRC[k.source]||k.source||'—')+'</span>'+
      '<span class="k">API Key</span><span class="v mono">'+esc(k.key_preview||'none')+'</span></div>'+
      (k.encryption_key_set?'':'<div class="alert warn" style="margin-top:8px">⚠️ На сервері не задано ENCRYPTION_KEY — зберегти ключі неможливо.</div>');
  }catch(e){el.textContent='⚠️ не вдалося завантажити';}}
async function ksave(){
  const k=document.getElementById('kk'),s=document.getElementById('ks');
  const d=await post('/api/me/keys',{api_key:k.value.trim(),api_secret:s.value.trim()});
  if(d.ok){k.value='';s.value='';msg('km','✅ '+(d.message||'Збережено'),true);kload();trload();}
  else msg('km','⚠️ '+(d.error||'не вдалося зберегти'),false);}
async function kact(a){
  if(a==='clear'&&!confirm('Видалити ключі Bybit з бота?'))return;
  const d=await j('/api/me/keys/'+a,{method:'POST'});
  if(a==='test') msg('km',d.ok?'✅ '+(d.message||'Ключі робочі'):'⚠️ '+(d.friendly||d.error||d.ret_msg||'перевірка не вдалась'),!!d.ok);
  else{msg('km',d.ok?'✅ '+(d.message||'Видалено'):'⚠️ '+(d.error||'не вдалося'),!!d.ok);kload();trload();}}
trload();kload();
"""


def render_cabinet(c, info_url):
    """👤 Повна сторінка особистого кабінету. `c` — дані з сервера (email,
    is_admin, role, tg, tg_linked, bot, access, ntf_btc, ntf_fnd, trade,
    home, home_lbl)."""
    e = escape
    adm = bool(c.get('is_admin'))
    home, home_lbl = c.get('home') or '/', c.get('home_lbl') or '← На головну'
    if adm:
        nav = None
    else:
        nav = [((info_url or '/info/'), '🛰 Головна', 'info'), ('/cabinet', '👤 Кабінет', 'cab')]
        if c.get('bot_ok'):
            nav.append(('/', '📊 Дашборд', 'dash'))
    tg_btn = '🔗 Перепривʼязати Telegram' if c.get('tg_linked') else '🔗 Привʼязати Telegram'
    left = (
        '<section class="card"><div class="card-h">👤 Профіль</div>'
        '<div class="hero" style="margin-bottom:14px">'
        f'<div class="ava lg">{e((c.get("email") or "?")[:1].upper())}</div>'
        f'<div><h1 style="font-size:18px">{e(c.get("email") or "")}</h1>'
        f'<div class="meta">{e(c.get("role") or "")}</div></div></div>'
        '<div class="kv">'
        f'<span class="k">Логін</span><span class="v">{e(c.get("email") or "")}</span>'
        f'<span class="k">📨 Telegram</span><span class="v">{e(c.get("tg") or "—")}</span>'
        f'<span class="k">⏳ Доступ</span><span class="v">{e(c.get("access") or "—")}</span>'
        f'<span class="k">🤖 Доступ до бота</span><span class="v">{e(c.get("bot") or "—")}</span>'
        '</div>'
        f'<div class="btns"><button class="btn" onclick="tglink()">{e(tg_btn)}</button></div>'
        '<div id="tgm" class="cab-msg"></div>'
        + ('' if c.get('tg_linked') else
           '<div class="alert warn" style="margin-top:10px">⚠️ Telegram ще не привʼязано — сповіщення не дійдуть.</div>')
        + '</section>'
        '<section class="card"><div class="card-h">🔑 Змінити пароль</div><div class="cab-stack">'
        '<input class="inp" id="np" type="password" placeholder="Новий пароль (мін. 8)" autocomplete="new-password">'
        '<input class="inp" id="np2" type="password" placeholder="Повторіть пароль" autocomplete="new-password">'
        '<button class="btn btn-primary" onclick="chpw()">Змінити пароль</button></div>'
        '<div id="pwm" class="cab-msg"></div></section>')
    if adm:
        ntf = ('<section class="card"><div class="card-h">📢 Групові теми Telegram<span class="r">для всіх учасників</span></div>'
               '<div class="cab-row"><div class="t"><b>🧮 VSV-монітор — банер · ₿ сеанс</b>'
               '<small>Майстер-вимикач теми в групі.</small></div>'
               f'<label class="sw"><input type="checkbox" id="nbtc" {"checked" if c.get("ntf_btc") else ""} onchange="saventf()"><span class="tr"></span></label></div>'
               '<div class="cab-row"><div class="t"><b>💰 Funding — поява монети з ММ</b>'
               '<small>Тема 💰 у групі.</small></div>'
               f'<label class="sw"><input type="checkbox" id="nfnd" {"checked" if c.get("ntf_fnd") else ""} onchange="saventf()"><span class="tr"></span></label></div>'
               '</section>')
    else:
        ntf = ('<section class="card"><div class="card-h">🔔 Сповіщення в Telegram</div>'
               '<p class="note" style="margin:0">Ринкові сигнали приходять у <b>Telegram-групу</b>, особисті '
               '(🔔 SBR, ⚠ Аномалії обсягу) — у приватний чат бота; підписки по монетах — у '
               'WATCHLIST на головній сторінці. Доступ до групи — після схвалення адміністратором.</p></section>')
    trade = ''
    if c.get('trade'):
        trade = ('<section class="card" id="trade"><div class="card-h">'
                 + ('🤖 Торгівля з кабінету' if adm else '👤 Мій Trade Manager')
                 + '</div>'
                 + ('' if adm else
                    '<p class="note" style="margin:0 0 10px">Особистий Trade Manager — дзеркало бота, що торгує '
                    'ЛИШЕ на вашому рахунку Bybit вашими ключами.</p>')
                 + '<div id="trs" class="note">⏳ …</div>'
                 f'<div class="btns"><a class="btn" href="{e(c.get("trade_href") or "/info/#trade-card")}">'
                 + ('⚙ Монети й книги на інфосайті →' if adm else '⚙ Керувати на головній сторінці →')
                 + '</a></div></section>')
        if adm:
            trade += ('<section class="card" id="keys"><div class="card-h">🔐 API-ключі біржі (Bybit)<span class="r">ключі БОТА</span></div>'
                      '<div id="kst" class="note">⏳ …</div><div class="cab-stack" style="margin-top:12px">'
                      '<input class="inp" id="kk" type="password" autocomplete="off" spellcheck="false" placeholder="API Key">'
                      '<input class="inp" id="ks" type="password" autocomplete="off" spellcheck="false" placeholder="API Secret">'
                      '<button class="btn btn-primary" onclick="ksave()">🔐 Зашифрувати і зберегти</button>'
                      '<div class="btns two" style="margin-top:0"><button class="btn" onclick="kact(\'test\')">🔎 Перевірити</button>'
                      '<button class="btn btn-danger" onclick="kact(\'clear\')">🗑 Видалити з бота</button></div></div>'
                      '<div id="km" class="cab-msg"></div>'
                      '<p class="note" style="margin:10px 0 0">Ключі шифруються на сервері (Fernet, <code>ENCRYPTION_KEY</code>) — '
                      'відкритий текст не повертається. Це ті самі ключі, якими торгує бот. Дозволи: Read + Trade, без виводу.</p></section>')
    body = (
        '<div class="page-h"><div><h1>👤 Особистий кабінет</h1>'
        f'<p>{e(c.get("role") or "")}</p></div>'
        f'<div class="act"><a class="btn" href="{e(home)}">{e(home_lbl)}</a>'
        + ('<a class="btn" href="/admin/users">🛡 Адмін-панель</a>' if adm else '')
        + '</div></div>'
        f'<div class="grid2"><div class="col">{left}</div><div class="col">{ntf}{trade}</div></div>')
    page = _shell('Кабінет', 'cab', c.get('email'), info_url, body, CAB_JS, nav=nav,
                  brand=('ADMIN' if adm else 'КАБІНЕТ'), home=home, admin_js=False)
    return page.replace('</style>', CAB_CSS + '</style>', 1)
