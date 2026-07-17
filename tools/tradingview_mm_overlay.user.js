// ==UserScript==
// @name         SVV ММ overlay for TradingView
// @namespace    svv-webhook-bot
// @version      1.2.1
// @description  Показує реальний ММ (liquidation-fuel) + стан ₿ BTC (і фандинг для funding-монет) із SVV WebHook BOT поверх графіка TradingView для поточної монети.
// @author       SVV
// @match        https://*.tradingview.com/chart/*
// @match        https://tradingview.com/chart/*
// @run-at       document-idle
// @grant        GM_xmlhttpRequest
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_registerMenuCommand
// @connect      *
// ==/UserScript==

/*
 Як це працює
 ────────────
 • Скрипт визначає монету, відкриту на графіку TradingView (з URL / заголовка).
 • Раз на кілька секунд запитує у твого бота ендпоінт
       GET <URL_бота>/api/fuel-filter/panel/<SYMBOL>
   і показує СПРАВЖНІЙ ММ (|fuel dir|×100), напрямок, рівень (немає/слабке/
   помірний/сильне), виснаженість і вердикт — ті самі числа, що в боті.
 • Дані рахуються з карти ліквідацій бота, тож для монети, яку бот ще не
   сканував, перший запит може бути «немає даних», а за ~хвилину зʼявиться
   (ендпоінт реєструє монету в liq-map на вимогу).

 Перше налаштування
 ──────────────────
 1. Постав Tampermonkey (Chrome/Edge/Firefox) або Violentmonkey.
 2. Додай цей файл як новий userscript.
 3. Задай URL свого бота: меню Tampermonkey → «SVV ММ: задати URL бота»
    (напр. https://svv-webhook-bot.onrender.com  або  http://localhost:10000).
    Без завершального «/».
 4. Відкрий будь-який графік на tradingview.com/chart/… — бейдж зʼявиться
    праворуч зверху. Його можна перетягувати мишкою.

 Примітки
 ────────
 • GM_xmlhttpRequest обходить CORS і mixed-content, тож http-localhost працює
   навіть на https-TradingView.
 • Символи TradingView виду BYBIT:VELVETUSDT.P нормалізуються у VELVETUSDT.
*/

(function () {
    'use strict';

    // ── Config ────────────────────────────────────────────────────────────
    const POLL_MS = 4000;           // як часто оновлювати ММ
    const K_URL = 'svv_bot_url';    // ключ зберігання URL бота
    const K_POS = 'svv_badge_pos';  // ключ позиції бейджа

    function getBotUrl() {
        return (GM_getValue(K_URL, '') || '').replace(/\/+$/, '');
    }
    function setBotUrl() {
        const cur = getBotUrl();
        const v = prompt('URL твого SVV-бота (без / в кінці):\n' +
            'напр. https://svv-webhook-bot.onrender.com  або  http://localhost:10000', cur);
        if (v != null) {
            GM_setValue(K_URL, v.trim().replace(/\/+$/, ''));
            lastSym = null;   // форс-оновлення
            tick();
        }
    }
    try { GM_registerMenuCommand('SVV ММ: задати URL бота', setBotUrl); } catch (e) {}

    // ── Symbol detection ──────────────────────────────────────────────────
    // TradingView symbol → base coin the bot knows (VANRYUSDT).
    // ВАЖЛИВО: беремо символ з ЖИВОГО DOM (легенда/заголовок графіка), бо
    // TradingView — SPA і НЕ оновлює ?symbol= в URL при зміні монети в застосунку.
    // Раніше URL стояв першим → оверлей «залипав» на старій монеті (BTCUSDT).
    function _cleanSym(raw) {
        if (!raw) return '';
        try { raw = decodeURIComponent(raw); } catch (e) {}
        if (raw.indexOf(':') >= 0) raw = raw.split(':').pop();  // прибрати "BYBIT:"
        raw = raw.trim().split(/[\s.·]/)[0];                    // до пробілу/крапки/·
        return raw.toUpperCase().trim();
    }
    function _symFromDom() {
        // Легенда графіка ("VANRYUSDT Perpetual Contract · 15 · Bybit") + кнопка
        // символу у верхній панелі. Пробуємо кілька селекторів (версії TV різні).
        const sels = [
            '[data-name="legend-source-title"]',
            '.js-button-text',
            '#header-toolbar-symbol-search',
            '[class*="symbolNameText"]',
            '[class*="mainSourceWrapper"] [class*="title"]',
        ];
        for (const s of sels) {
            try {
                const el = document.querySelector(s);
                const t = el && el.textContent ? el.textContent.trim() : '';
                if (!t) continue;
                const m = t.match(/[A-Z0-9]{2,20}USDT?(?:\.P)?/i) || t.match(/[A-Z0-9]{2,20}/);
                if (m) return m[0];
            } catch (e) {}
        }
        return '';
    }
    function currentSymbol() {
        // 1) DOM (оновлюється при зміні монети) → 2) заголовок вкладки → 3) URL.
        let raw = _symFromDom();
        if (!raw) {
            const m = (document.title || '').match(/[A-Z0-9]{2,20}(?:\.[A-Z]+)?/);
            raw = m ? m[0] : '';
        }
        if (!raw) {
            try { raw = new URLSearchParams(location.search).get('symbol') || ''; } catch (e) {}
        }
        return _cleanSym(raw);
    }

    // ── ММ bands (той самий поділ, що в боті) ──────────────────────────────
    function band(mm) {
        if (mm == null) return { label: '—' };
        if (mm < 10) return { label: 'немає' };
        if (mm < 30) return { label: 'слабке' };
        if (mm < 60) return { label: 'помірний' };
        return { label: 'сильне' };
    }
    function dirColor(dir) {
        if (dir === 'LONG') return '#22c55e';
        if (dir === 'SHORT') return '#ef4444';
        return '#8b93a7';
    }
    function dirLabel(dir) {
        if (dir === 'LONG') return '🟢 LONG';
        if (dir === 'SHORT') return '🔴 SHORT';
        return '⚪ —';
    }

    // ── Badge UI ──────────────────────────────────────────────────────────
    let badge, elSym, elMM, elDir, elBtc, elFund, elFoot;
    function buildBadge() {
        if (badge) return;
        badge = document.createElement('div');
        badge.id = 'svv-mm-badge';
        badge.style.cssText = [
            'position:fixed', 'z-index:2147483000', 'top:96px', 'right:18px',
            'min-width:172px', 'padding:9px 12px', 'border-radius:10px',
            'background:rgba(17,20,28,0.94)', 'border:1px solid rgba(255,255,255,0.14)',
            'box-shadow:0 6px 22px rgba(0,0,0,0.45)', 'color:#e5e7eb',
            'font-family:-apple-system,Segoe UI,Roboto,sans-serif', 'font-size:12px',
            'line-height:1.35', 'user-select:none', 'cursor:grab', 'backdrop-filter:blur(3px)'
        ].join(';');
        badge.innerHTML =
            '<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">' +
              '<span style="font-weight:800;letter-spacing:.3px">💰 ММ</span>' +
              '<span id="svv-mm-sym" style="font-weight:700;color:#cbd5e1;font-size:11px"></span>' +
            '</div>' +
            '<div style="display:flex;align-items:baseline;gap:8px">' +
              '<span id="svv-mm-val" style="font-weight:900;font-size:22px">—</span>' +
              '<span id="svv-mm-dir" style="font-weight:700;font-size:12px"></span>' +
            '</div>' +
            '<div id="svv-mm-btc" style="font-size:10.5px;color:#9aa3b5;margin-top:3px"></div>' +
            '<div id="svv-mm-fund" style="font-size:10.5px;color:#34d399;margin-top:2px"></div>' +
            '<div id="svv-mm-foot" style="font-size:9.5px;color:#6b7280;margin-top:3px"></div>';
        document.body.appendChild(badge);
        elSym = badge.querySelector('#svv-mm-sym');
        elMM = badge.querySelector('#svv-mm-val');
        elDir = badge.querySelector('#svv-mm-dir');
        elBtc = badge.querySelector('#svv-mm-btc');
        elFund = badge.querySelector('#svv-mm-fund');
        elFoot = badge.querySelector('#svv-mm-foot');
        badge.addEventListener('dblclick', setBotUrl);   // подвійний клік → налаштувати URL
        restorePos();
        makeDraggable();
    }

    function restorePos() {
        const p = GM_getValue(K_POS, null);
        if (p && typeof p.left === 'number') {
            badge.style.left = p.left + 'px';
            badge.style.top = p.top + 'px';
            badge.style.right = 'auto';
        }
    }
    function makeDraggable() {
        let sx, sy, ox, oy, dragging = false;
        badge.addEventListener('mousedown', (e) => {
            dragging = true; badge.style.cursor = 'grabbing';
            sx = e.clientX; sy = e.clientY;
            const r = badge.getBoundingClientRect(); ox = r.left; oy = r.top;
            e.preventDefault();
        });
        window.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            const left = Math.max(0, ox + e.clientX - sx);
            const top = Math.max(0, oy + e.clientY - sy);
            badge.style.left = left + 'px'; badge.style.top = top + 'px'; badge.style.right = 'auto';
        });
        window.addEventListener('mouseup', () => {
            if (!dragging) return;
            dragging = false; badge.style.cursor = 'grab';
            const r = badge.getBoundingClientRect();
            GM_setValue(K_POS, { left: Math.round(r.left), top: Math.round(r.top) });
        });
    }

    // Countdown "HH:MM:SS" until a future ms-timestamp (for funding settlement).
    function fmtLeft(ms) {
        if (ms == null) return '';
        let s = Math.floor((Number(ms) - Date.now()) / 1000);
        if (s < 0) s = 0;
        const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
        const pad = (n) => String(n).padStart(2, '0');
        return (h > 0 ? h + ':' : '') + pad(m) + ':' + pad(ss);
    }

    // Compact USD volume: $9.77M / $1.2K / $980 (same as the Vol 24h column).
    function fmtVol(v) {
        if (v == null || !(v > 0)) return '—';
        return v >= 1e9 ? '$' + (v / 1e9).toFixed(2) + 'B'
             : v >= 1e6 ? '$' + (v / 1e6).toFixed(1) + 'M'
             : v >= 1e3 ? '$' + (v / 1e3).toFixed(0) + 'K' : '$' + v.toFixed(0);
    }

    function render(sym, state) {
        buildBadge();
        elSym.textContent = sym || '—';
        // Reset the extra lines each render.
        elBtc.innerHTML = ''; elFund.innerHTML = '';
        if (state === 'noconfig') {
            elMM.textContent = '⚙'; elMM.style.color = '#f59e0b';
            elDir.textContent = ''; elBtc.textContent = 'Задай URL бота (2× клік по бейджу)';
            elFoot.textContent = '';
            return;
        }
        if (state === 'waking') {
            elMM.textContent = '⏳'; elMM.style.color = '#f59e0b';
            elDir.textContent = ''; elBtc.textContent = 'Бот прокидається… (Render sleep)';
            elFoot.textContent = getBotUrl() || '';
            return;
        }
        if (state === 'error') {
            elMM.textContent = '⚠'; elMM.style.color = '#f59e0b';
            elDir.textContent = ''; elBtc.textContent = 'Бот недоступний за URL';
            elFoot.textContent = getBotUrl() || '';
            return;
        }
        const d = state || {};
        const mm = (d.mm_str != null) ? Number(d.mm_str) : null;
        const dir = d.mm || d.fuel_status || null;
        elMM.textContent = (mm != null) ? (mm.toFixed(0) + '%') : '—';
        elMM.style.color = dirColor(dir);
        elDir.textContent = dirLabel(dir) + (mm != null ? ' · ' + band(mm).label : '');
        elDir.style.color = dirColor(dir);

        // ── ₿ BTC ММ line — the BTC liquidation-fuel state (напрямок + сила% +
        // рівень), same widget style as the coin's ММ. NOT the session START/
        // pause status. Mirrors the ₿ BTCUSDT banner ("LONG · 34% помірний"). ──
        if (d.enabled === false) {
            elBtc.style.color = '#9aa3b5';
            elBtc.textContent = 'FF вимкнено в боті';
        } else {
            const b = d.btc || {};
            const bs = (b.strength != null) ? Number(b.strength) : null;
            if (b.dir === 'LONG' || b.dir === 'SHORT') {
                const bc = dirColor(b.dir);
                const tail = (bs != null) ? ` · ${bs}% · ${band(bs).label}` : '';
                elBtc.innerHTML = `₿ BTC ММ: <span style="color:${bc};font-weight:700">${dirLabel(b.dir)}</span>${tail}`;
            } else {
                elBtc.style.color = '#9aa3b5';
                elBtc.innerHTML = (bs != null)
                    ? `₿ BTC ММ: ⚪ — · ${bs}% · ${band(bs).label}`
                    : '₿ BTC ММ: ⚪ —';
            }
        }

        // ── 💰 Funding line — only for coins in the «💰 Funding — ММ» table.
        // Funding is coloured by TREND exactly like the table: 🔴 глибше в мінус /
        // 🟢 до нуля / cyan стабільний / ✦ золоте (стабільне чітке 0.5-крокове
        // значення). Vol 24h (+ ↑/↓ vs ~2-хв базою) додається тим самим значенням,
        // що й у колонці Vol 24h таблиці. ──
        if (d.funding && d.funding_rate != null) {
            const r = Number(d.funding_rate);
            const prev = d.funding_prev_rate;
            let rc = '#cbd5e1', mark = '', extra = '';
            if (prev != null) {
                if (r < prev - 0.0005) { rc = '#f87171'; }        // глибше в мінус
                else if (r > prev + 0.0005) { rc = '#4ade80'; }   // до нуля
                else {
                    const av = Math.abs(r), step = Math.round(av / 0.5) * 0.5;
                    if (Math.abs(av - step) < 0.01 && step >= 0.5 && step <= 4) {
                        rc = '#fbbf24'; mark = '✦ ';
                        extra = ';text-shadow:0 0 8px rgba(251,191,36,0.9)';
                    } else { rc = '#7dd3fc'; }                     // стабільний
                }
            } else {
                rc = r >= 0 ? '#4ade80' : '#f87171';              // без тренду → знак
            }
            const left = d.funding_next_ms ? ` · ⏳ ${fmtLeft(d.funding_next_ms)}` : '';
            // Vol 24h segment (only for funding coins — matches the table column).
            let volSeg = '';
            if (d.vol24h != null && d.vol24h > 0) {
                let arr = '';
                if (d.vol24h_prev != null && d.vol24h_prev > 0) {
                    if (d.vol24h > d.vol24h_prev * 1.005) arr = ' <span style="color:#4ade80">▲</span>';
                    else if (d.vol24h < d.vol24h_prev * 0.995) arr = ' <span style="color:#f87171">▼</span>';
                    else arr = ' <span style="color:#9aa3b5">→</span>';
                }
                volSeg = ` · 📊 <span style="color:#cbd5e1;font-weight:700">${fmtVol(d.vol24h)}</span>${arr}`;
            }
            elFund.innerHTML = `💰 фандинг <span style="color:${rc};font-weight:700${extra}">${mark}${r >= 0 ? '+' : ''}${r.toFixed(4)}%</span>${left}${volSeg}`;
        }

        // ── ⚡ CTR line (замість «оновлено») — нахил STC по цій монеті ──
        const c = d.ctr;
        if (c && c.stc != null) {
            const tf = c.tf ? ('·' + String(c.tf).toUpperCase()) : '';
            const lean = c.lean;   // 'LONG' | 'SHORT' | null
            const icon = lean === 'LONG' ? '🟢' : (lean === 'SHORT' ? '🔴' : '⚪');
            const col = lean === 'LONG' ? '#4ade80' : (lean === 'SHORT' ? '#f87171' : '#9aa3b5');
            const label = lean ? (lean + '-нахил') : 'нейтрально';
            const pct = (c.lean_pct != null) ? c.lean_pct : 0;
            let foot = `⚡ CTR${tf} <span style="color:${col};font-weight:700">${icon} ${label} ${pct}%</span>`;
            // ⚡ 1H CTR alongside the primary TF (added by get_panel_status).
            if (c.stc_1h != null) {
                const l1 = c.lean_1h;
                const i1 = l1 === 'LONG' ? '🟢' : (l1 === 'SHORT' ? '🔴' : '⚪');
                const cc1 = l1 === 'LONG' ? '#4ade80' : (l1 === 'SHORT' ? '#f87171' : '#9aa3b5');
                const p1 = (c.lean_pct_1h != null) ? c.lean_pct_1h : 0;
                foot += ` <span style="color:${cc1};font-weight:700">· 1H ${i1} ${p1}%</span>`;
            }
            elFoot.innerHTML = foot;
        } else {
            elFoot.style.color = '#9aa3b5';
            elFoot.textContent = '⚡ CTR — (немає даних)';
        }
    }

    // ── Fetch ─────────────────────────────────────────────────────────────
    let _inFlight = false;   // avoid stacking requests during a slow wake-up
    function fetchPanel(sym) {
        const url = getBotUrl();
        if (!url) { render(sym, 'noconfig'); return; }
        if (_inFlight) return;
        _inFlight = true;
        const done = () => { _inFlight = false; };
        GM_xmlhttpRequest({
            method: 'GET',
            url: url + '/api/fuel-filter/panel/' + encodeURIComponent(sym),
            // Render free tier spins the bot DOWN after ~15 min idle; the first
            // request then triggers a cold start that can take 30-60s. Long
            // timeout so we ride the wake-up instead of flashing an error.
            timeout: 60000,
            onload: (r) => {
                done();
                try {
                    const j = JSON.parse(r.responseText);
                    if (j && j.ok !== false) render(sym, j);
                    else render(sym, 'error');
                } catch (e) { render(sym, 'error'); }
            },
            onerror: () => { done(); render(sym, 'error'); },
            ontimeout: () => { done(); render(sym, 'waking'); },
        });
    }

    // ── Loop ──────────────────────────────────────────────────────────────
    let lastSym = null;
    function tick() {
        const sym = currentSymbol();
        if (!sym) return;
        lastSym = sym;
        fetchPanel(sym);
    }
    // Реагуємо і на періодику, і на зміну символу без перезавантаження (TV SPA).
    setInterval(() => {
        const sym = currentSymbol();
        if (sym && sym !== lastSym) { tick(); }   // символ змінився → одразу
        else fetchPanel(sym || lastSym || '');     // інакше просто оновлюємо
    }, POLL_MS);

    // Старт (дати TV прогрузитись)
    setTimeout(tick, 1500);
})();
