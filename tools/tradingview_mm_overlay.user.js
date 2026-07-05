// ==UserScript==
// @name         SVV ММ overlay for TradingView
// @namespace    svv-webhook-bot
// @version      1.0.0
// @description  Показує реальний ММ (liquidation-fuel) із SVV WebHook BOT поверх графіка TradingView для поточної монети.
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
    // TradingView symbol → base coin the bot knows (VELVETUSDT).
    function currentSymbol() {
        let raw = '';
        try { raw = new URLSearchParams(location.search).get('symbol') || ''; } catch (e) {}
        if (!raw) {
            // Заголовок вкладки типу "VELVETUSDT.P · 15m — BYBIT"
            const m = (document.title || '').match(/[A-Z0-9]{2,20}(?:\.[A-Z]+)?/);
            raw = m ? m[0] : '';
        }
        try { raw = decodeURIComponent(raw); } catch (e) {}
        if (raw.indexOf(':') >= 0) raw = raw.split(':').pop();  // прибрати "BYBIT:"
        raw = raw.split('.')[0];                                // прибрати ".P" тощо
        return (raw || '').toUpperCase().trim();
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
    let badge, elSym, elMM, elDir, elSub, elFoot;
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
              '<span style="font-weight:800;letter-spacing:.3px">❤️ ММ</span>' +
              '<span id="svv-mm-sym" style="font-weight:700;color:#cbd5e1;font-size:11px"></span>' +
            '</div>' +
            '<div style="display:flex;align-items:baseline;gap:8px">' +
              '<span id="svv-mm-val" style="font-weight:900;font-size:22px">—</span>' +
              '<span id="svv-mm-dir" style="font-weight:700;font-size:12px"></span>' +
            '</div>' +
            '<div id="svv-mm-sub" style="font-size:10.5px;color:#9aa3b5;margin-top:2px"></div>' +
            '<div id="svv-mm-foot" style="font-size:9.5px;color:#6b7280;margin-top:3px"></div>';
        document.body.appendChild(badge);
        elSym = badge.querySelector('#svv-mm-sym');
        elMM = badge.querySelector('#svv-mm-val');
        elDir = badge.querySelector('#svv-mm-dir');
        elSub = badge.querySelector('#svv-mm-sub');
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

    function render(sym, state) {
        buildBadge();
        elSym.textContent = sym || '—';
        if (state === 'noconfig') {
            elMM.textContent = '⚙'; elMM.style.color = '#f59e0b';
            elDir.textContent = ''; elSub.textContent = 'Задай URL бота (2× клік по бейджу)';
            elFoot.textContent = '';
            return;
        }
        if (state === 'error') {
            elMM.textContent = '⚠'; elMM.style.color = '#f59e0b';
            elDir.textContent = ''; elSub.textContent = 'Бот недоступний за URL';
            elFoot.textContent = getBotUrl() || '';
            return;
        }
        const d = state || {};
        if (d.enabled === false) {
            elSub.textContent = 'FF вимкнено в боті';
        }
        const mm = (d.mm_str != null) ? Number(d.mm_str) : null;
        const dir = d.mm || d.fuel_status || null;
        elMM.textContent = (mm != null) ? (mm.toFixed(0) + '%') : '—';
        elMM.style.color = dirColor(dir);
        elDir.textContent = dirLabel(dir) + (mm != null ? ' · ' + band(mm).label : '');
        elDir.style.color = dirColor(dir);
        const parts = [];
        if (d.exhaustion != null) {
            const ex = Number(d.exhaustion).toFixed(1);
            const over = d.exhausted ? ' ⚠️>' + (d.max_exhaustion != null ? d.max_exhaustion + '%' : '') : '';
            parts.push('Виснаж ' + ex + '%' + over);
        }
        if (d.verdict) parts.push('🧠 ' + d.verdict);
        elSub.textContent = parts.join(' · ');
        const t = new Date();
        elFoot.textContent = 'оновлено ' + t.toLocaleTimeString();
    }

    // ── Fetch ─────────────────────────────────────────────────────────────
    function fetchPanel(sym) {
        const url = getBotUrl();
        if (!url) { render(sym, 'noconfig'); return; }
        GM_xmlhttpRequest({
            method: 'GET',
            url: url + '/api/fuel-filter/panel/' + encodeURIComponent(sym),
            timeout: 8000,
            onload: (r) => {
                try {
                    const j = JSON.parse(r.responseText);
                    if (j && j.ok !== false) render(sym, j);
                    else render(sym, 'error');
                } catch (e) { render(sym, 'error'); }
            },
            onerror: () => render(sym, 'error'),
            ontimeout: () => render(sym, 'error'),
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
