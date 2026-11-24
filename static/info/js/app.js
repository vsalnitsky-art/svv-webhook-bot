// ─────────────────────────────────────────────────────────────────────────
//  VSV — особистий кабінет користувача (інфосайт).
// ─────────────────────────────────────────────────────────────────────────

(function () {
  "use strict";

  var CFG = window.INFOSITE_CONFIG || {};
  var BASE = (CFG.BOT_API_BASE || "").replace(/\/+$/, "") || window.location.origin;
  var REFRESH = CFG.REFRESH_MS || 5000;
  var FORCED_SYMBOL = (CFG.POTENTIAL_SYMBOL || "").toUpperCase().trim();
  var BOT_LINK = (CFG.BOT_TELEGRAM_LINK || "").trim();
  var API_KEY = (CFG.INFO_API_KEY || "").trim();
  var CROSS = BASE !== window.location.origin;

  function botUrl(p) { return BASE + p; }

  var TOK_KEY = "vsv_info_token";
  function getTok() { try { return localStorage.getItem(TOK_KEY) || ""; } catch (e) { return ""; } }
  function setTok(t) { try { t ? localStorage.setItem(TOK_KEY, t) : localStorage.removeItem(TOK_KEY); } catch (e) {} }
  (function captureToken() {
    var h = window.location.hash || "";
    var m = h.match(/[#&]it=([^&]+)/);
    if (m) {
      setTok(decodeURIComponent(m[1]));
      try { history.replaceState(null, "", window.location.pathname + window.location.search); } catch (e) {}
    }
  })();
  function logout() {
    var go = function () {
      setTok("");
      window.location.href = botUrl("/logout?next=info");
    };
    apiPost("/api/me/logout", {}).then(go, go);
  }

  function $(sel) { return document.querySelector(sel); }

  function _headers(extra) {
    var h = { "Accept": "application/json" };
    var t = getTok();
    if (t) h["X-Info-Token"] = t;
    if (API_KEY) h["X-API-Key"] = API_KEY;
    if (extra) for (var k in extra) h[k] = extra[k];
    return h;
  }

  var CREDS = CROSS ? "omit" : "same-origin";
  var API_TIMEOUT = Number(CFG.API_TIMEOUT_MS) || 12000;
  function isSyncPath(path) {
    return path.indexOf("/api/info/") === 0 || path === "/api/me" ||
      path === "/api/me/watchlist" || path === "/api/me/notify" ||
      path === "/api/me/trading" || path === "/api/me/tm";
  }
  function markSync(path) {
    if (!isSyncPath(path)) return;
    var lu = $("#last-update");
    if (lu) lu.textContent = "оновлено: " + new Date().toLocaleTimeString(window.VSV_I18N ? window.VSV_I18N.locale() : "uk-UA");
    if (started) setConn("ok", "онлайн");
  }
  function markSyncError(path, e) {
    if (!isSyncPath(path)) return;
    if (e && (e.status === 401 || e.status === 403)) {
      if (!appRevealed) return;                 // первинний boot сам покаже вхід / відмову
      var deny = e.status === 403 && DENY_TXT[e.code];
      if (getTok()) setTok("");
      showData(false, deny ? e.code : "");
      renderAuth(null);
      setConn(deny ? "err" : "wait", deny ? "доступ закрито" : "потрібен вхід");
      return;
    }
    if (started) setConn("err", "немає зв’язку");
  }

  function api(path) {
    var ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
    var tm = ctl ? setTimeout(function () { ctl.abort(); }, API_TIMEOUT) : null;
    return fetch(BASE + path, {
      headers: _headers(), credentials: CREDS, cache: "no-store",
      signal: ctl ? ctl.signal : undefined
    }).then(function (r) {
        if (tm) { clearTimeout(tm); tm = null; }
        if (!r.ok) {
          var e = new Error("HTTP " + r.status); e.status = r.status;
          return r.json().then(function (j) { e.code = j && j.error; throw e; },
                               function () { throw e; });
        }
        return r.json();
      }).then(function (d) { markSync(path); return d; }, function (e) {
        if (tm) clearTimeout(tm);
        markSyncError(path, e);
        throw e;
      });
  }

  function apiPost(path, body) {
    return fetch(BASE + path, {
      method: "POST", credentials: CREDS,
      headers: _headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(body || {})
    }).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); });
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  /* SpotHQ Cryptocurrency Icons: SVG за тикером; невідомі / нові монети
     автоматично отримують локальний generic без битих зображень. */
  function coinIcon(sym, extra) {
    var s = String(sym || "").toUpperCase().replace(/\.P$/, "");
    var m = s.match(/^(.*?)(USDT|USDC|USD)$/), base = m && m[1] ? m[1] : s;
    base = base.replace(/^(?:1000000|100000|10000|1000|1M|1K)(?=[A-Z])/, "");
    if (base === "XBT") base = "BTC";
    var key = base.toLowerCase().replace(/[^a-z0-9]/g, "");
    var src = key ? "https://cdn.jsdelivr.net/gh/spothq/cryptocurrency-icons@master/svg/color/" + key + ".svg" : "img/coin-generic.svg";
    var seed = 0; for (var i = 0; i < s.length; i++) seed = (seed + s.charCodeAt(i) * (i + 3)) % 37;
    return '<span class="coin-ic' + (extra ? " " + esc(extra) : "") + '" data-coin="' + esc(s) + '" aria-hidden="true" style="--coin-delay:-' + (seed / 10).toFixed(1) + 's">' +
      '<img src="' + src + '" alt="" loading="eager" decoding="async" referrerpolicy="no-referrer" onerror="this.onerror=null;this.src=\'img/coin-generic.svg\'"></span>';
  }
  function keepCoinIcons(host, old) {
    host.querySelectorAll(".coin-ic[data-coin]").forEach(function (el) {
      var prev = old[el.getAttribute("data-coin")];
      if (prev) el.parentNode.replaceChild(prev, el);
    });
  }

  function tvSym(sym, inBot) {
    var s = String(sym || "").toUpperCase();
    var url = "https://ru.tradingview.com/chart/?symbol=BYBIT:" + s + ".P";
    var m = s.match(/^(.*?)(USDT|USDC\.P|USDC|USD)$/);
    var bCls = "sym-b" + (inBot ? " in-bot" : "");
    var inner = m && m[1] ? '<b class="' + bCls + '">' + esc(m[1]) + '</b><small class="sym-q">' + esc(m[2]) + "</small>" : esc(s);
    return '<a href="' + url + '" target="tvchart" rel="noopener" ' +
      'title="Графік BYBIT:' + s + '.P (USDT-перпетуал) у TradingView" ' +
      'style="color:inherit;text-decoration:none;border-bottom:1px dotted rgba(255,255,255,0.4)">' +
      inner + "</a>";
  }

  var SIGNAL_BADGES_JS = { "choch": "🟦 CHoCH", "choch_bos": "🟦 CHoCH+BOS", "vob_alert": "🟪 Volumized OB", "vob": "💰 Volumized OB", "mm_vob": "🧮 VOB з VSV-монітора", "liq_vob": "💧 VOB + Сканер ліквідності", "ob_alert": "🆕 Новий OB", "ai": "🤖 Сигнал ШІ", "anom": "⚠ Аномалії обсягу", "poc": "🎯 POC-сетап", "opp": "🔄 Реверс", "external": "🔌 Зовнішня", "manual": "✋ Ручний", "manual_ui": "✋ Ручний", "manual_ui_overflow": "✋ Ручний" };
  var ENGINE_BADGES_JS = { "Q1": "🎯 Черга-1", "Q2": "🎯 Черга-2", "Q3": "🎯 Готовність", "Q4": "🎯 Черга-4", "Q3-VOB(funding)": "💰 VOB+Шари", "POC-сетап": "🎯 POC-сетап", "MMM": "🧮 VSV-монітор", "MMN": "🧮 VSV-Trend", "MMC": "⚡ VSV-CTR", "MMF": "🔮 VSV-Local", "MMP": "🧭 VSV-Pull", "MMD": "🧠 VSV-Decision", "MMR": "🧩 VSV-Готовність", "MMS": "🔔 VSV-SBR", "XSPR": "🔭 Пружина-скан", "SPR": "🌀 Пружина", "MIR": "🪞 Дзеркало бота", "EXH": "🔥 Виснаженість", "FF": "🔥 FF", "direct": "⚡ Прямий" };
  var SIGNAL_ICON_JS = { "choch": "🟦", "choch_bos": "🟦", "vob_alert": "🟪", "vob": "💰", "mm_vob": "🧮", "liq_vob": "💧", "ob_alert": "🆕", "ai": "🤖", "anom": "⚠", "poc": "🎯", "opp": "🔄", "external": "🔌", "manual": "✋", "manual_ui": "✋", "manual_ui_overflow": "✋", "Q3-VOB(funding)": "💰", "POC-сетап": "🎯", "EXH": "🔥", "FF": "🔥", "MMM": "🧮", "MMN": "🧮", "MMC": "⚡", "MMF": "🔮", "MMP": "🧭", "MMD": "🧠", "MMR": "🧩", "MMS": "🔔", "XSPR": "🔭", "SPR": "🌀" };
  var ENGINE_ICON_OVERRIDE_JS = { "MMM": "🧮", "MMN": "🧮", "MMC": "⚡", "MMF": "🔮", "MMP": "🧭", "MMD": "🧠", "MMR": "🧩", "MMS": "🔔", "XSPR": "🔭", "SPR": "🌀" };
  
  function signalCodeOf(raw) { return raw ? String(raw).split(" · ")[0].split(" → ")[0].trim() : ""; }
  function engineCodeOf(raw) { if (!raw) return ""; var parts = String(raw).split(" · ")[0].split(" → "); return parts.length > 1 ? parts[parts.length - 1].trim() : ""; }
  function prettyOpenedBy(raw) {
    if (!raw) return "";
    raw = String(raw);
    var tail = "", ci = raw.indexOf(" · ");
    if (ci >= 0) { tail = raw.slice(ci); raw = raw.slice(0, ci); }
    var parts = raw.split(" → ");
    var out = parts.map(function (p, i) {
      p = p.trim();
      return i === 0 ? (SIGNAL_BADGES_JS[p] || ENGINE_BADGES_JS[p] || p) : (ENGINE_BADGES_JS[p] || SIGNAL_BADGES_JS[p] || p);
    });
    return out.join(" → ") + tail;
  }
  // 🤖 Фірмова іконка ШІ (вимога 22.11): усі дані ШІ — головним логотипом.
  var AI_BRAND = '<img class="ai-brand" src="/static/info/img/logo.png" alt="ШІ">';
  function aiBrand(title) { return '<span class="ai-mark" title="' + esc(title || "ШІ бота") + '">' + AI_BRAND + "</span>"; }
  function signalIconHtml(raw) {
    if (!raw) return "";
    if (signalCodeOf(raw) === "ai") return aiBrand(prettyOpenedBy(raw));
    var icon = ENGINE_ICON_OVERRIDE_JS[engineCodeOf(raw)] || SIGNAL_ICON_JS[signalCodeOf(raw)] || "🏷";
    return '<span title="' + prettyOpenedBy(raw).replace(/"/g, "&quot;") + '" style="margin-right:3px;font-size:0.8rem;cursor:help">' + icon + "</span>";
  }

  function fmtPrice(p) {
    if (p == null || isNaN(p) || Number(p) <= 0) return "—";
    p = Number(p);
    if (p < 0.0001) return "$" + p.toFixed(8);
    if (p < 0.01) return "$" + p.toFixed(6);
    if (p < 1) return "$" + p.toFixed(5);
    if (p < 100) return "$" + p.toFixed(4);
    return "$" + p.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function setConn(state, text) {
    var el = $("#conn");
    if (el) { el.className = "conn conn--" + state; el.textContent = "● " + text; }
  }

  var NTF_KEY = "vsv_ntf_open";
  function ntfOpen() { try { return localStorage.getItem(NTF_KEY) === "1"; } catch (e) { return false; } }
  function ntfApply() {
    var on = ntfOpen(), b = $("#notify-body"), h = $("#notify-head");
    if (b) b.style.display = on ? "" : "none";
    if (h) h.setAttribute("aria-expanded", on ? "true" : "false");
  }
  document.addEventListener("click", function (e) {
    if (!e.target.closest || !e.target.closest("#notify-head")) return;
    try { localStorage.setItem(NTF_KEY, ntfOpen() ? "0" : "1"); } catch (x) {}
    ntfApply();
  });

  function renderNotify(n) {
    var card = $("#notify-card"), body = $("#notify-body");
    if (!card || !body) return;
    if (!n || n.ok === false) { card.style.display = "none"; return; }
    card.style.display = "";
    ntfApply();
    var linked = !!n.tg_linked;
    var linkA = BOT_LINK ? ' <a href="' + esc(BOT_LINK) + '" target="_blank" rel="noopener">відкрити бота</a>' : "";
    var warn = linked ? "" :
      '<p class="ntf-warn">⚠️ Telegram ще не прив’язано. Акаунт уже є? Натисніть ' +
      '<button class="tglink-btn" data-tglink="1">🔗 Привʼязати Telegram</button> — бот привʼяже його ' +
      'саме до цього акаунта (повторна реєстрація не потрібна).' + linkA + '</p>' +
      '<p class="ntf-warn" id="tglink-msg" style="display:none"></p>';
    var nSub = Object.keys(IW.sbr || {}).length;
    body.innerHTML =
      '<div class="ntf-info">' +
        '<p><b>👤 Особисто вам</b> (чат із ботом): сигнали <b>🔔 SBR</b> по монетах вашого ' +
        'WATCHLIST — натисніть 🔔 біля монети й оберіть <b>1H</b> та/або <b>4H</b>. ' +
        (nSub ? "Зараз підписано монет: <b>" + nSub + "</b>." : "Зараз підписок немає.") + "</p>" +
        '<p><b>👥 У Telegram-групі</b> (теми): <b>🧮 VSV-монітор</b> — зміни банерів VSV, ' +
        '<b>🔔 Сигнали</b> — SBR 1H/4H по основних монетах, <b>📈 Угоди</b> — відкриття й закриття угод бота. ' +
        'Група — лише для читання.</p>' +
        '<p>Доступ до групи — <b>після реєстрації та схвалення адміністратором</b>; посилання прийде автоматично. ' +
        'Якщо доступ закрито (вимкнено / строк минув) — доступу до групи немає, ' +
        'а особисто бот нічого не надсилає. Видалений акаунт прибирається з групи повністю.</p>' +
        warn + '</div>';
  }

  document.addEventListener("click", function (e) {
    var b = e.target.closest ? e.target.closest("[data-tglink]") : null;
    if (!b || b.disabled) return;
    b.disabled = true;
    var m = $("#tglink-msg");
    apiPost("/api/me/tg-link", {}).then(function (d) {
      b.disabled = false;
      if (!d || !d.ok || !d.url) {
        if (m) { m.style.display = ""; m.textContent = "⚠️ " + ((d && d.error) || "Не вдалось отримати посилання"); }
        return;
      }
      if (m) {
        m.style.display = "";
        m.innerHTML = 'Відкрийте бота й натисніть <b>Start</b>: <a href="' + esc(d.url) +
          '" target="_blank" rel="noopener"><b>@' + esc(d.bot || "bot") + '</b></a> (посилання одноразове, діє 30 хв).';
      }
      try { window.open(d.url, "_blank", "noopener"); } catch (x) {}
    }, function () {
      b.disabled = false;
      if (m) { m.style.display = ""; m.textContent = "⚠️ Немає звʼязку з ботом"; }
    });
  });

  var notifyBusy = false;
  function refreshNotify() {
    if (notifyBusy) return Promise.resolve(null);
    notifyBusy = true;
    return api("/api/me/notify").then(renderNotify).catch(function () { return null; })
      .then(function (d) { notifyBusy = false; return d; });
  }

  /* 🤖 ТОРГІВЛЯ З КАБІНЕТУ (вимога 24.11): угоди за сигналом ⚠ «сценарій входу»,
     книга real / paper на монету, ключі Bybit із шифруванням. Поки що — лише
     адміністратору (сервер вирішує: allowed). Правило: немає угоди → відкрити ·
     той самий бік → нічого · протилежний → закрити поточну і відкрити нову. */
  /* 🛡 Права інфосайту (вимога 24.11): що показувати користувачу вирішує
     адміністратор у боті («🛡 Керування користувачами»). Сервер перевіряє
     кожне право сам; тут лише ховаємо розділи / кнопки без доступу.
     Поки права ще не прийшли — нічого не ховаємо (PERMS = null). */
  var PERMS = null;
  var PERM_KEYS = ["sec_terminal", "sec_signals", "sec_paper", "ly_struct", "ly_ob", "ly_vob", "ly_imb", "ly_sbr", "ly_anom", "ly_hl",
                   "ly_poc", "ly_shl", "ly_pd", "ly_vol", "ly_liq", "ntf_sbr", "ntf_anom", "trade_paper", "trade_real", "trade_mirror"];
  // 🧪 PAPER — ЛИШЕ АДМІНІСТРАТОР (вимога 24.11, `user_perms.ADMIN_ONLY`): поки
  // права не прийшли або не дано явно — не показуємо нічого про paper.
  var PERM_ADMIN_ONLY = ["sec_paper", "trade_paper"];
  function pOk(k) {
    if (PERM_ADMIN_ONLY.indexOf(k) >= 0) return !!(PERMS && PERMS[k] === true);
    return !PERMS || PERMS[k] !== false;
  }
  function permSig(p) { return PERM_KEYS.map(function (k) { return p && p[k] === false ? "0" : "1"; }).join(""); }
  function applyPerms(p) {
    if (!p || typeof p !== "object") return;
    var was = permSig(PERMS), now = permSig(p);
    PERMS = p;
    var b = document.body;
    if (b) PERM_KEYS.forEach(function (k) { b.classList.toggle("perm-no-" + k, !pOk(k)); });
    var none = p.sec_terminal === false && p.sec_signals === false && p.sec_paper === false;
    var pn = $("#perm-note"); if (pn) pn.style.display = none ? "" : "none";
    if (was === now) return;
    if (!started) return;
    if (p.ly_liq === false && typeof LQ !== "undefined" && LQ.on && typeof liqOff === "function") liqOff();
    renderTfBar(); renderList(); IW.sig = ""; loadCoin(true); loadRec(); loadPaper(); loadRows();
  }

  var TR = { allowed: null, enabled: false, coins: {}, tm: {}, log: [], keys: null, reason: "", books: ["real"], keys_allowed: false, keys_owner: "", is_admin: false };
  // 👤 Мій Trade Manager (дзеркало бота, ВЛАСНІ ключі) — стан з /api/me/tm
  var UT = { d: null, at: 0 };
  var TR_KEY = "vsv_trade_open";
  var TR_BOOK = { paper: "🧪 paper", real: "🏦 реальна" };
  function trOpen() { try { return localStorage.getItem(TR_KEY) === "1"; } catch (e) { return false; } }
  function trApplyAcc() {
    var on = trOpen(), b = $("#trade-body"), h = $("#trade-head");
    if (b) b.style.display = on ? "" : "none";
    if (h) h.setAttribute("aria-expanded", on ? "true" : "false");
  }
  document.addEventListener("click", function (e) {
    if (!e.target.closest || !e.target.closest("#trade-head")) return;
    try { localStorage.setItem(TR_KEY, trOpen() ? "0" : "1"); } catch (x) {}
    trApplyAcc();
  });
  // POST, що повертає JSON навіть на 4xx — причину відмови показуємо людині.
  function apiPostJ(path, body) {
    return fetch(BASE + path, {
      method: "POST", credentials: CREDS,
      headers: _headers({ "Content-Type": "application/json" }),
      body: JSON.stringify(body || {})
    }).then(function (r) { return r.json().catch(function () { return { ok: false, error: "HTTP " + r.status }; }); });
  }
  function trApply(d) {
    if (!d) return;
    TR.allowed = !!d.allowed; TR.reason = d.reason || "";
    TR.keys_allowed = !!d.keys_allowed; TR.keys_owner = d.keys_owner || (d.keys_allowed ? "bot" : "");
    TR.is_admin = !!d.is_admin;
    if (Array.isArray(d.books) && d.books.length) TR.books = d.books.slice();
    if (d.allowed) {
      TR.enabled = !!d.enabled; TR.coins = d.coins || {}; TR.tm = d.tm || {}; TR.log = d.log || [];
    }
  }
  function trCoinOn(sym) { return !!(TR.allowed && TR.coins[sym]); }
  function trDefBook(sym) { return TR.coins[sym] || TR.books[0] || "real"; }
  function trBookSel(sym, attr) {
    var b = trDefBook(sym);
    // 🧪 paper — лише адміністратор (24.11): одна дозволена книга → без вибору
    if (TR.books.length < 2) return '<span class="tr-book-one tr-book-' + b + '">' + esc(TR_BOOK[b] || b) + "</span>";
    return '<select ' + attr + '="' + esc(sym) + '" class="tr-book tr-book-' + b + '">' +
      TR.books.map(function (k) { return '<option value="' + k + '"' + (k === b ? " selected" : "") + ">" + TR_BOOK[k] + "</option>"; }).join("") + "</select>";
  }
  // ✋ підтвердження для реальних грошей — і при виборі книги, і при вмиканні
  function trConfirmReal(sym) {
    return window.confirm("🏦 РЕАЛЬНА угода по " + sym + " за сигналом ⚠ «сценарій входу»?\n\n" +
      (TR.is_admin ? "Trade Manager бота відкриватиме реальні угоди на Bybit (ключі бота)"
                   : "ВАШ Trade Manager відкриватиме реальні угоди на ВАШОМУ рахунку Bybit (ваші ключі)") +
      ": протилежний сигнал закриє поточну угоду і відкриє нову.");
  }
  function trSave(body) {
    return apiPostJ("/api/me/trading", body).then(function (d) {
      if (d && d.ok) trApply(d);
      else trMsg("⚠️ " + ((d && (d.reason || d.error)) || "не вдалося зберегти"), true);
      renderTrade(); renderList();
      return d;
    }, function () { trMsg("⚠️ Немає звʼязку з ботом", true); });
  }
  function trSetCoin(sym, on, book) {
    if (on && book === "real" && !trConfirmReal(sym)) { renderTrade(); renderList(); return; }
    return trSave({ coin: { symbol: sym, on: on, book: book } });
  }
  function trMsg(t, bad) {
    var m = $("#tr-msg"); if (!m) return;
    m.style.display = t ? "" : "none"; m.className = "tr-msg" + (bad ? " bad" : ""); m.textContent = t || "";
  }
  function trUserKeysHTML(k) {
    var has = !!k.has_keys, noMaster = !k.encryption_key_set;
    return '<div class="tr-keys">' +
      '<div class="tr-kv"><span>Біржа</span><b>Bybit (USDT-перпетуали) · ваш рахунок</b></div>' +
      '<div class="tr-kv"><span>Ваші ключі</span><b class="' + (has ? "pp-long" : "pp-short") + '">' + (has ? "🟢 підключено" : "🔴 не підключено") + "</b></div>" +
      (has ? '<div class="tr-kv"><span>API Key</span><code>' + esc(k.key_preview || "") + "</code></div>" : "") +
      (noMaster ? '<p class="ntf-warn">⚠️ На сервері не задано <code>ENCRYPTION_KEY</code> — зберегти ключі неможливо. Зверніться до адміністратора.</p>' : "") +
      '<form class="tr-kform" id="tr-kform" autocomplete="off">' +
        '<label>API Key<input type="password" id="tr-k" autocomplete="off" spellcheck="false" placeholder="ключ з вашого Bybit"></label>' +
        '<label>API Secret<input type="password" id="tr-s" autocomplete="off" spellcheck="false" placeholder="секрет з вашого Bybit"></label>' +
        '<div class="tr-kbtns"><button type="submit" class="tglink-btn"' + (noMaster ? " disabled" : "") + ">🔐 " + (has ? "Замінити" : "Підключити") + " ключі</button>" +
        (has ? '<button type="button" class="tglink-btn" data-trk="test">🔎 Перевірити</button><button type="button" class="tglink-btn tr-del" data-trk="clear">🗑 Видалити мої ключі</button>' : "") + "</div>" +
      "</form>" +
      '<p class="muted tr-note">🔐 Ваші ключі — лише для ВАШОГО Trade Manager: ними торгує тільки він і тільки на вашому рахунку. Ключі бота тут не використовуються ніколи. Шифруються на сервері (Fernet), відкритий текст не повертається. Потрібні дозволи <b>Read + Trade</b> (Unified Trading), без виводу коштів.</p>' +
      '<div id="tr-kmsg" class="tr-msg" style="display:none"></div></div>';
  }
  function trKeysHTML(k) {
    if (!k) return '<div class="muted">Завантаження стану ключів…</div>';
    if (k.owner === "user") return trUserKeysHTML(k);
    var SRC = { db_encrypted: "🟢 збережено в боті (зашифровано)", env_plain: "🟡 змінна середовища (без шифрування)",
                env_encrypted: "🟢 змінна середовища (зашифровано)", none: "🔴 не задано" };
    var noMaster = !k.encryption_key_set;
    return '<div class="tr-keys">' +
      '<div class="tr-kv"><span>Біржа</span><b>Bybit (USDT-перпетуали)</b></div>' +
      '<div class="tr-kv"><span>Джерело ключів</span><b>' + esc(SRC[k.source] || k.source || "—") + "</b></div>" +
      '<div class="tr-kv"><span>API Key</span><code>' + esc(k.key_preview || "none") + "</code></div>" +
      (noMaster ? '<p class="ntf-warn">⚠️ На сервері не задано <code>ENCRYPTION_KEY</code> — зашифрувати й зберегти ключі неможливо. Додайте змінну середовища й перезапустіть бота.</p>' : "") +
      '<form class="tr-kform" id="tr-kform" autocomplete="off">' +
        '<label>API Key<input type="password" id="tr-k" autocomplete="off" spellcheck="false" placeholder="ключ з Bybit"></label>' +
        '<label>API Secret<input type="password" id="tr-s" autocomplete="off" spellcheck="false" placeholder="секрет з Bybit"></label>' +
        '<div class="tr-kbtns"><button type="submit" class="tglink-btn"' + (noMaster ? " disabled" : "") + '>🔐 Зашифрувати і зберегти</button>' +
        '<button type="button" class="tglink-btn" data-trk="test">🧪 Перевірити</button>' +
        (k.db_has_stored ? '<button type="button" class="tglink-btn tr-del" data-trk="clear">🗑 Видалити з бота</button>' : "") + "</div>" +
      "</form>" +
      '<p class="muted tr-note">Ключі шифруються на сервері (Fernet) і зберігаються в базі бота — відкритий текст не повертається і не пишеться в лог. Це ті самі ключі, якими торгує бот. Потрібні дозволи <b>Read + Trade</b> (Unified Trading), без виводу коштів; бажано обмежити IP сервера.</p>' +
      '<div id="tr-kmsg" class="tr-msg" style="display:none"></div></div>';
  }
  function trLogHTML() {
    if (!TR.log.length) return '<div class="muted">Дій ще не було — угода зʼявиться на першому новому сигналі.</div>';
    var ST = { opened: "✅ відкрито", reversed: "🔄 реверс", hold: "⏸ уже в угоді", rejected: "⛔ не відкрито" };
    return '<table class="pp-tbl tr-log"><thead><tr><th>Час</th><th>Монета</th><th>Сигнал</th><th>Книга</th><th>Дія</th><th>Деталі</th></tr></thead><tbody>' +
      TR.log.map(function (r) {
        return "<tr><td>" + esc(new Date((r.ts || 0) * 1000).toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" })) +
          '</td><td class="sym">' + esc(String(r.symbol || "").replace(/USDT$/, "")) +
          '</td><td class="' + (r.side === "LONG" ? "pp-long" : "pp-short") + '">' + esc(r.side || "") +
          "</td><td>" + esc(TR_BOOK[r.book] || r.book || "") + "</td><td>" + esc(ST[r.status] || r.status || "") +
          '</td><td class="muted">' + esc(r.reason || "") + "</td></tr>";
      }).join("") + "</tbody></table>";
  }
  var UT_SET = [["fixed_usd_amount", "Сума на угоду, $", "real"], ["leverage", "Плече (реальні)", "real"],
                ["max_open_positions", "Макс. угод", "real"], ["paper_deposit", "Paper-депозит, $", "paper"],
                ["paper_margin_pct", "Paper: % на угоду", "paper"], ["paper_leverage", "Paper: плече", "paper"]];
  function utPx(v) { return v == null || v === "" ? "—" : (typeof fmtPrice === "function" ? fmtPrice(+v) : String(v)); }
  function utPosRows(rows, book) {
    return (rows || []).map(function (p) {
      var sym = String(p.symbol || ""), pnl = +p.pnl_pct || 0;
      return '<tr><td class="sym">' + esc(sym.replace(/USDT$/, "")) + (p.mirror_bot ? ' <span title="🪞 повтор угоди бота">🪞</span>' : "") +
        (p.cab_avg_n ? ' <span class="pp-avg" title="додавань">➕' + (+p.cab_avg_n) + "</span>" : "") +
        "</td><td>" + esc(TR_BOOK[book]) + '</td><td class="' + (p.side === "LONG" ? "pp-long" : "pp-short") + '">' + esc(p.side || "") +
        '</td><td class="r">' + utPx(p.entry_price) + '</td><td class="r">' + utPx(p.current_price) +
        '</td><td class="r ' + (pnl >= 0 ? "pp-long" : "pp-short") + '">' + (pnl >= 0 ? "+" : "") + pnl.toFixed(2) + "%</td>" +
        '<td class="r"><button class="tglink-btn ut-act" data-utadd="' + esc(sym) + '" data-utbook="' + book + '" title="Додати до угоди (розмір — як у нової)">➕</button> ' +
        '<button class="tglink-btn tr-del ut-act" data-utclose="' + esc(sym) + '" data-utbook="' + book + '" title="Закрити угоду">✕</button></td></tr>';
    }).join("");
  }
  function utHTML(d) {
    if (!d) return '<div class="muted">Завантаження вашого Trade Manager…</div>';
    var s = d.settings || {}, m = d.meta || {}, bk = d.books || [];
    var hasReal = bk.indexOf("real") >= 0, hasPaper = bk.indexOf("paper") >= 0;
    var w = d.wallet || {};
    var eq = w.ok ? (w.equity != null ? w.equity : w.wallet) : null;
    var tg = function (id, on, lbl, dis, tip) {
      return '<label class="tr-chk" title="' + esc(tip || "") + '"><input type="checkbox" id="' + id + '"' + (on ? " checked" : "") + (dis ? " disabled" : "") + "> " + lbl + "</label>";
    };
    var fields = UT_SET.filter(function (f) { return bk.indexOf(f[2]) >= 0; }).map(function (f) {
      return '<label>' + esc(f[1]) + '<input type="text" inputmode="decimal" data-utset="' + f[0] + '" value="' + esc(s[f[0]] == null ? "" : String(s[f[0]])) + '"></label>';
    }).join("");
    var pos = utPosRows(d.positions, "real") + utPosRows(d.shadow_positions, "paper");
    var cl = (d.closed || []).map(function (c) { c._b = "real"; return c; }).concat((d.shadow_closed || []).map(function (c) { c._b = "paper"; return c; }))
      .sort(function (a, b) { return (b.closed_at || 0) - (a.closed_at || 0); }).slice(0, 12);
    var lg = (d.log || []).slice(0, 15);
    return '<div class="ut-box">' +
      '<div class="tr-status">' +
        (hasReal ? tg("ut-real", d.enabled, "<b>🏦 Реальна торгівля</b>", !d.has_key, "Угоди на вашому рахунку Bybit, вашими ключами") : "") +
        (hasPaper ? tg("ut-paper", d.test_mode, "<b>🧪 Paper (Test Mode)</b>", false, "Тестова книга з вашим віртуальним депозитом") : "") +
        "<span>Ключі: <b class=\"" + (d.has_key ? "pp-long\">ваші ✓" : "pp-short\">немає") + "</b></span>" +
        (eq != null ? "<span>Баланс: <b>$" + (+eq).toFixed(2) + "</b></span>" : "") +
        '<span class="muted">' + (d.running ? "● працює" : "○ зупинено") + "</span></div>" +
      '<div class="tr-status">' +
        tg("ut-mset", m.mirror_settings, "🪞 Алгоритм бота (дзеркало налаштувань)", false,
           "Правила виходу, SL/TP, автопілот, трейли — як у бота. Ваші: увімкнення, розмір, плече, ліміт" + (hasPaper ? ", paper-депозит." : ".")) +
        tg("ut-msig", m.mirror_signals && d.mirror_allowed, "🪞 Дублювати угоди бота", !d.mirror_allowed,
           d.mirror_allowed ? "Бот відкрив / закрив угоду → ваш Trade Manager повторює її на вашому рахунку" : "Не дозволено адміністратором") +
        (d.mirror_allowed && bk.length > 1 ? '<select id="ut-mbook" class="tr-book">' + bk.map(function (b) { return '<option value="' + b + '"' + (b === m.mirror_book ? " selected" : "") + ">" + TR_BOOK[b] + "</option>"; }).join("") + "</select>" : "") +
      "</div>" +
      (fields ? '<form class="tr-kform ut-form" id="ut-form">' + fields + '<div class="tr-kbtns"><button type="submit" class="tglink-btn">💾 Зберегти</button></div></form>' : "") +
      '<h3 class="pp-h">✋ Відкрити угоду</h3>' +
      '<div class="ut-open"><input type="text" id="ut-sym" list="ut-syms" placeholder="BTC" spellcheck="false">' +
        '<datalist id="ut-syms">' + (IW.wl || []).map(function (x) { return '<option value="' + esc(x.replace(/USDT$/, "")) + '">'; }).join("") + "</datalist>" +
        '<select id="ut-book" class="tr-book"' + (bk.length < 2 ? ' style="display:none"' : "") + ">" + bk.map(function (b) { return '<option value="' + b + '">' + TR_BOOK[b] + "</option>"; }).join("") + "</select>" +
        '<button class="tglink-btn ut-long" data-utopen="LONG">▲ LONG</button><button class="tglink-btn ut-short" data-utopen="SHORT">▼ SHORT</button></div>' +
      '<div id="ut-msg" class="tr-msg" style="display:none"></div>' +
      '<h3 class="pp-h">Відкриті угоди</h3><div class="pp-wrap tr-wrap"><table class="pp-tbl"><thead><tr><th>Монета</th><th>Книга</th><th>Бік</th><th class="r">Вхід</th><th class="r">Зараз</th><th class="r">PnL</th><th class="r">Дія</th></tr></thead><tbody>' +
        (pos || '<tr><td colspan="7" class="muted">Відкритих угод немає</td></tr>') + "</tbody></table></div>" +
      '<h3 class="pp-h">Закриті угоди</h3><div class="pp-wrap tr-wrap"><table class="pp-tbl"><thead><tr><th>Закрито</th><th>Монета</th><th>Книга</th><th>Бік</th><th class="r">PnL</th><th>Причина</th></tr></thead><tbody>' +
        (cl.map(function (c) {
          var v = +c.pnl_pct || 0;
          return "<tr><td>" + esc(c.closed_at ? locDM(c.closed_at) : "—") + '</td><td class="sym">' + esc(String(c.symbol || "").replace(/USDT$/, "")) +
            "</td><td>" + esc(TR_BOOK[c._b]) + '</td><td class="' + (c.side === "LONG" ? "pp-long" : "pp-short") + '">' + esc(c.side || "") +
            '</td><td class="r ' + (v >= 0 ? "pp-long" : "pp-short") + '">' + (v >= 0 ? "+" : "") + v.toFixed(2) + "%</td><td>" + esc((typeof PP_REASON !== "undefined" && PP_REASON[c.reason]) || c.reason || "") + "</td></tr>";
        }).join("") || '<tr><td colspan="6" class="muted">Ще немає</td></tr>') + "</tbody></table></div>" +
      '<h3 class="pp-h">Лог мого Trade Manager</h3><div class="pp-wrap tr-wrap"><table class="pp-tbl"><tbody>' +
        (lg.map(function (e) {
          return "<tr><td>" + esc(locDM(e.t || 0)) + '</td><td class="sym">' + esc(String(e.symbol || "").replace(/USDT$/, "")) + '</td><td class="muted">' + esc(e.detail || e.event || "") + "</td></tr>";
        }).join("") || '<tr><td class="muted">Подій ще немає</td></tr>') + "</tbody></table></div></div>";
  }
  function utMsg(t, bad) {
    var m = $("#ut-msg"); if (!m) return;
    m.style.display = t ? "" : "none"; m.className = "tr-msg" + (bad ? " bad" : ""); m.textContent = t || "";
  }
  function utPaint() { var h = $("#ut-host"); if (h) h.innerHTML = utHTML(UT.d); }
  var utBusy = false;
  function refreshUt() {
    if (utBusy || TR.is_admin || !TR.allowed || (TR.tm || {}).need_keys) return Promise.resolve(null);
    utBusy = true;
    return api("/api/me/tm").then(function (d) {
      if (d && d.need_keys) { TR.tm = TR.tm || {}; TR.tm.need_keys = true; renderTrade(); return d; }
      UT.d = d; UT.at = Date.now(); utPaint(); return d;
    }).catch(function () { return null; }).then(function (d) { utBusy = false; return d; });
  }
  function utPost(path, body, okMsg) {
    return apiPostJ(path, body).then(function (d) {
      if (d && d.ok !== false) { if (d.settings || d.meta) UT.d = Object.assign(UT.d || {}, d); utPaint();
        utMsg(d.note ? "⚠️ " + d.note : (okMsg || ""), !!d.note); refreshUt(); }
      else utMsg("⚠️ " + ((d && (d.reason || d.error)) || "не вдалося"), true);
      return d;
    }, function () { utMsg("⚠️ Немає звʼязку з ботом", true); });
  }
  setInterval(function () { if (!document.hidden && trOpen() && UT.d) refreshUt(); }, 10000);
  document.addEventListener("submit", function (e) {
    if (!e.target || e.target.id !== "ut-form") return;
    e.preventDefault();
    var st = {};
    e.target.querySelectorAll("[data-utset]").forEach(function (i) {
      var v = parseFloat(String(i.value || "").replace(",", ".").trim());
      if (isFinite(v)) st[i.getAttribute("data-utset")] = v;
    });
    utPost("/api/me/tm/settings", { settings: st }, "✅ Збережено");
  });
  document.addEventListener("click", function (e) {
    var t = e.target.closest ? e.target : null; if (!t) return;
    var o = t.closest("[data-utopen]");
    if (o) {
      var sym = String(($("#ut-sym") || {}).value || "").trim().toUpperCase().replace(/[^A-Z0-9]/g, "");
      if (!sym) { utMsg("Вкажіть монету", true); return; }
      if (!/USDT$/.test(sym)) sym += "USDT";
      var sd = o.getAttribute("data-utopen"), bk = ($("#ut-book") || {}).value || "real";
      if (!window.confirm((bk === "real" ? "🏦 РЕАЛЬНА" : "🧪 Paper") + " угода " + sd + " " + sym + "?" + (bk === "real" ? "\n\nНа ВАШОМУ рахунку Bybit, вашими ключами." : ""))) return;
      utPost("/api/me/tm/open", { symbol: sym, side: sd, book: bk }, "✅ " + sym + " " + sd + " відкрито"); return;
    }
    var a = t.closest("[data-utadd]");
    if (a) {
      var s1 = a.getAttribute("data-utadd"), b1 = a.getAttribute("data-utbook");
      if (!window.confirm("➕ Додати до угоди " + s1 + " (" + TR_BOOK[b1] + ")?\nРозмір — як у нової угоди.")) return;
      utPost("/api/me/tm/add", { symbol: s1, book: b1 }, "✅ Додано до " + s1); return;
    }
    var c = t.closest("[data-utclose]");
    if (c) {
      var s2 = c.getAttribute("data-utclose"), b2 = c.getAttribute("data-utbook");
      if (!window.confirm("✕ Закрити угоду " + s2 + " (" + TR_BOOK[b2] + ")?")) return;
      utPost("/api/me/tm/close", { symbol: s2, book: b2 }, "✅ " + s2 + " закрито");
    }
  });
  document.addEventListener("change", function (e) {
    var t = e.target; if (!t || !t.id) return;
    if (t.id === "ut-real") {
      if (t.checked && !window.confirm("Увімкнути РЕАЛЬНУ торгівлю вашого Trade Manager?\n\nУгоди — на вашому рахунку Bybit, вашими ключами.")) { t.checked = false; return; }
      utPost("/api/me/tm/settings", { settings: { enabled: t.checked } }); return;
    }
    if (t.id === "ut-paper") { utPost("/api/me/tm/settings", { settings: { test_mode: t.checked } }); return; }
    if (t.id === "ut-mset") { utPost("/api/me/tm/settings", { meta: { mirror_settings: t.checked } }); return; }
    if (t.id === "ut-msig") {
      if (t.checked && ($("#ut-mbook") || {}).value === "real" && !window.confirm("Дублювати угоди бота на вашому РЕАЛЬНОМУ рахунку?")) { t.checked = false; return; }
      utPost("/api/me/tm/settings", { meta: { mirror_signals: t.checked, mirror_book: ($("#ut-mbook") || {}).value } }); return;
    }
    if (t.id === "ut-mbook") utPost("/api/me/tm/settings", { meta: { mirror_book: t.value } });
  });

  function renderTrade() {
    var card = $("#trade-card"), body = $("#trade-body");
    if (!card || !body || TR.allowed === null) return;
    // 🛡 немає права на торгівлю — картки не показуємо зовсім
    if (document.body) document.body.classList.toggle("perm-no-trade", !TR.allowed);
    if (!TR.allowed) { card.style.display = "none"; return; }
    card.style.display = ""; trApplyAcc();
    var st = $("#trade-state");
    var ttl = $("#trade-title"); if (ttl) ttl.textContent = TR.is_admin ? "🤖 Торгівля з кабінету" : "👤 Мій Trade Manager";
    // 🔐 TM користувача зʼявляється ЛИШЕ після підключення ВЛАСНИХ ключів API
    if (!TR.is_admin && (TR.tm || {}).need_keys) {
      if (st) { st.textContent = "🔐 ключі"; st.className = "tr-state"; }
      body.innerHTML = '<p class="tr-note">🔐 Підключіть <b>свої</b> ключі API Bybit — після цього зʼявиться <b>ваш особистий Trade Manager</b> (дзеркало бота, що торгує лише на вашому рахунку).</p>' +
        '<div id="tr-keys-host">' + trKeysHTML(TR.keys) + "</div>";
      if (!TR.keys) refreshKeys();
      return;
    }
    var n = Object.keys(TR.coins).length, nReal = Object.keys(TR.coins).filter(function (s) { return TR.coins[s] === "real"; }).length;
    if (st) st.textContent = TR.enabled ? (n ? "● " + n : "● 0") : "○ вимк.";
    if (st) st.className = "tr-state" + (TR.enabled ? " on" : "");
    var tm = TR.tm || {};
    var warn = [];
    var tmN = TR.is_admin ? "Trade Manager бота" : "ваш Trade Manager";
    if (TR.enabled && nReal && !tm.enabled) warn.push(tmN + " — реальна торгівля вимкнена: 🏦 реальні угоди не відкриються");
    if (TR.enabled && nReal && !tm.has_key) warn.push((TR.is_admin ? "ключі бота" : "ваші ключі") + " не задано — 🏦 реальні угоди не відкриються");
    if (TR.books.indexOf("paper") >= 0 && TR.enabled && n - nReal && !tm.test_mode) warn.push("Test Mode вимкнено — 🧪 paper-угоди не відкриються");
    var rows = (IW.wl || []).map(function (sym) {
      var on = !!TR.coins[sym];
      return "<tr><td class=\"sym\">" + esc(sym.replace(/USDT$/, "")) + '</td><td><label class="tr-chk"><input type="checkbox" data-trcoin="' + esc(sym) + '"' + (on ? " checked" : "") + "> торгувати</label></td><td>" +
        (on ? trBookSel(sym, "data-trbook") : '<span class="muted">—</span>') + "</td></tr>";
    }).join("");
    body.innerHTML =
      (TR.is_admin ? "" : '<div id="ut-host">' + utHTML(UT.d) + "</div>" +
        '<h3 class="pp-h">⚠ Сигнали «Аномалії обсягу» → ваш Trade Manager</h3>') +
      '<div class="tr-top"><label class="tr-master"><input type="checkbox" id="tr-master"' + (TR.enabled ? " checked" : "") +
        '> <b>Торгівля за сигналами</b></label><span class="muted">сигнал — ⚠ Аномалії обсягу · сценарій входу (LONG / SHORT)</span></div>' +
      '<div class="tr-rules"><span>🟢 немає угоди → <b>відкрити</b></span><span>⏸ угода в той самий бік → <b>нічого не робимо</b></span><span>🔄 протилежний сигнал → <b>закрити поточну і відкрити нову</b></span></div>' +
      (TR.is_admin ? '<div class="tr-status"><span>Trade Manager бота: <b class="' + (tm.enabled ? "pp-long" : "pp-short") + '">' + (tm.enabled ? "увімкнено" : "вимкнено") + "</b></span>" +
        '<span>Test Mode: <b class="' + (tm.test_mode ? "pp-long" : "pp-short") + '">' + (tm.test_mode ? "увімкнено" : "вимкнено") + "</b></span>" +
        '<span>Ключі бота: <b class="' + (tm.has_key ? "pp-long" : "pp-short") + '">' + (tm.has_key ? "є" : "немає") + "</b></span></div>" : "") +
      (warn.length ? '<p class="ntf-warn">⚠️ ' + esc(warn.join(" · ")) + "</p>" : "") +
      '<div id="tr-msg" class="tr-msg" style="display:none"></div>' +
      '<h3 class="pp-h">Монети WATCHLIST · куди йдуть сигнали</h3>' +
      '<div class="pp-wrap tr-wrap"><table class="pp-tbl"><thead><tr><th>Монета</th><th>Торгівля</th><th>Книга</th></tr></thead><tbody>' +
        (rows || '<tr><td colspan="3" class="muted">Додайте монети у WATCHLIST</td></tr>') + "</tbody></table></div>" +
      (TR.keys_allowed ? '<h3 class="pp-h">🔐 ' + (TR.is_admin ? "API-ключі БОТА" : "Ваші API-ключі") + '</h3><div id="tr-keys-host">' + trKeysHTML(TR.keys) + "</div>"
        : '<p class="muted tr-note">Дозволені книги: ' + esc(TR.books.map(function (b) { return TR_BOOK[b] || b; }).join(", ")) + ".</p>") +
      '<h3 class="pp-h">Останні дії</h3><div class="pp-wrap tr-wrap">' + trLogHTML() + "</div>";
  }
  var tradeBusy = false;
  function refreshTrade() {
    if (tradeBusy) return Promise.resolve(null);
    tradeBusy = true;
    return api("/api/me/trading").then(function (d) {
      trApply(d); renderTrade(); renderList();
      if (TR.allowed && TR.keys_allowed && !TR.keys) refreshKeys();
      if (TR.allowed && !TR.is_admin && !(TR.tm || {}).need_keys) refreshUt();
      return d;
    }).catch(function () { return null; }).then(function (d) { tradeBusy = false; return d; });
  }
  function refreshKeys() {
    api("/api/me/keys").then(function (k) {
      TR.keys = k; var h = $("#tr-keys-host"); if (h) h.innerHTML = trKeysHTML(k);
    }).catch(function () {});
  }
  function trKMsg(t, bad) {
    var m = $("#tr-kmsg"); if (!m) return;
    m.style.display = t ? "" : "none"; m.className = "tr-msg" + (bad ? " bad" : ""); m.textContent = t || "";
  }
  document.addEventListener("submit", function (e) {
    if (!e.target || e.target.id !== "tr-kform") return;
    e.preventDefault();
    var k = $("#tr-k"), s = $("#tr-s");
    apiPostJ("/api/me/keys", { api_key: (k.value || "").trim(), api_secret: (s.value || "").trim() }).then(function (d) {
      if (d && d.ok) { k.value = ""; s.value = ""; TR.keys = null; refreshKeys(); refreshTrade(); setTimeout(function () { trKMsg("✅ " + (d.message || "Збережено")); }, 400); }
      else trKMsg("⚠️ " + ((d && d.error) || "не вдалося зберегти"), true);
    }, function () { trKMsg("⚠️ Немає звʼязку з ботом", true); });
  });
  document.addEventListener("click", function (e) {
    var b = e.target.closest ? e.target.closest("[data-trk]") : null;
    if (!b) return;
    var a = b.getAttribute("data-trk");
    if (a === "clear" && !window.confirm(TR.is_admin ? "Видалити ключі Bybit з бота?\n\nБот візьме ключі зі змінних середовища (якщо задані), інакше реальна торгівля зупиниться."
        : "Видалити ВАШІ ключі Bybit?\n\nРеальна торгівля вашого Trade Manager зупиниться.")) return;
    b.disabled = true;
    apiPostJ(a === "test" ? "/api/me/keys/test" : "/api/me/keys/clear", {}).then(function (d) {
      b.disabled = false;
      if (a === "test") trKMsg(d && d.ok ? "✅ " + (d.message || "Ключі робочі") : "⚠️ " + ((d && (d.friendly || d.error || d.ret_msg)) || "перевірка не вдалась"), !(d && d.ok));
      else { TR.keys = null; refreshKeys(); refreshTrade(); setTimeout(function () { trKMsg(d && d.ok ? "✅ " + (d.message || "Видалено") : "⚠️ " + ((d && d.error) || "не вдалося"), !(d && d.ok)); }, 400); }
    }, function () { b.disabled = false; trKMsg("⚠️ Немає звʼязку з ботом", true); });
  });
  document.addEventListener("change", function (e) {
    var t = e.target;
    if (!t || !t.closest) return;
    if (t.id === "tr-master") {
      var nReal = Object.keys(TR.coins).filter(function (s) { return TR.coins[s] === "real"; });
      if (t.checked && nReal.length && !window.confirm("Увімкнути торгівлю за сигналами?\n\n🏦 Реальні угоди: " + nReal.join(", ").replace(/USDT/g, ""))) { t.checked = false; return; }
      trSave({ enabled: t.checked }); return;
    }
    var c = t.closest("[data-trcoin]");
    if (c) { trSetCoin(c.getAttribute("data-trcoin"), !!c.checked, trDefBook(c.getAttribute("data-trcoin"))); return; }
    var bk = t.closest("[data-trbook]") || t.closest("[data-belltradebook]");
    if (bk) {
      var sym = bk.getAttribute("data-trbook") || bk.getAttribute("data-belltradebook");
      if (bk.hasAttribute("data-belltradebook")) IW.bellOpen = sym;
      trSetCoin(sym, true, bk.value); return;
    }
    var bt = t.closest("[data-belltrade]");
    if (bt) {
      var s2 = bt.closest("[data-bellmenu]").getAttribute("data-bellmenu");
      IW.bellOpen = s2;
      trSetCoin(s2, !!bt.checked, trDefBook(s2));
    }
  });

  function renderAuth(me) {
    var el = $("#auth-links");
    if (!el) return;
    if (me && me.ok) {
      if (me.perms) applyPerms(me.perms);
      // ЗМІНА: Динамічна назва вкладки з ім'ям користувача
      var uname = me.email ? me.email.split('@')[0] : "Користувач";
      document.title = uname + " - Мій кабінет";
      
      var nc = $("#nav-cabinet"); if (nc) { nc.href = botUrl("/cabinet?from=info"); nc.hidden = false; }
      el.innerHTML =
        '<a class="who" href="' + botUrl("/cabinet?from=info") + '" title="Мій кабінет">👤 ' +
          esc(me.email || "") + (me.is_admin ? " · адмін" : "") + '</a>' +
        '<a href="#" class="lo">Вихід</a>';
    } else {
      var nc0 = $("#nav-cabinet"); if (nc0) nc0.hidden = true;
      var reg = BOT_LINK || botUrl("/register");
      var ext = BOT_LINK ? ' target="_blank" rel="noopener"' : "";
      el.innerHTML = '<a href="' + botUrl("/info-login") + '">Вхід</a>' +
                     '<a href="' + reg + '"' + ext + '>Реєстрація</a>';
    }
  }

  var authBusy = false;
  function refreshAuth() {
    if (authBusy) return Promise.resolve(null);
    authBusy = true;
    return api("/api/me").then(renderAuth).catch(function (e) {
      if (e && (e.status === 401 || e.status === 403)) renderAuth(null);
      return null;
    }).then(function (d) { authBusy = false; return d; });
  }

  var REC_MS = 10000, PROFILE_MS = 15000, ACCOUNT_MS = 30000, UNIVERSE_MS = 60000;
  var CHART_TFS = ["15m", "30m", "1h", "4h", "1d"];
  var OB_TFS = ["15m", "30m", "1h", "4h"];
  var VOB_TFS = ["5m", "15m", "30m", "1h", "4h"];
  // 📈 HH/HL/LH/LL — свінг-структура на графіку (15.11), свій TF, деф. вимкнено
  var HL_TFS = ["15m", "30m", "1h", "4h", "1d"];
  var POC_DAYS = [1, 2, 3, 5, 7, 14, 30];
  var POC_TFS = ["15m", "1h", "4h"];
  var IW = { wl: [], sel: "", rows: {}, max: 20, sig: "", coin: null, chart: null, series: null,
             prim: null, pocLine: null, chartKey: "", universe: [],
             tf: "15m", ob_tf: "1h", vob_tf: "15m", hl_tf: "off", sbr_tf: "chart", poc_days: 7, poc_tf: "1h", show_imb: true, show_sbr: false, show_shl: true, shl_tf: "1h", show_anom: false,
             sbr: {}, anom: [], tg_linked: true, bellOpen: "" };
  var SBR_TFS = ["15m", "30m", "45m", "1h", "4h", "1d"];   // TF особистих 🔔 підписок (17.11: усі TF рушія; сервер шле sbr_tfs)
  var SBR_MARK_PX = 14;                 // розмір позначки S/B на графіку (css px)
  /* 🔔 SBR на графіку — на ВСІХ TF графіка (вимога 16.11): «chart» = TF графіка. */
  var SBR_CHART_TFS = ["15m", "30m", "1h", "4h", "1d"];
  /* 👁 25.11: монета, відкрита подвійним кліком із VSV SIGNALS (може бути поза
     вашим WATCHLIST) — показується замість обраної, доки графік на весь екран. */
  function curSym() { return IW.view || IW.sel; }
  function iwKey(o) { o = o || IW; return [o.view || o.sel, o.tf, o.ob_tf, o.vob_tf, o.poc_days, o.poc_tf, o.hl_tf, o.sbr_tf, o.shl_tf, o.show_anom ? 1 : 0].join("|"); }
  function sbrTfOk(tf) { return SBR_TFS.indexOf(String(tf || "").toLowerCase()) >= 0; }
  function sbrEff() { return IW.sbr_tf && IW.sbr_tf !== "chart" ? IW.sbr_tf : IW.tf; }
  function sbrCtl() {
    var on = !!IW.show_sbr;
    return '<span class="iw-sel s-sbr"><button type="button" data-layer="show_sbr" class="ly-sbr ' + (on ? "on" : "") + '" title="SBR — Session Band Reversal: розворот від межі волатильної смуги">SBR</button>' +
      '<select id="iw-sbr-tf" title="TF сигналів SBR: «= графік» — TF графіка, або інший TF (мультитаймфреймний аналіз)" aria-label="Таймфрейм SBR"' + (on ? "" : " disabled") + ">" +
      '<option value="chart"' + (IW.sbr_tf === "chart" ? " selected" : "") + ">= графік</option>" +
      SBR_CHART_TFS.map(function (tf) { return '<option value="' + tf + '"' + (tf === IW.sbr_tf ? " selected" : "") + ">" + tfLbl(tf) + "</option>"; }).join("") + "</select></span>";
  }

  function tfLbl(tf) {
    tf = String(tf || "").toLowerCase();
    if (tf === "off") return "вимк.";
    return /[hd]$/.test(tf) ? tf.toUpperCase() : tf;
  }
  function tfWord(tf) {
    var m = String(tf || "").toLowerCase().match(/^(\d+)([mhd])$/);
    if (!m) return String(tf || "").toUpperCase();
    return m[1] + " " + ({ m: "Min", h: "H", d: "D" })[m[2]];
  }
  function fmtVol(v, dp) {
    v = Number(v); dp = dp == null ? 2 : dp;
    if (!isFinite(v) || v <= 0) return "";
    if (v >= 1e9) return (v / 1e9).toFixed(dp) + "B";
    if (v >= 1e6) return (v / 1e6).toFixed(dp) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(dp) + "K";
    return v >= 10 ? String(Math.round(v)) : v.toFixed(2);
  }

  function pdBadge(pd) {
    if (!pd) return '<span class="bdg mute" title="Premium / Discount ще розраховується">PD —</span>';
    var name = pd.zone === "discount" ? "Discount" : (pd.zone === "premium" ? "Premium" : "Equilibrium");
    var cls = pd.zone === "discount" ? "long" : (pd.zone === "premium" ? "short" : "mute");
    return '<span class="bdg hb ' + cls + '" title="Позиція ціни в дилінг-діапазоні ' + pd.tf + ': Premium — зона продажу, Discount — зона покупки, Equilibrium — 50%"><span class="d"></span>' +
      '<span class="hb-z">' + name + '</span><span class="hb-v">' + Math.round(pd.pct) + '%</span>' +
      '<span class="hb-tf">' + esc(pd.tf) + "</span></span>";
  }

  function pocBadge(poc, state) {
    if (!poc) {
      var t = state === "none" ? "POC немає даних" : "POC ⏳";
      return '<span class="bdg mute" title="Volume Profile: POC розраховується">' + t + "</span>";
    }
    var d = poc.dist_pct, rel = poc.rel;
    var cls = rel === "above" ? "short" : (rel === "below" ? "long" : "mute");
    var hint = rel === "above" ? "ціна над POC — перевага SHORT (mean reversion)" : (rel === "below" ? "ціна під POC — перевага LONG (mean reversion)" : "ціна на POC — баланс");
    var tip = "POC (Point of Control) — рівень максимального обсягу за " + poc.days + " дн · " + hint;
    return '<span class="bdg hb ' + cls + '" title="' + esc(tip) + '"><span class="hb-k k-poc">POC</span>' +
      '<span class="hb-px">' + fmtPrice(poc.poc).replace("$", "") + "</span>" +
      (d == null ? "" : '<span class="hb-v">' + (d >= 0 ? "+" : "") + Number(d).toFixed(2) + "%</span>") +
      '<span class="hb-tf">' + esc(poc.days) + "д·" + esc(poc.tf) + "</span></span>";
  }

  function sbrMove(side, trigger, price) {
    var tr = Number(trigger), px = Number(price);
    if (!(tr > 0) || !(px > 0)) return null;
    var pct = (px - tr) / tr * 100;
    var fav = Math.abs(pct) < 1e-9 ? null : (side === "LONG" ? pct > 0 : (side === "SHORT" ? pct < 0 : null));
    return { pct: pct, fav: fav };
  }

  function p2(n) { return ("0" + n).slice(-2); }
  function locHM(sec) { var d = new Date(sec * 1000); return p2(d.getHours()) + ":" + p2(d.getMinutes()); }
  function sct(s) { return s ? (s.ct || 0) : 0; }
  function locDM(sec) { var d = new Date(sec * 1000); return p2(d.getDate()) + "." + p2(d.getMonth() + 1); }
  var LOC_MON = ["січ", "лют", "бер", "кві", "тра", "чер", "лип", "сер", "вер", "жов", "лис", "гру"];
  function locTick(t, type) {
    var d = new Date(t * 1000);
    if (type === 0) return String(d.getFullYear());
    if (type === 1) return LOC_MON[d.getMonth()];
    if (type === 2) return String(d.getDate());
    if (type === 4) return locHM(t) + ":" + p2(d.getSeconds());
    return locHM(t);
  }
  function locFull(t) { return locDM(t) + "." + String(new Date(t * 1000).getFullYear()).slice(-2) + " " + locHM(t); }

  function sbrLastBadge(tf, s, price) {
    var T = tf.toUpperCase();
    if (!s) return '<span class="bdg mute sbr-last" title="SBR ' + T + ': сигнал розраховується">SBR ' + T + " ⏳</span>";
    if (s.none) return '<span class="bdg mute sbr-last" title="SBR ' + T + ': активного сигналу немає">SBR ' + T + " —</span>";
    var lg = s.side === "LONG", m = sbrMove(s.side, s.trigger, price);
    var cls = !m || m.fav == null ? "mute" : (m.fav ? "long" : "short");
    var pct = m ? (m.pct >= 0 ? "+" : "") + m.pct.toFixed(2) + "%" : "—";
    var tip = "Останній сигнал SBR " + T + (s.n ? " #" + s.n : "") + " · спрацював " + (sct(s) ? locDM(sct(s)) + " " + locHM(sct(s)) : "") +
      " · Trigger " + fmtPrice(s.trigger) + (s.score != null ? " · Score " + Number(s.score).toFixed(1) + "%" : "") +
      " → зараз " + (price ? fmtPrice(price) : "—") + " · " + (!m || m.fav == null ? "без руху" : (m.fav ? "у бік сигналу" : "проти сигналу"));
    return '<span class="bdg hb sbr-last ' + cls + '" title="' + esc(tip) + '">' +
      '<span class="hb-k k-sbr">SBR</span><span class="hb-tf">' + T + "</span>" +
      '<span class="sl-side ' + (lg ? "lg" : "sh") + '">' + (lg ? "▲ LONG" : "▼ SHORT") + "</span>" +
      '<span class="sl-pct">' + pct + "</span></span>";
  }

  function iwMsg(t, err) {
    var el = $("#iw-msg");
    if (el) { el.textContent = t || ""; el.className = "iw-msg" + (err ? " err" : ""); }
  }

  function livePx(sym) {
    var c = IW.coin;
    if (c && c.symbol === sym && c.ohlc && c.ohlc.length) return c.ohlc[c.ohlc.length - 1].close;
    return (IW.rows[sym] || {}).price;
  }
  function dirCls(sym) {
    var c = (IW.rows[sym] || {}).chg24;
    return c == null ? "" : (c > 0 ? " up" : (c < 0 ? " dn" : ""));
  }

  var MM_SYMS = ["BTCUSDT"];
  function mmCls(sym) {
    if (MM_SYMS.indexOf(sym) < 0) return "";
    var m = (IW.rows[sym] || {}).mm;
    if (!m) return "";
    return m.dir === "LONG" ? " mm-long" : (m.dir === "SHORT" ? " mm-short" : "");
  }
  function mmTip(sym) {
    if (MM_SYMS.indexOf(sym) < 0) return sym;
    var m = (IW.rows[sym] || {}).mm;
    if (!m) return "VSV (ліквідність): даних немає";
    if (m.dir === "LONG") return "VSV (ліквідність): кластери ліквідацій над ціною — тяга вгору, перевага LONG · сила " + m.str + "%";
    if (m.dir === "SHORT") return "VSV (ліквідність): кластери ліквідацій під ціною — тяга вниз, перевага SHORT · сила " + m.str + "%";
    return "VSV (ліквідність) ⚖ баланс — вираженої тяги немає · сила " + m.str + "%";
  }

  var IW_SORT = "default";
  try { IW_SORT = localStorage.getItem("vsv_watchlist_sort") || "default"; } catch (e) {}
  function renderList() {
    var host = $("#iw-list");
    if (!host) return;
    var c = $("#iw-count"); if (c) c.textContent = IW.wl.length + " / " + IW.max;
    if (!IW.wl.length) {
      host.innerHTML = '<div class="iw-empty">Список порожній — додайте монету вище.</div>';
      return;
    }
    var oldIcons = {};
    host.querySelectorAll(".coin-ic[data-coin]").forEach(function (el) { oldIcons[el.getAttribute("data-coin")] = el; });
    var sorted = IW.wl.slice();
    if (IW_SORT === "name") sorted.sort(function (a, b) { return a.localeCompare(b); });
    if (IW_SORT === "price") sorted.sort(function (a, b) { return (Number(livePx(b)) || 0) - (Number(livePx(a)) || 0); });
    if (IW_SORT === "change") sorted.sort(function (a, b) { var av = Number((IW.rows[a] || {}).chg24), bv = Number((IW.rows[b] || {}).chg24); return (isFinite(bv) ? bv : -Infinity) - (isFinite(av) ? av : -Infinity); });
    host.innerHTML = sorted.map(function (sym) {
      var r = IW.rows[sym] || {};
      var ext = r.src === "ext" ? '<small title="Поза WATCHLIST бота — розрахунок окремим потоком">🛰</small>' : "";
      var chg = r.chg24, bell = bellHTML(sym);
      var chgHtml = chg == null ? '<span class="iw-chg">—</span>'
        : '<span class="iw-chg ' + (chg > 0 ? "up" : (chg < 0 ? "dn" : "")) + '" title="Зміна за 24 год">' +
          (chg > 0 ? "+" : "") + Number(chg).toFixed(2) + "%</span>";
      return '<div class="iw-row' + (sym === IW.sel ? " sel" : "") + '" data-sym="' + esc(sym) + '">' +
        '<div class="iw-main"><span class="iw-sym' + mmCls(sym) + '" title="' + esc(mmTip(sym)) + '">' + coinIcon(sym) + '<b class="' + (r.src && r.src !== "ext" ? "in-bot" : "") + '">' + esc(sym.replace(/USDT$/, "")) + '</b><small>USDT</small>' + ext + "</span>" +
        '<span class="iw-quote"><span class="iw-price' + dirCls(sym) + '">' + (livePx(sym) ? fmtPrice(livePx(sym)) : "—") + "</span>" + chgHtml + '</span></div>' +
        bell.btn + '<button class="iw-del" data-del="' + esc(sym) + '" title="Видалити з WATCHLIST">×</button>' + bell.menu +
        (r.err ? '<div class="iw-err" title="' + esc(r.err) + '">⚠ ' + esc(r.err) + "</div>" : "") + "</div>";
    }).join("");
    keepCoinIcons(host, oldIcons);
  }

  /* Live-оновлення лише другого рядка: назва, іконка, дзвоник і весь
     перший рядок DOM не перестворюються та не перемальовуються. */
  function renderQuotes() {
    var host = $("#iw-list"); if (!host) return;
    host.querySelectorAll(".iw-row[data-sym]").forEach(function (row) {
      var sym = row.getAttribute("data-sym"), r = IW.rows[sym] || {}, px = livePx(sym);
      var name = row.querySelector(".iw-sym");
      if (name) {
        var title = mmTip(sym);
        if (name.title !== title) name.title = title;
        name.classList.toggle("mm-long", !!(r.mm && r.mm.dir === "LONG"));
        name.classList.toggle("mm-short", !!(r.mm && r.mm.dir === "SHORT"));
      }
      var price = row.querySelector(".iw-price"), chgEl = row.querySelector(".iw-chg");
      if (price) {
        var pt = px ? fmtPrice(px) : "—", pc = "iw-price" + dirCls(sym);
        if (price.textContent !== pt) price.textContent = pt;
        if (price.className !== pc) price.className = pc;
      }
      var chg = r.chg24;
      if (!chgEl && chg != null) {
        chgEl = document.createElement("span"); chgEl.className = "iw-chg";
        var quote = row.querySelector(".iw-quote"); if (quote) quote.appendChild(chgEl);
      }
      if (chgEl) {
        var ct = chg == null ? "—" : (chg > 0 ? "+" : "") + Number(chg).toFixed(2) + "%";
        var cc = "iw-chg" + (chg > 0 ? " up" : chg < 0 ? " dn" : "");
        if (chgEl.textContent !== ct) chgEl.textContent = ct;
        if (chgEl.className !== cc) chgEl.className = cc;
        if (chg == null) chgEl.removeAttribute("title"); else chgEl.title = "Зміна ціни за ~24 год";
      }
    });
    /* Живе сортування — переміщенням тих самих рядків, без innerHTML. */
    if (IW_SORT === "price" || IW_SORT === "change") {
      var rows = Array.prototype.slice.call(host.querySelectorAll(".iw-row[data-sym]")), previous = rows.slice();
      rows.sort(function (a, b) {
        var sa = a.getAttribute("data-sym"), sb = b.getAttribute("data-sym");
        if (IW_SORT === "price") return (Number(livePx(sb)) || 0) - (Number(livePx(a)) || 0);
        var av = Number((IW.rows[sa] || {}).chg24), bv = Number((IW.rows[sb] || {}).chg24);
        return (isFinite(bv) ? bv : -Infinity) - (isFinite(av) ? av : -Infinity);
      });
      var changed = rows.some(function (row, i) { return row !== previous[i]; });
      if (changed) rows.forEach(function (row) { host.appendChild(row); });
    }
  }

  var sortSelect = $("#iw-sort");
  if (sortSelect) {
    if (["default", "name", "price", "change"].indexOf(IW_SORT) < 0) IW_SORT = "default";
    sortSelect.value = IW_SORT;
    sortSelect.addEventListener("change", function () { IW_SORT = sortSelect.value; try { localStorage.setItem("vsv_watchlist_sort", IW_SORT); } catch (e) {} renderList(); });
  }

  function bellHTML(sym) {
    var okS = pOk("ntf_sbr"), okA = pOk("ntf_anom"), okT = !!TR.allowed;
    if (!okS && !okA && !okT) return { btn: "", menu: "" };      // 🛡 нічого не дозволено
    var tfs = okS ? (IW.sbr[sym] || []) : [];
    var an = okA && (IW.anom || []).indexOf(sym) >= 0;
    var trd = trCoinOn(sym), trBook = trDefBook(sym);
    var on = tfs.length > 0 || an || trd;
    var lbl = tfs.map(function (t) { return String(t).toUpperCase(); }).join("+") + (an ? (tfs.length ? "+" : "") + "⚠" : "") +
      (trd ? (tfs.length || an ? "+" : "") + (trBook === "real" ? "🏦" : "🧪") : "");
    var tip = on ? "У Telegram по " + sym + ": " + (tfs.length ? "SBR " + tfs.map(function (t) { return String(t).toUpperCase(); }).join("+") : "") +
      (an ? (tfs.length ? " · " : "") + "⚠ Аномалії обсягу" : "") : "Отримувати сигнали по " + sym + " у Telegram";
    var menu = IW.bellOpen === sym
      ? '<div class="iw-bell-menu" data-bellmenu="' + esc(sym) + '">' +
          '<div class="t">🔔 ' + esc(sym.replace(/USDT$/, "")) + "</div>" +
          (okS ? '<div class="t t2">SBR у Telegram</div><div class="iw-bell-tfs">' + SBR_TFS.map(function (tf) {
            return '<label><input type="checkbox" data-belltf="' + tf + '"' + (tfs.indexOf(tf) >= 0 ? " checked" : "") + "> " + tf.toUpperCase() + "</label>";
          }).join("") + "</div>" : "") +
          (okA ? '<div class="t t2">⚠ Аномалії обсягу у Telegram</div>' +
          '<label class="iw-bell-anom" title="Сетап на вхід за order flow: сценарій ШІ підтверджено (climax / абсорбція / follow-through), тригер свіжий — не старший за 2 свічки, фон order flow за 6 год не проти позиції."><input type="checkbox" data-bellanom="1"' + (an ? " checked" : "") + '><span class="iw-bell-anom-t">сценарій входу<small>LONG / SHORT</small></span></label>' +
          (an ? '<div class="iw-bell-fmt" title="Формат алерту в Telegram (для всіх інструментів): коротке — напрямок, інструмент, ціна сигналу й сетап; повне — ще й етапи сетапу та фон order flow за 6 год.">' +
            ["short", "full"].map(function (f) {
              return '<button type="button" data-bellanomfmt="' + f + '" class="' + ((IW.anom_fmt || "short") === f ? "on" : "") + '">' + (f === "short" ? "коротке" : "повне") + "</button>";
            }).join("") + "</div>" : "") : "") +
          (okT
            ? '<div class="t t2">🤖 Торгівля за сигналом ⚠</div><div class="iw-bell-trade"><label title="Автоторгівля за сетапом ⚠: немає позиції — вхід · сигнал у бік позиції — утримання, або усереднення (DCA) з кращої ціни · протилежний — реверс (закрити й відкрити зустрічну)"><input type="checkbox" data-belltrade="1"' + (trd ? " checked" : "") + "> торгувати</label>" +
                (trd ? trBookSel(sym, "data-belltradebook") : "") + "</div>" +
                (trd && !TR.enabled ? '<div class="w">⏸ Торгівлю вимкнено головним тумблером у картці «🤖 Торгівля з кабінету».</div>' : "")
            : "") +
          (IW.tg_linked ? "" : '<div class="w">⚠️ Telegram не прив’язано — сигнали не дійдуть. <button class="tglink-btn" data-tglink="1">🔗 Привʼязати</button></div>') +
          '<div class="n">Сигнали приходять у ваш особистий чат із ботом: SBR — після закриття бару, ⚠ — щойно склався сценарій входу.</div></div>'
      : "";
    return { btn: '<span class="iw-bell-wrap"><button type="button" class="iw-bell' + (on ? " on" : "") +
      '" data-bell="' + esc(sym) + '" title="' + esc(tip) + '"><svg class="iw-bell-ic" aria-hidden="true"><use href="#i-bell"/></svg>' +
      (lbl ? "<small>" + esc(lbl) + "</small>" : "") + "</button></span>", menu: menu };
  }

  var TF_LAST_KEY = "vsv_tf_last";
  function tfLast(k) {
    var d = k === "vob_tf" ? "15m" : "1h", L = k === "ob_tf" ? OB_TFS : (k === "hl_tf" ? HL_TFS : VOB_TFS);
    try { var v = (JSON.parse(localStorage.getItem(TF_LAST_KEY) || "{}") || {})[k]; if (L.indexOf(v) >= 0) return v; } catch (e) {}
    return d;
  }
  function tfLastSet(k, v) {
    try { var o = JSON.parse(localStorage.getItem(TF_LAST_KEY) || "{}") || {}; o[k] = v; localStorage.setItem(TF_LAST_KEY, JSON.stringify(o)); } catch (e) {}
  }

  function renderTfBar() {
    var host = $("#iw-tfs");
    if (host) host.innerHTML = CHART_TFS.map(function (tf) {
      return '<button type="button" data-tf="' + tf + '" class="' + (tf === IW.tf ? "on" : "") + '">' + tfLbl(tf) + "</button>";
    }).join("");
    ["ob_tf", "vob_tf"].forEach(function (k) {
      var on = IW[k] !== "off", bt = $("#iw-" + k.replace("_tf", "") + "-on");
      if (bt) { bt.className = "iw-tgl" + (on ? " on" : ""); bt.setAttribute("aria-pressed", on ? "true" : "false"); }
      var se = $("#iw-" + k.replace("_", "-")); if (se) se.disabled = !on;
    });
    var opt = function (list, cur) {
      if (cur === "off") cur = tfLast(list === OB_TFS ? "ob_tf" : "vob_tf");
      return list.map(function (tf) { return '<option value="' + tf + '"' + (tf === cur ? " selected" : "") + ">" + tfLbl(tf) + "</option>"; }).join("");
    };
    var o = $("#iw-ob-tf"); if (o) o.innerHTML = opt(OB_TFS, IW.ob_tf);
    var v = $("#iw-vob-tf"); if (v) v.innerHTML = opt(VOB_TFS, IW.vob_tf);
    var pd = $("#iw-poc-days");
    if (pd) pd.innerHTML = POC_DAYS.map(function (n) { return '<option value="' + n + '"' + (n === Number(IW.poc_days) ? " selected" : "") + ">" + n + " дн</option>"; }).join("");
    var pt = $("#iw-poc-tf");
    if (pt) pt.innerHTML = POC_TFS.map(function (tf) { return '<option value="' + tf + '"' + (tf === IW.poc_tf ? " selected" : "") + ">" + tfLbl(tf) + "</option>"; }).join("");
    var lay = $("#iw-layers");
    if (lay) lay.innerHTML =
      (pOk("ly_imb") ? '<button type="button" data-layer="show_imb" class="ly-imb ' + (IW.show_imb ? "on" : "") + '" title="IMB — імбаланс (Fair Value Gap) з розкладом обсягу покупців / продавців">IMB</button>' : "") +
      (pOk("ly_hl") ? hlCtl() : "") +
      (pOk("ly_sbr") ? sbrCtl() : "") +
      (pOk("ly_shl") ? shlCtl() : "") +
      (pOk("ly_anom") ? '<button type="button" data-layer="show_anom" class="ly-anom ' + (IW.show_anom ? "on" : "") + '" title="⚠ Сетапи на вхід за аномаліями обсягу (ті самі, що в Telegram): B — Buy / LONG, S — Sell / SHORT. Маркери накопичуються з моменту, коли інструмент узято на моніторинг (графік із цим шаром, Пружини або підписки).">⚠ Сигнали</button>' : "");
    liqUi();
  }

  // 🏁 High/Low + вибір TF (24.11, як у HH/HL і SBR).
  function shlCtl() {
    var on = !!IW.show_shl;
    return '<span class="iw-sel s-shl"><button type="button" data-layer="show_shl" class="ly-shl ' + (on ? "on" : "") + '" title="Strong / Weak High і Low — межі дилінг-діапазону (Weak — ціль ліквідності, Strong — захищений екстремум)">High/Low</button>' +
      '<select id="iw-shl-tf" title="Таймфрейм High/Low" aria-label="Таймфрейм High/Low"' + (on ? "" : " disabled") + ">" +
      HL_TFS.map(function (tf) { return '<option value="' + tf + '"' + (tf === IW.shl_tf ? " selected" : "") + ">" + tfLbl(tf) + "</option>"; }).join("") + "</select></span>";
  }

  // 📈 Кнопка HH/HL + вибір TF (поруч з IMB). Вимкнено = hl_tf "off" (сервер не рахує).
  function hlCtl() {
    var on = IW.hl_tf !== "off", cur = on ? IW.hl_tf : tfLast("hl_tf");
    return '<span class="iw-sel s-hl"><button type="button" id="iw-hl-on" class="iw-tgl ly-hl' + (on ? " on" : "") + '" data-tgl="hl_tf" aria-pressed="' + (on ? "true" : "false") +
      '" title="Ринкова структура: HH / HL — висхідна (bullish), LH / LL — низхідна (bearish)">HH/HL</button>' +
      '<select id="iw-hl-tf" title="Таймфрейм HH/HL" aria-label="Таймфрейм HH/HL"' + (on ? "" : " disabled") + ">" +
      HL_TFS.map(function (tf) { return '<option value="' + tf + '"' + (tf === cur ? " selected" : "") + ">" + tfLbl(tf) + "</option>"; }).join("") + "</select></span>";
  }

  function applyMine(d) {
    if (!d) return;
    IW.wl = d.wl || []; IW.max = d.max || IW.max;
    var changed = iwKey({ view: IW.view, sel: d.sel, tf: d.tf || IW.tf, ob_tf: d.ob_tf || IW.ob_tf, vob_tf: d.vob_tf || IW.vob_tf, poc_days: d.poc_days || IW.poc_days, poc_tf: d.poc_tf || IW.poc_tf, hl_tf: d.hl_tf || IW.hl_tf, sbr_tf: d.sbr_tf || IW.sbr_tf, shl_tf: d.shl_tf || IW.shl_tf, show_anom: typeof d.show_anom === "boolean" ? d.show_anom : IW.show_anom }) !== iwKey();
    IW.sel = d.sel || "";
    if (d.tf) IW.tf = d.tf;
    if (d.ob_tf) IW.ob_tf = d.ob_tf;
    if (d.vob_tf) IW.vob_tf = d.vob_tf;
    if (d.hl_tf) IW.hl_tf = d.hl_tf;
    if (d.sbr_tf) IW.sbr_tf = d.sbr_tf;
    if (d.poc_days) IW.poc_days = Number(d.poc_days);
    if (d.poc_tf) IW.poc_tf = d.poc_tf;
    if (typeof d.show_imb === "boolean") IW.show_imb = d.show_imb;
    if (typeof d.show_sbr === "boolean") IW.show_sbr = d.show_sbr;
    if (typeof d.show_shl === "boolean") IW.show_shl = d.show_shl;
    if (d.shl_tf) IW.shl_tf = d.shl_tf;
    if (typeof d.show_anom === "boolean") IW.show_anom = d.show_anom;
    if (d.sbr && typeof d.sbr === "object") IW.sbr = d.sbr;
    if (Array.isArray(d.anom)) IW.anom = d.anom.slice();
    if (d.anom_fmt) IW.anom_fmt = d.anom_fmt;
    if (Array.isArray(d.sbr_tfs) && d.sbr_tfs.length) SBR_TFS = d.sbr_tfs.slice();
    if (typeof d.tg_linked === "boolean") IW.tg_linked = d.tg_linked;
    if (d.perms) applyPerms(d.perms);
    renderList(); renderTfBar();
    if (changed) { IW.sig = ""; IW.coin = null; loadCoin(true); }
  }

  var mineBusy = false, mineVersion = 0;
  function saveMine(body) {
    mineVersion++;
    return apiPost("/api/me/watchlist", body).then(function (d) {
      if (d && d.error) iwMsg(d.error, true); else iwMsg("");
      applyMine(d); loadRows(); return d;
    }).catch(function () { iwMsg("не вдалося зберегти", true); });
  }

  function loadMine() {
    if (mineBusy) return Promise.resolve(null);
    mineBusy = true;
    var version = mineVersion;
    return api("/api/me/watchlist").then(function (d) {
      if (version === mineVersion) applyMine(d);
      return d;
    }).then(function (d) { mineBusy = false; return d; }, function (e) { mineBusy = false; throw e; });
  }

  var rowsBusy = false;
  function loadRows() {
    if (rowsBusy || !IW.wl.length || !pOk("sec_terminal")) return Promise.resolve(null);
    rowsBusy = true;
    return api("/api/info/rows?symbols=" + encodeURIComponent(IW.wl.join(","))).then(function (d) {
      (d.rows || []).forEach(function (r) { IW.rows[r.symbol] = r; });
      renderQuotes(); renderHead();
      return d;
    }).catch(function () { return null; }).then(function (d) { rowsBusy = false; return d; });
  }

  var universeBusy = false;
  function loadUniverse() {
    if (universeBusy) return Promise.resolve(null);
    universeBusy = true;
    return api("/api/info/universe").then(function (d) {
      IW.universe = d.symbols || [];
      var dl = $("#iw-universe");
      if (dl) dl.innerHTML = IW.universe.map(function (s) { return '<option value="' + esc(s.replace(/USDT$/, "")) + '">' + esc(s) + "</option>"; }).join("");
      return d;
    }).catch(function () { return null; }).then(function (d) { universeBusy = false; return d; });
  }

  document.addEventListener("submit", function (e) {
    if (!e.target || e.target.id !== "iw-add-form") return;
    e.preventDefault();
    var inp = $("#iw-add-in"); var v = (inp.value || "").trim();
    if (!v) return;
    inp.value = ""; saveMine({ add: v });
  });

  document.addEventListener("click", function (e) {
    if (!e.target.closest) return;
    var bell = e.target.closest("[data-bell]");
    if (bell) {
      e.stopPropagation();
      var bs = bell.getAttribute("data-bell");
      IW.bellOpen = IW.bellOpen === bs ? "" : bs;
      renderList(); return;
    }
    var fb = e.target.closest("[data-bellanomfmt]");
    if (fb) {
      e.stopPropagation();
      var fsym = fb.closest("[data-bellmenu]").getAttribute("data-bellmenu");
      IW.anom_fmt = fb.getAttribute("data-bellanomfmt");
      renderList();
      saveMine({ anom_fmt: IW.anom_fmt }).then(function () { IW.bellOpen = fsym; renderList(); });
      return;
    }
    if (e.target.closest("[data-bellmenu]")) { e.stopPropagation(); return; }
    if (IW.bellOpen) { IW.bellOpen = ""; renderList(); }
  }, true);

  document.addEventListener("change", function (e) {
    var ca = e.target && e.target.closest ? e.target.closest("[data-bellanom]") : null;
    if (!ca) return;
    var sym = ca.closest("[data-bellmenu]").getAttribute("data-bellmenu"), on = !!ca.checked;
    IW.anom = (IW.anom || []).filter(function (x) { return x !== sym; }).concat(on ? [sym] : []);
    saveMine({ anom: { symbol: sym, on: on } }).then(function () { IW.bellOpen = sym; renderList(); refreshNotify(); });
  });

  document.addEventListener("change", function (e) {
    var cb = e.target && e.target.closest ? e.target.closest("[data-belltf]") : null;
    if (!cb) return;
    var menu = cb.closest("[data-bellmenu]"); var sym = menu.getAttribute("data-bellmenu"); var tfs = [];
    menu.querySelectorAll("[data-belltf]").forEach(function (x) { if (x.checked) tfs.push(x.getAttribute("data-belltf")); });
    if (tfs.length) IW.sbr[sym] = tfs; else delete IW.sbr[sym];
    saveMine({ sbr: { symbol: sym, tfs: tfs } }).then(function () { IW.bellOpen = sym; renderList(); refreshNotify(); });
  });

  document.addEventListener("click", function (e) {
    var tfb = e.target.closest ? e.target.closest("#iw-tfs [data-tf]") : null;
    if (tfb) {
      var tf = tfb.getAttribute("data-tf");
      if (tf === IW.tf) return;
      IW.tf = tf; IW.sig = ""; IW.coin = null;
      renderTfBar(); loadCoin(true); saveMine({ tf: tf }); return;
    }
    var tg = e.target.closest ? e.target.closest("[data-tgl]") : null;
    if (tg) {
      var tk = tg.getAttribute("data-tgl");
      if (IW[tk] === "off") IW[tk] = tfLast(tk); else { tfLastSet(tk, IW[tk]); IW[tk] = "off"; }
      IW.sig = ""; renderTfBar(); loadCoin(false, true);
      var tb = {}; tb[tk] = IW[tk]; saveMine(tb); return;
    }
    var lyr = e.target.closest ? e.target.closest("#iw-layers [data-layer]") : null;
    if (lyr) {
      var k = lyr.getAttribute("data-layer");
      IW[k] = !IW[k]; renderTfBar();
      if (k === "show_anom") { IW.sig = ""; loadCoin(false, true); }   // сигнали ⚠ просимо в сервера лише з увімкненим шаром
      else if (IW.coin) drawCoin(IW.coin, false);
      var b = {}; b[k] = IW[k]; saveMine(b); return;
    }
    var del = e.target.closest ? e.target.closest("[data-del]") : null;
    if (del) { e.stopPropagation(); saveMine({ remove: del.getAttribute("data-del") }); return; }
    var row = e.target.closest ? e.target.closest(".iw-row[data-sym]") : null;
    if (row) {
      var sym = row.getAttribute("data-sym");
      if (sym === IW.sel && !IW.view) return;
      IW.sel = sym; IW.view = ""; IW.sig = ""; IW.coin = null;
      renderList(); loadCoin(true); saveMine({ sel: sym });
    }
  });

  document.addEventListener("change", function (e) {
    var id = e.target && e.target.id;
    var MAP = { "iw-ob-tf": "ob_tf", "iw-vob-tf": "vob_tf", "iw-hl-tf": "hl_tf", "iw-sbr-tf": "sbr_tf", "iw-shl-tf": "shl_tf", "iw-poc-days": "poc_days", "iw-poc-tf": "poc_tf" };
    var k = MAP[id]; if (!k) return;
    IW[k] = k === "poc_days" ? Number(e.target.value) : e.target.value; IW.sig = "";
    if (k === "ob_tf" || k === "vob_tf" || k === "hl_tf") tfLastSet(k, e.target.value);
    loadCoin(false, true); var b = {}; b[k] = e.target.value; saveMine(b);
  });

  // ═══ 💧 ПРОФІЛЬ ЛІКВІДНОСТІ ════════════════════════════════════════════
  var LIQ_KEY = "vsv_liq_ui";
  var LIQ_BARS = [24, 48, 72, 120, 168, 240, 336, 504, 720, 1000];
  var LQ = { ex: "binance", bars: 168, res: "mid", on: false, sym: "", data: null, busy: 0, seq: 0 };
  try { var _lq = JSON.parse(localStorage.getItem(LIQ_KEY) || "{}"); if (_lq.ex) LQ.ex = _lq.ex; if (_lq.bars) LQ.bars = Number(_lq.bars); if (_lq.res) LQ.res = _lq.res; } catch (e) {}
  function liqSaveUi() { try { localStorage.setItem(LIQ_KEY, JSON.stringify({ ex: LQ.ex, bars: LQ.bars, res: LQ.res })); } catch (e) {} }
  function liqUi() {
    if (!LQ) return;
    var b = $("#iw-liq-bars"); if (b && !b.options.length) b.innerHTML = LIQ_BARS.map(function (n) { return '<option value="' + n + '">' + liqBarsLbl(n) + "</option>"; }).join("");
    if (b) b.value = String(LQ.bars);
    var ex = $("#iw-liq-ex"); if (ex) ex.value = LQ.ex;
    var rs = $("#iw-liq-res"); if (rs) rs.value = LQ.res;
    var btn = $("#iw-liq-btn"); if (btn) { btn.classList.toggle("on", LQ.on); btn.classList.toggle("busy", !!LQ.busy); btn.setAttribute("aria-pressed", LQ.on ? "true" : "false"); }
    var box = $("#iw-liq"); if (box) box.classList.toggle("on", LQ.on);
  }
  function liqStatus(html, title) { var st = $("#iw-liq-st"); if (st) { st.innerHTML = html || ""; st.title = title || ""; } }
  function liqOffset(on) {
    if (!IW.chart) return;
    var ts = IW.chart.timeScale();
    try {
      if (on) {
        if (LQ.prevRO == null) LQ.prevRO = ts.options().rightOffset || 0;
        var css = Math.max(64, Math.min(148, (ts.width() || 600) * 0.15));
        ts.applyOptions({ rightOffset: Math.ceil(css / (ts.options().barSpacing || 6)) + 2 });
      } else if (LQ.prevRO != null) { ts.applyOptions({ rightOffset: LQ.prevRO }); LQ.prevRO = null; }
    } catch (e) {}
  }
  function liqOff() {
    var had = !!LQ.data;
    LQ.on = false; LQ.sym = ""; LQ.data = null; LQ.busy = 0;
    if (IW.prim) IW.prim.setLiq(null);
    liqOffset(false); if (had) liqLevelTitles();
    liqStatus(""); liqUi();
  }
  var LIQ_RES_PX = { fine: 5, mid: 9, coarse: 14 };
  function liqNiceStep(raw) {
    if (!(raw > 0)) return null;
    var mag = Math.pow(10, Math.floor(Math.log10(raw))), ms = [1, 2, 2.5, 5, 10];
    for (var i = 0; i < ms.length; i++) if (raw <= mag * ms[i] * 1.0000001) return +(mag * ms[i]).toPrecision(6);
    return +(mag * 10).toPrecision(6);
  }
  function liqStepFromChart() {
    if (!IW.chart || !IW.series) return null;
    var h = 0; try { h = IW.chart.paneSize ? IW.chart.paneSize().height : 0; } catch (e) { h = 0; }
    if (!(h > 50)) { var bx = $("#iw-box"); h = bx ? bx.clientHeight - 28 : 0; }
    if (!(h > 50)) return null;
    var p0 = IW.series.coordinateToPrice(0), p1 = IW.series.coordinateToPrice(h);
    if (p0 == null || p1 == null) return null;
    return liqNiceStep(Math.abs(p0 - p1) * (LIQ_RES_PX[LQ.res] || 9) / h);
  }
  var LIQ_BR_FAIL_MS = 30 * 60 * 1000, LIQ_BR_KEY = "vsv_liq_br_fail";
  function liqBrFailed(ex) { try { var m = JSON.parse(localStorage.getItem(LIQ_BR_KEY) || "{}"); return m[ex] && Date.now() - m[ex] < LIQ_BR_FAIL_MS; } catch (e) { return false; } }
  function liqBrMark(ex, bad) { try { var m = JSON.parse(localStorage.getItem(LIQ_BR_KEY) || "{}"); if (bad) m[ex] = Date.now(); else delete m[ex]; localStorage.setItem(LIQ_BR_KEY, JSON.stringify(m)); } catch (e) {} }
  function liqXGet(url) {
    var ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
    var tm = ctl ? setTimeout(function () { ctl.abort(); }, 12000) : null;
    return fetch(url, { credentials: "omit", cache: "no-store", signal: ctl ? ctl.signal : undefined })
      .then(function (r) {
        if (tm) clearTimeout(tm);
        if (!r.ok) { var e = new Error("HTTP " + r.status); e.net = true; throw e; }
        return r.json();
      }, function (e) { if (tm) clearTimeout(tm); e.net = true; throw e; });
  }
  function liqNotFound(sym, ex) { var e = new Error(sym + " не знайдено на " + ex.toUpperCase()); e.data = true; return e; }
  var LIQ_BROWSER = {
    binance: function (sym, bars) {
      var B = "https://fapi.binance.com/fapi/v1/";
      return Promise.all([liqXGet(B + "openInterest?symbol=" + sym), liqXGet(B + "ticker/price?symbol=" + sym), liqXGet(B + "klines?symbol=" + sym + "&interval=1h&limit=" + bars)]).then(function (a) {
        var px = Number(a[1] && a[1].price) || 0, oi = Number(a[0] && a[0].openInterest) || 0;
        if (!(px > 0) || !Array.isArray(a[2])) throw liqNotFound(sym, "binance");
        return { price: px, oi_usd: oi * px, klines: a[2].map(function (x) { return [x[2], x[3], x[4], x[7]]; }) };
      });
    },
    bybit: function (sym, bars) {
      var B = "https://api.bybit.com/v5/market/";
      return Promise.all([liqXGet(B + "tickers?category=linear&symbol=" + sym), liqXGet(B + "kline?category=linear&symbol=" + sym + "&interval=60&limit=" + bars)]).then(function (a) {
        if (a[0] && a[0].retCode === 10006) { var e = new Error("ліміт Bybit"); e.net = true; throw e; }
        var t = ((a[0] && a[0].result) || {}).list || []; if (!t.length) throw liqNotFound(sym, "bybit");
        var px = Number(t[0].lastPrice) || 0, oi = Number(t[0].openInterest) || 0;
        var kl = (((a[1] && a[1].result) || {}).list || []).slice().reverse();
        return { price: px, oi_usd: oi * px, klines: kl.map(function (x) { return [x[2], x[3], x[4], x[6]]; }) };
      });
    }
  };
  function liqFromBrowser(sym, st) {
    var ex = LQ.ex, f = LIQ_BROWSER[ex];
    if (!f || liqBrFailed(ex)) return Promise.reject(Object.assign(new Error("skip"), { net: true }));
    return f(sym, LQ.bars).then(function (raw) {
      if (!(raw.oi_usd > 0)) { var e = new Error(ex.toUpperCase() + " не віддає відкритий інтерес"); e.data = true; throw e; }
      liqBrMark(ex, false);
      return apiPost("/api/me/liq-calc", { symbol: sym, exchange: ex, bars: LQ.bars, res: LQ.res, step: st || null, price: raw.price, oi_usd: raw.oi_usd, klines: raw.klines });
    }, function (e) { if (e && e.net) liqBrMark(ex, true); throw e; });
  }
  function liqFromServer(sym, st) { return api("/api/info/liq?symbol=" + encodeURIComponent(sym) + "&exchange=" + LQ.ex + "&bars=" + LQ.bars + "&res=" + LQ.res + (st ? "&step=" + st : "")); }
  function liqSrcLbl(d) { return d.src === "browser" ? '<span class="src" title="Дані завантажено з біржі напряму (браузер)">🖥</span>' : '<span class="src" title="Дані з кешу сервера">☁ ' + (Number(d.age) || 0) + "с</span>"; }
  function liqLoad() {
    var sym = curSym(); if (!sym || !pOk("ly_liq")) return;
    var tok = ++LQ.seq;
    LQ.on = true; LQ.sym = sym; LQ.busy = tok; liqUi();
    liqStatus('<span class="wait">⏳ рахую ' + esc(sym) + "…</span>");
    var st = liqStepFromChart();
    liqFromBrowser(sym, st).then(null, function (e) { if (e && e.data) return { ok: false, reason: e.message }; return liqFromServer(sym, st); })
      .then(function (d) {
        if (LQ.busy !== tok || !LQ.on || LQ.sym !== curSym()) return;
        LQ.busy = 0;
        if (!d || !d.ok) { LQ.data = null; if (IW.prim) IW.prim.setLiq(null); liqStatus('<span class="bad">' + esc((d && d.reason) || "немає даних") + "</span>"); liqUi(); return; }
        LQ.data = d; if (IW.prim) IW.prim.setLiq(d); liqOffset(true); liqLevelTitles();
        var up = Number(d.above_pct) || 0, dn = Number(d.below_pct) || 0;
        var pull = d.pull === "up" ? '<b class="up">⬆ вгору</b>' : d.pull === "down" ? '<b class="dn">⬇ вниз</b>' : "<b>⚖ рівновага</b>";
        liqStatus('<span class="up">▲' + up.toFixed(1) + '%</span> <span class="dn">▼' + dn.toFixed(1) + "%</span> · " + pull + " " + liqSrcLbl(d), d.verdict || "");
        liqUi();
      }, function (e) { if (LQ.busy !== tok) return; LQ.busy = 0; liqStatus('<span class="bad">помилка: ' + esc(e && e.message || "") + "</span>"); liqUi(); });
  }
  document.addEventListener("click", function (e) {
    var b = e.target.closest ? e.target.closest("#iw-liq-btn") : null; if (!b) return;
    if (LQ.on) liqOff(); else liqLoad();
  });
  document.addEventListener("change", function (e) {
    var id = e.target && e.target.id;
    if (id !== "iw-liq-ex" && id !== "iw-liq-bars" && id !== "iw-liq-res") return;
    if (id === "iw-liq-ex") LQ.ex = e.target.value;
    if (id === "iw-liq-bars") LQ.bars = Number(e.target.value);
    if (id === "iw-liq-res") LQ.res = e.target.value;
    liqSaveUi(); if (LQ.on || LQ.busy) { liqOff(); liqStatus('<span class="wait">параметри змінено</span>'); }
  });

  function liqWidth(W, hr) { return Math.round(Math.max(64, Math.min(148, (W / hr) * 0.15)) * hr); }
  var LIQ_UP = "6,182,192", LIQ_DN = "194,24,91", LIQ_MAG = "245,158,11";
  function _liqPx(v) { var d = IW.prec != null ? IW.prec : 4; return Number(v).toFixed(d); }
  function _liqDraw(ctx, liq, sr, XR, W, H, hr, vr, font, dark, levels) {
    var rows = liq.rows, taken = [], px0 = XR + 3 * hr, px1 = W - 3 * hr, PW = px1 - px0, mx = 0;
    rows.forEach(function (r) { if (r.pct > mx) mx = r.pct; });
    if (mx <= 0 || PW <= 10 * hr) return taken;
    ctx.fillStyle = dark ? "rgba(13,17,26,.26)" : "rgba(248,250,252,.46)"; _rrect(ctx, px0, 2 * vr, PW, H - 4 * vr, 4 * vr); ctx.fill();
    ctx.strokeStyle = dark ? "rgba(6,182,192,.14)" : "rgba(14,116,144,.16)"; ctx.lineWidth = Math.max(1, Math.round(hr)); _rrect(ctx, px0, 2 * vr, PW, H - 4 * vr, 4 * vr); ctx.stroke();
    ctx.font = font(700, 10); ctx.textBaseline = "middle";
    var up = Number(liq.above_pct) || 0, dn = Number(liq.below_pct) || 0;
    var tUp = "▲" + up.toFixed(0) + "%", tDn = "▼" + dn.toFixed(0) + "%";
    var wDn = ctx.measureText(tDn).width, wUp = ctx.measureText(tUp).width;
    var wRight = wDn + wUp + 6 * hr, xRight = px1 - 6 * hr;
    var tTitle = "LIQUIDITY · " + String(liq.exchange || "").toUpperCase() + " · " + liqBarsLbl(liq.bars);
    var wTitle = ctx.measureText(tTitle).width, twoLines = false;
    if (px0 + 7 * hr + wTitle + 10 * hr > xRight - wRight) {
      tTitle = "LIQUIDITY · " + String(liq.exchange || "").toUpperCase(); wTitle = ctx.measureText(tTitle).width;
      if (px0 + 7 * hr + wTitle + 10 * hr > xRight - wRight) {
        tTitle = "LIQUIDITY"; wTitle = ctx.measureText(tTitle).width;
        if (px0 + 7 * hr + wTitle + 10 * hr > xRight - wRight) twoLines = true;
      }
    }
    var top = (twoLines ? 34 : 22) * vr;
    ctx.textAlign = "left"; ctx.fillStyle = dark ? "#7dd3fc" : "#0369a1";
    ctx.fillText(tTitle, px0 + 7 * hr, 12 * vr);
    var yPct = twoLines ? 26 * vr : 12 * vr;
    ctx.textAlign = "right";
    ctx.fillStyle = "rgb(" + LIQ_DN + ")"; ctx.fillText(tDn, xRight, yPct);
    ctx.fillStyle = "rgb(" + LIQ_UP + ")"; ctx.fillText(tUp, xRight - wDn - 6 * hr, yPct);
    var bx0 = px0 + 6 * hr, bx1 = px1 - 6 * hr, BW = bx1 - bx0;
    ctx.save(); ctx.beginPath(); ctx.rect(px0, top, PW, H - top - 4 * vr); ctx.clip();
    var mag = liq.magnet, magY = null, lbl = [];
    rows.forEach(function (r) {
      var yA = sr.priceToCoordinate(r.hi), yB = sr.priceToCoordinate(r.lo); if (yA == null || yB == null) return;
      var y0 = Math.min(yA, yB) * vr, y1 = Math.max(yA, yB) * vr; if (y1 < top || y0 > H) return;
      var mid = (y0 + y1) / 2, h = Math.max(3 * vr, (y1 - y0) - 2 * vr); if (h > y1 - y0 && y1 - y0 > 0) h = Math.max(3 * vr, y1 - y0);
      var len = Math.max(3 * hr, BW * r.pct / mx);
      var isMag = mag && Math.abs(r.lo - mag.lo) < 1e-12 * Math.max(1, Math.abs(mag.lo)) + 1e-9;
      var rgb = isMag ? LIQ_MAG : (r.dir === "up" ? LIQ_UP : LIQ_DN);
      var x = bx1 - len, yy = mid - h / 2, g = ctx.createLinearGradient(bx1, 0, x, 0);
      g.addColorStop(0, "rgba(" + rgb + ",.18)"); g.addColorStop(1, "rgba(" + rgb + "," + (isMag ? ".95" : ".72") + ")");
      ctx.fillStyle = g; _rrect(ctx, x, yy, len, h, Math.min(3 * vr, h / 2)); ctx.fill();
      if (isMag) { magY = mid; ctx.strokeStyle = "rgba(" + LIQ_MAG + ",1)"; ctx.lineWidth = Math.max(1, Math.round(1.5 * hr)); _rrect(ctx, x, yy, len, h, Math.min(3 * vr, h / 2)); ctx.stroke(); }
      lbl.push({ r: r, y: mid, x: x, isMag: isMag });
    });
    var cp = liq.price != null ? sr.priceToCoordinate(Number(liq.price)) : null;
    if (cp != null) {
      var cy = cp * vr; ctx.strokeStyle = dark ? "rgba(226,232,240,.75)" : "rgba(30,41,59,.65)"; ctx.lineWidth = Math.max(1, Math.round(hr));
      if (ctx.setLineDash) ctx.setLineDash([3 * hr, 3 * hr]);
      ctx.beginPath(); ctx.moveTo(bx0, Math.round(cy) + 0.5); ctx.lineTo(bx1, Math.round(cy) + 0.5); ctx.stroke();
      if (ctx.setLineDash) ctx.setLineDash([]);
    }
    var used = (levels || []).map(function (lv) { var y = sr.priceToCoordinate(lv.price); return y == null ? -1e9 : y * vr; });
    lbl.sort(function (a, b) { return (b.isMag - a.isMag) || (b.r.pct - a.r.pct); });
    var lh = 12 * vr, shown = 0; ctx.font = font(700, 10); ctx.textAlign = "right";
    lbl.forEach(function (q) {
      if (shown >= 4 && !q.isMag) return; if (used.some(function (u) { return Math.abs(u - q.y) < lh; })) return;
      used.push(q.y); shown++;
      var t = (q.isMag ? "🧲 " : "") + liqUsd(q.r.usd) + " · " + Number(q.r.pct).toFixed(1) + "%";
      ctx.lineWidth = 3 * hr; ctx.strokeStyle = dark ? "rgba(13,17,26,.9)" : "rgba(248,250,252,.9)"; ctx.strokeText(t, bx1 - 2 * hr, q.y);
      ctx.fillStyle = q.isMag ? (dark ? "#fcd34d" : "#92400e") : (dark ? "#e2e8f0" : "#1e293b"); ctx.fillText(t, bx1 - 2 * hr, q.y);
    });
    ctx.restore();
    if (mag && magY == null) { var ya = sr.priceToCoordinate(mag.hi), yb = sr.priceToCoordinate(mag.lo); if (ya != null && yb != null) magY = (ya + yb) / 2 * vr; }
    if (mag && magY != null && magY > top && magY < H) {
      ctx.strokeStyle = "rgba(" + LIQ_MAG + ",.55)"; ctx.lineWidth = Math.max(1, Math.round(hr));
      if (ctx.setLineDash) ctx.setLineDash([6 * hr, 5 * hr]);
      ctx.beginPath(); ctx.moveTo(0, Math.round(magY) + 0.5); ctx.lineTo(XR - 10 * hr, Math.round(magY) + 0.5); ctx.stroke();
      if (ctx.setLineDash) ctx.setLineDash([]);
      if (cp != null) {
        var ax = px0 + 3 * hr, y0a = cp * vr, y1a = magY, dirUp = y1a < y0a;
        if (Math.abs(y1a - y0a) > 8 * vr) {
          ctx.strokeStyle = "rgba(" + LIQ_MAG + ",.95)"; ctx.lineWidth = Math.max(1, Math.round(2 * hr));
          ctx.beginPath(); ctx.moveTo(ax, y0a); ctx.lineTo(ax, y1a + (dirUp ? 5 : -5) * vr); ctx.stroke();
          ctx.fillStyle = "rgba(" + LIQ_MAG + ",1)"; ctx.beginPath(); ctx.moveTo(ax - 4 * hr, y1a + (dirUp ? 6 : -6) * vr); ctx.lineTo(ax + 4 * hr, y1a + (dirUp ? 6 : -6) * vr); ctx.lineTo(ax, y1a); ctx.closePath(); ctx.fill();
        }
      }
      var dist = mag.dist != null ? (mag.dir === "up" ? "↑" : "↓") + Number(mag.dist).toFixed(2) + "%" : "";
      var t1 = "🧲 магніт " + _liqPx(mag.lo) + (mag.hi > mag.lo ? "–" + _liqPx(mag.hi) : ""), t2 = dist;
      ctx.font = font(700, 11); var w1 = ctx.measureText(t1).width;
      ctx.font = font(600, 10.5); var w2 = t2 ? ctx.measureText(t2).width + 8 * hr : 0;
      var cw = w1 + w2 + 16 * hr, ch = 18 * vr, cx = XR - cw - 10 * hr, cyy = magY - ch / 2;
      if (cx < 2 * hr) cx = 2 * hr;
      ctx.fillStyle = dark ? "rgba(28,20,4,.95)" : "rgba(255,251,235,.97)"; _rrect(ctx, cx, cyy, cw, ch, 4 * vr); ctx.fill();
      ctx.strokeStyle = "rgba(" + LIQ_MAG + ",.95)"; ctx.lineWidth = Math.max(1, Math.round(hr)); _rrect(ctx, cx, cyy, cw, ch, 4 * vr); ctx.stroke();
      ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.font = font(700, 11); ctx.fillStyle = dark ? "#fcd34d" : "#92400e";
      ctx.fillText(t1, cx + 8 * hr, cyy + ch / 2 + 0.5 * vr);
      if (t2) { ctx.font = font(600, 10.5); ctx.fillStyle = dark ? "#fde68a" : "#b45309"; ctx.fillText(t2, cx + 8 * hr + w1 + 8 * hr, cyy + ch / 2 + 0.5 * vr); }
      taken.push({ x0: cx - 2 * hr, x1: cx + cw + 2 * hr, y0: cyy - 2 * vr, y1: cyy + ch + 2 * vr });
    }
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; return taken;
  }
  function liqBarsLbl(n) { n = Number(n) || 0; return n >= 48 && n % 24 === 0 ? (n / 24) + " дн" : n + " год"; }
  function liqUsd(v) { v = Number(v) || 0; if (v >= 1e9) return "$" + (v / 1e9).toFixed(2) + "B"; if (v >= 1e6) return "$" + (v / 1e6).toFixed(v >= 1e8 ? 0 : 1) + "M"; if (v >= 1e3) return "$" + (v / 1e3).toFixed(0) + "K"; return "$" + v.toFixed(0); }

  function BoxPrimitive() {
    this._b = []; this._l = []; this._s = []; this._lv = []; this._hits = []; this._liq = null;
    this._chart = null; this._series = null; this._req = null;
  }
  BoxPrimitive.prototype.attached = function (p) { this._chart = p.chart; this._series = p.series; this._req = p.requestUpdate; };
  BoxPrimitive.prototype.detached = function () { this._chart = null; this._series = null; this._req = null; };
  BoxPrimitive.prototype.set = function (boxes, lines, sigs, levels) { this._b = boxes || []; this._l = lines || []; this._s = sigs || []; this._lv = levels || []; if (this._req) this._req(); };
  BoxPrimitive.prototype.setHl = function (pts) { this._hl = pts || []; if (this._req) this._req(); };
  BoxPrimitive.prototype.setLiq = function (liq) { this._liq = liq && liq.rows && liq.rows.length ? liq : null; if (this._req) this._req(); };
  BoxPrimitive.prototype.hitAt = function (x, y) {
    for (var i = this._hits.length - 1; i >= 0; i--) { var h = this._hits[i]; if (x >= h.x0 - 3 && x <= h.x1 + 3 && y >= h.y0 - 3 && y <= h.y1 + 3) return h.sig; }
    return null;
  };
  function _rrect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h); ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath(); }

  /* 15.11 IMB «як у TradingView»: зверху помаранчева смуга продажу, знизу бірюзова купівлі
     (довжина = %), обсяг — праворуч у зоні світлим текстом, зона тягнеться за останню свічку. */
  /* Кольори й розміри — рівно з Pine «PRO Volumatic FVG» (Bull FVG #1ac2d8 / Bear
     FVG #d8761a з прозорістю 50; тло зони — та сама барва з прозорістю 70;
     смуги з обводкою кольору тла графіка; бокс до останньої свічки + 25). */
  var IMB_SELL = "rgba(216,118,26,.5)", IMB_BUY = "rgba(26,194,216,.5)", IMB_MIN_PX = 1.5, IMB_EXT_BARS = 25;
  var IMB_BG_SELL = "rgba(216,118,26,.3)", IMB_BG_BUY = "rgba(26,194,216,.3)";
  function _imbBar(ctx, X1, w, y, bh, pct, color, hr, vr, font, dark, tv) {
    if (pct == null || pct <= 0 || bh < 1) return;
    var sw = Math.max(2 * hr, w * Math.min(100, Math.max(0, pct)) / 100);
    ctx.fillStyle = color; ctx.fillRect(X1, y, sw, bh);
    ctx.strokeStyle = tv ? (dark ? "#000" : "#fff") : (dark ? "rgba(0,0,0,.55)" : "rgba(0,0,0,.25)"); ctx.lineWidth = Math.max(1, Math.round(hr)); ctx.strokeRect(X1, y, sw, bh);
    if (bh < 8 * vr) return;
    /* 15.11 як у TV: % білий, розмір від висоти смуги (у високій зоні — великий) */
    var fp = tv ? Math.max(7, Math.min(44, bh / vr * 0.5)) : Math.max(8, Math.min(32, bh / vr * 0.62));
    ctx.font = font(tv ? 500 : 600, fp); ctx.textBaseline = "middle"; ctx.fillStyle = dark ? "rgba(240,242,248,.94)" : "rgba(20,24,32,.9)";   // Pine: chart.fg_color
    var t = pct + "%", tw = ctx.measureText(t).width;
    if (sw >= tw + 10 * hr) { ctx.textAlign = "right"; ctx.fillText(t, X1 + sw - 5 * hr, y + bh / 2); }
    else if (w - sw >= tw + 8 * hr) { ctx.textAlign = "left"; ctx.fillText(t, X1 + sw + 4 * hr, y + bh / 2); }
  }
  function _imbBody(ctx, b, X1, X2, yT, yB, hr, vr, font, dark, wms) {
    var w = X2 - X1, h = yB - yT;
    /* 15.11 як у TV: обсяг IMB — на ЗАДНЬОМУ плані (під смугами), а не поверх */
    if (b.kind === "imb") _imbVol(ctx, b, X1, X2, yT, yB, hr, vr, font, dark);
    if (w > 4 * hr && (b.topPct != null || b.botPct != null)) {
      var bh = h / 2;
      _imbBar(ctx, X1, w, yT, bh, b.topPct, b.topCol || IMB_SELL, hr, vr, font, dark, b.kind === "imb");
      _imbBar(ctx, X1, w, yT + bh, bh, b.botPct, b.botCol || IMB_BUY, hr, vr, font, dark, b.kind === "imb");
    }
    if (b.kind === "imb") { ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; return; }
    if (b.volTxt && h >= 18 * vr && w > 40 * hr) {
      var wk = b.volTxt + "|" + Math.round(w) + "|" + Math.round(h), fv;
      if (b._wk === wk) { fv = b._fv; ctx.font = font(500, fv); }
      else { fv = b.kind === "vob" ? Math.max(10, Math.min(30, h / vr * 0.32)) : Math.max(10, Math.min(64, h / vr * 0.82)); ctx.font = font(500, fv); while (fv > 10 && ctx.measureText(b.volTxt).width > w * 0.92) { fv -= 2; ctx.font = font(500, fv); } b._wk = wk; b._fv = fv; }
      var tw = ctx.measureText(b.volTxt).width, cx = (X1 + X2) / 2, cy = (yT + yB) / 2;
      var wm = { txt: b.volTxt, fv: fv, x: cx, y: cy + fv * vr * 0.04, color: b.kind === "vob" ? b.text : null, r: { x0: cx - tw / 2, x1: cx + tw / 2, y0: cy - fv * vr * 0.45, y1: cy + fv * vr * 0.45 } };
      if (wms) wms.push(wm); else wmDraw(ctx, [wm], font, dark);
    }
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  }
  /* Обсяг IMB — як у TV: праворуч у зоні, світлий, розмір від висоти зони. */
  function _imbVol(ctx, b, X1, X2, yT, yB, hr, vr, font, dark) {
    var w = X2 - X1, h = yB - yT;
    if (!b.volTxt || h < 7 * vr || w < 30 * hr) return;
    /* 15.11 як у TV: обсяг ТЕМНИЙ, ПІД смугами (малюється першим), праворуч, розмір від висоти зони */
    var fv = Math.max(7, Math.min(120, h / vr * 0.6));
    ctx.font = font(400, fv);
    while (fv > 7 && ctx.measureText(b.volTxt).width > w * 0.62) { fv -= 1; ctx.font = font(400, fv); }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillStyle = dark ? "#000" : "#fff";   // Pine: set_text_color(chart.bg_color)
    ctx.fillText(b.volTxt, X2 - 5 * hr, (yT + yB) / 2 + fv * vr * 0.04);
  }
  function wmDraw(ctx, wms, font, dark) {
    var done = [];
    for (var i = wms.length - 1; i >= 0; i--) {
      var w = wms[i], r = w.r;
      if (done.some(function (q) { return r.x0 < q.x1 && r.x1 > q.x0 && r.y0 < q.y1 && r.y1 > q.y0; })) continue;
      done.push(r); ctx.font = font(500, w.fv); ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = w.color || (dark ? "rgba(4,6,12,.5)" : "rgba(0,0,0,.16)"); ctx.fillText(w.txt, w.x, w.y);
    }
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; return done;
  }

  BoxPrimitive.prototype.paneViews = function () {
    var self = this;
    return [{ renderer: function () { return { draw: function (target) {
      if (!self._chart || !self._series) return;
      var ts = self._chart.timeScale(), sr = self._series;
      var dark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
      target.useBitmapCoordinateSpace(function (sc) {
        var ctx = sc.context, hr = sc.horizontalPixelRatio, vr = sc.verticalPixelRatio;
        var W = sc.bitmapSize.width, H = sc.bitmapSize.height;
        var FONT = "-apple-system,Segoe UI,Roboto,sans-serif";
        var font = function (w, px) { return w + " " + Math.round(px * vr) + "px " + FONT; };
        var LW = self._liq ? liqWidth(W, hr) : 0, XR = W - LW;
        var liqTaken = LW ? _liqDraw(ctx, self._liq, sr, XR, W, H, hr, vr, font, dark, self._lv) : [];
        var sLbl = [];
        (self._hl || []).forEach(function (p) {
          var x = ts.timeToCoordinate(p.t), y = sr.priceToCoordinate(p.price);
          if (x == null || y == null || x * hr > XR) return;
          ctx.font = font(800, HL_PX); ctx.textAlign = "center"; ctx.textBaseline = p.high ? "bottom" : "top";
          ctx.fillStyle = p.color; ctx.fillText(p.text, x * hr, (y + (p.high ? -5 : 5)) * vr);
        });
        ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
        self._l.forEach(function (l) {
          var x1 = ts.timeToCoordinate(l.t1), x2 = ts.timeToCoordinate(l.t2), y = sr.priceToCoordinate(l.price);
          if (x1 == null || x2 == null || y == null) return;
          ctx.strokeStyle = l.color; ctx.lineWidth = Math.max(1, Math.round(hr));
          if (l.dashed && ctx.setLineDash) ctx.setLineDash([4 * hr, 3 * hr]);
          ctx.beginPath(); ctx.moveTo(x1 * hr, Math.round(y * vr) + 0.5); ctx.lineTo(x2 * hr, Math.round(y * vr) + 0.5); ctx.stroke();
          if (ctx.setLineDash) ctx.setLineDash([]);
          if (l.text) sLbl.push({ l: l, x: (x1 + x2) / 2, y: y });
        });
        var placed = [], wms = [];
        for (var i = self._b.length - 1; i >= 0; i--) {
          var b = self._b[i];
          var x1 = ts.timeToCoordinate(b.t);
          var yA = sr.priceToCoordinate(b.top), yB = sr.priceToCoordinate(b.bottom);
          if (x1 == null || yA == null || yB == null) continue;
          var X1 = Math.max(0, x1 * hr), X2 = XR - (LW ? 3 * hr : 0);
          if (b.ext) X2 = Math.min(X2, (x1 + b.ext * ts.options().barSpacing) * hr);
          var yT = Math.min(yA, yB) * vr, yBt = Math.max(yA, yB) * vr;
          if (X2 <= X1) continue;
          if (b.kind === "imb" && (yBt - yT) < IMB_MIN_PX * vr) continue;
          if (yBt - yT < 1) yBt = yT + 1;
          ctx.fillStyle = b.fill; ctx.fillRect(X1, yT, X2 - X1, yBt - yT);
          if (b.kind === "imb" || b.kind === "vob") _imbBody(ctx, b, X1, X2, yT, yBt, hr, vr, font, dark, wms);
          ctx.strokeStyle = b.stroke; ctx.lineWidth = Math.max(1, Math.round((b.thin ? 1 : 1.2) * hr));
          if (b.dashed && ctx.setLineDash) ctx.setLineDash([6 * hr, 4 * hr]);
          ctx.strokeRect(X1, yT, X2 - X1, yBt - yT);
          if (ctx.setLineDash) ctx.setLineDash([]);
          if (b.edgeCol) {
            var ey = b.edge === "top" ? yT : yBt;
            ctx.strokeStyle = b.edgeCol; ctx.lineWidth = Math.max(2, Math.round(2 * hr));
            ctx.beginPath(); ctx.moveTo(X1, Math.round(ey) + 0.5); ctx.lineTo(X2, Math.round(ey) + 0.5); ctx.stroke();
          }
          if (b.centerCol) {
            var cym = (yT + yBt) / 2;
            ctx.strokeStyle = b.centerCol; ctx.lineWidth = Math.max(1, Math.round(hr));
            if (ctx.setLineDash) ctx.setLineDash([3 * hr, 3 * hr]);
            ctx.beginPath(); ctx.moveTo(X1, Math.round(cym) + 0.5); ctx.lineTo(X2, Math.round(cym) + 0.5); ctx.stroke();
            if (ctx.setLineDash) ctx.setLineDash([]);
          }
          placed.push({ b: b, X1: X1, X2: X2, yT: yT, yB: yBt });
        }
        var wmTaken = wmDraw(ctx, wms, font, dark);
        var bs = SBR_MARK_PX, marks = [];   // 16.11: позначку S/B зменшено (було 18)
        self._s.forEach(function (s) {
          var x = ts.timeToCoordinate(s.t);
          var y = sr.priceToCoordinate(s.side === "SHORT" ? s.hi : s.lo);
          if (x == null || y == null) return;
          var up = s.side === "SHORT";
          var cy = up ? y - 10 - bs / 2 : y + 10 + bs / 2;
          for (var g = 0; g < 6; g++) {
            var hit = marks.some(function (m) { return Math.abs(m.x - x) < bs + 2 && Math.abs(m.cy - cy) < bs + 2; });
            if (!hit) break;
            cy += up ? -(bs + 3) : (bs + 3);
          }
          marks.push({ s: s, x: x, y: y, cy: cy, x0: x - bs / 2, y0: cy - bs / 2 });
        });
        var taken = marks.map(function (m) { return { x0: (m.x0 - 2) * hr, x1: (m.x0 + bs + 2) * hr, y0: (m.y0 - 2) * vr, y1: (m.y0 + bs + 7) * vr }; });
        if (LW) taken.push({ x0: XR - 2 * hr, x1: W, y0: 0, y1: H });
        liqTaken.forEach(function (r) { taken.push(r); });
        self._lv.forEach(function (lv) { var y = sr.priceToCoordinate(lv.price); if (y == null) return; ctx.font = font(500, 12); var tw = ctx.measureText(lv.title || "").width + 24 * hr; taken.push({ x0: W - tw, x1: W, y0: (y - 10) * vr, y1: (y + 10) * vr }); });
        var overlaps = function (r) { for (var k = 0; k < taken.length; k++) { var q = taken[k]; if (r.x0 < q.x1 && r.x1 > q.x0 && r.y0 < q.y1 && r.y1 > q.y0) return true; } return false; };
        var LIQ_GAP = 10 * hr;
        if (LW) self._lv.forEach(function (lv) {
          var y = sr.priceToCoordinate(lv.price); if (y == null) return;
          ctx.strokeStyle = lv.color || "#f5f5f5"; ctx.lineWidth = Math.max(1, Math.round(hr));
          ctx.beginPath(); ctx.moveTo(0, Math.round(y * vr) + 0.5); ctx.lineTo(XR - LIQ_GAP, Math.round(y * vr) + 0.5); ctx.stroke();
        });
        if (LW) self._lv.forEach(function (lv) {
          if (!lv.title) return;
          var y = sr.priceToCoordinate(lv.price); if (y == null) return;
          ctx.font = font(700, 11); var tw = ctx.measureText(lv.title).width + 14 * hr, chh = 17 * vr;
          var r = { x0: XR - LIQ_GAP - tw, x1: XR - LIQ_GAP, y0: y * vr - chh / 2, y1: y * vr + chh / 2 };
          for (var g = 0; g < 6 && overlaps(r); g++) { var mv = 0; taken.forEach(function (q) { if (r.x0 < q.x1 && r.x1 > q.x0 && r.y0 < q.y1 && r.y1 > q.y0) mv = Math.max(mv, r.x1 - q.x0 + 4 * hr); }); r.x0 -= mv; r.x1 -= mv; }
          if (r.x0 < 0) return; taken.push(r);
          ctx.fillStyle = lv.color || "#f5f5f5"; _rrect(ctx, r.x0, r.y0, tw, chh, 3 * vr); ctx.fill();
          ctx.fillStyle = "#0b0f17"; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(lv.title, r.x0 + 7 * hr, y * vr + 0.5 * vr); ctx.textBaseline = "alphabetic";
        });
        var prio = { vob: 0, ob: 1, imb: 2 }; var pr = function (k) { return Object.prototype.hasOwnProperty.call(prio, k) ? prio[k] : 9; };
        placed.sort(function (a, c) { return pr(a.b.kind) - pr(c.b.kind); });
        var padX = 6 * hr, chipH = 17 * vr, gap = 2 * vr;
        var vobBodies = placed.filter(function (p) { return p.b.kind === "vob"; }).map(function (p) { return { x0: p.X1, x1: p.X2, y0: p.yT, y1: p.yB }; });
        placed.forEach(function (p) {
          var b = p.b; if (!b.label || b.kind === "imb") return;
          ctx.font = font(700, 11); var w1 = ctx.measureText(b.label).width; ctx.font = font(500, 10.5); var w2 = b.sub ? ctx.measureText(b.sub).width + 6 * hr : 0;
          var w = w1 + w2 + padX * 2 + 4 * hr, right = b.kind === "vob";
          var x0 = right ? (XR - w - (LW ? 10 : 8) * hr) : Math.min(p.X1 + 4 * hr, XR - w - (LW ? 10 : 8) * hr);
          if (x0 < 2 * hr) x0 = 2 * hr;
          var mid = (p.yT + p.yB) / 2, tries = [0, -1, 1, -2, 2, -3, 3], ok = null, avoid = b.kind === "ob" ? vobBodies : [];
          for (var pass = avoid.length ? 0 : 1; pass < 2 && !ok; pass++) {
            for (var k = 0; k < tries.length; k++) {
              var y0 = mid - chipH / 2 + tries[k] * (chipH + gap); if (y0 < 0 || y0 + chipH > H) continue;
              var r = { x0: x0, x1: x0 + w, y0: y0, y1: y0 + chipH };
              if (overlaps(r)) continue;
              if (pass === 0 && avoid.some(function (q) { return r.x0 < q.x1 && r.x1 > q.x0 && r.y0 < q.y1 && r.y1 > q.y0; })) continue;
              ok = r; break;
            }
          }
          if (!ok) return; taken.push(ok);
          ctx.globalAlpha = 0.92; ctx.fillStyle = dark ? "rgba(13,17,26,.88)" : "rgba(255,255,255,.92)"; _rrect(ctx, ok.x0, ok.y0, w, chipH, 4 * vr); ctx.fill(); ctx.globalAlpha = 1;
          ctx.strokeStyle = b.stroke; ctx.lineWidth = Math.max(1, Math.round(hr)); _rrect(ctx, ok.x0, ok.y0, w, chipH, 4 * vr); ctx.stroke();
          ctx.fillStyle = b.stroke; ctx.fillRect(ok.x0, ok.y0 + 3 * vr, 3 * hr, chipH - 6 * vr);
          ctx.textBaseline = "middle"; ctx.textAlign = "left"; ctx.font = font(700, 11); ctx.fillStyle = b.text;
          var tx = ok.x0 + padX + 2 * hr, ty = ok.y0 + chipH / 2 + 0.5 * vr;
          ctx.fillText(b.label, tx, ty);
          if (b.sub) { ctx.font = font(500, 10.5); ctx.fillStyle = dark ? "#94a3b8" : "#5b6475"; ctx.fillText(b.sub, tx + w1 + 6 * hr, ty); }
          ctx.textBaseline = "alphabetic";
        });
        wmTaken.forEach(function (r) { taken.push(r); });
        ctx.font = font(600, 10.5); sLbl.sort(function (a, c) { return c.l.t2 - a.l.t2; });
        sLbl.forEach(function (q) {
          var tw = ctx.measureText(q.l.text).width, th = 12 * vr, cx = q.x * hr, up = q.l.up;
          var y0 = up ? q.y * vr - 3 * vr - th : q.y * vr + 3 * vr;
          var r = { x0: cx - tw / 2 - 2 * hr, x1: cx + tw / 2 + 2 * hr, y0: y0, y1: y0 + th };
          if (r.y0 < 0 || r.y1 > H || r.x1 > XR || overlaps(r)) return;
          taken.push(r); ctx.fillStyle = q.l.color; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(q.l.text, cx, y0 + th / 2);
        });
        ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; self._hits = [];
        marks.forEach(function (m) {
          var s = m.s, x = m.x, cy = m.cy, x0 = m.x0, y0 = m.y0, col = s.kind === "anom" ? (s.side === "SHORT" ? ANOM_SHORT : ANOM_LONG) : (s.side === "SHORT" ? "#f23645" : "#22c55e");
          _rrect(ctx, x0 * hr, y0 * vr, bs * hr, bs * vr, 3 * vr); ctx.fillStyle = col; ctx.fill();
          ctx.strokeStyle = dark ? "rgba(255,255,255,.35)" : "rgba(0,0,0,.25)"; ctx.lineWidth = Math.max(1, Math.round(hr)); ctx.stroke();
          ctx.beginPath();
          if (s.side === "SHORT") { ctx.moveTo((x - 3) * hr, (y0 + bs) * vr); ctx.lineTo((x + 3) * hr, (y0 + bs) * vr); ctx.lineTo(x * hr, (y0 + bs + 4) * vr); }
          else { ctx.moveTo((x - 3) * hr, y0 * vr); ctx.lineTo((x + 3) * hr, y0 * vr); ctx.lineTo(x * hr, (y0 - 4) * vr); }
          ctx.closePath(); ctx.fillStyle = col; ctx.fill();
          ctx.font = font(800, 9); ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(s.side === "SHORT" ? "S" : "B", x * hr, (cy + 0.5) * vr);
          ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; self._hits.push({ x0: x0, x1: x0 + bs, y0: y0, y1: y0 + bs, sig: s });
        });
      });
    } }; } }];
  };

  function snapSec(sec, ohlc) {
    if (!ohlc.length || !sec) return sec;
    if (sec <= ohlc[0].time) return ohlc[0].time;
    var last = ohlc[ohlc.length - 1].time;
    if (sec >= last) return last;
    var lo = 0, hi = ohlc.length - 1, ans = ohlc[0].time;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (ohlc[mid].time <= sec) { ans = ohlc[mid].time; lo = mid + 1; } else hi = mid - 1;
    }
    return ans;
  }

  function autoScaleOn() { if (IW.fixedRange) return false; try { return IW.chart.priceScale("right").options().autoScale !== false; } catch (e) { return true; } }
  function setAutoScale(on) {
    if (!IW.chart) return;
    if (on) IW.fixedRange = null;
    IW.chart.priceScale("right").applyOptions({ autoScale: !!on });
    var b = $("#iw-auto"); if (b) b.classList.toggle("on", !!on);
  }
  function zoomRange(lo, hi, k) { var mid = (lo + hi) / 2, half = (hi - lo) / 2 * k; if (!(half > 0)) return null; return { lo: mid - half, hi: mid + half }; }
  function wheelPrice(k, paneH) {
    var sr = IW.series, top = sr.coordinateToPrice(0), bot = sr.coordinateToPrice(paneH);
    if (top == null || bot == null) return;
    var v = zoomRange(Math.min(top, bot), Math.max(top, bot), k); if (!v) return;
    var m = (IW.chart.priceScale("right").options().scaleMargins) || { top: 0.2, bottom: 0.1 }, span = v.hi - v.lo;
    IW.fixedRange = { minValue: v.lo + span * (m.bottom || 0), maxValue: v.hi - span * (m.top || 0) };
    IW.chart.priceScale("right").applyOptions({ autoScale: true });
    var b = $("#iw-auto"); if (b) b.classList.remove("on");
    sr.applyOptions({ autoscaleInfoProvider: priceRangeProvider });
  }
  function priceRangeProvider(base) { if (IW.fixedRange) return { priceRange: IW.fixedRange }; return base(); }

  function tvPanInit(box) {
    var y0 = null, x0 = null;
    box.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      if (e.target.closest && e.target.closest("#iw-sig")) { y0 = null; return; }
      var r = box.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      var psw = 0, tsh = 0; try { psw = IW.chart.priceScale("right").width(); tsh = IW.chart.timeScale().height(); } catch (er) {}
      if (x > r.width - psw || y > r.height - tsh) { y0 = null; return; }
      x0 = e.clientX; y0 = e.clientY;
    }, true);
    box.addEventListener("pointermove", function (e) {
      if (y0 == null || !(e.buttons & 1)) return;
      var dy = Math.abs(e.clientY - y0), dx = Math.abs(e.clientX - x0);
      if (dy > 5 && dy > dx * 0.6 && (autoScaleOn() || IW.fixedRange)) {
        IW.fixedRange = null; IW.chart.priceScale("right").applyOptions({ autoScale: false });
        var ab = $("#iw-auto"); if (ab) ab.classList.remove("on");
      }
    }, true);
    window.addEventListener("pointerup", function () { y0 = null; }, true);
    box.addEventListener("dblclick", function (e) {
      var r = box.getBoundingClientRect(), psw = 0, tsh = 0; try { psw = IW.chart.priceScale("right").width(); tsh = IW.chart.timeScale().height(); } catch (er) {}
      var x = e.clientX - r.left, y = e.clientY - r.top;
      if (psw > 0 && x >= r.width - psw) {
        setAutoScale(true);
        setTimeout(function () { var b = $("#iw-auto"); if (b) b.classList.toggle("on", autoScaleOn()); }, 0);
        return;
      }
      if (chartDblFree(e, x, y, r.height - tsh)) setFull(!isFull());
    });
    box.addEventListener("wheel", function (e) {
      if (!IW.chart || !IW.series || (e.target.closest && e.target.closest("#iw-sig"))) return;
      var r = box.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      var psw = 0, tsh = 0; try { psw = IW.chart.priceScale("right").width(); tsh = IW.chart.timeScale().height(); } catch (er) {}
      if (!(psw > 0) || x < r.width - psw || y > r.height - tsh) return;
      e.preventDefault(); e.stopPropagation(); var dy = e.deltaY || 0;
      if (e.deltaMode === 1) dy *= 16; if (!dy) return;
      wheelPrice(Math.pow(1.0015, Math.max(-300, Math.min(300, dy))), r.height - tsh);
    }, { capture: true, passive: false });
    var b = document.createElement("button");
    b.id = "iw-auto"; b.type = "button"; b.className = "iw-auto on"; b.textContent = "A"; b.title = "Автомасштаб ціни";
    b.addEventListener("click", function () { setAutoScale(!autoScaleOn()); });
    IW.series.applyOptions({ autoscaleInfoProvider: priceRangeProvider });
    box.appendChild(b);
  }

  /* 🖱 25.11: подвійний клік по ЧИСТОМУ простору графіка → на весь екран і назад.
     Не чіпаємо: шкалу часу, кнопки / картку сигналу, позначки S/B (клік по
     них відкриває картку) і зону обсягу (там подвійний клік — «Σ від свічки»). */
  function chartDblFree(e, x, y, plotH) {
    if (e.target && e.target.closest && e.target.closest("#iw-sig, #iw-sig-arrow, button, a, input, select, label")) return false;
    if (!(plotH > 0) || y < 0 || y > plotH) return false;
    if (VOL.on && y >= plotH * VOL_TOP - 12) return false;
    try { if (IW.prim && IW.prim.hitAt(x, y)) return false; } catch (er) {}
    return true;
  }
  /* 🛰 25.11: подвійний клік по рядку VSV SIGNALS → ця монета на графіку на весь
     екран. Монета з вашого WATCHLIST стає обраною; інша — показується, доки
     графік розгорнутий (вихід із повного екрана повертає вашу монету). */
  function recOpenOnChart(sym) {
    sym = String(sym || "").toUpperCase(); if (!sym || !pOk("sec_terminal")) return;
    if ((IW.wl || []).indexOf(sym) >= 0) {
      if (sym !== IW.sel || IW.view) { IW.sel = sym; IW.view = ""; IW.sig = ""; IW.coin = null; renderList(); loadCoin(true); saveMine({ sel: sym }); }
    } else if (sym !== curSym()) { IW.view = sym; IW.sig = ""; IW.coin = null; renderList(); loadCoin(true); }
    setFull(true);
  }
  document.addEventListener("dblclick", function (e) {
    var tr = e.target && e.target.closest ? e.target.closest("#rec-tbl tr.rec-row[data-sym]") : null;
    if (!tr) return;
    try { var sel = window.getSelection && window.getSelection(); if (sel) sel.removeAllRanges(); } catch (er) {}
    recOpenOnChart(tr.getAttribute("data-sym"));
  });

  function setFull(on) {
    var card = $("#iw-chart"), b = $("#iw-fs"); if (!card) return;
    on = !!on; card.classList.toggle("iw-full", on); document.body.classList.toggle("iw-full-on", on);
    if (!on && IW.view) { IW.view = ""; IW.sig = ""; IW.coin = null; renderList(); loadCoin(true); }
    if (b) { b.setAttribute("aria-pressed", on ? "true" : "false"); b.textContent = on ? "✕" : "⛶"; b.title = on ? "Згорнути (Esc)" : "На весь екран"; }
  }
  function isFull() { var c = $("#iw-chart"); return !!(c && c.classList.contains("iw-full")); }
  document.addEventListener("click", function (e) { var b = e.target.closest ? e.target.closest("#iw-fs") : null; if (b) { e.preventDefault(); setFull(!isFull()); } });
  document.addEventListener("keydown", function (e) { if ((e.key === "Escape" || e.key === "Esc") && isFull()) { setFull(false); } });

  function standardCandles() { try { return localStorage.getItem("vsv_candle_standard") === "1"; } catch (e) { return false; } }
  function applyCandlePalette() {
    if (!IW.series) return;
    var dark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark", standard = standardCandles();
    var up = standard ? (dark ? "#00bb7f" : "#26a69a") : "#c7ccd6";
    var down = standard ? (dark ? "#fb2c36" : "#ef5350") : "#5b6475";
    IW.series.applyOptions({ upColor: up, downColor: down, borderVisible: !standard, borderUpColor: up, borderDownColor: standard ? down : "#8b94a5", wickUpColor: up, wickDownColor: standard ? down : "#8b94a5" });
  }

  function ensureChart() {
    if (IW.chart || !window.LightweightCharts) return !!IW.chart;
    var box = $("#iw-box"), dark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
    IW.chart = LightweightCharts.createChart(box, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: dark ? "#8f9aad" : "#4b5565", fontFamily: "Inter,Segoe UI,sans-serif", fontSize: 13 },
      grid: { vertLines: { color: dark ? "rgba(42,46,57,.55)" : "rgba(225,229,235,.75)" }, horzLines: { color: dark ? "rgba(42,46,57,.55)" : "rgba(225,229,235,.75)" } },
      rightPriceScale: { borderColor: dark ? "#2a2e39" : "#d7dde5" },
      timeScale: { borderColor: dark ? "#2a2e39" : "#d7dde5", timeVisible: true, secondsVisible: false, rightOffset: 18, tickMarkFormatter: locTick },
      crosshair: { mode: 0 }, kineticScroll: { mouse: true, touch: true },
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
      handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: { time: true, price: true }, axisDoubleClickReset: { time: true, price: true } },
      localization: { locale: "uk-UA", timeFormatter: locFull },
    });
    IW.series = IW.chart.addCandlestickSeries({ priceFormat: { type: "price", precision: 2, minMove: 0.01 } });
    applyCandlePalette();
    IW.prim = new BoxPrimitive(); IW.series.attachPrimitive(IW.prim); IW.pocLine = null;
    IW.chart.subscribeClick(function (p) { var s = p && p.point ? IW.prim.hitAt(p.point.x, p.point.y) : null; sigCard(s, p && p.point ? p.point.x : 0, p && p.point ? p.point.y : 0); });
    IW.chart.subscribeCrosshairMove(function (p) { var hit = p && p.point ? IW.prim.hitAt(p.point.x, p.point.y) : null; box.style.cursor = hit ? "pointer" : ""; });
    tvPanInit(box);

    if (window.VSV_PRO) { try { window.VSV_PRO.attach(IW); } catch (err) { console.error("[VSV_PRO]", err); } }
    return true;
  }

  function precisionOf(p) { p = Number(p) || 0; return p < 0.0001 ? 8 : p < 0.01 ? 6 : p < 1 ? 5 : p < 100 ? 4 : 2; }

  /* OB і VOB утворює бот. Тут не змінюємо їхні межі чи модель виявлення:
     лише відсіюємо пошкоджені дані, прив'язуємо до реальної ділянки OHLC
     та прибираємо звичайний блок після закриття за протилежною межею. */
  function blockGeometry(z, o) {
    var side = String(z.side || "").toUpperCase(), top = Number(z.top), bottom = Number(z.bottom), start = Number(z.start);
    if ((side !== "LONG" && side !== "SHORT") || !o.length || !isFinite(start) || !isFinite(top) || !isFinite(bottom) || top <= bottom) return null;
    if (start < Number(o[0].time) || start > Number(o[o.length - 1].time)) return null;
    return { start: snapSec(start, o), top: top, bottom: bottom, side: side };
  }
  function blockMitigated(z, o) {
    var step = o.length > 1 ? Number(o[1].time) - Number(o[0].time) : 0;
    var from = Number(z.start) + step; // свічка утворення не вважається ретестом
    for (var i = 0; i < o.length; i++) {
      var b = o[i]; if (Number(b.time) < from) continue;
      var close = Number(b.close); if (!isFinite(close)) continue;
      if (z.side === "LONG" && close < Number(z.bottom)) return true;
      if (z.side === "SHORT" && close > Number(z.top)) return true;
    }
    return false;
  }

  function boxesOf(c) {
    var o = c.ohlc || [], out = [];
    (c.vob || []).forEach(function (raw) {
      var g = blockGeometry(raw, o); if (!g) return;
      var v = Object.assign({}, raw, g);
      if (!v.breaker && blockMitigated(v, o)) return;
      var L = v.side === "LONG";
      out.push({ kind: "vob", t: v.start, top: v.top, bottom: v.bottom, dashed: v.breaker,
        fill: L ? "rgba(34,197,94,.30)" : "rgba(242,54,69,.32)", stroke: L ? "rgba(34,197,94,.55)" : "rgba(242,54,69,.6)", text: L ? "#86efac" : "#fca5a5",
        volTxt: fmtVol(v.volume, 3), topPct: v.hi_pct, botPct: v.lo_pct,
        topCol: L ? "rgba(34,197,94,.45)" : "rgba(242,54,69,.45)", botCol: L ? "rgba(34,197,94,.18)" : "rgba(242,54,69,.18)",
        label: tfWord(v.tf || c.vob_tf) + " VOB" + (v.breaker ? " ✕" : ""),
        sub: [fmtVol(v.volume, 3), v.hi_pct != null ? Math.round(Math.max(v.hi_pct, v.lo_pct)) + "%" : (v.pct != null ? v.pct + "%" : "")].filter(Boolean).join(" · ") });
    });
    (c.ob || []).forEach(function (raw) {
      var g = blockGeometry(raw, o); if (!g) return;
      var b = Object.assign({}, raw, g);
      if (blockMitigated(b, o)) return;
      var L = b.side === "LONG", sw = b.fam === "swing";
      out.push({ kind: "ob", thin: true, t: b.start, top: b.top, bottom: b.bottom, dashed: false,
        fill: L ? "rgba(41,98,255,.16)" : "rgba(190,90,105,.18)", stroke: L ? "rgba(96,140,255,.4)" : "rgba(224,122,138,.45)", text: L ? "#93c5fd" : "#fda4af",
        label: "OB " + tfLbl(c.ob_tf), sub: (sw ? "swing" : "int") + (b.tag ? " · " + b.tag : "") });
    });
    if (IW.show_imb) (c.imb || []).forEach(function (z) {
      var L = z.side === "LONG";
      var t0 = snapSec(z.start, o), k0 = o.length - 1;
      while (k0 > 0 && Number(o[k0].time) > t0) k0--;
      out.push({ kind: "imb", thin: true, t: t0, top: z.top, bottom: z.bottom, ext: (o.length - 1 - k0) + IMB_EXT_BARS,
        fill: L ? IMB_BG_BUY : IMB_BG_SELL, stroke: "rgba(0,0,0,0)", text: L ? "#5eead4" : "#fdba74",
        volTxt: fmtVol(z.vol, 3), topPct: z.sell_pct, botPct: z.buy_pct, topCol: IMB_SELL, botCol: IMB_BUY, label: "IMB" });
    });
    return out;
  }

  function sbrOf(c) {
    if (!IW.show_sbr) return [];
    var o = c.ohlc || [], byT = {}, out = [], same = !c.sbr_eff || c.sbr_eff === (c.tf || IW.tf);
    if (!o.length) return [];
    o.forEach(function (b) { byT[b.time] = b; });
    (c.sbr || []).forEach(function (s) {
      /* Свій TF — свічка сигналу; інший TF — свічка ГРАФІКА, на якій сигнал
         спрацював (закриття його бару), щоб позначка стояла в момент сигналу. */
      var t = same ? s.t : snapSec((s.ct || s.t) - 1, o);
      if (!same && ((s.ct || s.t) - 1 < o[0].time)) return;      // раніше за графік
      var bar = byT[t]; if (!bar) return;
      out.push(Object.assign({ hi: bar.high, lo: bar.low, close: bar.close }, s, { t: t, tf: String(s.tf || "").toUpperCase() }));
    });
    return out;
  }

  /* ⚠ 24.11: сигнали «Аномалії обсягу» — та сама позначка B/S, що SBR, лише
     ІНШОГО кольору; стоїть на свічці графіка, де сигнал спрацював (ct). */
  var ANOM_LONG = "#38bdf8", ANOM_SHORT = "#f59e0b";
  function anomOf(c) {
    if (!IW.show_anom) return [];
    var o = c.ohlc || [], byT = {}, out = [];
    if (!o.length) return [];
    o.forEach(function (b) { byT[b.time] = b; });
    (c.anom_sig || []).forEach(function (s) {
      var ct = s.ct || s.t; if (ct < o[0].time) return;
      var t = snapSec(ct, o), bar = byT[t]; if (!bar) return;
      out.push(Object.assign({ hi: bar.high, lo: bar.low, close: bar.close }, s, { t: t, kind: "anom" }));
    });
    return out;
  }

  function sigCard(s, x, y) {
    var box = $("#iw-box"), el = $("#iw-sig");
    if (!box) return;
    if (!el) { el = document.createElement("div"); el.id = "iw-sig"; el.className = "iw-sig"; box.appendChild(el); sigDragInit(el, box); }
    if (!s) { el.style.display = "none"; IW.sigSel = null; sigArrow(); return; }
    IW.sigSel = { t: s.t, side: s.side, kind: s.kind || "" };
    if (s.kind === "anom") { anomCard(el, box, s, x, y); return; }
    var S = s.side === "SHORT", ct = sct(s), dd = ct ? locDM(ct) : "", slPct = s.sl && s.close ? Math.abs(s.sl / s.close - 1) * 100 : null;
    var row = function (k, v, cls) { return '<div class="r"><span>' + k + '</span><b class="' + (cls || "") + '">' + v + "</b></div>"; };
    var sc = s.score != null ? Number(s.score) : null, scCls = sc == null ? "" : (sc >= 85 ? "v-hi" : (sc >= 65 ? "v-mid" : "v-lo"));
    el.innerHTML =
      '<div class="h ' + (S ? "short" : "long") + '"><span class="tag">' + (S ? "S" : "B") + "</span>" + (S ? "SHORT" : "LONG") + (s.n ? " #" + s.n : "") +
      '<span class="tf">SBR · ' + esc(s.tf || "") + " · " + esc(dd) + " " + esc(ct ? locHM(ct) : "") + "</span>" +
      '<button class="x" aria-label="Закрити">×</button></div>' +
      row("Trigger", fmtPrice(s.trigger), "v-trg") +
      '<div class="r"><span>Score<small>(сила)</small></span><b class="' + scCls + '">' + (sc != null ? sc.toFixed(2) + "%" : "—") + "</b></div>" +
      row("Ризик", esc(s.risk || "—"), "v-risk") +
      row("SL", fmtPrice(s.sl) + (slPct != null ? ' <i>' + slPct.toFixed(2) + "%</i>" : ""), "v-sl");
    el.classList.toggle("is-short", S); el.classList.toggle("is-long", !S); el.classList.remove("is-anom"); el.style.display = "block";
    var bw = box.clientWidth, bh = box.clientHeight, ew = el.offsetWidth, eh = el.offsetHeight;
    var left = Math.min(Math.max(8, x + 14), bw - ew - 8), top = Math.min(Math.max(8, y - eh / 2), bh - eh - 8);
    el.style.left = left + "px"; el.style.top = top + "px"; sigArrowLoop();
  }

  function anomCard(el, box, s, x, y) {
    var S = s.side === "SHORT", ct = s.ct || s.t;
    var row = function (k, v, cls) { return '<div class="r"><span>' + k + '</span><b class="' + (cls || "") + '">' + v + "</b></div>"; };
    el.innerHTML =
      '<div class="h ' + (S ? "short" : "long") + '"><span class="tag">' + (S ? "S" : "B") + "</span>" + (S ? "SHORT" : "LONG") +
      '<span class="tf">⚠ Аномалії · ' + esc(locDM(ct)) + " " + esc(locHM(ct)) + "</span>" +
      '<button class="x" aria-label="Закрити">×</button></div>' +
      row("Сетап", esc(s.name || "—")) +
      row("Ціна сигналу", s.px ? fmtPrice(s.px) : "—", "v-trg") +
      (s.adir ? row("Order flow 6 год", (s.adir === "LONG" ? "▲ " : "▼ ") + esc(s.adir)) : "") +
      ((s.stages || []).length ? '<div class="r st"><span>Етапи</span><b>' + s.stages.map(function (x) { return "✓ " + esc(x); }).join("<br>") + "</b></div>" : "");
    el.classList.toggle("is-short", S); el.classList.toggle("is-long", !S); el.classList.add("is-anom"); el.style.display = "block";
    var bw = box.clientWidth, bh = box.clientHeight, ew = el.offsetWidth, eh = el.offsetHeight;
    var left = Math.min(Math.max(8, x + 14), bw - ew - 8), top = Math.min(Math.max(8, y - eh / 2), bh - eh - 8);
    el.style.left = left + "px"; el.style.top = top + "px"; sigArrowLoop();
  }

  function sigAnchor() {
    var sel = IW.sigSel, hits = (IW.prim && IW.prim._hits) || []; if (!sel) return null;
    for (var i = 0; i < hits.length; i++) {
      var h = hits[i], g = h.sig || {};
      if (g.t === sel.t && g.side === sel.side && (g.kind || "") === (sel.kind || "")) return { x: (h.x0 + h.x1) / 2, y: (h.y0 + h.y1) / 2, r: (h.x1 - h.x0) / 2 };
    }
    return null;
  }

  function sigEdge(L, T, W, H, tx, ty) {
    var cx = L + W / 2, cy = T + H / 2, dx = tx - cx, dy = ty - cy;
    if (!dx && !dy) return null;
    var k = Math.min(dx ? (W / 2) / Math.abs(dx) : Infinity, dy ? (H / 2) / Math.abs(dy) : Infinity);
    if (k >= 1) return null;
    return { x: cx + dx * k, y: cy + dy * k, ux: dx, uy: dy };
  }

  function sigArrow() {
    var box = $("#iw-box"), el = $("#iw-sig"); if (!box) return false;
    var svg = $("#iw-sig-arrow");
    if (!svg) { svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.id = "iw-sig-arrow"; svg.setAttribute("class", "iw-sig-arrow"); svg.innerHTML = '<path class="tail"/><path class="edge"/><circle class="ring" r="0"/>'; box.appendChild(svg); }
    var a = el && el.style.display !== "none" ? sigAnchor() : null;
    if (!a) { svg.style.display = "none"; return !!(el && el.style.display !== "none"); }
    var L = el.offsetLeft, T = el.offsetTop, W = el.offsetWidth, H = el.offsetHeight, e = sigEdge(L, T, W, H, a.x, a.y);
    if (!e) { svg.style.display = "none"; return true; }
    
    var len = Math.hypot(a.x - e.x, a.y - e.y) || 1, ux = (a.x - e.x) / len, uy = (a.y - e.y) / len;
    var tx = a.x - ux * (a.r + 3), ty = a.y - uy * (a.r + 3);
    var half = Math.min(13, Math.max(6, len / 4)), px = -uy * half, py = ux * half;
    
    // ЗМІНА: Заводимо основу стрілки глибоко всередину картки, щоб вона зливалася з нею
    var bx = e.x - ux * 24, by = e.y - uy * 24;
    
    var mx = (bx + tx) / 2, my = (by + ty) / 2;
    var c1x = mx + px * 0.35, c1y = my + py * 0.35, c2x = mx - px * 0.35, c2y = my - py * 0.35;
    var A = (bx + px) + "," + (by + py), B = (bx - px) + "," + (by - py), P = tx + "," + ty;
    
    svg.style.display = "block"; svg.setAttribute("width", box.clientWidth); svg.setAttribute("height", box.clientHeight);
    var path = svg.querySelector(".tail"), edge = svg.querySelector(".edge"), ring = svg.querySelector(".ring");
    
    path.setAttribute("d", "M" + A + " Q" + c1x + "," + c1y + " " + P + " Q" + c2x + "," + c2y + " " + B + " Z");
    edge.setAttribute("d", "M" + A + " Q" + c1x + "," + c1y + " " + P + " Q" + c2x + "," + c2y + " " + B);
    
    ring.setAttribute("cx", a.x); ring.setAttribute("cy", a.y); ring.setAttribute("r", a.r + 3);
    var S = IW.sigSel && IW.sigSel.side === "SHORT"; svg.classList.toggle("short", S); svg.classList.toggle("long", !S); svg.classList.toggle("anom", !!(IW.sigSel && IW.sigSel.kind === "anom"));
    return true;
  }

  function sigArrowLoop() {
    if (IW.sigRaf) return;
    var step = function () { IW.sigRaf = null; if (sigArrow()) IW.sigRaf = requestAnimationFrame(step); };
    IW.sigRaf = requestAnimationFrame(step);
  }

  function sigDragInit(el, box) {
    var d = null;
    el.addEventListener("pointerdown", function (e) {
      if (e.button !== 0 || !(e.target.closest && e.target.closest(".h")) || e.target.closest(".x")) return;
      d = { id: e.pointerId, sx: e.clientX, sy: e.clientY, l: el.offsetLeft, t: el.offsetTop };
      try { el.setPointerCapture(e.pointerId); } catch (x) {} el.classList.add("drag"); e.preventDefault();
    });
    el.addEventListener("pointermove", function (e) {
      if (!d || e.pointerId !== d.id) return;
      var bw = box.clientWidth, bh = box.clientHeight, ew = el.offsetWidth, eh = el.offsetHeight;
      var l = Math.min(Math.max(0, d.l + e.clientX - d.sx), Math.max(0, bw - ew)), t = Math.min(Math.max(0, d.t + e.clientY - d.sy), Math.max(0, bh - eh));
      el.style.left = l + "px"; el.style.top = t + "px"; sigArrow();
    });
    var end = function (e) { if (!d || (e && e.pointerId !== d.id)) return; d = null; el.classList.remove("drag"); };
    el.addEventListener("pointerup", end); el.addEventListener("pointercancel", end);
    ["wheel", "dblclick"].forEach(function (ev) { el.addEventListener(ev, function (e) { e.stopPropagation(); }); });
  }

  document.addEventListener("click", function (e) { if (e.target.closest && e.target.closest("#iw-sig .x")) sigCard(null); });

  var HL_BULL = "#00e676", HL_BEAR = "#ff1744", HL_PX = 9;
  function structOf(c) {
    var o = c.ohlc || [], lines = [], marks = [], hl = []; if (!o.length) return { lines: lines, marks: marks, hl: hl };
    var first = o[0].time;
    (c.events || []).forEach(function (e) {
      if (!e.to || e.to < first) return;
      var bull = e.dir === "bull", col = bull ? "#22c55e" : "#ef4444", t2 = snapSec(e.to, o);
      lines.push({ t1: snapSec(e.from, o), t2: t2, price: e.level, color: col, dashed: e.tag !== "BOS", text: e.tag, up: bull });
    });
    // 📈 HH/HL/LH/LL: мітка над (high) / під (low) свічкою півота, яскравий
    // колір напрямку структури; малює примітив (свій, менший шрифт HL_PX). Півот іншого TF прив'язується до свічки графіка.
    (c.hl || []).forEach(function (h) {
      if (!h.t || h.t < first) return;
      var bull = h.dir === "bull";
      hl.push({ t: snapSec(h.t, o), price: h.price, high: !!h.high, color: bull ? HL_BULL : HL_BEAR, text: h.type });
    });
    return { lines: lines, marks: marks, hl: hl };
  }

  function setShl(shl) {
    if (!IW.series) return;
    var key = shl && IW.show_shl ? [shl.tf, (shl.high || {}).price, (shl.low || {}).price, (shl.high || {}).label, (shl.low || {}).label, _lvTitleOff()].join("|") : "";
    if (key === IW.shlKey) return;
    (IW.shlLines || []).forEach(function (l) { try { IW.series.removePriceLine(l); } catch (x) {} });
    IW.shlLines = []; IW.shlKey = key; if (!shl || !IW.show_shl) return;
    [["high", "#f87171"], ["low", "#4ade80"]].forEach(function (p) {
      var v = shl[p[0]]; if (!v || !(v.price > 0)) return;
      IW.shlLines.push(IW.series.createPriceLine({ price: Number(v.price), color: p[1], lineWidth: 1, lineStyle: 0, axisLabelVisible: true, lineVisible: !_lvTitleOff(), title: _lvTitleOff() ? "" : (v.label || "") + " " + (shl.tf || "") }));
    });
  }

  function setPoc(poc) {
    if (!IW.series) return;
    if (IW.pocLine) { try { IW.series.removePriceLine(IW.pocLine); } catch (x) {} IW.pocLine = null; }
    if (!poc || !poc.poc) return;
    IW.pocLine = IW.series.createPriceLine({ price: Number(poc.poc), color: "#f5f5f5", lineWidth: 1, lineStyle: 0, axisLabelVisible: true, lineVisible: !_lvTitleOff(), title: _lvTitleOff() ? "" : "POC" });
  }
  function _lvTitleOff() { return !!(LQ && LQ.on && LQ.data); }
  function liqLevelTitles() { if (!IW.series) return; var c = IW.coin || {}; setShl(c.shl); setPoc(c.poc); if (IW.prim && IW.prim._req) IW.prim._req(); }

  function renderHead(quotesOnly) {
    var h = $("#iw-head"), lg = $("#iw-legend");
    if (!h) return;
    var sym = curSym();
    if (!sym) { h.innerHTML = '<span class="iw-title">—</span>'; IW.layerLg = ""; if (lg) legendPaint(); return; }
    var c = IW.coin || {}, px = livePx(sym);
    if (quotesOnly) {
      var quote = h.querySelector(".iw-px");
      if (quote) {
        var qt = px ? fmtPrice(px) : "—", qc = "iw-px" + dirCls(sym);
        if (quote.textContent !== qt) quote.textContent = qt;
        if (quote.className !== qc) quote.className = qc;
        return;
      }
    }
    /* OHLC-легенда переїхала у шапку графіка; SBR-бейджі 1H/4H видалені. */
    /* 17.11: OHLC-легенду НЕ перестворюємо — шапка перемальовується щосекунди
       (жива ціна), і порожній новий вузол легенди блимав «то є, то нема», доки
       її не домальовувало наступне оновлення. Тепер той самий вузол
       переноситься в нову розмітку, а незмінна шапка не чіпається взагалі. */
    var headHtml = '<span class="iw-title' + mmCls(sym) + '" title="' + esc(mmTip(sym)) + '">' + tvSym(sym, !!(c.src && c.src !== "ext")) + "</span>" +
      (IW.view ? '<span class="iw-peek" title="Відкрито подвійним кліком із VSV SIGNALS. Вихід із повного екрана (Esc / ✕ / подвійний клік) повертає вашу монету.">🛰 VSV SIGNALS</span>' : "") +
      '<span class="iw-px' + dirCls(sym) + '">' + (px ? fmtPrice(px) : "—") + "</span>" + (pOk("ly_pd") ? pdBadge(c.pd) : "") + (pOk("ly_poc") ? pocBadge(c.poc, c.poc_state) : "") +
      (c.src === "ext" ? '<span class="bdg warn ext-badge" title="Поза WATCHLIST бота — розрахунок окремим потоком">🛰 NOT IN BOT</span>' : "");
    var keepLg = document.getElementById("tv-ohlc"), keepCoin = document.getElementById("iw-head-coin");
    if (h._headSig !== headHtml || !keepLg || keepLg.parentNode !== h || !keepCoin || keepCoin.parentNode !== h) {
      h.innerHTML = headHtml;
      h._headSig = headHtml;
      if (!keepCoin || keepCoin.getAttribute("data-coin") !== sym) {
        var coinWrap = document.createElement("span");
        coinWrap.innerHTML = coinIcon(sym, "coin-ic-head"); keepCoin = coinWrap.firstChild; keepCoin.id = "iw-head-coin";
      }
      h.insertBefore(keepCoin, h.firstChild);
      if (!keepLg) {
        keepLg = document.createElement("span");
        keepLg.className = "tv-ohlc"; keepLg.id = "tv-ohlc";
        keepLg.innerHTML = '<span class="ohlc-legend"></span>';
      }
      h.appendChild(keepLg);
    }
    /* Легенда шарів переїхала в нижній рядок графіка (chart-footer). */
    IW.layerLg =
      '<span title="Volumized Order Block"><i style="background:rgba(34,197,94,.55)"></i>VOB ' + tfLbl(IW.vob_tf) + "</span>" +
      '<span title="Order Block"><i style="background:rgba(41,98,255,.2);outline:1px solid rgba(120,160,255,.7)"></i>OB ' + tfLbl(IW.ob_tf) + "</span>" +
      '<span title="CHoCH — зміна характеру руху · BOS — злам структури"><i style="background:#22c55e;height:2px"></i>CHoCH/BOS</span>' +
      '<span title="IMB — імбаланс (Fair Value Gap) з розкладом обсягу покупців / продавців"><i style="background:rgba(234,128,38,.6)"></i>IMB</span>' +
      '<span title="SBR — Session Band Reversal"><i style="background:#f23645;width:9px"></i><i style="background:#22c55e;width:9px;margin-left:-3px"></i>SBR</span>' +
      '<span title="POC — Point of Control"><i style="background:#f5f5f5;height:2px"></i>POC</span>' +
      '<span title="High / Low — межі дилінг-діапазону"><i style="background:#f87171;height:2px;width:7px"></i><i style="background:#4ade80;height:2px;width:7px;margin-left:-3px"></i>High/Low</span>';
    if (lg) legendPaint();
  }

  function setNote(t) {
    var n = $("#iw-note"); if (!n) return;
    /* Усі повідомлення про процес завантаження — лише у новому вигляді
       лоадера; тексти помилок лишаються текстовими. */
    var s = String(t || "");
    if (s && /завантаж|очіку|рахуєть|loading|загруз/i.test(s) && !/не вдалося|не завантажилась|помилка|error|fail/i.test(s)) { setLoading(); return; }
    n.textContent = s; n.style.display = s ? "" : "none";
  }
  function setLoading() {
    /* Єдиний вигляд процесу завантаження скрізь: лого → LOADING → круг.
       Жодних додаткових текстів. */
    var n = $("#iw-note"); if (!n) return;
    n.innerHTML = '<span class="chart-loading"><img src="/static/info/img/logo-loading.svg?v=214" alt="VSV"><b>LOADING</b><span class="boot-spin" aria-hidden="true"></span></span>';
    n.style.display = "";
  }
  function setWait(c) {
    var w = $("#iw-tf-wait"); if (!w) return;
    var parts = []; if (c && c.ob_wait) parts.push("OB " + tfLbl(c.ob_tf)); if (c && c.vob_wait) parts.push("VOB " + tfLbl(c.vob_tf)); if (c && c.hl_wait) parts.push("HH/HL " + tfLbl(c.hl_tf)); if (c && c.sbr_wait) parts.push("SBR " + tfLbl(c.sbr_eff));
    w.innerHTML = parts.length ? '<span class="wait">⏳ рахується: ' + esc(parts.join(", ")) + "</span>" : "";
  }

  /* ⚡ 15.11 ЖИВА СВІЧКА як у TradingView: публічний WebSocket Bybit
     (kline.<iv>.<SYM>) прямо з браузера — ціна рухається щосекунди, сервер
     бота не навантажується. Мережа/обрив → перепідключення з паузою; полл
     сервера (раз на 15-30с) лишається запасним шляхом. */
  var LIVE_WS_URL = "wss://stream.bybit.com/v5/public/linear";
  var LIVE_IV = { "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720", "1d": "D" };
  function liveIv(tf) { return LIVE_IV[String(tf || "").toLowerCase()] || null; }
  function liveBarOf(k) {
    if (!k) return null;
    var t = Math.floor(Number(k.start) / 1000), o = Number(k.open), h = Number(k.high), l = Number(k.low), c = Number(k.close);
    if (!(t > 0) || !(c > 0) || !(h > 0) || !(l > 0)) return null;
    return { time: t, open: o, high: h, low: l, close: c, confirm: !!k.confirm };
  }
  function LiveKline(onBar, onTrade) { this.onBar = onBar; this.onTrade = onTrade || null; this.ws = null; this.key = ""; this.tkey = ""; this.t0 = 0; this.retry = 0; this.ping = null; this.rt = null; this.last = 0; }
  /* trades=true → у тому самому зʼєднанні ще й publicTrade.<SYM> (кожна угода
     зі стороною тейкера) — для розкладу обсягу живої свічки. */
  LiveKline.prototype.set = function (sym, tf, trades) {
    var iv = liveIv(tf), s = String(sym || "").toUpperCase();
    var key = (iv && /^[A-Z0-9]+USDT$/.test(s)) ? "kline." + iv + "." + s : "";
    var tkey = (key && trades) ? "publicTrade." + s : "";
    if (key === this.key && tkey === this.tkey) return;
    this.key = key; this.tkey = tkey; this.retry = 0; this._close();
    if (key) this._open();
  };
  LiveKline.prototype._close = function () {
    clearInterval(this.ping); this.ping = null; clearTimeout(this.rt); this.rt = null;
    if (this.ws) { var w = this.ws; this.ws = null; w.onclose = w.onmessage = w.onerror = null; try { w.close(); } catch (e) {} }
  };
  LiveKline.prototype._open = function () {
    if (typeof WebSocket === "undefined" || !this.key) return;
    var self = this, key = this.key, ws;
    try { ws = new WebSocket(LIVE_WS_URL); } catch (e) { return this._later(); }
    this.ws = ws;
    ws.onopen = function () {
      if (self.ws !== ws) return;
      self.retry = 0; self.t0 = 0;
      ws.send(JSON.stringify({ op: "subscribe", args: self.tkey ? [key, self.tkey] : [key] }));
      self.ping = setInterval(function () { try { ws.send('{"op":"ping"}'); } catch (e) {} }, 20000);
    };
    ws.onmessage = function (ev) {
      if (self.ws !== ws) return;
      var m; try { m = JSON.parse(ev.data); } catch (e) { return; }
      if (!m || !m.topic || !m.data) return;
      /* t0 — час БІРЖІ першого повідомлення після (пере)підключення: від нього
         потік без розривів (годинник ПК тут не бере участі). */
      if (!self.t0 && Number(m.ts) > 0) self.t0 = Number(m.ts);
      if (self.tkey && m.topic === self.tkey) { if (self.onTrade) { try { self.onTrade(self.tkey, m.data, self.t0); } catch (e) { console.error("[live]", e); } } return; }
      if (m.topic !== key) return;
      self.last = Date.now();
      for (var i = 0; i < m.data.length; i++) { var b = liveBarOf(m.data[i]); if (b) { var q = Number(m.data[i].volume); if (q >= 0) b.volume = q; try { self.onBar(key, b); } catch (e) { console.error("[live]", e); } } }
    };
    ws.onerror = function () {};
    ws.onclose = function () { if (self.ws === ws) { self.ws = null; clearInterval(self.ping); self.ping = null; self._later(); } };
  };
  LiveKline.prototype._later = function () {
    var self = this; if (!this.key) return;
    var d = Math.min(30000, 1000 * Math.pow(2, this.retry++));
    clearTimeout(this.rt); this.rt = setTimeout(function () { self._open(); }, d);
  };
  LiveKline.prototype.ok = function () { return !!this.ws && Date.now() - this.last < 15000; };

  function liveKeyOf(sym, tf) { var iv = liveIv(tf); return iv ? "kline." + iv + "." + String(sym || "").toUpperCase() : ""; }
  /* Свіжий бар потоку поверх даних сервера: той самий бар → ціни з потоку;
     новіший → дописується (ahead: true — сервер його ще не бачив). */
  function liveMerge(c) {
    var b = IW.liveBar, o = c && c.ohlc;
    if (!b || !o || !o.length || !IW.live || !IW.live.ok() || b.key !== liveKeyOf(c.symbol, c.tf)) return {};
    var last = o[o.length - 1];
    if (b.time === last.time) { last.open = b.open; last.high = b.high; last.low = b.low; last.close = b.close; if (b.volume != null) last.volume = b.volume; return { same: true }; }
    if (b.time > last.time) { o.push({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume || 0 }); return { ahead: true, bar: b }; }
    return {};
  }
  function liveSet(c) {
    if (!IW.live) IW.live = new LiveKline(onLiveBar, onLiveTrades);
    IW.live.set(c && c.symbol, c && c.tf, VOL.on);
  }
  var liveUiAt = 0;
  function onLiveBar(key, b) {
    var c = IW.coin;
    if (!IW.series || !c || !c.ohlc || !c.ohlc.length || key !== liveKeyOf(c.symbol, c.tf)) return;
    var o = c.ohlc, last = o[o.length - 1];
    if (b.time < last.time) return;
    b.key = key; IW.liveBar = b;
    var bar = { time: b.time, open: b.open, high: b.high, low: b.low, close: b.close };
    if (b.time === last.time) { last.open = b.open; last.high = b.high; last.low = b.low; last.close = b.close; if (b.volume != null) last.volume = b.volume; }
    else o.push({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume || 0 });
    try { IW.series.update(bar); } catch (e) { return; }
    volLive();
    var now = Date.now();
    if (now - liveUiAt >= 1000) { liveUiAt = now; try { renderHead(true); renderQuotes(); } catch (e) {} }
  }

  /* 📊 16.11 ОБСЯГ: ПОКУПЦІ / ПРОДАВЦІ В КОЖНІЙ СВІЧЦІ.
     Стовпчик = обсяг свічки (монети, як у TradingView); знизу — частка
     агресивних ПОКУПЦІВ (зелена), зверху — ПРОДАВЦІВ (червона); бік, якого
     було більше, — насичений. Джерела розкладу, за точністю:
       ⚡ tape — справжні угоди Bybit (publicTrade, сторона тейкера), коли потік
                 покрив свічку з її відкриття: точно;
       ⚡≈ mix — свічка, посеред якої відкрили сторінку: оцінка сервера до
                 моменту asof + справжні угоди після;
       ≈ sub  — історія: підсвічки Bybit (зелена → покупці, червона → продавці),
                 рахує сервер бота, спільний кеш на всіх.
     Малює РІДНА серія графіка (custom series): прокрутка, масштаб, зміна
     розміру й автошкала — тим самим проходом, що й свічки. */
  var VOL_UP = "0,187,127", VOL_DN = "251,44,54", VOL_NA = "128,138,157", VOL_TOP = 0.82;
  /* 24.11: «Обсяг» / «Аномалії обсягу» на відкритті сторінки ЗАВЖДИ вимкнені (вибір не памʼятаємо — діє, поки вікно відкрите). */
  function volInd(k) { return false; }
  function volOn() { return volInd("volume"); }
  var VOL = { on: volOn(), anom: volInd("volanom"), chart: null, series: null, plain: false, key: "", srv: {}, sub: 0, fill: false, wait: false, base: null, tape: null, rows: [], idx: {}, hover: null, html: "", sel: null, tm: null };

  /* ⚠ 16.11 АНОМАЛІЇ ОБСЯГУ — те, на що варто звернути увагу перед входом.
     Лише з даних самого індикатора (обсяг свічки + розклад покупці/продавці),
     без нових запитів. База — МЕДІАНА обсягу попередніх VOL_AN.base свічок
     (медіана, а не середнє: один сплеск не роздуває базу для сусідів).
     Пріоритет (одна позначка на свічку):
       🧽 absorb — поглинання: агресія одного боку ≥ 65% при обсязі ≥ ×1.5, а
                   свічка закрилась ПРОТИ неї (тиск поглинули лімітні ордери —
                   часта ознака розвороту);
       🔥 climax — кульмінація: обсяг ≥ ×3 медіани (вершина / дно руху,
                   ліквідації, новина);
       ⚡ side   — одностороння агресія: ≥ 75% в один бік при обсязі ≥ ×1.5;
       📈 spike  — сплеск обсягу ≥ ×2 без явної переваги боку.
     Свічки без розкладу (≈ ще рахується) мають лише 🔥 / 📈. */
  var VOL_AN = { base: 20, minBase: 10, rv: 1.5, spike: 2, climax: 3, side: 0.75, absorb: 0.65 };
  var VOL_AN_COL = { absorb: "#c084fc", climax: "#f59e0b", spike: "#94a3b8", side_up: "#00bb7f", side_dn: "#fb2c36" };
  function volMedian(a) { var b = a.slice().sort(function (x, y) { return x - y; }), n = b.length; return n ? (n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2) : 0; }
  function volAnomalyAt(rows, i) {
    var r = rows[i], v = r ? Number(r.value) : 0;
    if (!(v > 0)) return null;
    var prev = [];
    for (var j = i - 1; j >= 0 && prev.length < VOL_AN.base; j--) { var pv = Number(rows[j].value); if (pv > 0) prev.push(pv); }
    if (prev.length < VOL_AN.minBase) return null;
    var med = volMedian(prev); if (!(med > 0)) return null;
    var rv = v / med, sh = r.buy, up = r.up;
    if (sh != null && rv >= VOL_AN.rv && up != null) {
      if (sh >= VOL_AN.absorb && !up) return { k: "absorb", rv: rv, sh: sh, by: "sell" };
      if (sh <= 1 - VOL_AN.absorb && up) return { k: "absorb", rv: rv, sh: sh, by: "buy" };
    }
    if (rv >= VOL_AN.climax) return { k: "climax", rv: rv, sh: sh };
    if (sh != null && rv >= VOL_AN.rv && (sh >= VOL_AN.side || sh <= 1 - VOL_AN.side)) return { k: "side", rv: rv, sh: sh, dir: sh >= 0.5 ? "up" : "dn" };
    if (rv >= VOL_AN.spike) return { k: "spike", rv: rv, sh: sh };
    return null;
  }
  function volMarkAnomalies(rows, from) {
    for (var i = Math.max(0, from || 0); i < rows.length; i++) rows[i].an = volAnomalyAt(rows, i);
  }
  /* Σ 16.11 СУМА ВІД СВІЧКИ ДО ПОТОЧНОЇ (подвійний клік по стовпчику обсягу).
     ЧИСТА: рядки обсягу + свічки графіка → підсумок діапазону [t0 … остання].
     Покупці/продавці — лише зі свічок, де розклад уже є (`split` / `n`). */
  function volRangeSum(rows, t0, ohlc) {
    var i0 = -1;
    for (var i = 0; i < (rows || []).length; i++) if (rows[i].time === t0) { i0 = i; break; }
    if (i0 < 0) return null;
    var vol = 0, buy = 0, sell = 0, split = 0, an = 0, n = 0;
    for (var j = i0; j < rows.length; j++) {
      var r = rows[j], v = Number(r.value) || 0;
      n++; vol += v;
      if (r.buy != null && v > 0) { buy += v * r.buy; sell += v * (1 - r.buy); split++; }
      if (r.an) an++;
    }
    var o0 = null, cN = null;
    (ohlc || []).forEach(function (b) { if (b.time === t0) o0 = Number(b.open); });
    if (ohlc && ohlc.length) cN = Number(ohlc[ohlc.length - 1].close);
    var chg = (o0 > 0 && cN > 0) ? (cN / o0 - 1) * 100 : null;
    return { t0: t0, n: n, vol: vol, buy: buy, sell: sell, split: split, delta: buy - sell,
             share: (buy + sell) > 0 ? buy / (buy + sell) : null, anoms: an, chg: chg };
  }
  function volRangeHTML(m) {
    var h = '<span class="vl-k vl-sum">Σ від ' + esc(locFull(m.t0)) + "</span>" +
      ' <span class="vl-n">' + m.n + " св.</span>" +
      ' <b class="vl-v">' + esc(fmtVol(m.vol, 3) || "0") + "</b>";
    if (m.share != null) {
      var bp = Math.round(m.share * 100);
      h += ' <span class="vl-up' + (bp >= 50 ? " vl-win" : "") + '">▲ покупці ' + bp + "% · " + esc(fmtVol(m.buy, 3) || "0") + "</span>" +
        ' <span class="vl-dn' + (bp < 50 ? " vl-win" : "") + '">▼ продавці ' + (100 - bp) + "% · " + esc(fmtVol(m.sell, 3) || "0") + "</span>" +
        ' <span class="' + (m.delta >= 0 ? "vl-up" : "vl-dn") + '">Δ ' + (m.delta >= 0 ? "+" : "−") + esc(fmtVol(Math.abs(m.delta), 3) || "0") + "</span>";
      if (m.split < m.n) h += ' <span class="vl-n" title="Дельта ще розраховується: розклад buy / sell є для ' + m.split + ' свічок">розклад ' + m.split + "/" + m.n + "</span>";
    }
    if (m.chg != null) h += ' <span class="' + (m.chg >= 0 ? "vl-up" : "vl-dn") + '" title="Зміна ціни від open першої свічки діапазону">ціна ' + (m.chg >= 0 ? "+" : "") + m.chg.toFixed(2) + "%</span>";
    if (VOL.anom && m.anoms) h += ' <span class="vl-n">⚠ ' + m.anoms + "</span>";
    return h + ' <span class="vl-x" title="Скинути діапазон (Esc або повторний подвійний клік)">✕</span>';
  }
  function volSetRange(t) {
    VOL.sel = (t == null || VOL.sel === t) ? null : t;
    /* Нові обʼєкти рядків → бібліотека гарантовано перемальовує панель
       (setData з тими самими обʼєктами вона вважає «без змін»). */
    if (VOL.series && IW.coin) volRebuild(IW.coin);
    volLegend();
  }
  /* Подвійний клік — лише в зоні стовпчиків обсягу (нижні 18% панелі), щоб
     не плутати з подвійним кліком по шкалі цін (автомасштаб). */
  function volDbl(p) {
    if (!VOL.on || !VOL.series || !p || p.time == null || !p.point) return;
    var box = $("#iw-box"); if (!box) return;
    var axis = 0; try { axis = IW.chart.timeScale().height(); } catch (e) {}
    if (p.point.y < (box.clientHeight - axis) * VOL_TOP - 12) return;
    volSetRange(p.time);
  }
  function volAnColor(a) { return a.k === "side" ? VOL_AN_COL[a.dir === "up" ? "side_up" : "side_dn"] : VOL_AN_COL[a.k]; }
  function volAnText(a) {
    if (!a) return ["", ""];
    var x = "×" + a.rv.toFixed(1), bp = a.sh != null ? Math.round(a.sh * 100) : null;
    if (a.k === "absorb") return a.by === "sell"
      ? ["🧽 абсорбція покупок", "Агресивні ринкові покупки " + bp + "%, обсяг " + x + ", але свічка закрилась ЧЕРВОНОЮ: лімітні продавці абсорбували попит — ризик розвороту вниз. LONG без підтвердження не брати."]
      : ["🧽 абсорбція продажів", "Агресивні ринкові продажі " + (100 - bp) + "%, обсяг " + x + ", але свічка закрилась ЗЕЛЕНОЮ: лімітні покупці абсорбували пропозицію — ймовірний розворот угору. SHORT без підтвердження не брати."];
    if (a.k === "climax") return ["🔥 climax-обсяг " + x, "Обсяг " + x + " до медіани " + VOL_AN.base + " свічок: кульмінація руху (buying / selling climax) — каскад ліквідацій або новина. Вхід навздогін — ризик купити вершину / продати дно; чекати реакції наступних свічок." + (bp != null ? " Ринкові покупки " + bp + "%." : "")];
    if (a.k === "side") return a.dir === "up"
      ? ["⚡ агресія покупців " + bp + "%", "Ринкові покупки " + bp + "% при обсязі " + x + ": сильний дисбаланс дельти вгору — підтримує LONG, проти SHORT."]
      : ["⚡ агресія продавців " + (100 - bp) + "%", "Ринкові продажі " + (100 - bp) + "% при обсязі " + x + ": сильний дисбаланс дельти вниз — підтримує SHORT, проти LONG."];
    return ["📈 сплеск обсягу " + x, "Обсяг " + x + " до медіани " + VOL_AN.base + " свічок без вираженої дельти — пожвавлення інтересу, стежити за реакцією ціни."];
  }
  function volTfSec(tf) { var t = String(tf || "").toLowerCase(), n = parseInt(t, 10) || 1, u = t.slice(-1); return n * (u === "d" ? 86400 : u === "h" ? 3600 : 60); }
  /* Стрічка угод: t0 — час біржі, з якого потік БЕЗ розривів; bars[t] = {b, s,
     full, p}. full — потік покрив свічку з відкриття. Для неповної свічки
     угоди памʼятаються шматками по 5 с (p), щоб додати до оцінки сервера лише
     те, що було ПІСЛЯ її asof. */
  function volTapeNew(key, t0) { return { key: key, t0: Number(t0) || 0, bars: {}, last: 0 }; }
  function volTapeAdd(tape, T, side, v, bs) {
    T = Number(T); v = Number(v);
    if (!tape || !(T > 0) || !(v > 0) || !(bs > 0)) return;
    var t = Math.floor(T / 1000 / bs) * bs, r = tape.bars[t];
    if (!r) r = tape.bars[t] = { b: 0, s: 0, full: tape.t0 > 0 && tape.t0 <= t * 1000, p: [] };
    var buy = String(side) === "Buy";
    if (buy) r.b += v; else r.s += v;
    if (!r.full) {
      var k = Math.floor(T / 5000) * 5, q = r.p.length ? r.p[r.p.length - 1] : null;
      if (!q || q[0] !== k) { q = [k, 0, 0]; r.p.push(q); }
      if (buy) q[1] += v; else q[2] += v;
    }
  }
  /* Розрив потоку: точні ЗАКРИТІ свічки зберігаються, поточна — починає знову. */
  function volTapeCarry(old, fresh, liveT) {
    if (!old || !fresh) return fresh;
    Object.keys(old.bars).forEach(function (t) { var r = old.bars[t]; if (r.full && Number(t) < liveT) fresh.bars[t] = { b: r.b, s: r.s, full: true, p: [] }; });
    return fresh;
  }
  /* Розклад однієї свічки → {buy, sell, src} | null (даних ще немає). */
  function volSplit(t, srv, base, tape) {
    var r = tape && tape.bars[t];
    if (r && r.full && r.b + r.s > 0) return { buy: r.b, sell: r.s, src: "tape" };
    if (base && base.t === t) {
      var b = Number(base.b) || 0, s = Number(base.s) || 0, mix = false;
      if (r) for (var i = 0; i < r.p.length; i++) if (r.p[i][0] * 1000 >= base.asof) { b += r.p[i][1]; s += r.p[i][2]; mix = true; }
      if (b + s > 0) return { buy: b, sell: s, src: mix ? "mix" : "sub" };
    }
    var v = srv && srv[t];
    if (v && v[0] + v[1] > 0) return { buy: v[0], sell: v[1], src: "sub" };
    if (r && r.b + r.s > 0) return { buy: r.b, sell: r.s, src: "part" };
    return null;
  }
  /* Рядок серії: value — обсяг свічки; buy — ЧАСТКА покупців 0..1 (null — невідомо). */
  function volRow(bar, srv, base, tape) {
    var sp = volSplit(bar.time, srv, base, tape), sum = sp ? sp.buy + sp.sell : 0, tot = Number(bar.volume) || 0;
    if (!(tot > 0)) tot = sum;
    var share = sp ? sp.buy / sum : null;
    return { time: bar.time, ts: bar.time, value: tot, buy: share, up: (bar.close != null && bar.open != null) ? Number(bar.close) >= Number(bar.open) : null, src: sp ? sp.src : "", color: "rgba(" + (share == null ? VOL_NA : share >= 0.5 ? VOL_UP : VOL_DN) + ",.55)" };
  }
  /* Ширина стовпчика = ширина тіла свічки бібліотеки (той самий розрахунок),
     тож стовпчики стоять рівно під свічками на будь-якому масштабі. */
  function volBarW(spacing, pr) {
    if (spacing >= 2.5 && spacing <= 4) return Math.max(1, Math.floor(3 * pr));
    var k = 1 - 0.2 * Math.atan(Math.max(4, spacing) - 4) / (Math.PI * 0.5);
    var w = Math.max(Math.floor(pr), Math.min(Math.floor(spacing * k * pr), Math.floor(spacing * pr)));
    if (w >= 2 && (Math.floor(pr) % 2) !== (w % 2)) w--;
    return Math.max(1, w);
  }
  /* Геометрія одного стовпчика у пікселях полотна: покупці знизу, продавці зверху. */
  function volBarGeom(y0, yTop, share) {
    var h = Math.max(1, y0 - yTop);
    if (share == null) return { h: h, buy: 0, sell: h, gap: 0 };
    var hb = Math.round(h * Math.max(0, Math.min(1, share)));
    if (share > 0 && hb < 1 && h >= 2) hb = 1;
    if (share < 1 && hb > h - 1 && h >= 2) hb = h - 1;
    var gap = (h >= 8 && hb > 1 && h - hb > 1) ? 1 : 0;
    return { h: h, buy: hb, sell: h - hb - gap, gap: gap };
  }
  /* Анімація лише тоді, коли у видимій області справді є аномалії.
     Новий обʼєкт останнього рядка гарантовано інвалідовує полотно серії,
     не змінюючи жодного значення обсягу чи розкладу покупців / продавців. */
  function volAnFrame(now) {
    VOL.raf = 0;
    if (!VOL.on || !VOL.anom || !VOL.anVisible || !VOL.series || VOL.plain || document.hidden) return;
    if (!VOL.anFrame || now - VOL.anFrame >= 40) {
      VOL.anFrame = now;
      var last = VOL.rows[VOL.rows.length - 1];
      if (!last) return;
      try { VOL.series.update(Object.assign({}, last)); } catch (e) { return; }
    }
    if (!VOL.raf) VOL.raf = requestAnimationFrame(volAnFrame);
  }
  function volAnimate() {
    if (VOL.raf || !VOL.on || !VOL.anom || !VOL.anVisible || !VOL.series || VOL.plain || document.hidden || !window.requestAnimationFrame) return;
    VOL.raf = requestAnimationFrame(volAnFrame);
  }
  document.addEventListener("visibilitychange", function () { if (!document.hidden) volAnimate(); });
  function volAnMarker(ctx, cx, top, col, a, idx, hr, vr, now) {
    var cy = top - Math.round(11 * vr), phase = ((now / 1500) + idx * 0.173) % 1;
    var rot = now / 620 + idx * 0.71, pulse = 1 + Math.sin(now / 170 + idx) * 0.12;
    ctx.save();
    /* Хвиля реакції: мʼяко розходиться й зникає, не обводячи стовпчик. */
    ctx.globalAlpha = (1 - phase) * 0.52;
    ctx.strokeStyle = col; ctx.lineWidth = Math.max(1, 0.85 * hr);
    ctx.beginPath(); ctx.arc(cx, cy, (3.2 + phase * 8.5) * hr, 0, Math.PI * 2); ctx.stroke();
    /* Обертова дуга — спокійний «радар» навколо ядра. */
    ctx.globalAlpha = 0.82; ctx.lineWidth = Math.max(1, 1.05 * hr); ctx.lineCap = "round";
    ctx.beginPath(); ctx.arc(cx, cy, 5.2 * hr, rot, rot + Math.PI * 0.82); ctx.stroke();
    if (a.k === "absorb" || a.k === "climax") {
      ctx.globalAlpha = 0.42;
      ctx.beginPath(); ctx.arc(cx, cy, 7.1 * hr, -rot * 0.72, -rot * 0.72 + Math.PI * 0.55); ctx.stroke();
    }
    /* Три іскри підіймаються з різним зсувом фази. */
    for (var j = 0; j < 3; j++) {
      var p = ((now / 1150) + idx * 0.11 + j / 3) % 1;
      var sx = cx + Math.sin(rot * 0.35 + j * 2.1) * (2.2 + j * 0.6) * hr;
      var sy = cy - (3 + p * 10) * vr;
      ctx.globalAlpha = Math.sin(p * Math.PI) * 0.78; ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(sx, sy, Math.max(0.65, (1.05 - p * 0.45) * hr), 0, Math.PI * 2); ctx.fill();
    }
    /* Діамантове ядро з легким світінням у кольорі типу аномалії. */
    ctx.translate(cx, cy); ctx.rotate(Math.PI / 4);
    ctx.globalAlpha = 1; ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 6 * hr;
    var sz = (a.k === "climax" ? 2.7 : 2.35) * hr * pulse;
    ctx.fillRect(-sz, -sz, sz * 2, sz * 2);
    ctx.restore();
  }
  function VolRenderer() { this.d = null; }
  VolRenderer.prototype.update = function (d) { this.d = d; };
  VolRenderer.prototype.draw = function (target, toY) {
    var d = this.d;
    if (!d || !d.bars || !d.bars.length || !d.visibleRange) return;
    target.useBitmapCoordinateSpace(function (sc) {
      var ctx = sc.context, hr = sc.horizontalPixelRatio, vr = sc.verticalPixelRatio, y0c = toY(0);
      if (y0c == null) return;
      var y0 = Math.round(y0c * vr), w = volBarW(d.barSpacing, hr), half = Math.floor(w / 2);
      var from = Math.max(0, d.visibleRange.from), to = Math.min(d.bars.length, d.visibleRange.to);
      if (VOL.sel != null) {
        /* Σ діапазон від обраної свічки до поточної — смуга над стовпчиками. */
        var si = -1, top0 = y0;
        for (var q = 0; q < d.bars.length; q++) if (d.bars[q].originalData.ts === VOL.sel) { si = q; break; }   // `time` бібліотека з рядка прибирає
        for (var q2 = from; q2 < to; q2++) { var qy = toY(Number(d.bars[q2].originalData.value) || 0); if (qy != null) top0 = Math.min(top0, Math.round(qy * vr)); }
        if (si >= 0 && si < to) {
          var xs = si >= from ? Math.round(d.bars[si].x * hr) - half - Math.round(2 * hr) : 0;
          var yt = Math.max(0, top0 - Math.round(14 * vr));
          ctx.fillStyle = "rgba(56,189,248,.16)"; ctx.fillRect(xs, yt, sc.bitmapSize.width - xs, y0 - yt);
          if (si >= from) {
            ctx.fillStyle = "rgba(56,189,248,.95)"; ctx.fillRect(xs, yt, Math.max(2, Math.round(1.5 * hr)), y0 - yt);
            ctx.font = "700 " + Math.round(10 * vr) + "px JetBrains Mono, monospace"; ctx.textBaseline = "top";
            ctx.fillText("Σ", xs + Math.round(4 * hr), yt + Math.round(2 * vr));
          }
        }
      }
      for (var i = from; i < to; i++) {
        var b = d.bars[i], r = b.originalData, v = Number(r.value);
        if (!(v > 0)) continue;
        var yc = toY(v); if (yc == null) continue;
        var g = volBarGeom(y0, Math.min(Math.round(yc * vr), y0 - 1), r.buy), x = Math.round(b.x * hr) - half;
        if (r.buy == null) { ctx.fillStyle = "rgba(" + VOL_NA + ",.38)"; ctx.fillRect(x, y0 - g.h, w, g.h); continue; }
        var up = r.buy >= 0.5;
        if (g.sell > 0) { ctx.fillStyle = "rgba(" + VOL_DN + "," + (up ? ".30" : ".80") + ")"; ctx.fillRect(x, y0 - g.h, w, g.sell); }
        if (g.buy > 0) { ctx.fillStyle = "rgba(" + VOL_UP + "," + (up ? ".80" : ".30") + ")"; ctx.fillRect(x, y0 - g.buy, w, g.buy); }
      }
      VOL.anVisible = false;
      if (!VOL.anom) return;
      var now = window.performance && performance.now ? performance.now() : Date.now();
      for (var k = from; k < to; k++) {
        var bb = d.bars[k], rr = bb.originalData, a = rr.an;
        if (!a || !(Number(rr.value) > 0)) continue;
        var yk = toY(Number(rr.value)); if (yk == null) continue;
        var top = Math.min(Math.round(yk * vr), y0 - 1), cx = Math.round(bb.x * hr);
        VOL.anVisible = true;
        volAnMarker(ctx, cx, top, volAnColor(a), a, k, hr, vr, now);
      }
      volAnimate();
    });
  };
  function VolPane() { this.r = new VolRenderer(); }
  VolPane.prototype.priceValueBuilder = function (row) { return [0, Number(row.value) || 0]; };
  VolPane.prototype.isWhitespace = function (row) { return row.value == null; };
  VolPane.prototype.renderer = function () { return this.r; };
  VolPane.prototype.update = function (data) { this.r.update(data); };
  VolPane.prototype.defaultOptions = function () { return Object.assign({}, window.LightweightCharts.customSeriesDefaultOptions || {}); };

  function volAnHit(p) {
    if (!VOL.on || !VOL.anom || VOL.plain || !p || p.time == null || !p.point || !VOL.series || !IW.chart) return null;
    var r = VOL.idx[p.time];
    if (!r || !r.an) return null;
    var x = IW.chart.timeScale().timeToCoordinate(p.time), y = VOL.series.priceToCoordinate(Number(r.value));
    if (x == null || y == null) return null;
    var cy = y - 11;
    if (Math.abs(p.point.x - x) > 13 || Math.abs(p.point.y - cy) > 17) return null;
    return { r: r, x: x, y: cy };
  }
  function volAnTipHTML(r) {
    var a = r.an, txt = volAnText(a), bp = r.buy == null ? null : Math.round(r.buy * 100);
    var h = '<div class="va-head"><i style="background:' + volAnColor(a) + '"></i><b>' + esc(txt[0]) + "</b></div>" +
      '<div class="va-time">' + esc(locFull(r.time)) + ' · обсяг ' + esc(fmtVol(r.value, 3)) + ' · ×' + a.rv.toFixed(2) + "</div>";
    if (bp != null) h += '<div class="va-split"><span class="va-up">▲ Покупці ' + bp + '%</span><span class="va-dn">▼ Продавці ' + (100 - bp) + '%</span></div>';
    var o = (IW.coin && IW.coin.ohlc) || [], bar = null;
    for (var i = 0; i < o.length; i++) if (o[i].time === r.time) { bar = o[i]; break; }
    if (bar) h += '<div class="va-ohlc"><span>O ' + esc(fmtPrice(bar.open)) + '</span><span>H ' + esc(fmtPrice(bar.high)) + '</span><span>L ' + esc(fmtPrice(bar.low)) + '</span><span>C ' + esc(fmtPrice(bar.close)) + "</span></div>";
    return h + '<div class="va-note">' + esc(txt[1]) + "</div>";
  }
  function volAnHover(p) {
    var box = $("#iw-box"), tip = $("#vol-an-tip");
    if (!box) return;
    if (!tip) {
      tip = document.createElement("div"); tip.id = "vol-an-tip"; tip.className = "vol-an-tip";
      tip.setAttribute("role", "tooltip"); tip.style.display = "none"; box.appendChild(tip);
    }
    var hit = volAnHit(p);
    if (!hit) { tip.style.display = "none"; return; }
    tip.innerHTML = volAnTipHTML(hit.r); tip.style.borderColor = volAnColor(hit.r.an);
    tip.style.display = "block"; tip.style.visibility = "hidden";
    var bw = box.clientWidth, bh = box.clientHeight, tw = tip.offsetWidth, th = tip.offsetHeight;
    var left = hit.x + 16;
    if (left + tw > bw - 8) left = hit.x - tw - 16;
    left = Math.max(8, Math.min(left, bw - tw - 8));
    var top = hit.y - th - 15;
    if (top < 8) top = hit.y + 19;
    top = Math.max(8, Math.min(top, bh - th - 8));
    tip.style.left = Math.round(left) + "px"; tip.style.top = Math.round(top) + "px"; tip.style.visibility = "visible";
  }
  function volEnsure() {
    if (!IW.chart) return;
    if (VOL.chart !== IW.chart) {
      VOL.chart = IW.chart; VOL.series = null; VOL.hover = null;
      IW.chart.subscribeCrosshairMove(function (p) { VOL.hover = (p && p.time != null && p.point) ? p.time : null; volAnHover(p); volLegend(); });
      if (IW.chart.subscribeDblClick) IW.chart.subscribeDblClick(volDbl);
    }
    if (VOL.on && !VOL.series) {
      var opt = { priceScaleId: "volume", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false };
      VOL.plain = !IW.chart.addCustomSeries;
      VOL.series = VOL.plain ? IW.chart.addHistogramSeries(opt) : IW.chart.addCustomSeries(new VolPane(), opt);
      IW.chart.priceScale("volume").applyOptions({ scaleMargins: { top: VOL_TOP, bottom: 0 }, visible: false });
    } else if (!VOL.on && VOL.series) {
      try { IW.chart.removeSeries(VOL.series); } catch (e) {}
      VOL.series = null; VOL.rows = []; VOL.idx = {};
    }
    if (!VOL.on) volLegend();
  }
  function volRebuild(c) {
    var o = (c && c.ohlc) || [], rows = [], idx = {};
    for (var i = 0; i < o.length; i++) { var r = volRow(o[i], VOL.srv, VOL.base, VOL.tape); rows.push(r); idx[r.time] = r; }
    volMarkAnomalies(rows, 0);
    VOL.rows = rows; VOL.idx = idx;
    VOL.anVisible = rows.some(function (r) { return !!r.an; });
    try { VOL.series.setData(rows); } catch (e) { console.error("[vol]", e); }
    volAnimate();
  }
  /* Нові дані сервера: розклад історії + оцінка живої свічки. */
  function volApply(c) {
    volEnsure();
    if (!VOL.on || !VOL.series || !c) return;
    var key = c.symbol + "|" + c.tf;
    if (VOL.key !== key) { VOL.key = key; VOL.srv = {}; VOL.base = null; VOL.sub = 0; VOL.sel = null; if (VOL.tape && VOL.tape.key !== volTapeKey(c)) VOL.tape = null; }
    var vs = c.vsplit;
    VOL.wait = !!c.vol_wait; VOL.fill = !!(vs && vs.fill);
    if (vs) {
      var m = {}, a = vs.m || [];
      for (var i = 0; i < a.length; i++) m[a[i][0]] = [Number(a[i][1]) || 0, Number(a[i][2]) || 0];
      VOL.srv = m; VOL.sub = Number(vs.sub) || 0;
      var lv = vs.live, tp = VOL.tape, alive = !!tp && Date.now() - tp.last < 20000;
      /* Оцінка живої свічки береться ОДИН раз (далі її уточнюють угоди); якщо
         потоку немає — завжди найсвіжіша. Свічку, покриту потоком повністю,
         оцінка не чіпає. */
      if (lv && lv.t > 0 && !(tp && tp.bars[lv.t] && tp.bars[lv.t].full) && (!VOL.base || VOL.base.t !== lv.t || !alive))
        VOL.base = { t: lv.t, b: Number(lv.b) || 0, s: Number(lv.s) || 0, asof: Number(lv.asof) || 0 };
    }
    volRebuild(c); volLegend();
  }
  function volTapeKey(c) { return "publicTrade." + String((c && c.symbol) || "").toUpperCase() + "|" + volTfSec(c && c.tf); }
  function onLiveTrades(tkey, data, t0) {
    var c = IW.coin;
    if (!VOL.on || !c || !data || tkey !== "publicTrade." + String(c.symbol || "").toUpperCase()) return;
    var bs = volTfSec(c.tf), id = volTapeKey(c);
    if (!VOL.tape || VOL.tape.key !== id || VOL.tape.t0 !== t0) {
      var old = (VOL.tape && VOL.tape.key === id) ? VOL.tape : null, o = c.ohlc || [];
      VOL.tape = volTapeCarry(old, volTapeNew(id, t0), o.length ? o[o.length - 1].time : 0);
      if (old) { VOL.base = null; IW.sig = ""; setTimeout(function () { try { loadCoin(false, true); } catch (e) {} }, 1500); }
    }
    for (var i = 0; i < data.length; i++) volTapeAdd(VOL.tape, data[i].T, data[i].S, data[i].v, bs);
    VOL.tape.last = Date.now();
    volLive();
  }
  /* Жива свічка: не частіше ніж раз на 400 мс (угод — сотні на секунду). */
  function volLive() {
    if (!VOL.on || !VOL.series || VOL.tm) return;
    VOL.tm = setTimeout(function () { VOL.tm = null; try { volLiveNow(); } catch (e) { console.error("[vol]", e); } }, 400);
  }
  function volLiveNow() {
    var c = IW.coin, o = c && c.ohlc;
    if (!VOL.on || !VOL.series || !o || !o.length || VOL.key !== c.symbol + "|" + c.tf) return;
    var last = o[o.length - 1], n = VOL.rows.length;
    /* нова свічка → попередня дістає остаточний розклад (повна перебудова, 400 рядків) */
    if (!n || VOL.rows[n - 1].time !== last.time) { volRebuild(c); volLegend(); return; }
    var r = volRow(last, VOL.srv, VOL.base, VOL.tape);
    VOL.rows[n - 1] = r; VOL.idx[r.time] = r;
    r.an = volAnomalyAt(VOL.rows, n - 1);
    try { VOL.series.update(r); } catch (e) { volRebuild(c); }
    if (VOL.hover == null || VOL.hover === r.time || VOL.sel != null) volLegend();
  }
  var VOL_SRC = {
    tape: ["⚡ угоди", "Точно: кожна угода Bybit за цю свічку, сторона — хто бив по ринку (тейкер)."],
    mix: ["⚡≈ угоди + оцінка", "Свічку відкрито до завантаження сторінки: початок — оцінка за підсвічками, далі — справжні угоди Bybit."],
    part: ["⚡ частково", "Лише угоди, що пройшли після відкриття сторінки (оцінка сервера ще не надійшла)."],
    sub: ["≈ підсвічки", "Оцінка за підсвічками Bybit всередині свічки: зелена підсвічка → покупці, червона → продавці."]
  };
  function volLegendHTML(r, sub, note) {
    var h = '<span class="vl-k">Обсяг</span>';
    if (!r || !(r.value > 0)) return note ? h + ' <span class="vl-n">' + esc(note) + "</span>" : h + ' <span class="vl-n">—</span>';
    h += ' <b class="vl-v">' + esc(fmtVol(r.value, 3) || "0") + "</b>";
    if (r.buy != null) {
      var bp = Math.round(r.buy * 100), bv = r.value * r.buy, sv = r.value - bv, d = bv - sv, src = VOL_SRC[r.src] || ["", ""];
      var lbl = r.src === "sub" && sub ? src[0] + " " + sub + " хв" : src[0];
      h += ' <span class="vl-up' + (bp >= 50 ? " vl-win" : "") + '">▲ покупці ' + bp + "% · " + esc(fmtVol(bv, 3) || "0") + "</span>" +
        ' <span class="vl-dn' + (bp < 50 ? " vl-win" : "") + '">▼ продавці ' + (100 - bp) + "% · " + esc(fmtVol(sv, 3) || "0") + "</span>" +
        ' <span class="' + (d >= 0 ? "vl-up" : "vl-dn") + '">Δ ' + (d >= 0 ? "+" : "−") + esc(fmtVol(Math.abs(d), 3) || "0") + "</span>" +
        (lbl ? ' <span class="vl-s" title="' + esc(src[1]) + '">' + esc(lbl) + "</span>" : "");
    }
    if (VOL.anom && r.an) {
      var t = volAnText(r.an);
      h += ' <span class="vl-an" style="color:' + volAnColor(r.an) + ";border-color:" + volAnColor(r.an) + '" title="' + esc(t[1]) + '">' + esc(t[0]) + "</span>";
    }
    return h + (note ? ' <span class="vl-n">' + esc(note) + "</span>" : "");
  }
  /* 📊 Дані обсягу — у НИЖНЬОМУ РЯДКУ графіка ЗАМІСТЬ легенди шарів (вимога
     16.11); індикатор вимкнено → легенда шарів повертається на місце.
     Показуємо свічку під курсором, інакше останню. */
  function volLegend() {
    var on = !!(VOL.on && VOL.series && IW.chart);
    var r = VOL.hover != null ? VOL.idx[VOL.hover] : null;
    if (!r && VOL.hover == null) r = VOL.rows.length ? VOL.rows[VOL.rows.length - 1] : null;
    var note = VOL.wait ? "⏳ рахується розклад покупці / продавці" : (r && r.buy == null && VOL.fill) ? "⏳ історія дозаповнюється" : "";
    var rng = (on && VOL.sel != null) ? volRangeSum(VOL.rows, VOL.sel, (IW.coin || {}).ohlc) : null;
    if (on && VOL.sel != null && !rng) VOL.sel = null;      // свічка випала з графіка
    VOL.html = !on ? "" : (rng ? volRangeHTML(rng) : volLegendHTML(r, VOL.sub, note));
    VOL.rngOn = !!rng;
    legendPaint();
  }
  function legendPaint() {
    var lg = $("#iw-legend"); if (!lg) return;
    if (!lg._volX) {
      lg._volX = true;
      lg.addEventListener("click", function (e) { if (e.target.closest && e.target.closest(".vl-x")) volSetRange(null); });
      document.addEventListener("keydown", function (e) {
        if ((e.key === "Escape" || e.key === "Esc") && VOL.sel != null) volSetRange(null);
      });
    }
    var vol = !!(typeof VOL !== "undefined" && VOL && VOL.on && VOL.html);
    var html = vol ? VOL.html : (IW.layerLg || "");
    lg.classList.toggle("vol", vol);
    lg.classList.toggle("rng", vol && !!VOL.rngOn);     // Σ-режим — підсвічена панель
    if (lg._h !== html) { lg.innerHTML = html; lg._h = html; }
  }
  /* Перемикач «Обсяг» у меню індикаторів (chart-tools). */
  function volToggle(on, anom) {
    var was = VOL.on; VOL.on = !!on;
    if (anom != null && VOL.anom !== !!anom) {
      VOL.anom = !!anom;
      if (VOL.series && IW.coin) volRebuild(IW.coin);
    }
    if (!VOL.on) { VOL.tape = null; VOL.base = null; }   // потік угод зупиняється → у стрічці була б діра
    volEnsure();
    if (IW.coin) { liveSet(IW.coin); if (VOL.on) volApply(IW.coin); }
    if (VOL.on && !was && curSym()) { IW.sig = ""; try { loadCoin(false, true); } catch (e) {} }
  }

  function drawCoin(c, fit) {
    if (!ensureChart()) { setNote("графічна бібліотека не завантажилась"); return; }
    var o = c.ohlc || [], key = c.symbol + "|" + c.tf;
    /* Точність ціни застосовуємо лише за наявності даних: інакше precisionOf(0)
       давав би 8 знаків, а після перестворення графіка (зміна теми) кеш IW.prec
       мусить бути скинутий — тоді формат перевстановлюється гарантовано. */
    if (o.length) {
      var prec = precisionOf(o[o.length - 1].close);
      if (IW.prec !== prec) { IW.series.applyOptions({ priceFormat: { type: "price", precision: prec, minMove: Math.pow(10, -prec) } }); IW.prec = prec; }
    }
    var ttl = String(c.symbol || "");
    if (IW.seriesTitle !== ttl) { IW.series.applyOptions({ title: ttl }); IW.seriesTitle = ttl; }
    var j = -1, lv0 = liveMerge(c);
    if (!fit && IW.chartKey === key && IW.lastT != null && o.length && !lv0.ahead) { for (var q = o.length - 1; q >= Math.max(0, o.length - 4); q--) { if (o[q].time === IW.lastT) { j = q; break; } } }
    try { if (j >= 0) { for (var u = j; u < o.length; u++) IW.series.update(o[u]); } else IW.series.setData(o); } catch (e) { IW.series.setData(o); }
    if (lv0.ahead) { try { IW.series.update(lv0.bar); } catch (e) {} }
    IW.lastT = o.length ? o[o.length - 1].time : null;
    liveSet(c); volApply(c);
    var st = structOf(c); IW.series.setMarkers(st.marks);
    var lv = [];
    if (c.poc && c.poc.poc) lv.push({ price: Number(c.poc.poc), title: "POC", color: "#f5f5f5" });
    if (c.shl && IW.show_shl) ["high", "low"].forEach(function (k) { var v = c.shl[k]; if (v && v.price > 0) lv.push({ price: Number(v.price), title: (v.label || "") + " " + (c.shl.tf || ""), color: k === "high" ? "#f87171" : "#4ade80" }); });
    IW.prim.setHl(st.hl); IW.prim.set(boxesOf(c), st.lines, sbrOf(c).concat(anomOf(c)), lv); setPoc(c.poc); setShl(c.shl);
    if (fit || IW.chartKey !== key) { setAutoScale(true); IW.chart.timeScale().fitContent(); IW.chartKey = key; }
    /* Єдиний вигляд завантаження на графіку: будь-який текстовий варіант
       (зокрема reason із бекенду на кшталт «завантаження барів 1H»)
       замінюється на лого-лоадер. */
    if (o.length) setNote("");
    else {
      var rsn = String(c.reason || "");
      if (rsn) setLoading();
      else setNote("немає даних");
    }
    setWait(c);
    
    if (window.VSV_PRO) { try { window.VSV_PRO.onData(IW, c); } catch (err) { console.error("[VSV_PRO]", err); } }
  }

  var coinBusy = false, coinSeq = 0, coinPromise = null;
  function loadCoin(force, keepBars) {
    if (!pOk("sec_terminal")) return Promise.resolve(null);
    var sym = curSym();
    if (!sym) { setNote("оберіть монету зі списку"); renderHead(); return Promise.resolve(null); }
    if (coinBusy && !force) return coinPromise || Promise.resolve(null);
    coinBusy = true;
    var req = ++coinSeq;
    if (force) {
      setLoading();
      if (IW.series) { IW.series.setData([]); IW.series.setMarkers([]); IW.prim.setHl([]); IW.prim.set([], [], []); setPoc(null); setShl(null); IW.lastT = null; }
      sigCard(null); if (LQ && LQ.on && LQ.sym !== sym) liqOff();
      IW.liveBar = null;
    }
    var want = iwKey();
    var q = "/api/info/coin?symbol=" + encodeURIComponent(sym) + "&tf=" + IW.tf + "&ob_tf=" + IW.ob_tf + "&vob_tf=" + IW.vob_tf + "&poc_days=" + IW.poc_days + "&poc_tf=" + IW.poc_tf + "&hl_tf=" + IW.hl_tf + "&sbr_tf=" + IW.sbr_tf + "&shl_tf=" + IW.shl_tf + (IW.show_anom ? "&anom=1" : "") + (VOL.on ? "&vol=1" : "") + (IW.sig && !force ? "&sig=" + encodeURIComponent(IW.sig) : "");
    var request = api(q).then(function (c) {
      if (req !== coinSeq || want !== iwKey() || c.same) return c;
      IW.coin = c; IW.sig = c.sig || "";
      renderHead(); renderQuotes(); drawCoin(c, force);
      if (!c.ok || c.ob_wait || c.vob_wait || c.hl_wait || c.sbr_wait || c.vol_wait || (c.vsplit && c.vsplit.fill)) setTimeout(function () { if (req === coinSeq && want === iwKey()) loadCoin(false, true); }, 4000);
      return c;
    }).catch(function (e) {
      if (req === coinSeq) setNote("не вдалося завантажити графік");
      if (e && (e.status === 401 || e.status === 403)) throw e;
      return null;
    });
    coinPromise = request.then(function (result) {
      if (req === coinSeq) { coinBusy = false; coinPromise = null; }
      return result;
    }, function (e) {
      if (req === coinSeq) { coinBusy = false; coinPromise = null; }
      throw e;
    });
    return coinPromise;
  }

  document.addEventListener("click", function (e) {
    if (!e.target.closest || !e.target.closest("#theme-btn") || !IW.chart) return;
    setTimeout(function () {
      try { IW.chart.remove(); } catch (x) {}
      IW.chart = null; IW.series = null; IW.prim = null; IW.pocLine = null; IW.chartKey = ""; IW.prec = null;
      if (IW.coin) drawCoin(IW.coin, true);
    }, 30);
  });

  const TCAL_MON = ['січень', 'лютий', 'березень', 'квітень', 'травень', 'червень', 'липень', 'серпень', 'вересень', 'жовтень', 'листопад', 'грудень'];
  const TCal = { el: null, inp: null, y: 0, m: 0, days: {}, onPick: null, max: '', usd: true };
  function tcalYmd(d) { const p = n => String(n).padStart(2, '0'); return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`; }
  function tcalBucket(items) {
    const out = {};
    (items || []).forEach(it => {
      const t = Number(it && it[0]); if (!(t > 0)) return;
      const k = tcalYmd(new Date(t * 1000)), o = out[k] || (out[k] = { n: 0, w: 0, l: 0, usd: 0, pct: 0 });
      const u = Number(it[1]) || 0, pc = Number(it[2]) || 0;
      o.n += 1; o.usd += u; o.pct += pc; if ((u || pc) > 0) o.w += 1; else if ((u || pc) < 0) o.l += 1;
    });
    return out;
  }
  function tcalFmt(ymd) { const [y, m, d] = String(ymd).split('-'); return `${d}.${m}.${y}`; }
  function tcalClose() {
    if (TCal.el) TCal.el.remove();
    TCal.el = null; TCal.inp = null;
    document.removeEventListener('mousedown', tcalOutside, true); document.removeEventListener('keydown', tcalKey, true);
  }
  function tcalOutside(e) { if (TCal.el && !TCal.el.contains(e.target) && e.target !== TCal.inp) tcalClose(); }
  function tcalKey(e) { if (e.key === 'Escape') tcalClose(); }
  function tcalOpen(inp, days, onPick, usd) {
    if (TCal.el && TCal.inp === inp) { tcalClose(); return; }
    tcalClose();
    const cur = inp.dataset.ymd || tcalYmd(new Date()), [y, m] = cur.split('-').map(Number);
    Object.assign(TCal, { inp, days: days || {}, onPick, y, m: m - 1, max: tcalYmd(new Date()), usd: usd !== false });
    const el = document.createElement('div'); el.className = 'tcal'; document.body.appendChild(el); TCal.el = el;
    tcalRender();
    const r = inp.getBoundingClientRect(), w = el.offsetWidth, h = el.offsetHeight;
    let left = r.left + window.scrollX, top = r.bottom + window.scrollY + 4;
    if (left + w > window.scrollX + document.documentElement.clientWidth - 8) left = window.scrollX + document.documentElement.clientWidth - w - 8;
    if (r.bottom + h + 8 > window.innerHeight && r.top > h) top = r.top + window.scrollY - h - 4;
    el.style.left = Math.max(8, left) + 'px'; el.style.top = top + 'px';
    document.addEventListener('mousedown', tcalOutside, true); document.addEventListener('keydown', tcalKey, true);
  }
  function tcalNav(step) {
    let m = TCal.m + step, y = TCal.y;
    if (m < 0) { m = 11; y -= 1; } else if (m > 11) { m = 0; y += 1; }
    TCal.m = m; TCal.y = y; tcalRender();
  }
  function tcalPick(ymd) {
    if (!ymd || ymd > TCal.max) return;
    const inp = TCal.inp, fn = TCal.onPick;
    if (inp) { inp.dataset.ymd = ymd; inp.value = tcalFmt(ymd); }
    tcalClose(); if (fn) fn(ymd);
  }
  function tcalRender() {
    const el = TCal.el; if (!el) return;
    const y = TCal.y, m = TCal.m, first = new Date(y, m, 1), shift = (first.getDay() + 6) % 7;
    const start = new Date(y, m, 1 - shift), sel = TCal.inp ? TCal.inp.dataset.ymd : '', today = tcalYmd(new Date());
    let mn = 0, musd = 0, cells = '';
    for (let i = 0; i < 42; i++) {
      const d = new Date(start.getFullYear(), start.getMonth(), start.getDate() + i), k = tcalYmd(d), info = TCal.days[k];
      const out = d.getMonth() !== m, fut = k > TCal.max;
      if (info && !out) { mn += info.n; musd += info.usd; }
      const v = info ? (info.usd || info.pct) : 0;
      let cls = 'tcal-d' + (out ? ' out' : '') + (fut ? ' fut' : '') + (k === today ? ' today' : '') + (k === sel ? ' sel' : '');
      let tip = tcalFmt(k), badge = '';
      if (info) {
        cls += v > 0 ? ' up' : (v < 0 ? ' dn' : ' eq');
        const ap = info.pct / info.n;
        tip += ` · угод: ${info.n} · ${info.w} W / ${info.l} L` + (TCal.usd ? ` · ${info.usd >= 0 ? '+' : '−'}$${Math.abs(info.usd).toFixed(2)}` : '') + ` · сер. рух ${ap >= 0 ? '+' : ''}${ap.toFixed(2)}%`;
        badge = `<i class="tcal-n">${info.n}</i>`;
      } else if (!fut) tip += ' · угод не було';
      cells += `<button type="button" class="${cls}" title="${tip}"` + (fut ? ' disabled' : ` onclick="tcalPick('${k}')"`) + `><span>${d.getDate()}</span>${badge}</button>`;
    }
    const sum = mn ? `угод за місяць: <b>${mn}</b>` + (TCal.usd ? ` · <b class="${musd >= 0 ? 'up' : 'dn'}">${musd >= 0 ? '+' : '−'}$${Math.abs(musd).toFixed(2)}</b>` : '') : 'угод цього місяця не було';
    const nextOk = tcalYmd(new Date(y, m + 1, 1)) <= TCal.max;
    el.innerHTML =
      `<div class="tcal-h"><button type="button" onclick="tcalNav(-1)" title="Попередній місяць">‹</button>` +
      `<b>${TCAL_MON[m]} ${y}</b><button type="button" onclick="tcalNav(1)" title="Наступний місяць"${nextOk ? '' : ' disabled'}>›</button></div>` +
      `<div class="tcal-w">${['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Нд'].map(x => `<span>${x}</span>`).join('')}</div>` +
      `<div class="tcal-g">${cells}</div><div class="tcal-s">${sum}</div>` +
      `<div class="tcal-f"><span class="tcal-lg"><i class="up"></i>плюс <i class="dn"></i>мінус · N угод</span><button type="button" onclick="tcalPick('${today}')">Сьогодні</button></div>`;
  }
  window.tcalPick = tcalPick; window.tcalNav = tcalNav;

  var PP = { period: { mode: "all" }, last: null, days: {}, daysAt: 0, shown: false, busy: false };
  var PP_KEY = "vsv_pp_period", PP_AMT = "vsv_pp_amt", PP_CL = "vsv_pp_closed_open";
  try { PP.period = JSON.parse(localStorage.getItem(PP_KEY) || "") || PP.period; } catch (e) {}
  try { PP.shown = localStorage.getItem(PP_AMT) === "1"; } catch (e) {}
  var PP_MON = 86400;
  function ppDayStart(ymd) { var a = String(ymd || "").split("-").map(Number); return a[0] ? new Date(a[0], a[1] - 1, a[2]).getTime() / 1000 : null; }
  function ppNextDay(ts) { return ppDayStart(tcalYmd(new Date((ts + PP_MON + 3600) * 1000))); }
  function ppRange(p) {
    var now = new Date(), t0 = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
    switch (p.mode) {
      case "today": return { from: t0, to: t0 + PP_MON, label: "сьогодні" };
      case "7d": return { from: t0 - 6 * PP_MON, to: t0 + PP_MON, label: "за 7 днів" };
      case "30d": return { from: t0 - 29 * PP_MON, to: t0 + PP_MON, label: "за 30 днів" };
      case "day": { var f = ppDayStart(p.day); return f == null ? null : { from: f, to: ppNextDay(f), label: "за " + tcalFmt(p.day) }; }
      case "range": {
        var a = ppDayStart(p.from), b = ppDayStart(p.to); if (a == null || b == null) return null;
        var lo = Math.min(a, b), hi = Math.max(a, b), x = a <= b ? p.from : p.to, y = a <= b ? p.to : p.from;
        return { from: lo, to: ppNextDay(hi), label: x === y ? "за " + tcalFmt(x) : tcalFmt(x) + " — " + tcalFmt(y) };
      }
      default: return { from: null, to: null, label: "за весь час" };
    }
  }
  function ppAmt(h) { return PP.shown ? h : '<span class="pp-mask">••••</span>'; }
  function ppUsd(v) { v = Number(v || 0); return ppAmt('<span class="pp-cur">' + (v >= 0 ? "+" : "−") + "$</span>" + Math.abs(v).toFixed(2)); }
  function ppMoney(v) { return ppAmt("$" + Number(v || 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })); }
  function ppCls(v) { v = Number(v || 0); return v > 0 ? "pp-up" : (v < 0 ? "pp-dn" : ""); }
  function ppPct(v, dp) { v = Number(v || 0); return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(dp == null ? 2 : dp) + '<span class="pp-pc">%</span>'; }

  function ppSync() {
    var P = PP.period, today = tcalYmd(new Date());
    document.querySelectorAll("#pp-period button[data-p]").forEach(function (b) { b.classList.toggle("on", b.getAttribute("data-p") === P.mode); });
    $("#pp-day-wrap").style.display = P.mode === "day" ? "inline-flex" : "none";
    $("#pp-range-wrap").style.display = P.mode === "range" ? "inline-flex" : "none";
    [["#pp-day", P.day], ["#pp-from", P.from], ["#pp-to", P.to]].forEach(function (x) { var el = $(x[0]); if (el) { el.dataset.ymd = x[1] || today; el.value = tcalFmt(x[1] || today); } });
    var eb = $("#pp-amt"); if (eb) eb.innerHTML = '<svg><use href="#i-eye"/></svg>' + (PP.shown ? "Сховати суми" : "Показати суми");
  }
  function ppSave() { try { localStorage.setItem(PP_KEY, JSON.stringify(PP.period)); } catch (e) {} }

  function ppCard(l, v, sub, vc, tip) { return '<div class="pp-kpi" title="' + esc(tip || "") + '"><div class="pp-kpi-l">' + l + '</div><div class="pp-kpi-v ' + (vc || "") + '">' + v + "</div>" + (sub ? '<div class="pp-kpi-s">' + sub + "</div>" : "") + "</div>"; }
  function ppNum(v) { var n = Number(v || 0); return n % 1 ? n.toFixed(2) : String(n); }
  function ppInlineHTML(a) {
    a = a || {};
    var usd = function (v) { return '<b class="ppi-usd">$' + ppNum(v) + "</b>"; }, pct = function (v) { return '<b class="ppi-pct">' + ppNum(v) + "%</b>"; };
    var deal = a.mode === "pct" && a.margin_pct != null ? pct(a.margin_pct) + ' <span class="ppi-l" title="Вільна маржа (available): гаманець − маржа відкритих позицій + нереалізований збиток (нереалізований прибуток не враховується) — як на біржі">від вільних</span> ' + usd(a.available != null ? a.available : a.balance) + ' <span class="ppi-l">=</span> ' + usd(a.margin_usd) : usd(a.margin_usd);
    return '<span class="ppi-l">депозит</span> ' + usd(a.deposit) + ' <span class="ppi-sep">·</span> <span class="ppi-l">угода</span> ' + deal + ' <span class="ppi-l">×</span> <b class="ppi-lev">' + (a.leverage || 0) + "x</b>" + ' <span class="ppi-sep">·</span> <span class="ppi-l">комісія</span> ' + pct(a.fee_pct);
  }
  function ppRenderKpis(d, label) {
    var st = d.stats || {}, pa = d.account || {}, host = $("#pp-kpis");
    var since = st.pnl_since_from_usd != null ? st.pnl_since_from_usd : st.total_pnl_usd, base = (pa.closed_balance != null ? pa.closed_balance : (pa.balance || 0)) - (since || 0);
    var dp = base > 0 ? (st.total_pnl_usd || 0) / base * 100 : null;
    var bal = ppCard("Тестовий баланс", ppMoney(pa.equity), dp == null ? "депозит " + ppMoney(pa.deposit) : '<span class="' + ppCls(dp) + '">' + ppPct(dp) + "</span> депозиту", "", "");
    var n = st.total_closed || 0;
    if (!n) { host.innerHTML = bal + '<div class="pp-empty">Закритих тестових угод ' + label + " немає</div>"; return; }
    var wr = st.win_rate || 0, pf = st.profit_factor;
    host.innerHTML = ppCard("Чистий P&amp;L", ppUsd(st.total_pnl_usd), '<span class="pp-up">' + ppUsd(st.gross_profit_usd) + '</span> · <span class="pp-dn">' + ppUsd(st.gross_loss_usd) + "</span>", PP.shown ? ppCls(st.total_pnl_usd) : "", "") + bal +
      ppCard("Відсоток успішних", wr + "%", n + ' угод · <span class="pp-up">' + (st.wins || 0) + ' W</span> · <span class="pp-dn">' + (st.losses || 0) + " L</span>" + '<div class="pp-bar"><i style="width:' + Math.max(0, Math.min(100, wr)) + '%"></i></div>', wr >= 50 ? "pp-up" : "pp-dn", "") +
      ppCard("Profit factor", pf == null ? (st.gross_profit_usd > 0 ? "∞" : "—") : Number(pf).toFixed(2), 'сер. <span class="pp-up">' + ppUsd(st.avg_win_usd) + '</span> / <span class="pp-dn">' + ppUsd(st.avg_loss_usd) + "</span>", pf == null ? (st.gross_profit_usd > 0 ? "pp-up" : "") : (pf >= 1 ? "pp-up" : "pp-dn"), "") +
      ppCard("Найкраща / найгірша", '<span class="' + (PP.shown ? ppCls(st.best_usd) : "") + '">' + ppUsd(st.best_usd) + "</span>", '<span class="' + ppCls(st.worst_usd) + '">' + ppUsd(st.worst_usd) + "</span> · сер. рух " + ppPct(st.avg_pnl_pct), "", "") +
      ppCard("Комісії (оцінка)", ppAmt("$" + Number(st.total_fees_usd || 0).toFixed(2)), "вхід + вихід", "", "");
    var es = st.entry_score_breakdown, eh = $("#pp-es"), parts = [], NM = { good: "✅ сильний", marginal: "🟡 середній", poor: "⚠️ слабкий" };
    ["good", "marginal", "poor"].forEach(function (k) {
      var b = es && es[k]; if (!b || !b.count) return;
      parts.push("<span>" + NM[k] + ": " + b.count + ' (<b class="' + (b.win_rate >= 50 ? "pp-up" : "pp-dn") + '">' + b.win_rate + '% у плюсі</b>, <b class="' + ppCls(b.avg_pnl_pct) + '">сер. ' + ppPct(b.avg_pnl_pct) + "</b>)</span>");
    });
    if (eh) { eh.innerHTML = parts.length ? "🧪 Оцінка входу: " + parts.join("") : ""; eh.style.display = parts.length ? "" : "none"; }
  }
  var PP_ALERT = { ok: ["✓ OK", "ok"], against: ["⚠ ПРОТИ", "against"] };
  function ppTimer(t) { var s = Math.max(0, Math.floor(Date.now() / 1000 - t)), h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); var p = function (n) { return String(n).padStart(2, "0"); }; return (h >= 24 ? Math.floor(h / 24) + "д " + p(h % 24) : p(h)) + ":" + p(m) + ":" + p(s % 60); }
  function exhColor(x) { x = Math.max(0, Math.min(100, Number(x) || 0)); return x >= 100 ? "#f472b6" : (x >= 75 ? "#fb923c" : (x >= 50 ? "#facc15" : "#60a5fa")); }
  // 🎯 «Ціль» (дзеркало бота springFireCellHTML, 11.11): знаковий прогрес угоди до
  // цілі Пружини — смужка 0..100%, мінус = угода відхиляється від цілі.
  function ppExh(p) {
    var e = p.exh, f = p.spring_fire, fd = f && f.fired;
    if (!e) return '<span class="muted">—</span>';
    var prog = (e.prog != null && isFinite(Number(e.prog))) ? Number(e.prog) : (Number(e.pct) || 0);
    var pct = Math.max(0, Math.min(100, prog)), col = prog < 0 ? "#f87171" : exhColor(pct);
    var tip = "🎯 Ціль (measured move) " + fmtPrice(e.target) + ": " + (prog >= 0 ? "пройдено +" : "просідання −") + Math.abs(Math.round(prog)) + "% шляху від входу";
    var lab = '<b style="color:' + col + '">' + (prog >= 0 ? "+" : "−") + Math.abs(Math.round(prog)) + "%</b>" + (e.climax ? '<b style="color:#f472b6">💥</b>' : "") + (fd && fd.side !== p.side ? '<b class="pp-blink" style="color:#f87171">⚠</b>' : "");
    return '<span class="pp-exh' + (prog < 0 ? " neg" : "") + '" title="' + esc(tip) + '"><i><b style="width:' + pct + "%;background-size:" + (pct > 0 ? 10000 / pct : 100) + '% 100%"></b></i>' + lab + (p.exh_trail && p.exh_trail.stop_px != null ? ' <b class="pp-exh-st" title="🎯 Climax-трейл: стоп підтягнуто за кульмінацією">🎯 ' + fmtPrice(p.exh_trail.stop_px).replace("$", "") + "</b>" : "") + "</span>";
  }
  function ppLvl(v, done) { return !v ? '<span class="muted">—</span>' : fmtPrice(v).replace("$", "") + (done ? ' <span class="pp-up">✓</span>' : ""); }
  function ppPnlCols(pc, usd) { return '<span class="pp-pnl"><span class="pp-pnl-p">' + ppPct(pc) + "</span>" + (usd != null ? '<span class="pp-pnl-s">·</span><span class="pp-pnl-u"><b class="pp-usd ' + (Number(usd) >= 0 ? "up" : "dn") + '">' + ppUsd(usd) + "</b></span>" : "") + "</span>"; }
  // 15.11 — поля paper-таблиці ЯК У БОТІ: PnL у два рядки (рух % · нетто $ ·
  // ROI ×плече), OB 4H/1H у полі «Бік», 🚨 завжди з реальним станом SBR.
  function ppPnlCell(p) {
    var pc = Number(p.pnl_pct || 0), n = p.pnl_net, z = p.paper_size || {};
    if (!n) return '<td class="r ' + ppCls(pc) + '">' + ppPct(pc) + "</td>";
    var roi = n.roi_pct != null ? "ROI " + (n.roi_pct >= 0 ? "+" : "−") + Math.abs(Number(n.roi_pct)).toFixed(1) + "%" : "";
    var tip = "Рух ціни " + (pc >= 0 ? "+" : "−") + Math.abs(pc).toFixed(2) + "% · маржа $" + Number(z.margin || 0).toFixed(2) + " × " + (z.lev || "?") + "x · комісії −$" + Number(n.fees_usd || 0).toFixed(2) + " (taker: вхід + вихід)";
    return '<td class="r pp-pnl2 ' + ppCls(n.net_usd) + '" title="' + esc(tip) + '"><b>' + ppPct(pc) + "</b><br>" +
      '<b class="pp-usd ' + (Number(n.net_usd) >= 0 ? "up" : "dn") + '">' + ppUsd(n.net_usd) + "</b>" +
      (roi ? ' <span class="pp-roi">' + roi + (z.lev ? " ×" + z.lev : "") + "</span>" : "") + "</td>";
  }
  function ppObMark(p, tf) {
    var o = p["ob" + tf], L = tf.toUpperCase();
    if (!o) return "";
    if (!o.bias) return '<span class="pp-ob" style="color:#6b7280;border-color:#4b5563" title="OB ' + L + ': активного order block немає">' + L + " —</span>";
    var bull = o.bias === "BULLISH", col = bull ? "#22c55e" : "#ef4444", agree = (bull && p.side === "LONG") || (!bull && p.side === "SHORT");
    var tip = "Останній OB " + L + ": " + (bull ? "BULLISH (LONG)" : "BEARISH (SHORT)") + (o.tag ? " · " + o.tag : "") + (o.bar_time ? " · свічка " + locDM(o.bar_time) + " " + locHM(o.bar_time) : "") + (agree ? " — у бік позиції" : " — проти позиції") + (o.restored ? " · ↻ з минулого запуску" : "");
    return '<span class="pp-ob" style="color:' + col + ";border-color:" + (agree ? col : "#fbbf24") + (o.restored ? ";opacity:.5" : "") + '" title="' + esc(tip) + '">' + L + (bull ? "▲" : "▼") + "</span>";
  }
  function ppAlertCell(p) {
    var t = p.sbr_trail;
    if (t && t.armed_px) {
      var armed = Number(t.armed_px), cur = Number(p.current_price) || armed, stop = Number(t.stop_px) || armed;
      var dist = armed > 0 ? (cur - armed) / armed * 100 * (p.side === "SHORT" ? -1 : 1) : 0;
      return '<td class="nw"><span class="pp-al pre" title="🔔⏳ Зустрічний сигнал SBR — позицію ведемо трейлінг-стопом · стоп ' + esc(fmtPrice(stop)) + '">🔔⏳ ' + esc(t.tf || "") + " 🛡" + fmtPrice(stop).replace("$", "") + ' <b class="' + ppCls(dist) + '">' + (dist >= 0 ? "+" : "−") + Math.abs(dist).toFixed(2) + "%</b></span></td>";
    }
    var m = p.mmd, a = m && m.alert;
    if (!a) return '<td class="nw"><span class="muted" title="SBR по інструменту ще не розраховано">—</span></td>';
    var al = PP_ALERT[a.level];
    if (al) return '<td class="nw"><span class="pp-al ' + al[1] + '" title="' + esc(a.text || "") + '">' + al[0] + (a.level === "against" && a.tf ? " " + esc(a.tf) : "") + "</span></td>";
    if (a.level === "pre" && a.side) {
      return '<td class="nw"><span class="pp-al pp-al-now ' + (a.side === "LONG" ? "pp-long" : "pp-short") + '" title="' + esc(a.text || "") + '">🔔 ' + (a.side === "LONG" ? "▲ LONG" : "▼ SHORT") + (a.tf ? " " + esc(a.tf) : "") + "</span></td>";
    }
    return '<td class="nw"><span class="pp-al pp-al-now muted" title="' + esc(a.text || "активного сигналу SBR немає") + '">🔔 без сигналу</span></td>';
  }
  function ppSlCell(p) {
    var v = p.manual_sl; if (!v) return '<td class="r"><span class="muted">—</span></td>';
    var tag = p.sl_breakeven ? ' <span class="pp-sltf be" title="Стоп перенесено в беззбиток (BE)">⚖️БЗ</span>'
      : (p.manual_sl_src === "auto" && p.manual_sl_tf ? ' <span class="pp-sltf" title="Стоп за структурою ' + esc(p.manual_sl_tf) + ' (виставив бот)">' + esc(p.manual_sl_tf) + "</span>" : "");
    return '<td class="r">' + fmtPrice(v).replace("$", "") + tag + "</td>";
  }
  // ➕ Усереднення угоди за ⚠ сигналами аномалій (24.11) — дзеркало cabAvgMarkHTML бота.
  function ppAvgMark(p) {
    var n = Number((p && p.cab_avg_n) || 0);
    if (!n) return "";
    return '<span class="pp-avg" title="➕ Усереднення (DCA): ' + n + ' докупівель з кращої ціни за сетапом ⚠ · вихід закриває всю позицію">➕' + n + "</span>";
  }
  function ppPeak(c) {
    var pk = c.peak_pnl_pct != null ? Number(c.peak_pnl_pct) : null;
    if (pk == null || !isFinite(pk)) return '<td class="r"><span class="muted">—</span></td>';
    var fin = c.pnl_pct != null ? Number(c.pnl_pct) : null;
    if (fin != null && fin > pk) pk = fin;
    var tip = "MFE — максимальний сприятливий рух від входу" + (fin != null ? " · віддано від піку " + Math.max(0, pk - fin).toFixed(2) + " п.п." : "");
    return '<td class="r ' + ppCls(pk) + '" title="' + esc(tip) + '">' + ppPct(pk) + "</td>";
  }
  var PP_PIN = '<span class="pp-pin" title="Ручне керування (Manual mode)">📌</span>';
  function ppRenderOpen(d) {
    var rows = d.positions || [], tb = $("#pp-open tbody"); $("#pp-open-n").textContent = rows.length;
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="11" class="muted">Відкритих угод немає</td></tr>'; return; }
    tb.innerHTML = rows.map(function (p) {
      return "<tr><td class=\"sym\">" + (p.manual_mode ? PP_PIN : "") + signalIconHtml(p.opened_by) + tvSym(p.symbol) + ppAvgMark(p) + '</td><td class="nw ' + (p.side === "LONG" ? "pp-long" : "pp-short") + '">' + ppObMark(p, "4h") + ppObMark(p, "1h") + esc(p.side) + "</td>" +
        '<td class="r">' + fmtPrice(p.entry_price) + '</td><td class="r">' + fmtPrice(p.current_price) + "</td>" +
        ppPnlCell(p) + "<td>" + ppExh(p) + "</td>" + ppAlertCell(p) +
        ppSlCell(p) + '<td class="r">' + ppLvl(p.manual_tp1, p.tp1_done) + '</td><td class="r">' + ppLvl(p.manual_tp) + "</td>" +
        '<td class="r pp-timer" data-since="' + Math.floor(p.opened_at || 0) + '">' + (p.opened_at ? ppTimer(p.opened_at) : "—") + "</td></tr>";
    }).join("");
  }
  var PP_REASON = { manual: "✋ Закрито вручну", stop_loss: "🛡 Stop Loss", take_profit: "🎯 Take Profit", manual_sl: "🛡 Manual SL", manual_tp: "🎯 Manual TP-2", pilot_target: "🎯 Автопілот: ціль досягнута", breakeven: "🟰 Беззбиток", breakeven_close: "⚖️ Беззбиток", sbr_exit: "🔔 Протилежний сигнал SBR", sbr_trail_exit: "🔔⏳ Трейл після SBR", opposite_ob_exit: "🧱 Новий протилежний OB", mm_flat_exit: "🧮 VSV LiQ ⚖ рівновага", mm_against_exit: "🧮 VSV LiQ проти", reverse_signal: "🔄 Протилежний сигнал (реверс)", mirror_close: "🪞 Дзеркало: бот закрив угоду", reverse_smc: "🔄 Розворот структури", exh_exit: "🎯 Досягнення вершини", spring_fired_exit: "💥 Пружина вистрілила", ai_exit: "🤖 Вихід ШІ", ai_pro_exit: "🧠 Проф. вихід ШІ", forecast_1h_exit: "🔮 Прогноз 1H проти", forecast_4h_exit: "🔮 Прогноз 4H проти", forecast_1h_confluence: "🔮 Прогноз 1H + розворот", decision_exit: "🧠 Decision проти", signal_exit_and: "🔗 Усі вердикти проти", htf_flip: "📡 Розворот старшого TF", time_stop: "⏱ Вихід за часом", trailing_stop: "📈 Трейлінг-стоп", external_close: "🌐 Закрито на біржі", auto_gate_wait: "⏸ Пауза Auto-gate", bos_2_partial: "✂️ Часткове BOS-2", bos_3_partial: "✂️ Часткове BOS-3", bos_4_partial: "✂️ Часткове BOS-4", fuel_flipped: "⛽ VSV розвернувся", fuel_faded: "⛽ VSV згас" };
  // 🎯 «Досягнення вершини»: частка піку угоди, взята на виході (результат ÷ пік).
  function ppPeakShare(c) {
    if (!c || c.reason !== "exh_exit") return null;
    var pk = Number(c.peak_pnl_pct), fin = Number(c.pnl_pct);
    if (!isFinite(pk) || !isFinite(fin)) return null;
    if (fin > pk) pk = fin;
    if (!(pk > 0)) return null;
    return Math.max(0, Math.min(100, Math.round(fin / pk * 100)));
  }
  function ppReason(c) {
    var t = esc(PP_REASON[c.reason] || c.reason || "—"), v = ppPeakShare(c);
    if (v == null) return t;
    return t + ' <b class="pp-share ' + (v >= 80 ? "hi" : v >= 50 ? "mid" : "lo") + '" title="Capture: зафіксовано ' + v + '% від MFE угоди">' + v + "%</b>";
  }
  function ppRenderClosed(d) {
    var rows = d.closed || [], tb = $("#pp-closed tbody"); $("#pp-closed-n").textContent = rows.length;
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="8" class="muted">Закритих угод немає</td></tr>'; return; }
    tb.innerHTML = rows.map(function (c) {
      var pc = Number(c.pnl_pct || 0), t = new Date((c.closed_at || 0) * 1000), p2 = function (n) { return String(n).padStart(2, "0"); };
      return "<tr><td><span class=\"dt-d\">" + p2(t.getDate()) + "." + p2(t.getMonth() + 1) + "</span> <span class=\"dt-t\">" + p2(t.getHours()) + ":" + p2(t.getMinutes()) + "</span></td>" +
        '<td class="sym">' + (c.was_manual ? PP_PIN : "") + signalIconHtml(c.opened_by) + tvSym(c.symbol) + ppAvgMark(c) + '</td><td class="' + (c.side === "LONG" ? "pp-long" : "pp-short") + '">' + esc(c.side) + "</td>" +
        '<td class="r">' + fmtPrice(c.entry_price) + '</td><td class="r">' + fmtPrice(c.exit_price) + "</td>" +
        '<td class="r ' + ppCls(c.net_usd != null ? c.net_usd : pc) + '">' + ppPnlCols(pc, c.net_usd) + "</td>" +
        ppPeak(c) +
        '<td style="font-family:var(--font)">' + ppReason(c) + "</td></tr>";
    }).join("");
  }
  function ppClosedApply() { var on = false; try { on = localStorage.getItem(PP_CL) === "1"; } catch (e) {} $("#pp-closed-body").style.display = on ? "" : "none"; $("#pp-closed-head").setAttribute("aria-expanded", on ? "true" : "false"); }
  function loadPaper() {
    var card = $("#pp-card"); if (!card || PP.busy || !pOk("sec_paper")) return Promise.resolve(null);
    var r = ppRange(PP.period); if (!r) return Promise.resolve(null);
    var q = r.from != null ? "?from=" + r.from + "&to=" + r.to : "";
    PP.busy = true;
    return api("/api/info/paper" + q).then(function (d) {
      if (!d || !d.ok) { card.style.display = "none"; return d; }
      card.style.display = ""; var cs = (d.closed || []).length + "|" + ((d.closed || [])[0] || {}).closed_at;
      if (cs !== PP.closedSig) { PP.closedSig = cs; PP.daysAt = 0; }
      PP.last = d; PP.label = r.label; ppSync(); ppClosedApply(); $("#pp-note").textContent = r.label;
      var sa = d.account || {}; $("#pp-inline").innerHTML = ppInlineHTML(sa); ppRenderKpis(d, r.label); ppRenderOpen(d); ppRenderClosed(d);
      return d;
    }).catch(function () { return null; }).then(function (result) { PP.busy = false; return result; });
  }
  function ppRerender() { if (PP.last) { ppSync(); ppRenderKpis(PP.last, PP.label || ""); ppRenderOpen(PP.last); ppRenderClosed(PP.last); } }
  function ppOpenCal(inp) {
    var go = function () { tcalOpen(inp, PP.days, function (ymd) { PP.period[inp.dataset.k] = ymd; ppSave(); loadPaper(); }, PP.shown); };
    if (Date.now() - PP.daysAt < 10000) { go(); return; }
    api("/api/info/paper/days").then(function (j) { if (j && j.ok) { PP.days = tcalBucket(j.items); PP.daysAt = Date.now(); } go(); }, go);
  }
  document.addEventListener("click", function (e) {
    if (!e.target.closest) return;
    var b = e.target.closest("#pp-period button[data-p]");
    if (b) { var P = PP.period, today = tcalYmd(new Date()); P.mode = b.getAttribute("data-p"); if (P.mode === "day" && !P.day) P.day = today; if (P.mode === "range") { if (!P.from) P.from = today; if (!P.to) P.to = today; } ppSave(); ppSync(); loadPaper(); return; }
    var di = e.target.closest(".pp-dateinp"); if (di) { ppOpenCal(di); return; }
    if (e.target.closest("#pp-amt")) { PP.shown = !PP.shown; try { localStorage.setItem(PP_AMT, PP.shown ? "1" : "0"); } catch (x) {} ppRerender(); return; }
    if (e.target.closest("#pp-closed-head")) { var on = false; try { on = localStorage.getItem(PP_CL) === "1"; } catch (x) {} try { localStorage.setItem(PP_CL, on ? "0" : "1"); } catch (x) {} ppClosedApply(); }
  });
  setInterval(function () { document.querySelectorAll(".pp-timer[data-since]").forEach(function (el) { var t = Number(el.getAttribute("data-since")); if (t > 0) el.textContent = ppTimer(t); }); }, 1000);

  // 🛰 VSV SIGNALS (вимога 24.11): монети ОБОХ Пружин з найвищим VSV Score —
  // radar волатильності; 25.11 — ЙМОВІРНИЙ напрямок з фактичною точністю.
  var REC = { f: "all", rows: [], meta: {} };
  var REC_F = ["all", "charged", "impulse", "ai"];
  try { var _rf = localStorage.getItem("vsv_rec_f"); if (REC_F.indexOf(_rf) >= 0) REC.f = _rf; } catch (e) {}
  function recPass(r, f) { return f === "all" || (f === "impulse" ? !!r.impulse : f === "ai" ? !!r.ai_ready : !r.impulse); }
  var REC_PART_UA = { charge: "заряд", squeeze: "стиснення", volat: "низька волатильність", volume: "висихання обсягу", ai: "VSV·AI" };
  var REC_PART_W = { charge: 45, squeeze: 15, volat: 15, volume: 10, ai: 15 };
  function recScoreCol(v) { return v >= 80 ? "#00e676" : v >= 65 ? "#facc15" : "#94a3b8"; }
  function recScoreCell(r) {
    var v = Number(r.score);
    if (!isFinite(v)) return '<span class="muted">—</span>';
    var p = r.parts || {}, tip = ["VSV Score " + v.toFixed(0) + " · клас " + (r.grade || "") + " (" + (r.grade_ua || "") + " готовність)"];
    Object.keys(REC_PART_W).forEach(function (k) { if (p[k] != null) tip.push(REC_PART_UA[k] + " " + Number(p[k]).toFixed(0) + "% × " + REC_PART_W[k] + "%"); });
    var c = recScoreCol(v);
    return '<span class="rec-score" title="' + esc(tip.join("\n")) + '"><span class="bar"><i style="width:' + Math.max(0, Math.min(100, v)).toFixed(0) + "%;background:" + c + '"></i></span><b style="color:' + c + '">' + v.toFixed(0) + '</b><span class="rec-grade g-' + esc(r.grade || "C") + '">' + esc(r.grade || "C") + "</span></span>";
  }
  function recChargeCell(r) {
    var c = Number(r.charge); if (!isFinite(c)) return '<span class="muted">—</span>';
    var ci = Number(r.charge_in), d = isFinite(ci) ? c - ci : null;
    var dh = d != null && Math.abs(d) >= 0.5 ? ' <small class="' + (d > 0 ? "rec-up" : "rec-dn") + '">' + (d > 0 ? "+" : "−") + Math.abs(d).toFixed(0) + "</small>" : "";
    return '<span class="rec-chg" title="Енергія стиснення «Пружини»' + (d != null ? " · при потраплянні в список " + ci.toFixed(0) : "") + '"><b style="color:' + exhColor(c) + '">' + c.toFixed(0) + "</b>" + dh + "</span>";
  }
  function recSqCell(r) {
    var n = Number(r.squeeze_bars);
    if (!isFinite(n) || n <= 0) return '<span class="muted" title="Зараз не в TTM squeeze">—</span>';
    return '<span class="' + (n >= 12 ? "rec-good" : "") + '" title="Смуги Боллінджера всередині каналу Кельтнера ' + n + ' год поспіль">' + n + " год</span>";
  }
  function recVolatCell(r) {
    var a = Number(r.atr_rank); if (!isFinite(a)) return '<span class="muted">—</span>';
    return '<span class="' + (a <= 25 ? "rec-good" : a >= 75 ? "rec-hot" : "") + '" title="Поточний ATR вищий за ' + a.toFixed(0) + '% власної історії">ATR p' + a.toFixed(0) + "</span>";
  }
  function recVolCell(r) {
    var v = Number(r.vol_ratio); if (!isFinite(v)) return '<span class="muted">—</span>';
    var cls = v <= 0.75 ? "rec-good" : v >= 1.5 ? "rec-hot" : "", t = v <= 0.75 ? "обсяг висихає — ліквідність накопичується" : v >= 1.5 ? "обсяг вище середнього — учасники вже активні" : "обсяг близький до середнього";
    return '<span class="' + cls + '" title="' + t + '">×' + v.toFixed(2) + "</span>";
  }
  function recAmpCell(r) {
    var a = Number(r.amp_pct), b = Number(r.box_pct);
    if (!isFinite(a)) return isFinite(b) ? '<span title="Ширина 24-год діапазону">±' + b.toFixed(2) + "%</span>" : '<span class="muted">—</span>';
    return '<span title="Measured move у будь-який бік' + (isFinite(b) ? " · діапазон 24 год " + b.toFixed(2) + "%" : "") + '">±' + a.toFixed(2) + "%</span>";
  }
  function recPhaseCell(r) {
    var ph = r.impulse ? ["imp", "💥 Імпульс"] : Number(r.charge) >= 65 ? ["sq", "🌀 Стиснення"] : ["acc", "⏳ Накопичення"];
    var notes = (r.notes || []).filter(function (x) { return x && x.indexOf("💥") !== 0; });
    var txt = notes.join(" · ");
    return '<span class="rec-ph"><span class="rec-phase ' + ph[0] + '">' + ph[1] + '</span><span class="rec-note" title="' + esc(txt) + '">' + esc(txt) + "</span></span>";
  }
  // 🧭 Ймовірний напрямок (25.11): зведення доказів бота + впевненість моделі +
  // фактична точність журналу перевірки; 🔄 — напрямок нещодавно розвернувся.
  function recAccCol(a) { return a >= 60 ? "#00e676" : a >= 50 ? "#facc15" : "#ff4d5e"; }
  function recDirCell(r) {
    var h = r.hint;
    if (!h) return '<span class="muted" title="Напрямок ще не порахований">—</span>';
    var tip = [], parts = h.parts || [];
    parts.forEach(function (e) {
      var s = Number(e.s) || 0;
      tip.push(e.label + ": " + e.txt + " (" + (s > 0 ? "▲ " : s < 0 ? "▼ " : "· ") + Math.abs(s * 100).toFixed(0) + "%, вага " + Math.round((Number(e.w) || 0) * 100) + "%)");
    });
    if (h.why) tip.push("→ " + h.why);
    tip.push("Напрямок перераховується постійно і може змінитися: хибний пробій, збір ліквідності, новий учасник чи рух BTC розвертають картину. Не 100% і не остаточний — див. «ⓘ Що це?».");
    var fl = h.flip ? '<small class="rec-flip" title="Напрямок розвернувся ' + esc(locHM(h.flip.t)) + '">🔄 був ' + esc(h.flip.prev) + " · " + (h.flip.ago_min < 60 ? h.flip.ago_min + " хв" : Math.floor(h.flip.ago_min / 60) + " год") + "</small>" : "";
    if (!h.side) return '<span class="rec-dir" title="' + esc(tip.join("\n")) + '"><span class="rec-dir-n">⚖ не визначено</span>' + fl + "</span>";
    var L = h.side === "LONG", a = h.acc || {}, acc;
    if (a.acc != null) {
      tip.push("Точність: " + a.acc + "% із " + a.n + " перевірених" + (a.scope === "bucket" ? " (впевненість " + a.range + "%)" : " (усі рядки)") + " · 50% = випадкове вгадування");
      acc = '<small class="rec-acc">точність <b style="color:' + recAccCol(Number(a.acc)) + '">' + Number(a.acc).toFixed(0) + "%</b> · " + a.n + "</small>";
    } else {
      tip.push("Точність: збираємо статистику — перевірено " + (a.n || 0) + " з " + (a.need || 20) + " потрібних");
      acc = '<small class="rec-acc muted">точність: збираємо ' + (a.n || 0) + "/" + (a.need || 20) + "</small>";
    }
    return '<span class="rec-dir" title="' + esc(tip.join("\n")) + '"><span class="rec-dir-s ' + (L ? "l" : "s") + '">' + (L ? "▲ LONG" : "▼ SHORT") + ' <small title="Впевненість моделі">~' + Number(h.prob).toFixed(0) + "%</small></span>" + acc + fl + "</span>";
  }
  function recRow(r) {
    var ex = String(r.ex || (r.src === "scan" ? "binance" : "bybit")).toLowerCase();
    var icon = '<span class="rec-ic" title="' + (ex === "binance" ? "Binance" : "Bybit") + " · " + (r.src === "scan" ? "скан біржі" : "WATCHLIST бота") + '">' + (SPB_ICON[ex] || "") + "</span>";
    var nb = r.bybit === false ? ' <span class="rec-nb" title="Інструмент не торгується на Bybit">⚠</span>' : "";
    var aiCell = '<span class="rec-ai-slot">' + (r.ai_ready ? '<span class="rec-ai on" title="' + esc("VSV·AI бачить сформований сетап" + (r.ai_scenario ? " · " + r.ai_scenario : "")) + '">' + AI_BRAND + "<b>СЕТАП</b></span>" : "") + "</span>";
    return '<tr class="rec-row' + (r.impulse ? " imp" : "") + (r.ai_ready ? " ai-pick" : "") + '" data-sym="' + esc(r.symbol) + '" title="Подвійний клік — відкрити на графіку на весь екран"><td class="sym"><span class="rec-symw">' + icon + '<span class="rec-sym">' + tvSym(r.symbol) + nb + "</span>" + aiCell + "</span></td><td>" + recDirCell(r) + "</td>" +
      '<td class="r">' + recScoreCell(r) + '</td><td class="r">' + recChargeCell(r) + '</td><td class="r">' + recSqCell(r) + '</td><td class="r">' + recVolatCell(r) +
      '</td><td class="r">' + recVolCell(r) + '</td><td class="r">' + recAmpCell(r) + "</td><td>" + recPhaseCell(r) + "</td></tr>";
  }
  function recRender() {
    var rows = REC.rows || [], n = {};
    REC_F.forEach(function (f) { n[f] = rows.filter(function (r) { return recPass(r, f); }).length; });
    var lbl = { all: "Усі", charged: "🌀 Стиснення", impulse: "💥 Імпульс", ai: "VSV·AI" };
    document.querySelectorAll("#rec-tabs button[data-d]").forEach(function (b) { var d = b.getAttribute("data-d"); b.textContent = lbl[d] + " (" + (n[d] || 0) + ")"; b.classList.toggle("on", d === REC.f); });
    var vis = rows.filter(function (r) { return recPass(r, REC.f); }), tb = $("#rec-tbl tbody");
    if (tb) tb.innerHTML = vis.length ? vis.map(recRow).join("") : '<tr><td colspan="9" class="muted">' + (rows.length ? "У цій вкладці зараз немає монет" : "Зараз немає монет у фазі стиснення — список оновиться, щойно «Пружина» зарядиться") + "</td></tr>";
    var c = (REC.meta || {}).counts || {}, rn = $("#rec-note");
    var da = (REC.meta || {}).dir_acc || {};
    if (rn) rn.title = da.acc != null ? "Фактична точність напрямку: " + da.acc + "% (✅ " + da.win + " · ❌ " + da.loss + " · без руху " + (da.flat || 0) + " · перевіряється " + (da.open || 0) + ")" : "Точність напрямку: збираємо статистику (" + (da.n || 0) + "/" + (da.min_n || 20) + ")";
    if (rn) rn.textContent = (c.total > rows.length ? "топ " + rows.length + " з " + c.total + " · " : "") + (da.acc != null ? "🧭 точність " + Number(da.acc).toFixed(0) + "% · " : "") + new Date().toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" });
  }
  // ⚖ Банер «SMC Forecast» (вимога 11.11): куди накручено пружину — 🔭 скан біржі
  // і 🌀 WATCHLIST, у відсотках і з таймером «відколи тримається напрямок».
  var SPB = { skew: 0 };
  // Локальні векторні знаки бірж; зовнішні запити під час роботи не потрібні.
  var SPB_ICON = {
    binance: '<img class="exchange-logo binance-logo" src="/static/info/img/binance.svg" alt="" aria-hidden="true">',
    bybit: '<img class="exchange-logo bybit-logo" src="/static/info/img/bybit.svg" alt="" aria-hidden="true">'
  };
  // ⏱ Перекидний таймер (спільний для обох банерів): кожна цифра — окрема плитка.
  function spbParts(since) {
    var s = since > 0 ? Math.max(0, Math.floor(Date.now() / 1000 - SPB.skew - since)) : 0, d = Math.floor(s / 86400);
    var p = function (n) { return String(n).padStart(2, "0"); };
    return { d: d ? String(d) : "", t: p(Math.floor(s % 86400 / 3600)) + p(Math.floor(s % 3600 / 60)) + p(s % 60) };
  }
  function spbFlipHTML(since) {
    var x = spbParts(since), h = "", i;
    var tile = function (ch) { return '<span class="fl"><b>' + ch + "</b></span>"; };
    if (x.d) { h += '<span class="fl-g fl-day">'; for (i = 0; i < x.d.length; i++) h += tile(x.d[i]); h += '<em>д</em></span>'; }
    for (i = 0; i < 6; i++) { if (i === 2 || i === 4) h += '<i class="fl-sep">:</i>'; if (i % 2 === 0) h += '<span class="fl-g">'; h += tile(x.t[i]); if (i % 2 === 1) h += "</span>"; }
    return h;
  }
  function spbFlipTick(el) {
    var t = Number(el.getAttribute("data-since")); if (!(t > 0)) return;
    var x = spbParts(t), digits = (x.d + x.t).split(""), tiles = el.querySelectorAll(".fl b");
    if (tiles.length !== digits.length) { el.innerHTML = spbFlipHTML(t); return; }
    for (var i = 0; i < digits.length; i++) {
      if (tiles[i].textContent === digits[i]) continue;
      tiles[i].textContent = digits[i];
      var f = tiles[i].parentNode; f.classList.remove("flip"); void f.offsetWidth; f.classList.add("flip");
    }
  }
  // ⚡ Напруженість пружини (вимога 18.11): середній заряд усіх монет 0-100,
  // пороги — стани Пружини (⏳ ≥45 · 🌀 ≥65). Рахує бот (`tension`).
  function spbTension(b) {
    var t = Number(b && b.tension);
    if (!b || b.tension == null || !isFinite(t)) return "";
    var lv = t >= 65 ? "hi" : t >= 45 ? "mid" : "lo";
    var n = Math.max(0, Math.min(100, Math.round(t)));
    return '<span class="spb-ready ' + lv + '" role="img" aria-label="Readiness ' + n + ' percent">' +
      '<span class="spb-ready-ring" style="--p:' + n + '"><b>' + n + '<i>%</i></b></span>' +
      '<small>READINESS</small></span>';
  }
  function spbSide(b) {
    var icon = SPB_ICON[b.ex] || "", head = '<div class="spb-h">' + icon + '<span class="spb-lab">' + esc(b.label) + "</span>";
    if (!b.ok) return head + '</div><div class="spb-empty muted">ще немає даних</div>';
    var dir = b.lean === "LONG" || b.lean === "SHORT", pl = Math.max(0, Math.min(100, Number(b.prob_long) || 50)), ps = 100 - pl;
    var st = b.lean === "LONG" ? ['<span class="spb-dot up"></span>LONG', "up"] : (b.lean === "SHORT" ? ['<span class="spb-dot dn"></span>SHORT', "dn"] : ["⚖ " + (b.enough ? "РІВНОВАГА" : "ЗАМАЛО"), "flat"]);
    var tip = "▲ LONG " + pl.toFixed(1) + "% · ▼ SHORT " + ps.toFixed(1) + "%";
    // Ширина кольору відповідає відсотку напрямку; центральний колір — колір LONG/SHORT.
    var right = b.ex === "bybit", longCol = "#08775b", shortCol = "#a92b3c";
    var innerLong = b.lean === "LONG", innerCol = innerLong ? longCol : shortCol, outerCol = innerLong ? shortCol : longCol;
    var innerPct = innerLong ? pl : ps, boundary = right ? innerPct : 100 - innerPct;
    var a = Math.max(0, boundary - 1.5).toFixed(1), z = Math.min(100, boundary + 1.5).toFixed(1);
    var grad = right
      ? "linear-gradient(90deg," + innerCol + " 0%," + innerCol + " " + a + "%," + outerCol + " " + z + "%," + outerCol + " 100%)"
      : "linear-gradient(90deg," + outerCol + " 0%," + outerCol + " " + a + "%," + innerCol + " " + z + "%," + innerCol + " 100%)";
    var longLabel = '<em class="r' + (pl >= ps ? " win" : "") + '">' + Math.round(pl) + "% ▲</em>", shortLabel = '<em class="l' + (ps > pl ? " win" : "") + '">▼ ' + Math.round(ps) + "%</em>";
    if (!right) { longLabel = '<em class="l' + (pl >= ps ? " win" : "") + '">▲ ' + Math.round(pl) + "%</em>"; shortLabel = '<em class="r' + (ps > pl ? " win" : "") + '">' + Math.round(ps) + "% ▼</em>"; }
    var labels = innerLong ? (right ? longLabel + shortLabel : shortLabel + longLabel) : (right ? shortLabel + longLabel : longLabel + shortLabel);
    return head + '<span class="spb-grp">' + spbTension(b) + '<span class="spb-st ' + st[1] + '">' + st[0] + "</span></span></div>" +
      '<div class="spb-bar' + (dir ? "" : " mute") + '" title="' + esc(tip) + '" style="background:' + grad + '">' + labels + "</div>";
  }
  function spbRender(d) {
    var card = $("#spb-card");
    /* The bot integrates the terminal without the InfoSite spring banner.
       Its two-row menu metrics still consume the same server payload. */
    if (!card) { mm4Render(d && d.mm4); return; }
    var bs = (d && d.banners) || [];
    if (!bs.length) { card.style.display = "none"; return; }
    if (d.ts) SPB.skew = Date.now() / 1000 - Number(d.ts);
    card.style.display = "";
    mm4Render(d.mm4);
    var by = {}; bs.forEach(function (b) { by[b.key] = b; });
    [["scan", "#spb-scan"], ["wl", "#spb-wl"]].forEach(function (k) {
      var el = $(k[1]); if (!el) return;
      el.innerHTML = by[k[0]] ? spbSide(by[k[0]]) : ""; el.style.visibility = by[k[0]] ? "" : "hidden";
    });
    // ⏱ ОДИН таймер (вимога 11.11): відколи тримається спільний напрямок обох пружин.
    var tm = $("#spb-tm"), a = by.all || {};
    if (tm) {
      var lc = a.lean === "LONG" ? "up" : a.lean === "SHORT" ? "dn" : "flat";
      tm.className = "spb-tm " + lc;
      tm.title = "Спільний напрямок обох пружин: " + (a.lean === "LONG" ? "LONG" : a.lean === "SHORT" ? "SHORT" : "⚖ рівновага / розбіжність") + " — тримається стільки";
      if (a.since) { if (Number(tm.getAttribute("data-since")) !== Number(a.since) || !tm.querySelector(".fl")) { tm.setAttribute("data-since", a.since); tm.innerHTML = spbFlipHTML(a.since); } }
      else { tm.removeAttribute("data-since"); tm.innerHTML = '<span class="muted">—</span>'; }
    }
  }
  // 🧭 Компактне зведення чотирьох VSV-модулів у панелі меню.
  var MM4_SHORT = { "new": "Trend", "fc": "Local", "dec": "Decision", "rdy": "Готовність" };
  var MM4_ICONS = {
    total: '<path d="M4 17h3V9H4v8Zm6 0h4V4h-4v13Zm7 0h3v-6h-3v6Z"/><path class="mm4-signal" d="m3 6 5-3 5 4 7-5"/>',
    "new": '<path d="M3 17 9 8l4 4 8-9"/><path class="mm4-signal" d="M16 3h5v5"/>',
    fc: '<circle cx="12" cy="12" r="3"/><circle class="mm4-signal" cx="12" cy="12" r="8"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3"/>',
    dec: '<path d="M8 4a4 4 0 0 0-4 4v2a4 4 0 0 0 2 7v3h5V4H8Zm8 0a4 4 0 0 1 4 4v2a4 4 0 0 1-2 7v3h-5V4h3Z"/><path class="mm4-signal" d="M11 8h2m-2 4h2m-2 4h2"/>',
    rdy: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/><path class="mm4-signal" d="m16 3 1 2 2 1-2 1-1 2-1-2-2-1 2-1 1-2Z"/>',
    agree: '<path d="m12 2 2.4 5.2L20 8l-4 4 .9 5.8L12 15.2l-4.9 2.6L8 12 4 8l5.6-.8L12 2Z"/><path class="mm4-signal" d="m9 11 2 2 4-4"/>'
  };
  function mm4Icon(key) { return '<svg class="mm4-ic" viewBox="0 0 24 24" aria-hidden="true">' + (MM4_ICONS[key] || MM4_ICONS.total) + "</svg>"; }
  function mm4DirTxt(dir, pct) {
    var p = pct != null && isFinite(Number(pct)) ? " " + Number(pct).toFixed(0) + "%" : "";
    var body = '<span class="mm4-dir">' + (dir === "LONG" ? "▲ LONG" : dir === "SHORT" ? "▼ SHORT" : "⚖") + '</span><span class="mm4-pct">' + p + "</span>";
    return '<span class="mm4-v ' + (dir === "LONG" ? "up" : dir === "SHORT" ? "dn" : "flat") + '">' + body + "</span>";
  }
  function mm4Chip(it) {
    var nm = MM4_SHORT[it.key] || String(it.name || "").replace(/^VSV-?/, "VSV-"), tip = String(it.name || nm).replace(/VSV/g, "VSV") + " — ";
    if (!it.on) return '<span class="mm4-chip off" title="' + esc(tip + "блок вимкнено в боті — у зведенні не бере участі") + '">' + mm4Icon(it.key) + '<span class="mm4-name">' + esc(nm) + '</span><span class="mm4-v flat">вимк</span></span>';
    if (!it.data) return '<span class="mm4-chip off" title="' + esc(tip + "даних ще немає") + '">' + mm4Icon(it.key) + '<span class="mm4-name">' + esc(nm) + '</span><span class="mm4-v flat">—</span></span>';
    var cls = it.dir === "LONG" ? "up" : it.dir === "SHORT" ? "dn" : "flat", pct = Math.max(0, Math.min(100, Number(it.pct) || 0));
    /* Напрямок завжди домінує: LONG — зелений, SHORT — червоний; відсоток посилює перевагу. */
    var split = it.dir === "LONG" ? 50 - pct / 2 : 50 + pct / 2;
    tip += (it.dir === "LONG" ? "LONG" : it.dir === "SHORT" ? "SHORT" : "⚖ рівновага") + " · важіль " + (it.pct != null ? Number(it.pct).toFixed(0) + "%" : "—");
    return '<span class="mm4-chip ' + cls + '" style="--split:' + split.toFixed(1) + '%" title="' + esc(tip) + '">' + mm4Icon(it.key) + '<span class="mm4-name">' + esc(nm) + "</span>" + mm4DirTxt(it.dir, it.pct) + "</span>";
  }
  function mm4Render(m) {
    var box = $("#spb-mm4"), card = $("#spb-card"); if (!box) return;
    var items = (m && m.items) || [];
    if (!items.length) { box.hidden = true; box.innerHTML = ""; if (card) card.classList.remove("mm4-up", "mm4-dn"); return; }
    box.hidden = false;
    box.innerHTML = '<div class="mm4-row mm4-summary"><div class="mm4-total" title="VSV · СИНХРОН">' + mm4Icon("total") + '<span class="mm4-lbl">VSV · СИНХРОН</span></div></div>' +
      '<div class="mm4-row mm4-chips">' + items.map(mm4Chip).join("") + '</div>';
    if (card) { card.classList.toggle("mm4-up", m.agree === "LONG"); card.classList.toggle("mm4-dn", m.agree === "SHORT"); }
  }
  setInterval(function () { document.querySelectorAll(".spb-tm[data-since]").forEach(spbFlipTick); }, 1000);
  var recBusy = false;
  function loadRec() {
    var card = $("#rec-card"); if (!card || recBusy || !pOk("sec_signals")) return Promise.resolve(null);
    recBusy = true;
    return api("/api/info/recommended").then(function (d) {
      if (!d || !d.ok) { card.style.display = "none"; spbRender(null); return d; }
      card.style.display = ""; REC.rows = d.rows || []; REC.meta = d; recRender(); spbRender(d); return d;
    }).catch(function () { return null; }).then(function (d) { recBusy = false; return d; });
  }
  document.addEventListener("click", function (e) {
    if (!e.target.closest) return;
    var ib = e.target.closest("#rec-info-btn");
    if (ib) { var box = $("#rec-info"); if (box) { box.hidden = !box.hidden; ib.setAttribute("aria-expanded", box.hidden ? "false" : "true"); ib.classList.toggle("on", !box.hidden); } return; }
    var b = e.target.closest("#rec-tabs button[data-d]"); if (!b) return;
    REC.f = b.getAttribute("data-d"); try { localStorage.setItem("vsv_rec_f", REC.f); } catch (x) {} recRender();
  });

  var DENY_TXT = {
    blocked: "⛔ Ваш акаунт заблоковано адміністратором — доступ до кабінету закрито.",
    expired: "⏳ Строк вашого доступу завершився — зверніться до адміністратора для продовження.",
    not_active: "⏳ Акаунт очікує схвалення адміністратора."
  };
  function showData(ok, code) {
    var hero = $("#auth-hero"); if (hero) hero.style.display = ok ? "none" : "";
    var dn = $("#auth-deny"); if (dn) { dn.textContent = (!ok && DENY_TXT[code]) || ""; dn.style.display = (!ok && DENY_TXT[code]) ? "" : "none"; }
    var lay = $("#iw-layout"); if (lay) lay.style.display = ok ? "" : "none";
    if (!ok) { var n = $("#notify-card"); if (n) n.style.display = "none"; }
    if (!ok) { var pc = $("#pp-card"); if (pc) pc.style.display = "none"; }
    if (!ok) { var rc = $("#rec-card"); if (rc) rc.style.display = "none"; }
    if (!ok) { var sb = $("#spb-card"); if (sb) sb.style.display = "none"; }
    var tb = $(".topbar"); if (tb) tb.style.display = ok ? "" : "none";
  }

  var appRevealed = false, bootTimer = null;
  function revealApp() {
    if (appRevealed) return Promise.resolve();
    var language = window.VSV_I18N && window.VSV_I18N.ready ? window.VSV_I18N.ready : Promise.resolve();
    var fonts = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
    var loaded = document.readyState === "complete" ? Promise.resolve() : new Promise(function (resolve) {
      window.addEventListener("load", resolve, { once: true });
    });
    return Promise.all([
      loaded,
      Promise.resolve(language).catch(function () { return null; }),
      Promise.resolve(fonts).catch(function () { return null; })
    ]).then(function () {
      return new Promise(function (resolve) {
        requestAnimationFrame(function () { requestAnimationFrame(resolve); });
      });
    }).then(function () {
      if (appRevealed) return;
      appRevealed = true;
      document.body.classList.add("app-ready");
      document.dispatchEvent(new CustomEvent("vsv:terminal-ready"));
    });
  }

  /* Старт лише після реальних барів і завершення всіх pending-розрахунків.
     Порожній WATCHLIST або заборонений термінал завантаження не блокують. */
  function initialCoinReady() {
    if (!pOk("sec_terminal") || !curSym()) return true;
    var c = IW.coin;
    return !!(c && c.ok && (c.ohlc || []).length && IW.series &&
      !c.ob_wait && !c.vob_wait && !c.hl_wait && !c.sbr_wait &&
      !c.vol_wait && !(c.vsplit && c.vsplit.fill));
  }
  function waitInitialCoin() {
    return new Promise(function (resolve, reject) {
      var lastTry = 0, done = false;
      function finish(fn, value) { if (done) return; done = true; fn(value); }
      function tick() {
        if (done) return;
        if (initialCoinReady()) { finish(resolve); return; }
        var now = Date.now();
        if (!coinBusy && now - lastTry >= 4000) {
          lastTry = now;
          loadCoin(false, true).catch(function (e) { finish(reject, e); });
        }
        if (!done) setTimeout(tick, 150);
      }
      tick();
    });
  }

  function syncMarket() { return Promise.all([loadRows(), loadCoin(false), loadPaper()]); }
  function syncProfile() { return Promise.all([loadMine().catch(function () { return null; }), refreshAuth()]); }
  function syncAccount() { return Promise.all([refreshNotify(), refreshTrade()]); }
  function syncAll() {
    if (document.hidden) return Promise.resolve([]);
    return Promise.all([syncMarket(), loadRec(), syncProfile(), syncAccount(), loadUniverse()]);
  }

  var started = false;
  function startSync() {
    if (started) return;
    started = true;
    setInterval(function () { if (!document.hidden) syncMarket(); }, REFRESH);
    setInterval(function () { if (!document.hidden) loadRec(); }, REC_MS);
    setInterval(function () { if (!document.hidden) syncProfile(); }, PROFILE_MS);
    setInterval(function () { if (!document.hidden) syncAccount(); }, ACCOUNT_MS);
    setInterval(function () { if (!document.hidden) loadUniverse(); }, UNIVERSE_MS);
    document.addEventListener("visibilitychange", function () { if (!document.hidden) syncAll(); });
    window.addEventListener("online", function () { syncAll(); });
  }

  function boot() {
    if (bootTimer) { clearTimeout(bootTimer); bootTimer = null; }
    loadMine().then(function () {
      showData(true); setConn("ok", "онлайн");
      var first = !started;
      startSync();
      if (first) syncAccount();
      return Promise.all([loadUniverse(), loadRows(), loadCoin(true), loadPaper(), loadRec()])
        .then(waitInitialCoin)
        .then(revealApp);
    }).catch(function (e) {
      if (e && (e.status === 401 || e.status === 403)) {
        var deny = e.status === 403 && DENY_TXT[e.code];
        if (getTok()) setTok("");
        showData(false, deny ? e.code : "");
        setConn(deny ? "err" : "wait", deny ? "доступ закрито" : "потрібен вхід");
        return revealApp();
      }
      /* Неготову сторінку не відкриваємо: LOADING лишається до успішного
         повного циклу, а сервер повторно опитується без перезавантаження. */
      setConn("err", "бот недоступний");
      bootTimer = setTimeout(boot, 4000);
      return null;
    });
  }

  (function fixChrome() {
    var imgs = document.querySelectorAll(".brand-mark img, .hero-mark img, .foot-brand img");
    for (var i = 0; i < imgs.length; i++) {
      (function (img) {
        img.onerror = function () {
          if (!img._triedBot && BASE) { img._triedBot = true; img.src = BASE + "/favicon.ico"; return; }
          img.onerror = null; img.style.display = "none";
          var d = img.nextElementSibling; if (d && d.classList && d.classList.contains("dot")) d.style.display = "inline-block";
        };
        img.src = "img/logo.png";
      })(imgs[i]);
    }
    var lg = $("#hero-login"); if (lg) lg.href = botUrl("/info-login");
    var rg = $("#hero-register"); if (rg) { rg.href = BOT_LINK || botUrl("/register"); if (BOT_LINK) { rg.target = "_blank"; rg.rel = "noopener"; } }
    var bl = $("#hero-bot"); if (bl) bl.style.display = "none";
  })();

  document.addEventListener("click", function (e) { var lo = e.target.closest ? e.target.closest(".lo") : null; if (lo) { e.preventDefault(); logout(); } });

  setConn("wait", "підключення…");
  boot();

  // ХУК ДЛЯ CHART-PRO.JS (доступ до внутрішніх змінних)
  window.VSV_HOOK = { get IW() { return IW; }, applyCandles: applyCandlePalette, volume: volToggle };
})();