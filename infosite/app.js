// ─────────────────────────────────────────────────────────────────────────
//  VSV — особистий кабінет користувача (інфосайт).
//  Кожен користувач бачить СВІЙ список монет (зберігається в БД бота на його
//  акаунт) і графік обраної монети зі структурою OB 1H та Volumized OB.
//
//  API бота:
//    GET  /api/me                     — хто увійшов
//    GET  /api/me/watchlist           — мій список монет + обрана монета
//    POST /api/me/watchlist           — {add|remove|sel|tf|ob_tf|vob_tf} (лише свій акаунт)
//    GET  /api/info/universe          — монети WATCHLIST бота (підказки)
//    GET  /api/info/rows?symbols=…    — ціна і зміна за добу по моїх монетах
//    GET  /api/info/coin?symbol=&tf=&ob_tf=&vob_tf=&sig= — графік: бари + CHoCH/BOS + OB + VOB + PD + POC
//    POST /api/me/logout               — вихід (відкликає всі токени)
//    GET  /api/me/notify              — стан Telegram
//  Сервер віддає ЛИШЕ готові дані зі спільного кешу — сторінка біржу не смикає.
// ─────────────────────────────────────────────────────────────────────────

(function () {
  "use strict";

  var CFG = window.INFOSITE_CONFIG || {};
  var BASE = (CFG.BOT_API_BASE || "").replace(/\/+$/, "") || window.location.origin;
  var REFRESH = CFG.REFRESH_MS || 5000;
  var FORCED_SYMBOL = (CFG.POTENTIAL_SYMBOL || "").toUpperCase().trim();
  var BOT_LINK = (CFG.BOT_TELEGRAM_LINK || "").trim();
  var API_KEY = (CFG.INFO_API_KEY || "").trim();
  // Cross-origin when the bot lives on a different domain than this site.
  var CROSS = BASE !== window.location.origin;

  function botUrl(p) { return BASE + p; }   // absolute link to the bot domain

  // ── Auth token (cross-origin, cookie-free) ─────────────────────────────────
  var TOK_KEY = "vsv_info_token";
  function getTok() { try { return localStorage.getItem(TOK_KEY) || ""; } catch (e) { return ""; } }
  function setTok(t) { try { t ? localStorage.setItem(TOK_KEY, t) : localStorage.removeItem(TOK_KEY); } catch (e) {} }
  // Capture the token handed back by the bot after login (URL hash «#it=…»).
  (function captureToken() {
    var h = window.location.hash || "";
    var m = h.match(/[#&]it=([^&]+)/);
    if (m) {
      setTok(decodeURIComponent(m[1]));
      // Clean the hash so the token isn't left in the address bar / history.
      try { history.replaceState(null, "", window.location.pathname + window.location.search); } catch (e) {}
    }
  })();
  // Вихід (вимога 08.11: «Вихід не відбувається… потім на сайт відбувається
  // автоматично»). Раніше лише стирався локальний токен, а сесія БОТА лишалась
  // живою → «Вхід» миттєво видавав новий токен. Тепер: (1) бот відкликає ВСІ
  // токени цього користувача, (2) сесію бота закриваємо переходом на /logout,
  // який повертає назад на сайт.
  function logout() {
    var go = function () {
      setTok("");
      window.location.href = botUrl("/logout?next=info");
    };
    apiPost("/api/me/logout", {}).then(go, go);
  }

  // ── helpers ──────────────────────────────────────────────────────────────
  function $(sel) { return document.querySelector(sel); }

  function _headers(extra) {
    var h = { "Accept": "application/json" };
    var t = getTok();
    if (t) h["X-Info-Token"] = t;             // per-user cross-origin auth
    if (API_KEY) h["X-API-Key"] = API_KEY;    // optional read-only public access
    if (extra) for (var k in extra) h[k] = extra[k];
    return h;
  }

  // Cross-origin reads use the API key (no cookies) — «omit» is REQUIRED because
  // the bot returns Access-Control-Allow-Origin: * which the browser refuses to
  // pair with credentials. Same-origin (bot-served) uses the session cookie.
  var CREDS = CROSS ? "omit" : "same-origin";

  function api(path) {
    return fetch(BASE + path, { headers: _headers(), credentials: CREDS })
      .then(function (r) {
        if (!r.ok) {
          var e = new Error("HTTP " + r.status); e.status = r.status;
          return r.json().then(function (j) { e.code = j && j.error; throw e; },
                               function () { throw e; });
        }
        return r.json();
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

  // Symbol → TradingView link (Bybit perpetual, .P) — same as the bot's tvSym.
  // target="tvchart" reuses one tab across clicks.
  function tvSym(sym) {
    var s = String(sym || "").toUpperCase();
    var url = "https://ru.tradingview.com/chart/?symbol=BYBIT:" + s + ".P";
    return '<a href="' + url + '" target="tvchart" rel="noopener" ' +
      'title="Відкрити BYBIT:' + s + '.P (ф’ючерс) у TradingView" ' +
      'style="color:inherit;text-decoration:none;border-bottom:1px dotted rgba(255,255,255,0.4)">' +
      esc(s) + "</a>";
  }

  // 🏷️ Мітки «Сигнал → Двигун» — JS-ДЗЕРКАЛО detection/signal_labels.py
  // (тримати синхронно з бекендом і smart_money.html). Компактна іконка
  // сигналу поряд із назвою монети + повна мітка у підказці.
  var SIGNAL_BADGES_JS = {
    "choch": "🟦 CHoCH", "choch_bos": "🟦 CHoCH+BOS",
    "vob_alert": "🟪 Volumized OB", "vob": "💰 Volumized OB",
    "mm_vob": "🧮 VOB з МММ-монітора",
    "liq_vob": "💧 VOB + Сканер ліквідності",
    "ob_alert": "🆕 Новий OB",
    "poc": "🎯 POC-сетап",
    "opp": "🔄 Реверс", "external": "🔌 Зовнішня", "manual": "✋ Ручний",
    "manual_ui": "✋ Ручний", "manual_ui_overflow": "✋ Ручний"
  };
  var ENGINE_BADGES_JS = {
    "Q1": "🎯 Черга-1", "Q2": "🎯 Черга-2", "Q3": "🎯 Готовність", "Q4": "🎯 Черга-4",
    "Q3-VOB(funding)": "💰 VOB+Шари", "POC-сетап": "🎯 POC-сетап",
    "MMM": "🧮 МММ-монітор", "MMN": "🧮 МММ-Trend", "MMC": "⚡ МММ-CTR", "MMF": "🔮 МММ-Local", "MMP": "🧭 МММ-Pull", "MMD": "🧠 МММ-Decision", "MMR": "🧩 МММ-Готовність", "MMS": "🔔 МММ-SBR",
    "EXH": "🔥 Виснаженість", "FF": "🔥 FF", "direct": "⚡ Прямий"
  };
  var SIGNAL_ICON_JS = {
    "choch": "🟦", "choch_bos": "🟦", "vob_alert": "🟪", "vob": "💰",
    "mm_vob": "🧮", "liq_vob": "💧",
    "ob_alert": "🆕",
    "poc": "🎯", "opp": "🔄", "external": "🔌", "manual": "✋",
    "manual_ui": "✋", "manual_ui_overflow": "✋",
    "Q3-VOB(funding)": "💰", "POC-сетап": "🎯", "EXH": "🔥", "FF": "🔥",
    "MMM": "🧮", "MMN": "🧮", "MMC": "⚡", "MMF": "🔮", "MMP": "🧭", "MMD": "🧠", "MMR": "🧩", "MMS": "🔔"
  };
  // Двигун, чия картинка ПЕРЕКРИВАЄ картинку сигналу: угоди 🧮 МММ-монітора
  // йдуть від того самого `vob_alert`, що й угоди Черги-4, тож 🟪 їх не
  // розрізняло. Мітка у підказці лишається повною.
  var ENGINE_ICON_OVERRIDE_JS = { "MMM": "🧮", "MMN": "🧮", "MMC": "⚡", "MMF": "🔮", "MMP": "🧭", "MMD": "🧠", "MMR": "🧩", "MMS": "🔔" };
  function signalCodeOf(raw) {
    if (!raw) return "";
    return String(raw).split(" · ")[0].split(" → ")[0].trim();
  }
  function engineCodeOf(raw) {
    if (!raw) return "";
    var parts = String(raw).split(" · ")[0].split(" → ");
    return parts.length > 1 ? parts[parts.length - 1].trim() : "";
  }
  function prettyOpenedBy(raw) {
    if (!raw) return "";
    raw = String(raw);
    var tail = "", ci = raw.indexOf(" · ");
    if (ci >= 0) { tail = raw.slice(ci); raw = raw.slice(0, ci); }
    var parts = raw.split(" → ");
    var out = parts.map(function (p, i) {
      p = p.trim();
      return i === 0 ? (SIGNAL_BADGES_JS[p] || ENGINE_BADGES_JS[p] || p)
                     : (ENGINE_BADGES_JS[p] || SIGNAL_BADGES_JS[p] || p);
    });
    return out.join(" → ") + tail;
  }
  function signalIconHtml(raw) {
    if (!raw) return "";
    var icon = ENGINE_ICON_OVERRIDE_JS[engineCodeOf(raw)]
      || SIGNAL_ICON_JS[signalCodeOf(raw)] || "🏷";
    var full = prettyOpenedBy(raw).replace(/"/g, "&quot;");
    return '<span title="' + full + '" style="margin-right:3px;font-size:0.8rem;cursor:help">' + icon + "</span>";
  }

  // Price with adaptive precision — 1:1 with the bot's fmtPriceJS so cheap
  // coins (DOGE $0.07429) aren't flattened to $0.07.
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
    el.className = "conn conn--" + state;
    el.textContent = "● " + text;
  }

  // ── 🔔 Мої сповіщення (per-user Telegram opt-in) ────────────────────────────
  function toggleRow(key, label, on, disabled) {
    return '<div class="ntf-row">' +
      '<span class="ntf-lbl">' + esc(label) + '</span>' +
      '<button class="switch' + (on ? " on" : "") + '" data-ntf="' + key + '"' +
        (disabled ? " disabled" : "") + ' aria-pressed="' + (on ? "true" : "false") + '">' +
        '<span class="knob"></span></button>' +
    '</div>';
  }

  function renderNotify(n) {
    var card = $("#notify-card"), body = $("#notify-body");
    if (!card || !body) return;
    if (!n || n.ok === false) { card.style.display = "none"; return; }
    card.style.display = "";
    // Ринкові сигнали йдуть у Telegram-ГРУПУ (не персонально). Персональних
    // опт-інів більше немає — тому показуємо пояснення, а не перемикачі.
    var linked = !!n.tg_linked;
    var linkA = BOT_LINK
      ? ' <a href="' + esc(BOT_LINK) + '" target="_blank" rel="noopener">відкрити бота</a>' : "";
    var warn = linked ? "" :
      '<p class="ntf-warn">⚠️ Telegram ще не прив’язано — напишіть боту <b>/start</b>' + linkA + '.</p>';
    var nSub = Object.keys(IW.sbr || {}).length;
    body.innerHTML =
      '<div class="ntf-info">' +
        '<p><b>👤 Особисто вам</b> (чат із ботом): сигнали <b>🔔 SBR</b> по монетах вашого ' +
        'WATCHLIST — натисніть 🔔 біля монети й оберіть <b>1H</b> та/або <b>4H</b>. ' +
        (nSub ? "Зараз підписано монет: <b>" + nSub + "</b>." : "Зараз підписок немає.") + "</p>" +
        '<p><b>👥 У Telegram-групі</b> (теми): <b>🧮 МММ-монітор</b> — зміни банерів МММ, ' +
        '<b>🔔 Сигнали</b> — SBR 1H/4H по основних монетах, <b>📈 Угоди</b> — відкриття й закриття угод бота. ' +
        'Під повідомленнями є кнопка <b>💬 Коментар</b> — коментар публікує бот із вашим імʼям.</p>' +
        '<p>Доступ до групи — <b>після реєстрації та схвалення адміністратором</b>; посилання прийде автоматично. ' +
        'Якщо доступ закрито (вимкнено / строк минув) — бот видаляє з групи і нічого не надсилає.</p>' +
        warn +
      '</div>';
  }

  document.addEventListener("click", function (e) {
    var btn = e.target.closest ? e.target.closest(".switch[data-ntf]") : null;
    if (!btn || btn.disabled) return;
    var key = btn.getAttribute("data-ntf");
    var next = !btn.classList.contains("on");
    btn.classList.toggle("on", next);
    btn.setAttribute("aria-pressed", next ? "true" : "false");
    var payload = {}; payload[key] = next;
    apiPost("/api/me/notify", payload).then(renderNotify).catch(function () {
      btn.classList.toggle("on", !next);   // revert on failure
    });
  });

  function refreshNotify() {
    api("/api/me/notify").then(renderNotify).catch(function () { renderNotify({ ok: false }); });
  }

  // ── 👤 auth links (вхід / реєстрація / кабінет) ─────────────────────────────
  function renderAuth(me) {
    var el = $("#auth-links");
    if (!el) return;
    if (me && me.ok) {
      el.innerHTML =
        '<span class="who">👤 ' + esc(me.email || "") + (me.is_admin ? " · адмін" : "") + '</span>' +
        '<a href="' + botUrl("/cabinet?from=info") + '">Кабінет</a>' +
        '<a href="#" class="lo">Вихід</a>';
    } else {
      var reg = BOT_LINK || botUrl("/register");
      var ext = BOT_LINK ? ' target="_blank" rel="noopener"' : "";
      el.innerHTML = '<a href="' + botUrl("/info-login") + '">Вхід</a>' +
                     '<a href="' + reg + '"' + ext + '>Реєстрація</a>';
    }
  }

  function refreshAuth() {
    api("/api/me").then(renderAuth).catch(function () { renderAuth(null); });
  }


  // ═══ 👤 МІЙ WATCHLIST ════════════════════════════════════════════════════
  var ROWS_MS = 15000, COIN_MS = 15000;
  var CHART_TFS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"];
  var OB_TFS = ["15m", "30m", "1h", "4h"];
  var VOB_TFS = ["1m", "5m", "15m", "30m", "1h", "4h"];
  var IW = { wl: [], sel: "", rows: {}, max: 20, sig: "", coin: null, chart: null, series: null,
             prim: null, pocLine: null, chartKey: "", universe: [],
             tf: "15m", ob_tf: "1h", vob_tf: "5m", show_imb: true, show_sbr: true,
             sbr: {}, tg_linked: true, bellOpen: "" };
  var SBR_TFS = ["1h", "4h"];          // SBR веде бот і Telegram лише на цих TF
  function sbrTfOk(tf) { return SBR_TFS.indexOf(String(tf || "").toLowerCase()) >= 0; }

  function tfLbl(tf) {
    tf = String(tf || "").toLowerCase();
    if (tf === "off") return "вимк.";
    return /[hd]$/.test(tf) ? tf.toUpperCase() : tf;
  }
  // «5 Min OB» / «1 H OB» — як підпис боксу в TradingView.
  function tfWord(tf) {
    var m = String(tf || "").toLowerCase().match(/^(\d+)([mhd])$/);
    if (!m) return String(tf || "").toUpperCase();
    return m[1] + " " + ({ m: "Min", h: "H", d: "D" })[m[2]];
  }
  function fmtVol(v) {
    v = Number(v);
    if (!isFinite(v) || v <= 0) return "";
    if (v >= 1e9) return (v / 1e9).toFixed(2) + "B";
    if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(2) + "K";
    return String(Math.round(v));
  }

  function pdBadge(pd) {
    if (!pd) return '<span class="bdg mute" title="PD-зона ще не порахована">PD —</span>';
    var name = pd.zone === "discount" ? "Discount" : (pd.zone === "premium" ? "Premium" : "Equilibrium");
    var cls = pd.zone === "discount" ? "long" : (pd.zone === "premium" ? "short" : "mute");
    var tip = "Premium / Discount: де ціна в дилінг-діапазоні " + pd.tf + " (0% — низ, 100% — верх). " +
      "Discount — дешева зона (краще для LONG), Premium — дорога (краще для SHORT).";
    return '<span class="bdg ' + cls + '" title="' + esc(tip) + '"><span class="d"></span>' +
      name + " " + Math.round(pd.pct) + "% " + esc(pd.tf) + "</span>";
  }

  function pocBadge(poc, state) {
    if (!poc) {
      var t = state === "none" ? "POC немає даних" : "POC ⏳";
      return '<span class="bdg mute" title="POC (ціна найбільшого обсягу) рахується окремо — до хвилини">' + t + "</span>";
    }
    var d = poc.dist_pct, rel = poc.rel;
    var cls = rel === "above" ? "short" : (rel === "below" ? "long" : "mute");
    var hint = rel === "above" ? "краще SHORT" : (rel === "below" ? "краще LONG" : "на POC");
    var tip = "POC " + poc.days + "д · " + poc.tf + " · " + poc.market + " — ціна найбільшого обсягу (лінія на графіку). " +
      "Ціна ВИЩЕ POC → краще SHORT, НИЖЧЕ → краще LONG. " + hint + ".";
    return '<span class="bdg ' + cls + '" title="' + esc(tip) + '">POC ' +
      fmtPrice(poc.poc).replace("$", "") +
      (d == null ? "" : " " + (d >= 0 ? "+" : "") + Number(d).toFixed(2) + "%") + "</span>";
  }

  function iwMsg(t, err) {
    var el = $("#iw-msg");
    if (el) { el.textContent = t || ""; el.className = "iw-msg" + (err ? " err" : ""); }
  }

  // Рядок списку — ЛИШЕ назва, ціна і зміна (PD і POC тепер на графіку).
  function renderList() {
    var host = $("#iw-list");
    $("#iw-count").textContent = IW.wl.length + " / " + IW.max;
    if (!IW.wl.length) {
      host.innerHTML = '<div class="iw-empty">Список порожній — додайте монету вище.</div>';
      return;
    }
    host.innerHTML = IW.wl.map(function (sym) {
      var r = IW.rows[sym] || {};
      var ext = r.src === "ext"
        ? '<small title="Монети немає у WATCHLIST бота — сайт рахує її окремо">🛰</small>' : "";
      var c = r.chg24, bell = bellHTML(sym);
      var chg = c == null ? '<span class="iw-chg">—</span>'
        : '<span class="iw-chg ' + (c > 0 ? "up" : (c < 0 ? "dn" : "")) + '" title="Зміна ціни за ~24 год">' +
          (c > 0 ? "+" : "") + Number(c).toFixed(2) + "%</span>";
      return '<div class="iw-row' + (sym === IW.sel ? " sel" : "") + '" data-sym="' + esc(sym) + '">' +
        '<span class="iw-sym">' + esc(sym.replace(/USDT$/, "")) + '<small>USDT</small>' + ext + "</span>" +
        '<span class="iw-price">' + (r.price ? fmtPrice(r.price) : "—") + "</span>" + chg +
        bell.btn +
        '<button class="iw-del" data-del="' + esc(sym) + '" title="Прибрати зі списку">×</button>' + bell.menu +
        (r.err ? '<div class="iw-err" title="' + esc(r.err) + '">⚠ ' + esc(r.err) + "</div>" : "") +
      "</div>";
    }).join("");
  }

  // 🔔 Особисті SBR-сповіщення по монеті: 1H / 4H (вимога 09.11). Сигнал
  // приходить у ВАШ особистий чат із ботом, а не в групу.
  function bellHTML(sym) {
    var tfs = IW.sbr[sym] || [];
    var on = tfs.length > 0;
    var lbl = on ? tfs.map(tfLbl).join("+") : "";
    var tip = on ? "SBR-сигнали " + lbl + " по " + sym + " приходять вам у Telegram. Клік — змінити."
                 : "Отримувати сигнали SBR по " + sym + " у свій Telegram (обрати 1H / 4H)";
    var menu = IW.bellOpen === sym
      ? '<div class="iw-bell-menu" data-bellmenu="' + esc(sym) + '">' +
          '<div class="t">🔔 SBR у Telegram · ' + esc(sym.replace(/USDT$/, "")) + "</div>" +
          SBR_TFS.map(function (tf) {
            return '<label><input type="checkbox" data-belltf="' + tf + '"' + (tfs.indexOf(tf) >= 0 ? " checked" : "") +
              "> " + tfLbl(tf) + "</label>";
          }).join("") +
          (IW.tg_linked ? "" : '<div class="w">⚠️ Telegram не прив’язано — напишіть боту /start, інакше сигнали не дійдуть.</div>') +
          '<div class="n">Сигнал приходить у ваш особистий чат із ботом одразу після закриття бару.</div>' +
        "</div>"
      : "";
    return { btn: '<span class="iw-bell-wrap"><button type="button" class="iw-bell' + (on ? " on" : "") +
      '" data-bell="' + esc(sym) + '" title="' + esc(tip) + '">🔔' + (lbl ? "<small>" + esc(lbl) + "</small>" : "") +
      "</button></span>", menu: menu };
  }

  function renderTfBar() {
    var host = $("#iw-tfs");
    if (host) host.innerHTML = CHART_TFS.map(function (tf) {
      return '<button type="button" data-tf="' + tf + '" class="' + (tf === IW.tf ? "on" : "") + '">' + tfLbl(tf) + "</button>";
    }).join("");
    var opt = function (list, cur) {
      return list.concat(["off"]).map(function (tf) {
        return '<option value="' + tf + '"' + (tf === cur ? " selected" : "") + ">" + tfLbl(tf) + "</option>";
      }).join("");
    };
    var o = $("#iw-ob-tf"); if (o) o.innerHTML = opt(OB_TFS, IW.ob_tf);
    var v = $("#iw-vob-tf"); if (v) v.innerHTML = opt(VOB_TFS, IW.vob_tf);
    var lay = $("#iw-layers");
    if (lay) lay.innerHTML =
      '<button type="button" data-layer="show_imb" class="' + (IW.show_imb ? "on" : "") +
        '" title="Імбаланси (Fair Value Gap): розрив між 1-ю і 3-ю свічками, який ціна ще не заповнила">IMB</button>' +
      (sbrTfOk(IW.tf)
        ? '<button type="button" data-layer="show_sbr" class="' + (IW.show_sbr ? "on" : "") +
          '" title="Сигнали Session Band Reversal на ' + tfLbl(IW.tf) + ': S — SHORT, B — LONG (BUY). Клік по позначці — подробиці.">SBR</button>'
        : '<button type="button" disabled class="off" title="SBR показується лише на 1H і 4H: саме на цих таймфреймах індикатор калібровано і саме їх веде бот та Telegram-сигнали. Перемкніть графік на 1H або 4H.">SBR</button>');
  }

  function applyMine(d) {
    if (!d) return;
    IW.wl = d.wl || [];
    IW.max = d.max || IW.max;
    var key = [d.sel, d.tf, d.ob_tf, d.vob_tf].join("|");
    var changed = key !== [IW.sel, IW.tf, IW.ob_tf, IW.vob_tf].join("|");
    IW.sel = d.sel || "";
    if (d.tf) IW.tf = d.tf;
    if (d.ob_tf) IW.ob_tf = d.ob_tf;
    if (d.vob_tf) IW.vob_tf = d.vob_tf;
    if (typeof d.show_imb === "boolean") IW.show_imb = d.show_imb;
    if (typeof d.show_sbr === "boolean") IW.show_sbr = d.show_sbr;
    if (d.sbr && typeof d.sbr === "object") IW.sbr = d.sbr;
    if (typeof d.tg_linked === "boolean") IW.tg_linked = d.tg_linked;
    renderList();
    renderTfBar();
    if (changed) { IW.sig = ""; IW.coin = null; loadCoin(true); }
  }

  function saveMine(body) {
    return apiPost("/api/me/watchlist", body).then(function (d) {
      if (d && d.error) iwMsg(d.error, true); else iwMsg("");
      applyMine(d);
      loadRows();
      return d;
    }).catch(function () { iwMsg("не вдалося зберегти — перевірте звʼязок", true); });
  }

  function loadMine() {
    return api("/api/me/watchlist").then(function (d) { applyMine(d); return d; });
  }

  function loadRows() {
    if (!IW.wl.length) return Promise.resolve();
    return api("/api/info/rows?symbols=" + encodeURIComponent(IW.wl.join(","))).then(function (d) {
      (d.rows || []).forEach(function (r) { IW.rows[r.symbol] = r; });
      renderList();
    }).catch(function () {});
  }

  function loadUniverse() {
    api("/api/info/universe").then(function (d) {
      IW.universe = d.symbols || [];
      var dl = $("#iw-universe");
      if (dl) dl.innerHTML = IW.universe.map(function (s) {
        return '<option value="' + esc(s.replace(/USDT$/, "")) + '">' + esc(s) + "</option>";
      }).join("");
    }).catch(function () {});
  }

  document.addEventListener("submit", function (e) {
    if (!e.target || e.target.id !== "iw-add-form") return;
    e.preventDefault();
    var inp = $("#iw-add-in");
    var v = (inp.value || "").trim();
    if (!v) return;
    inp.value = "";
    saveMine({ add: v });
  });

  document.addEventListener("click", function (e) {
    if (!e.target.closest) return;
    var bell = e.target.closest("[data-bell]");
    if (bell) {
      e.stopPropagation();
      var bs = bell.getAttribute("data-bell");
      IW.bellOpen = IW.bellOpen === bs ? "" : bs;
      renderList();
      return;
    }
    if (e.target.closest("[data-bellmenu]")) { e.stopPropagation(); return; }
    if (IW.bellOpen) { IW.bellOpen = ""; renderList(); }
  }, true);

  document.addEventListener("change", function (e) {
    var cb = e.target && e.target.closest ? e.target.closest("[data-belltf]") : null;
    if (!cb) return;
    var menu = cb.closest("[data-bellmenu]");
    var sym = menu.getAttribute("data-bellmenu");
    var tfs = [];
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
      renderTfBar(); loadCoin(true);
      saveMine({ tf: tf });
      return;
    }
    var lyr = e.target.closest ? e.target.closest("#iw-layers [data-layer]") : null;
    if (lyr) {
      var k = lyr.getAttribute("data-layer");
      IW[k] = !IW[k];
      renderTfBar();
      if (IW.coin) drawCoin(IW.coin, false);
      var b = {}; b[k] = IW[k];
      saveMine(b);
      return;
    }
    var del = e.target.closest ? e.target.closest("[data-del]") : null;
    if (del) { e.stopPropagation(); saveMine({ remove: del.getAttribute("data-del") }); return; }
    var row = e.target.closest ? e.target.closest(".iw-row[data-sym]") : null;
    if (row) {
      var sym = row.getAttribute("data-sym");
      if (sym === IW.sel) return;
      IW.sel = sym; IW.sig = ""; IW.coin = null;
      renderList(); loadCoin(true);
      saveMine({ sel: sym });
    }
  });

  document.addEventListener("change", function (e) {
    var id = e.target && e.target.id;
    if (id !== "iw-ob-tf" && id !== "iw-vob-tf") return;
    var k = id === "iw-ob-tf" ? "ob_tf" : "vob_tf";
    IW[k] = e.target.value; IW.sig = "";
    loadCoin(false, true);
    var b = {}; b[k] = e.target.value;
    saveMine(b);
  });

  // ═══ 📈 ГРАФІК: бари + структура + OB + VOB + IMB + POC + сигнали SBR ═════
  // Примітив малює: (а) лінії структури; (б) зони (бокси) від свічки зони до
  // правого краю; (в) ПІДПИСИ зон — окремим проходом, «чипами» з фоном і
  // розкладкою БЕЗ НАКЛАДАНЬ (VOB — праворуч, OB/IMB — біля початку зони;
  // чип, що наліз би на вже поставлений, зсувається вгору/вниз, а коли місця
  // немає — не малюється, повний опис лишається в підказці легенди);
  // (г) позначки сигналів SBR «S»/«B», клікабельні.
  function BoxPrimitive() {
    this._b = []; this._l = []; this._s = []; this._hits = [];
    this._chart = null; this._series = null; this._req = null;
  }
  BoxPrimitive.prototype.attached = function (p) { this._chart = p.chart; this._series = p.series; this._req = p.requestUpdate; };
  BoxPrimitive.prototype.detached = function () { this._chart = null; this._series = null; this._req = null; };
  BoxPrimitive.prototype.set = function (boxes, lines, sigs) {
    this._b = boxes || []; this._l = lines || []; this._s = sigs || [];
    if (this._req) this._req();
  };
  BoxPrimitive.prototype.hitAt = function (x, y) {
    for (var i = this._hits.length - 1; i >= 0; i--) {
      var h = this._hits[i];
      if (x >= h.x0 - 3 && x <= h.x1 + 3 && y >= h.y0 - 3 && y <= h.y1 + 3) return h.sig;
    }
    return null;
  };

  function _rrect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // 📐 IMB «як на скріні» (вимога 09.11): зона + смуга у верхній половині,
  // довжина якої = % обсягу імпульсної свічки (число — праворуч у смузі), і
  // великий напівпрозорий обсяг розриву «водяним знаком» на всю висоту зони.
  // Зона нижча за ~14px — лише смуга без тексту (інакше текст нечитабельний).
  function _imbBody(ctx, b, X1, X2, yT, yB, hr, vr, font, dark) {
    var w = X2 - X1, h = yB - yT;
    if (b.pct != null && w > 4 * hr) {
      var sw = Math.max(2 * hr, w * Math.min(100, Math.max(0, b.pct)) / 100);
      ctx.fillStyle = b.strip; ctx.fillRect(X1, yT, sw, h / 2);
      ctx.strokeStyle = dark ? "rgba(0,0,0,.55)" : "rgba(0,0,0,.25)";
      ctx.lineWidth = Math.max(1, Math.round(hr));
      ctx.strokeRect(X1, yT, sw, h / 2);
      if (h >= 12 * vr) {
        var fp = Math.max(8.5, Math.min(34, h / vr * 0.38));
        ctx.font = font(600, fp); ctx.textBaseline = "middle";
        ctx.fillStyle = dark ? "rgba(235,238,245,.95)" : "rgba(20,24,32,.88)";
        var tw = ctx.measureText(b.pct + "%").width;
        if (sw >= tw + 10 * hr) { ctx.textAlign = "right"; ctx.fillText(b.pct + "%", X1 + sw - 5 * hr, yT + h / 4); }
        else if (X2 - (X1 + sw) >= tw + 8 * hr) { ctx.textAlign = "left"; ctx.fillText(b.pct + "%", X1 + sw + 4 * hr, yT + h / 4); }
      }
    }
    if (b.volTxt && h >= 18 * vr && w > 40 * hr) {
      var fv = Math.max(10, Math.min(110, h / vr * 0.82));
      ctx.font = font(500, fv);
      while (fv > 10 && ctx.measureText(b.volTxt).width > w * 0.92) { fv -= 2; ctx.font = font(500, fv); }
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = dark ? "rgba(4,6,12,.78)" : "rgba(0,0,0,.2)";
      ctx.fillText(b.volTxt, (X1 + X2) / 2, (yT + yB) / 2 + fv * vr * 0.04);
    }
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
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
        // ── лінії структури ──
        self._l.forEach(function (l) {
          var x1 = ts.timeToCoordinate(l.t1), x2 = ts.timeToCoordinate(l.t2), y = sr.priceToCoordinate(l.price);
          if (x1 == null || x2 == null || y == null) return;
          ctx.strokeStyle = l.color; ctx.lineWidth = Math.max(1, Math.round(hr));
          if (l.dashed && ctx.setLineDash) ctx.setLineDash([4 * hr, 3 * hr]);
          ctx.beginPath(); ctx.moveTo(x1 * hr, Math.round(y * vr) + 0.5); ctx.lineTo(x2 * hr, Math.round(y * vr) + 0.5); ctx.stroke();
          if (ctx.setLineDash) ctx.setLineDash([]);
        });
        // ── зони (найновіша — зверху) ──
        var placed = [];        // геометрія для підписів
        for (var i = self._b.length - 1; i >= 0; i--) {
          var b = self._b[i];
          var x1 = ts.timeToCoordinate(b.t);
          var yA = sr.priceToCoordinate(b.top), yB = sr.priceToCoordinate(b.bottom);
          if (x1 == null || yA == null || yB == null) continue;
          var X1 = Math.max(0, x1 * hr), X2 = W;
          // IMB тягнемо лише `ext` барів (як FVG-індикатори TV), а не до краю —
          // інакше десяток розривів перетворюється на суцільну штриховку.
          if (b.ext) X2 = Math.min(W, (x1 + b.ext * ts.options().barSpacing) * hr);
          var yT = Math.min(yA, yB) * vr, yBt = Math.max(yA, yB) * vr;
          if (X2 <= X1) continue;
          if (yBt - yT < 1) yBt = yT + 1;
          ctx.fillStyle = b.fill; ctx.fillRect(X1, yT, X2 - X1, yBt - yT);
          if (b.kind === "imb") _imbBody(ctx, b, X1, X2, yT, yBt, hr, vr, font, dark);
          ctx.strokeStyle = b.stroke; ctx.lineWidth = Math.max(1, Math.round((b.thin ? 1 : 1.2) * hr));
          if (b.dashed && ctx.setLineDash) ctx.setLineDash([6 * hr, 4 * hr]);
          ctx.strokeRect(X1, yT, X2 - X1, yBt - yT);
          if (ctx.setLineDash) ctx.setLineDash([]);
          placed.push({ b: b, X1: X1, X2: X2, yT: yT, yB: yBt });
        }
        // ── геометрія позначок SBR — ДО підписів, щоб чипи їх обходили ──
        var bs = 18, marks = [];   // css px
        self._s.forEach(function (s) {
          var x = ts.timeToCoordinate(s.t);
          var y = sr.priceToCoordinate(s.side === "SHORT" ? s.hi : s.lo);
          if (x == null || y == null) return;
          var cy = s.side === "SHORT" ? y - 13 - bs / 2 : y + 13 + bs / 2;
          marks.push({ s: s, x: x, y: y, cy: cy, x0: x - bs / 2, y0: cy - bs / 2 });
        });
        // ── підписи: чипи без накладань ──
        var taken = marks.map(function (m) {
          return { x0: (m.x0 - 2) * hr, x1: (m.x0 + bs + 2) * hr, y0: (m.y0 - 2) * vr, y1: (m.y0 + bs + 7) * vr };
        });
        var overlaps = function (r) {
          for (var k = 0; k < taken.length; k++) {
            var q = taken[k];
            if (r.x0 < q.x1 && r.x1 > q.x0 && r.y0 < q.y1 && r.y1 > q.y0) return true;
          }
          return false;
        };
        var prio = { vob: 0, ob: 1, imb: 2 };
        var pr = function (k) { return Object.prototype.hasOwnProperty.call(prio, k) ? prio[k] : 9; };
        placed.sort(function (a, c) { return pr(a.b.kind) - pr(c.b.kind); });
        var padX = 6 * hr, chipH = 17 * vr, gap = 2 * vr;
        placed.forEach(function (p) {
          var b = p.b;
          if (!b.label) return;
          if (b.kind === "imb") return;   // обсяг і % намальовано всередині зони (_imbBody)
          ctx.font = font(700, 11);
          var w1 = ctx.measureText(b.label).width;
          ctx.font = font(500, 10.5);
          var w2 = b.sub ? ctx.measureText(b.sub).width + 6 * hr : 0;
          var w = w1 + w2 + padX * 2 + 4 * hr;
          var right = b.kind === "vob";
          var x0 = right ? (W - w - 8 * hr) : Math.min(p.X1 + 4 * hr, W - w - 8 * hr);
          if (x0 < 2 * hr) x0 = 2 * hr;
          var mid = (p.yT + p.yB) / 2;
          var tries = [0, -1, 1, -2, 2, -3, 3], ok = null;
          for (var k = 0; k < tries.length; k++) {
            var y0 = mid - chipH / 2 + tries[k] * (chipH + gap);
            if (y0 < 0 || y0 + chipH > H) continue;
            var r = { x0: x0, x1: x0 + w, y0: y0, y1: y0 + chipH };
            if (!overlaps(r)) { ok = r; break; }
          }
          if (!ok) return;            // місця немає — не малюємо (опис — у легенді)
          taken.push(ok);
          ctx.globalAlpha = 0.92;
          ctx.fillStyle = dark ? "rgba(13,17,26,.88)" : "rgba(255,255,255,.92)";
          _rrect(ctx, ok.x0, ok.y0, w, chipH, 4 * vr); ctx.fill();
          ctx.globalAlpha = 1;
          ctx.strokeStyle = b.stroke; ctx.lineWidth = Math.max(1, Math.round(hr));
          _rrect(ctx, ok.x0, ok.y0, w, chipH, 4 * vr); ctx.stroke();
          ctx.fillStyle = b.stroke;
          ctx.fillRect(ok.x0, ok.y0 + 3 * vr, 3 * hr, chipH - 6 * vr);
          ctx.textBaseline = "middle"; ctx.textAlign = "left";
          ctx.font = font(700, 11); ctx.fillStyle = b.text;
          var tx = ok.x0 + padX + 2 * hr, ty = ok.y0 + chipH / 2 + 0.5 * vr;
          ctx.fillText(b.label, tx, ty);
          if (b.sub) {
            ctx.font = font(500, 10.5); ctx.fillStyle = dark ? "#94a3b8" : "#5b6475";
            ctx.fillText(b.sub, tx + w1 + 6 * hr, ty);
          }
          ctx.textBaseline = "alphabetic";
        });
        // ── сигнали SBR: «S» / «B» ──
        self._hits = [];
        marks.forEach(function (m) {
          var s = m.s, x = m.x, cy = m.cy, x0 = m.x0, y0 = m.y0;
          var col = s.side === "SHORT" ? "#f23645" : "#22c55e";
          _rrect(ctx, x0 * hr, y0 * vr, bs * hr, bs * vr, 4 * vr);
          ctx.fillStyle = col; ctx.fill();
          ctx.strokeStyle = dark ? "rgba(255,255,255,.35)" : "rgba(0,0,0,.25)"; ctx.lineWidth = Math.max(1, Math.round(hr)); ctx.stroke();
          // хвостик до свічки
          ctx.beginPath();
          if (s.side === "SHORT") { ctx.moveTo((x - 4) * hr, (y0 + bs) * vr); ctx.lineTo((x + 4) * hr, (y0 + bs) * vr); ctx.lineTo(x * hr, (y0 + bs + 5) * vr); }
          else { ctx.moveTo((x - 4) * hr, y0 * vr); ctx.lineTo((x + 4) * hr, y0 * vr); ctx.lineTo(x * hr, (y0 - 5) * vr); }
          ctx.closePath(); ctx.fillStyle = col; ctx.fill();
          ctx.font = font(800, 11.5); ctx.fillStyle = "#fff";
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(s.side === "SHORT" ? "S" : "B", x * hr, (cy + 0.5) * vr);
          ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
          self._hits.push({ x0: x0, x1: x0 + bs, y0: y0, y1: y0 + bs, sig: s });
        });
      });
    } }; } }];
  };

  // Час блоку → найближчий бар графіка ≤ цього часу (інакше timeToCoordinate
  // повертає null для часу МІЖ барами — напр. 1H-блок на графіку 15m).
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

  function ensureChart() {
    if (IW.chart || !window.LightweightCharts) return !!IW.chart;
    var box = $("#iw-box");
    var dark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
    IW.chart = LightweightCharts.createChart(box, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: dark ? "#aab6c8" : "#3d4759",
                fontFamily: "-apple-system,Segoe UI,Roboto,sans-serif" },
      grid: { vertLines: { color: dark ? "rgba(38,50,74,.45)" : "rgba(211,218,230,.6)" },
              horzLines: { color: dark ? "rgba(38,50,74,.45)" : "rgba(211,218,230,.6)" } },
      rightPriceScale: { borderColor: dark ? "#26324a" : "#d3dae6" },
      timeScale: { borderColor: dark ? "#26324a" : "#d3dae6", timeVisible: true, secondsVisible: false,
                   rightOffset: 18 },
      crosshair: { mode: 0 },
      // Мова явно: деякі браузери віддають тег на кшталт «en-US@posix», на
      // якому бібліотека падає і графік лишається порожнім.
      localization: { locale: "uk-UA" },
    });
    IW.series = IW.chart.addCandlestickSeries({
      upColor: "#c7ccd6", downColor: "#5b6475", borderVisible: true,
      borderUpColor: "#c7ccd6", borderDownColor: "#8b94a5",
      wickUpColor: "#c7ccd6", wickDownColor: "#8b94a5",
      priceFormat: { type: "price", precision: 6, minMove: 0.000001 },
    });
    if (!dark) IW.series.applyOptions({ upColor: "#26a69a", downColor: "#ef5350", borderUpColor: "#26a69a",
      borderDownColor: "#ef5350", wickUpColor: "#26a69a", wickDownColor: "#ef5350" });
    IW.prim = new BoxPrimitive();
    IW.series.attachPrimitive(IW.prim);
    IW.pocLine = null;
    IW.chart.subscribeClick(function (p) {
      var s = p && p.point ? IW.prim.hitAt(p.point.x, p.point.y) : null;
      sigCard(s, p && p.point ? p.point.x : 0, p && p.point ? p.point.y : 0);
    });
    IW.chart.subscribeCrosshairMove(function (p) {
      var hit = p && p.point ? IW.prim.hitAt(p.point.x, p.point.y) : null;
      box.style.cursor = hit ? "pointer" : "";
    });
    return true;
  }

  function precisionOf(p) {
    p = Number(p) || 0;
    return p < 0.0001 ? 8 : p < 0.01 ? 6 : p < 1 ? 5 : p < 100 ? 4 : 2;
  }

  // Підписи короткі: число — головне, решта — приглушено (детальний опис
  // кожного типу зони — у підказці легенди над графіком).
  function boxesOf(c) {
    var o = c.ohlc || [], out = [];
    (c.vob || []).forEach(function (v) {
      var L = v.side === "LONG", vol = fmtVol(v.volume);
      out.push({ kind: "vob", t: snapSec(v.start, o), top: v.top, bottom: v.bottom, dashed: v.breaker,
        fill: L ? "rgba(34,171,94,.16)" : "rgba(242,54,69,.16)",
        stroke: L ? "rgba(34,197,94,.95)" : "rgba(242,54,69,.95)",
        text: L ? "#86efac" : "#fca5a5",
        label: (vol || "VOB") + (v.pct != null ? " (" + v.pct + "%)" : "") + (v.breaker ? " ✕" : ""),
        sub: tfWord(v.tf || c.vob_tf) + " OB" });
    });
    (c.ob || []).forEach(function (b) {
      var L = b.side === "LONG", sw = b.fam === "swing";
      out.push({ kind: "ob", t: snapSec(b.start, o), top: b.top, bottom: b.bottom, dashed: true,
        fill: L ? (sw ? "rgba(45,212,191,.12)" : "rgba(45,212,191,.06)") : (sw ? "rgba(251,146,60,.12)" : "rgba(251,146,60,.06)"),
        stroke: L ? "rgba(45,212,191,.8)" : "rgba(251,146,60,.8)",
        text: L ? "#5eead4" : "#fdba74",
        label: "OB " + tfLbl(c.ob_tf), sub: (sw ? "swing" : "int") + (b.tag ? " · " + b.tag : "") });
    });
    if (IW.show_imb) (c.imb || []).forEach(function (z) {
      var L = z.side === "LONG";
      out.push({ kind: "imb", thin: true, ext: 30, t: snapSec(z.start, o), top: z.top, bottom: z.bottom,
        fill: L ? "rgba(59,130,246,.36)" : "rgba(234,128,38,.36)",
        strip: L ? "rgba(59,130,246,.72)" : "rgba(234,128,38,.72)",
        stroke: L ? "rgba(59,130,246,.65)" : "rgba(234,128,38,.65)",
        text: L ? "#93c5fd" : "#fdba74",
        volTxt: fmtVol(z.vol), pct: z.pct, label: "IMB" });
    });
    return out;
  }

  // Сигнали SBR → позначки біля свічки (S над high, B під low).
  function sbrOf(c) {
    if (!IW.show_sbr || !sbrTfOk(c.tf || IW.tf)) return [];
    var o = c.ohlc || [], byT = {};
    o.forEach(function (b) { byT[b.time] = b; });
    var out = [];
    (c.sbr || []).forEach(function (s) {
      var bar = byT[s.t];
      if (!bar) return;
      out.push(Object.assign({ hi: bar.high, lo: bar.low, close: bar.close }, s));
    });
    return out;
  }

  // Картка сигналу SBR (клік по «S»/«B»).
  function sigCard(s, x, y) {
    var box = $("#iw-box"), el = $("#iw-sig");
    if (!box) return;
    if (!el) { el = document.createElement("div"); el.id = "iw-sig"; el.className = "iw-sig"; box.appendChild(el); }
    if (!s) { el.style.display = "none"; return; }
    var S = s.side === "SHORT";
    // дата — у тому самому часовому поясі, що й час сигналу (Europe/Kyiv)
    var dd = "";
    try { dd = new Date(s.t * 1000).toLocaleDateString("uk-UA", { timeZone: "Europe/Kyiv", day: "2-digit", month: "2-digit" }); }
    catch (x) { var d = new Date(s.t * 1000); dd = ("0" + d.getDate()).slice(-2) + "." + ("0" + (d.getMonth() + 1)).slice(-2); }
    var slPct = s.sl && s.close ? Math.abs(s.sl / s.close - 1) * 100 : null;
    var row = function (k, v) { return '<div class="r"><span>' + k + "</span><b>" + v + "</b></div>"; };
    el.innerHTML =
      '<div class="h ' + (S ? "short" : "long") + '"><span class="tag">' + (S ? "S" : "B") + "</span>" +
        (S ? "SHORT" : "LONG") + (s.n ? " #" + s.n : "") + '<span class="tf">SBR · ' + esc(s.tf || "") + "</span>" +
        '<button class="x" aria-label="Закрити">×</button></div>' +
      row("Бар", dd + " " + esc(s.time_kyiv || "") + " (Київ)") +
      row("Trigger", fmtPrice(s.trigger)) +
      row("Score", s.score != null ? Number(s.score).toFixed(2) : "—") +
      row("Ризик", esc(s.risk || "—")) +
      row("SL", fmtPrice(s.sl) + (slPct != null ? ' <i>' + slPct.toFixed(2) + "%</i>" : "")) +
      '<div class="note">Session Band Reversal — розворот від смуги EMA5 ± 0.5·ATR14 на закритому барі. ' +
        "Той самий алгоритм, що сигнали бота 1H/4H.</div>";
    el.style.display = "block";
    var bw = box.clientWidth, bh = box.clientHeight, ew = el.offsetWidth, eh = el.offsetHeight;
    var left = Math.min(Math.max(8, x + 14), bw - ew - 8), top = Math.min(Math.max(8, y - eh / 2), bh - eh - 8);
    el.style.left = left + "px"; el.style.top = top + "px";
  }

  document.addEventListener("click", function (e) {
    if (e.target.closest && e.target.closest("#iw-sig .x")) sigCard(null);
  });

  // CHoCH / BOS: лінія від півота до бару пробою + стрілка з підписом (як на
  // графіку бота / LuxAlgo). CHoCH — пунктир, BOS — суцільна.
  function structOf(c) {
    var o = c.ohlc || [], lines = [], marks = [];
    if (!o.length) return { lines: lines, marks: marks };
    var first = o[0].time;
    (c.events || []).forEach(function (e) {
      if (!e.to || e.to < first) return;
      var bull = e.dir === "bull", col = bull ? "#22c55e" : "#ef4444";
      var t2 = snapSec(e.to, o);
      lines.push({ t1: snapSec(e.from, o), t2: t2, price: e.level, color: col, dashed: e.tag !== "BOS" });
      marks.push({ time: t2, position: bull ? "aboveBar" : "belowBar", color: col,
                   shape: bull ? "arrowDown" : "arrowUp", text: e.tag });
    });
    marks.sort(function (a, b) { return a.time - b.time; });
    return { lines: lines, marks: marks };
  }

  function setPoc(poc) {
    if (!IW.series) return;
    if (IW.pocLine) { try { IW.series.removePriceLine(IW.pocLine); } catch (x) {} IW.pocLine = null; }
    if (!poc || !poc.poc) return;
    IW.pocLine = IW.series.createPriceLine({ price: Number(poc.poc), color: "#f5f5f5", lineWidth: 1,
      lineStyle: 0, axisLabelVisible: true, title: "POC" });
  }

  function renderHead() {
    var h = $("#iw-head");
    if (!h) return;
    var sym = IW.sel;
    if (!sym) { h.innerHTML = '<span class="iw-title">—</span>'; return; }
    var c = IW.coin || {}, r = IW.rows[sym] || {};
    var o = c.ohlc || [], px = o.length ? o[o.length - 1].close : r.price;
    h.innerHTML =
      '<span class="iw-title">' + tvSym(sym) + "</span>" +
      '<span class="iw-px">' + (px ? fmtPrice(px) : "—") + "</span>" +
      pdBadge(c.pd) + pocBadge(c.poc, c.poc_state) +
      (c.src === "ext" ? '<span class="bdg warn" title="Монети немає у WATCHLIST бота — сайт рахує її окремо тими самими алгоритмами">🛰 поза ботом</span>' : "") +
      '<span class="iw-legend">' +
        '<span title="Volumized Order Block: обсяг блоку і (%) — баланс обсягу між половинами; пунктир із ✕ — блок пробитий (breaker)"><i style="background:rgba(34,197,94,.55)"></i>VOB ' + tfLbl(IW.vob_tf) + "</span>" +
        '<span title="Order Block — пунктир; густіший — swing, світліший — internal"><i style="background:rgba(45,212,191,.35);outline:1px dashed rgba(45,212,191,.9)"></i>OB ' + tfLbl(IW.ob_tf) + "</span>" +
        '<span title="Структура: CHoCH — пунктир (розворот), BOS — суцільна (продовження)"><i style="background:#22c55e;height:2px"></i>CHoCH/BOS</span>' +
        '<span title="IMB — імбаланс (Fair Value Gap): розрив між 1-ю і 3-ю свічками, ще не заповнений ціною (заповнення звужує зону). Велике число — обсяг трьох свічок розриву; смуга і % — частка обсягу імпульсної (середньої) свічки. Синій — бичачий, помаранчевий — ведмежий"><i style="background:rgba(234,128,38,.6)"></i>IMB</span>' +
        '<span title="Сигнали SBR на цьому таймфреймі: S — SHORT, B — LONG; клік по позначці — подробиці"><i style="background:#f23645;width:9px"></i><i style="background:#22c55e;width:9px;margin-left:-3px"></i>SBR</span>' +
        '<span title="POC — ціна найбільшого обсягу"><i style="background:#f5f5f5;height:2px"></i>POC</span>' +
      "</span>";
  }

  function setNote(t) {
    var n = $("#iw-note");
    if (!n) return;
    n.textContent = t || "";
    n.style.display = t ? "" : "none";
  }

  function setWait(c) {
    var w = $("#iw-tf-wait");
    if (!w) return;
    var parts = [];
    if (c && c.ob_wait) parts.push("OB " + tfLbl(c.ob_tf));
    if (c && c.vob_wait) parts.push("VOB " + tfLbl(c.vob_tf));
    w.innerHTML = parts.length ? '<span class="wait">⏳ рахується: ' + esc(parts.join(", ")) + "</span>" : "";
  }

  function drawCoin(c, fit) {
    if (!ensureChart()) { setNote("графічна бібліотека не завантажилась"); return; }
    var o = c.ohlc || [];
    var prec = precisionOf(o.length ? o[o.length - 1].close : 0);
    IW.series.applyOptions({ priceFormat: { type: "price", precision: prec, minMove: Math.pow(10, -prec) } });
    IW.series.setData(o);
    var st = structOf(c);
    IW.series.setMarkers(st.marks);
    IW.prim.set(boxesOf(c), st.lines, sbrOf(c));
    setPoc(c.poc);
    var key = c.symbol + "|" + c.tf;
    if (fit || IW.chartKey !== key) {
      IW.chart.timeScale().fitContent();
      IW.chartKey = key;
    }
    setNote(o.length ? "" : (c.reason || "немає даних"));
    setWait(c);
  }

  var coinBusy = false;
  function loadCoin(force, keepBars) {
    var sym = IW.sel;
    if (!sym) { setNote("оберіть монету зі списку"); renderHead(); return; }
    if (coinBusy && !force && !keepBars) return;
    coinBusy = true;
    if (force) {
      setNote("завантаження " + sym + " · " + tfLbl(IW.tf) + "…");
      if (IW.series) { IW.series.setData([]); IW.series.setMarkers([]); IW.prim.set([], [], []); setPoc(null); }
      sigCard(null);
    }
    var want = [IW.sel, IW.tf, IW.ob_tf, IW.vob_tf].join("|");
    var q = "/api/info/coin?symbol=" + encodeURIComponent(sym) + "&tf=" + IW.tf +
      "&ob_tf=" + IW.ob_tf + "&vob_tf=" + IW.vob_tf +
      (IW.sig && !force ? "&sig=" + encodeURIComponent(IW.sig) : "");
    api(q).then(function (c) {
      if (want !== [IW.sel, IW.tf, IW.ob_tf, IW.vob_tf].join("|")) return;  // поки чекали — змінили вибір
      if (c.same) return;
      IW.coin = c; IW.sig = c.sig || "";
      renderHead();
      drawCoin(c, force);
      // щось ще рахується (монета поза ботом / свій TF) — перепитуємо швидше
      if (!c.ok || c.ob_wait || c.vob_wait)
        setTimeout(function () { if (want === [IW.sel, IW.tf, IW.ob_tf, IW.vob_tf].join("|")) loadCoin(false, true); }, 4000);
    }).catch(function () { setNote("не вдалося завантажити графік"); })
      .then(function () { coinBusy = false; });
  }

  // Тема змінилась → перемалювати кольори графіка.
  document.addEventListener("click", function (e) {
    if (!e.target.closest || !e.target.closest("#theme-btn") || !IW.chart) return;
    setTimeout(function () {
      try { IW.chart.remove(); } catch (x) {}
      IW.chart = null; IW.series = null; IW.prim = null; IW.pocLine = null; IW.chartKey = "";
      if (IW.coin) drawCoin(IW.coin, true);
    }, 30);
  });

  // ═══ показ / автооновлення ═════════════════════════════════════════════
  // Акаунт є, але доступ закрито (вимога 09.11): сторінка не показується,
  // лише пояснення. Повідомлення — від бота (`error` у відповіді 403).
  var DENY_TXT = {
    blocked: "⛔ Ваш акаунт заблоковано адміністратором — доступ до кабінету закрито.",
    expired: "⏳ Строк вашого доступу завершився — зверніться до адміністратора для продовження.",
    not_active: "⏳ Акаунт очікує схвалення адміністратора."
  };
  function showData(ok, code) {
    document.body.classList.add("app-ready");
    var hero = $("#auth-hero"); if (hero) hero.style.display = ok ? "none" : "";
    var dn = $("#auth-deny");
    if (dn) { dn.textContent = (!ok && DENY_TXT[code]) || ""; dn.style.display = (!ok && DENY_TXT[code]) ? "" : "none"; }
    var lay = $("#iw-layout"); if (lay) lay.style.display = ok ? "" : "none";
    if (!ok) { var n = $("#notify-card"); if (n) n.style.display = "none"; }
    var tb = $(".topbar"); if (tb) tb.style.display = ok ? "" : "none";
  }

  var started = false;
  function boot() {
    loadMine().then(function () {
      showData(true);
      setConn("ok", "онлайн");
      if (!started) {
        started = true;
        loadUniverse();
        loadRows();
        loadCoin(true);
        setInterval(function () { if (!document.hidden) loadRows(); }, ROWS_MS);
        setInterval(function () { if (!document.hidden) loadCoin(false); }, COIN_MS);
        setInterval(function () { if (!document.hidden) { refreshAuth(); refreshNotify(); } }, 60000);
      }
      $("#last-update").textContent = "оновлено: " + new Date().toLocaleTimeString("uk-UA");
    }).catch(function (e) {
      if (e && (e.status === 401 || e.status === 403)) {
        var deny = e.status === 403 && DENY_TXT[e.code];
        if (getTok()) setTok("");
        showData(false, deny ? e.code : "");
        setConn(deny ? "err" : "wait", deny ? "доступ закрито" : "потрібен вхід");
      } else {
        showData(true);
        setConn("err", "бот недоступний");
        setTimeout(boot, 10000);
      }
    });
    refreshAuth();
    refreshNotify();
  }

  // Logos: try the LOCAL file first (vsv-logo.png next to index.html), then the
  // bot's /favicon.ico, then the accent dot. Auth links point at the bot domain.
  (function fixChrome() {
    var imgs = document.querySelectorAll("img.logo, img.hero-logo");
    for (var i = 0; i < imgs.length; i++) {
      (function (img) {
        img.onerror = function () {
          if (!img._triedBot && BASE) { img._triedBot = true; img.src = BASE + "/favicon.ico"; return; }
          img.onerror = null; img.style.display = "none";
          var d = img.nextElementSibling;
          if (d && d.classList && d.classList.contains("dot")) d.style.display = "inline-block";
        };
        img.src = "vsv-logo.png";
      })(imgs[i]);
    }
    var lg = $("#hero-login"); if (lg) lg.href = botUrl("/info-login");
    // Registration is Telegram-only → point it at the bot chat (fallback: /register).
    var rg = $("#hero-register");
    if (rg) { rg.href = BOT_LINK || botUrl("/register"); if (BOT_LINK) { rg.target = "_blank"; rg.rel = "noopener"; } }
    // The extra «Telegram-бот» button is redundant now — hide it.
    var bl = $("#hero-bot"); if (bl) bl.style.display = "none";
  })();

  // Logout link (clears the local token).
  document.addEventListener("click", function (e) {
    var lo = e.target.closest ? e.target.closest(".lo") : null;
    if (lo) { e.preventDefault(); logout(); }
  });

  setConn("wait", "підключення…");
  boot();
})();
