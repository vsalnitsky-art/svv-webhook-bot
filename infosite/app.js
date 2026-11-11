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

  function api(path) {
    return fetch(BASE + path, { headers: _headers(), credentials: CREDS, cache: "no-store" })
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

  function tvSym(sym) {
    var s = String(sym || "").toUpperCase();
    var url = "https://ru.tradingview.com/chart/?symbol=BYBIT:" + s + ".P";
    return '<a href="' + url + '" target="tvchart" rel="noopener" ' +
      'title="Відкрити BYBIT:' + s + '.P (ф’ючерс) у TradingView" ' +
      'style="color:inherit;text-decoration:none;border-bottom:1px dotted rgba(255,255,255,0.4)">' +
      esc(s) + "</a>";
  }

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
    "MMM": "🧮 МММ-монітор", "MMN": "🧮 МММ-Trend", "MMC": "⚡ МММ-CTR", "MMF": "🔮 МММ-Local", "MMP": "🧭 МММ-Pull", "MMD": "🧠 МММ-Decision", "MMR": "🧩 МММ-Готовність", "MMS": "🔔 МММ-SBR", "XSPR": "🔭 Пружина-скан",
    "EXH": "🔥 Виснаженість", "FF": "🔥 FF", "direct": "⚡ Прямий"
  };
  var SIGNAL_ICON_JS = {
    "choch": "🟦", "choch_bos": "🟦", "vob_alert": "🟪", "vob": "💰",
    "mm_vob": "🧮", "liq_vob": "💧",
    "ob_alert": "🆕",
    "poc": "🎯", "opp": "🔄", "external": "🔌", "manual": "✋",
    "manual_ui": "✋", "manual_ui_overflow": "✋",
    "Q3-VOB(funding)": "💰", "POC-сетап": "🎯", "EXH": "🔥", "FF": "🔥",
    "MMM": "🧮", "MMN": "🧮", "MMC": "⚡", "MMF": "🔮", "MMP": "🧭", "MMD": "🧠", "MMR": "🧩", "MMS": "🔔", "XSPR": "🔭"
  };
  var ENGINE_ICON_OVERRIDE_JS = { "MMM": "🧮", "MMN": "🧮", "MMC": "⚡", "MMF": "🔮", "MMP": "🧭", "MMD": "🧠", "MMR": "🧩", "MMS": "🔔", "XSPR": "🔭" };
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

  function toggleRow(key, label, on, disabled) {
    return '<div class="ntf-row">' +
      '<span class="ntf-lbl">' + esc(label) + '</span>' +
      '<button class="switch' + (on ? " on" : "") + '" data-ntf="' + key + '"' +
        (disabled ? " disabled" : "") + ' aria-pressed="' + (on ? "true" : "false") + '">' +
        '<span class="knob"></span></button>' +
    '</div>';
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
    var linkA = BOT_LINK
      ? ' <a href="' + esc(BOT_LINK) + '" target="_blank" rel="noopener">відкрити бота</a>' : "";
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
        '<p><b>👥 У Telegram-групі</b> (теми): <b>🧮 МММ-монітор</b> — зміни банерів МММ, ' +
        '<b>🔔 Сигнали</b> — SBR 1H/4H по основних монетах, <b>📈 Угоди</b> — відкриття й закриття угод бота. ' +
        'Група — лише для читання.</p>' +
        '<p>Доступ до групи — <b>після реєстрації та схвалення адміністратором</b>; посилання прийде автоматично. ' +
        'Якщо доступ закрито (вимкнено / строк минув) — доступу до групи немає, ' +
        'а особисто бот нічого не надсилає. Видалений акаунт прибирається з групи повністю.</p>' +
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
      btn.classList.toggle("on", !next);
    });
  });

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

  function refreshNotify() {
    api("/api/me/notify").then(renderNotify).catch(function () { renderNotify({ ok: false }); });
  }

  function renderAuth(me) {
    var el = $("#auth-links");
    if (!el) return;
    if (me && me.ok) {
      el.innerHTML =
        '<a class="who" href="' + botUrl("/cabinet?from=info") + '" title="Мій кабінет">👤 ' +
          esc(me.email || "") + (me.is_admin ? " · адмін" : "") + '</a>' +
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

  var ROWS_MS = 15000, COIN_MS = 15000;
  var CHART_TFS = ["15m", "30m", "1h", "4h", "1d"];
  var OB_TFS = ["15m", "30m", "1h", "4h"];
  var VOB_TFS = ["5m", "15m", "30m", "1h", "4h"];
  var POC_DAYS = [1, 2, 3, 5, 7, 14, 30];
  var POC_TFS = ["15m", "1h", "4h"];
  var IW = { wl: [], sel: "", rows: {}, max: 20, sig: "", coin: null, chart: null, series: null,
             prim: null, pocLine: null, chartKey: "", universe: [],
             tf: "15m", ob_tf: "1h", vob_tf: "5m", poc_days: 7, poc_tf: "1h", show_imb: true, show_sbr: true, show_shl: true,
             sbr: {}, tg_linked: true, bellOpen: "" };
  var SBR_TFS = ["1h", "4h"];
  function iwKey(o) { o = o || IW; return [o.sel, o.tf, o.ob_tf, o.vob_tf, o.poc_days, o.poc_tf].join("|"); }
  function sbrTfOk(tf) { return SBR_TFS.indexOf(String(tf || "").toLowerCase()) >= 0; }

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
    if (!pd) return '<span class="bdg mute" title="PD-зона ще не порахована">PD —</span>';
    var name = pd.zone === "discount" ? "Discount" : (pd.zone === "premium" ? "Premium" : "Equilibrium");
    var cls = pd.zone === "discount" ? "long" : (pd.zone === "premium" ? "short" : "mute");
    var tip = "Premium / Discount · " + pd.tf;
    return '<span class="bdg hb ' + cls + '" title="' + esc(tip) + '"><span class="d"></span>' +
      '<span class="hb-z">' + name + '</span><span class="hb-v">' + Math.round(pd.pct) + '%</span>' +
      '<span class="hb-tf">' + esc(pd.tf) + "</span></span>";
  }

  function pocBadge(poc, state) {
    if (!poc) {
      var t = state === "none" ? "POC немає даних" : "POC ⏳";
      return '<span class="bdg mute" title="POC рахується">' + t + "</span>";
    }
    var d = poc.dist_pct, rel = poc.rel;
    var cls = rel === "above" ? "short" : (rel === "below" ? "long" : "mute");
    var hint = rel === "above" ? "краще SHORT" : (rel === "below" ? "краще LONG" : "на POC");
    var tip = "POC — ціна з найбільшим обсягом торгів за останні " + poc.days + " доби (свічки " + poc.tf +
      ", " + String(poc.exchange || "binance").toUpperCase() + " " + poc.market + ") · " + hint;
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
    if (!s) return '<span class="bdg mute sbr-last" title="SBR ' + T + ' рахується">SBR ' + T + " ⏳</span>";
    if (s.none) return '<span class="bdg mute sbr-last" title="Сигналів SBR ' + T + ' за останні бари немає">SBR ' + T + " —</span>";
    var lg = s.side === "LONG", m = sbrMove(s.side, s.trigger, price);
    var cls = !m || m.fav == null ? "mute" : (m.fav ? "long" : "short");
    var pct = m ? (m.pct >= 0 ? "+" : "") + m.pct.toFixed(2) + "%" : "—";
    var tip = "Останній сигнал SBR " + T + (s.n ? " #" + s.n : "") + " · спрацював " + (sct(s) ? locDM(sct(s)) + " " + locHM(sct(s)) : "") +
      " · Trigger " + fmtPrice(s.trigger) + (s.score != null ? " · Score " + Number(s.score).toFixed(1) + "%" : "") +
      " → зараз " + (price ? fmtPrice(price) : "—") + " · " +
      (!m || m.fav == null ? "без руху" : (m.fav ? "ціна йде В БІК сигналу" : "ціна йде ПРОТИ сигналу"));
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
    if (!m) return "МММ-бабло: даних немає";
    if (m.dir === "LONG") return "МММ-бабло ↑ зверху → тягне в LONG · " + m.str + "%";
    if (m.dir === "SHORT") return "МММ-бабло ↓ знизу → тягне в SHORT · " + m.str + "%";
    return "МММ-бабло ⚖ рівновага — напрямку немає · " + m.str + "%";
  }

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
        ? '<small title="Поза WATCHLIST бота">🛰</small>' : "";
      var c = r.chg24, bell = bellHTML(sym);
      var chg = c == null ? '<span class="iw-chg">—</span>'
        : '<span class="iw-chg ' + (c > 0 ? "up" : (c < 0 ? "dn" : "")) + '" title="Зміна ціни за ~24 год">' +
          (c > 0 ? "+" : "") + Number(c).toFixed(2) + "%</span>";
      return '<div class="iw-row' + (sym === IW.sel ? " sel" : "") + '" data-sym="' + esc(sym) + '">' +
        '<span class="iw-sym' + mmCls(sym) + '" title="' + esc(mmTip(sym)) + '">' + esc(sym.replace(/USDT$/, "")) + '<small>USDT</small>' + ext + "</span>" +
        '<span class="iw-price' + dirCls(sym) + '">' + (livePx(sym) ? fmtPrice(livePx(sym)) : "—") + "</span>" + chg +
        bell.btn +
        '<button class="iw-del" data-del="' + esc(sym) + '" title="Прибрати зі списку">×</button>' + bell.menu +
        (r.err ? '<div class="iw-err" title="' + esc(r.err) + '">⚠ ' + esc(r.err) + "</div>" : "") +
      "</div>";
    }).join("");
  }

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
          (IW.tg_linked ? "" : '<div class="w">⚠️ Telegram не прив’язано — сигнали не дійдуть. <button class="tglink-btn" data-tglink="1">🔗 Привʼязати</button></div>') +
          '<div class="n">Сигнал приходить у ваш особистий чат із ботом одразу після закриття бару.</div>' +
        "</div>"
      : "";
    return { btn: '<span class="iw-bell-wrap"><button type="button" class="iw-bell' + (on ? " on" : "") +
      '" data-bell="' + esc(sym) + '" title="' + esc(tip) + '">🔔' + (lbl ? "<small>" + esc(lbl) + "</small>" : "") +
      "</button></span>", menu: menu };
  }

  var TF_LAST_KEY = "vsv_tf_last";
  function tfLast(k) {
    var d = k === "ob_tf" ? "1h" : "5m", L = k === "ob_tf" ? OB_TFS : VOB_TFS;
    try { var v = (JSON.parse(localStorage.getItem(TF_LAST_KEY) || "{}") || {})[k]; if (L.indexOf(v) >= 0) return v; } catch (e) {}
    return d;
  }
  function tfLastSet(k, v) {
    try { var o = JSON.parse(localStorage.getItem(TF_LAST_KEY) || "{}") || {}; o[k] = v;
      localStorage.setItem(TF_LAST_KEY, JSON.stringify(o)); } catch (e) {}
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
      return list.map(function (tf) {
        return '<option value="' + tf + '"' + (tf === cur ? " selected" : "") + ">" + tfLbl(tf) + "</option>";
      }).join("");
    };
    var o = $("#iw-ob-tf"); if (o) o.innerHTML = opt(OB_TFS, IW.ob_tf);
    var v = $("#iw-vob-tf"); if (v) v.innerHTML = opt(VOB_TFS, IW.vob_tf);
    var pd = $("#iw-poc-days");
    if (pd) pd.innerHTML = POC_DAYS.map(function (n) {
      return '<option value="' + n + '"' + (n === Number(IW.poc_days) ? " selected" : "") + ">" + n + " дн</option>";
    }).join("");
    var pt = $("#iw-poc-tf");
    if (pt) pt.innerHTML = POC_TFS.map(function (tf) {
      return '<option value="' + tf + '"' + (tf === IW.poc_tf ? " selected" : "") + ">" + tfLbl(tf) + "</option>";
    }).join("");
    var lay = $("#iw-layers");
    if (lay) lay.innerHTML =
      '<button type="button" data-layer="show_imb" class="ly-imb ' + (IW.show_imb ? "on" : "") +
        '" title="IMB">IMB</button>' +
      (sbrTfOk(IW.tf)
        ? '<button type="button" data-layer="show_sbr" class="ly-sbr ' + (IW.show_sbr ? "on" : "") +
          '" title="SBR · ' + tfLbl(IW.tf) + '">SBR</button>'
        : '<button type="button" disabled class="ly-sbr off" title="SBR — лише на 1H і 4H">SBR</button>') +
      '<button type="button" data-layer="show_shl" class="ly-shl ' + (IW.show_shl ? "on" : "") +
        '" title="Swing High / Low (як у боті)">High/Low</button>';
    liqUi();
  }

  function applyMine(d) {
    if (!d) return;
    IW.wl = d.wl || [];
    IW.max = d.max || IW.max;
    var changed = iwKey({ sel: d.sel, tf: d.tf || IW.tf, ob_tf: d.ob_tf || IW.ob_tf, vob_tf: d.vob_tf || IW.vob_tf,
                          poc_days: d.poc_days || IW.poc_days, poc_tf: d.poc_tf || IW.poc_tf }) !== iwKey();
    IW.sel = d.sel || "";
    if (d.tf) IW.tf = d.tf;
    if (d.ob_tf) IW.ob_tf = d.ob_tf;
    if (d.vob_tf) IW.vob_tf = d.vob_tf;
    if (d.poc_days) IW.poc_days = Number(d.poc_days);
    if (d.poc_tf) IW.poc_tf = d.poc_tf;
    if (typeof d.show_imb === "boolean") IW.show_imb = d.show_imb;
    if (typeof d.show_sbr === "boolean") IW.show_sbr = d.show_sbr;
    if (typeof d.show_shl === "boolean") IW.show_shl = d.show_shl;
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
      renderHead();
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
    var tg = e.target.closest ? e.target.closest("[data-tgl]") : null;
    if (tg) {
      var tk = tg.getAttribute("data-tgl");
      if (IW[tk] === "off") IW[tk] = tfLast(tk);
      else { tfLastSet(tk, IW[tk]); IW[tk] = "off"; }
      IW.sig = ""; renderTfBar(); loadCoin(false, true);
      var tb = {}; tb[tk] = IW[tk]; saveMine(tb);
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
    var MAP = { "iw-ob-tf": "ob_tf", "iw-vob-tf": "vob_tf", "iw-poc-days": "poc_days", "iw-poc-tf": "poc_tf" };
    var k =