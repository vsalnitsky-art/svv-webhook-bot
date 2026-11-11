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

  var SIGNAL_BADGES_JS = { "choch": "🟦 CHoCH", "choch_bos": "🟦 CHoCH+BOS", "vob_alert": "🟪 Volumized OB", "vob": "💰 Volumized OB", "mm_vob": "🧮 VOB з МММ-монітора", "liq_vob": "💧 VOB + Сканер ліквідності", "ob_alert": "🆕 Новий OB", "poc": "🎯 POC-сетап", "opp": "🔄 Реверс", "external": "🔌 Зовнішня", "manual": "✋ Ручний", "manual_ui": "✋ Ручний", "manual_ui_overflow": "✋ Ручний" };
  var ENGINE_BADGES_JS = { "Q1": "🎯 Черга-1", "Q2": "🎯 Черга-2", "Q3": "🎯 Готовність", "Q4": "🎯 Черга-4", "Q3-VOB(funding)": "💰 VOB+Шари", "POC-сетап": "🎯 POC-сетап", "MMM": "🧮 МММ-монітор", "MMN": "🧮 МММ-Trend", "MMC": "⚡ МММ-CTR", "MMF": "🔮 МММ-Local", "MMP": "🧭 МММ-Pull", "MMD": "🧠 МММ-Decision", "MMR": "🧩 МММ-Готовність", "MMS": "🔔 МММ-SBR", "XSPR": "🔭 Пружина-скан", "EXH": "🔥 Виснаженість", "FF": "🔥 FF", "direct": "⚡ Прямий" };
  var SIGNAL_ICON_JS = { "choch": "🟦", "choch_bos": "🟦", "vob_alert": "🟪", "vob": "💰", "mm_vob": "🧮", "liq_vob": "💧", "ob_alert": "🆕", "poc": "🎯", "opp": "🔄", "external": "🔌", "manual": "✋", "manual_ui": "✋", "manual_ui_overflow": "✋", "Q3-VOB(funding)": "💰", "POC-сетап": "🎯", "EXH": "🔥", "FF": "🔥", "MMM": "🧮", "MMN": "🧮", "MMC": "⚡", "MMF": "🔮", "MMP": "🧭", "MMD": "🧠", "MMR": "🧩", "MMS": "🔔", "XSPR": "🔭" };
  var ENGINE_ICON_OVERRIDE_JS = { "MMM": "🧮", "MMN": "🧮", "MMC": "⚡", "MMF": "🔮", "MMP": "🧭", "MMD": "🧠", "MMR": "🧩", "MMS": "🔔", "XSPR": "🔭" };
  
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
  function signalIconHtml(raw) {
    if (!raw) return "";
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
        '<p><b>👥 У Telegram-групі</b> (теми): <b>🧮 МММ-монітор</b> — зміни банерів МММ, ' +
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

  function refreshNotify() {
    api("/api/me/notify").then(renderNotify).catch(function () { renderNotify({ ok: false }); });
  }

  function renderAuth(me) {
    var el = $("#auth-links");
    if (!el) return;
    if (me && me.ok) {
      // ЗМІНА: Динамічна назва вкладки з ім'ям користувача
      var uname = me.email ? me.email.split('@')[0] : "Користувач";
      document.title = uname + " - Мій кабінет";
      
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
    return '<span class="bdg hb ' + cls + '" title="Premium / Discount · ' + pd.tf + '"><span class="d"></span>' +
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
    var tip = "POC — ціна з найбільшим обсягом торгів за останні " + poc.days + " доби · " + hint;
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
    if (s.none) return '<span class="bdg mute sbr-last" title="Сигналів SBR ' + T + ' немає">SBR ' + T + " —</span>";
    var lg = s.side === "LONG", m = sbrMove(s.side, s.trigger, price);
    var cls = !m || m.fav == null ? "mute" : (m.fav ? "long" : "short");
    var pct = m ? (m.pct >= 0 ? "+" : "") + m.pct.toFixed(2) + "%" : "—";
    var tip = "Останній сигнал SBR " + T + (s.n ? " #" + s.n : "") + " · спрацював " + (sct(s) ? locDM(sct(s)) + " " + locHM(sct(s)) : "") +
      " · Trigger " + fmtPrice(s.trigger) + (s.score != null ? " · Score " + Number(s.score).toFixed(1) + "%" : "") +
      " → зараз " + (price ? fmtPrice(price) : "—") + " · " + (!m || m.fav == null ? "без руху" : (m.fav ? "в бік" : "проти"));
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
    if (!host) return;
    var c = $("#iw-count"); if (c) c.textContent = IW.wl.length + " / " + IW.max;
    if (!IW.wl.length) {
      host.innerHTML = '<div class="iw-empty">Список порожній — додайте монету вище.</div>';
      return;
    }
    host.innerHTML = IW.wl.map(function (sym) {
      var r = IW.rows[sym] || {};
      var ext = r.src === "ext" ? '<small title="Поза WATCHLIST бота">🛰</small>' : "";
      var chg = r.chg24, bell = bellHTML(sym);
      var chgHtml = chg == null ? '<span class="iw-chg">—</span>'
        : '<span class="iw-chg ' + (chg > 0 ? "up" : (chg < 0 ? "dn" : "")) + '" title="Зміна ціни за ~24 год">' +
          (chg > 0 ? "+" : "") + Number(chg).toFixed(2) + "%</span>";
      return '<div class="iw-row' + (sym === IW.sel ? " sel" : "") + '" data-sym="' + esc(sym) + '">' +
        '<span class="iw-sym' + mmCls(sym) + '" title="' + esc(mmTip(sym)) + '">' + esc(sym.replace(/USDT$/, "")) + '<small>USDT</small>' + ext + "</span>" +
        '<span class="iw-price' + dirCls(sym) + '">' + (livePx(sym) ? fmtPrice(livePx(sym)) : "—") + "</span>" + chgHtml +
        bell.btn + '<button class="iw-del" data-del="' + esc(sym) + '" title="Прибрати зі списку">×</button>' + bell.menu +
        (r.err ? '<div class="iw-err" title="' + esc(r.err) + '">⚠ ' + esc(r.err) + "</div>" : "") + "</div>";
    }).join("");
  }

  function bellHTML(sym) {
    var tfs = IW.sbr[sym] || [];
    var on = tfs.length > 0;
    var lbl = on ? tfs.map(tfLbl).join("+") : "";
    var tip = on ? "SBR-сигнали " + lbl + " по " + sym + " приходять вам у Telegram." : "Отримувати сигнали SBR по " + sym + " у Telegram";
    var menu = IW.bellOpen === sym
      ? '<div class="iw-bell-menu" data-bellmenu="' + esc(sym) + '">' +
          '<div class="t">🔔 SBR у Telegram · ' + esc(sym.replace(/USDT$/, "")) + "</div>" +
          SBR_TFS.map(function (tf) {
            return '<label><input type="checkbox" data-belltf="' + tf + '"' + (tfs.indexOf(tf) >= 0 ? " checked" : "") + "> " + tfLbl(tf) + "</label>";
          }).join("") +
          (IW.tg_linked ? "" : '<div class="w">⚠️ Telegram не прив’язано — сигнали не дійдуть. <button class="tglink-btn" data-tglink="1">🔗 Привʼязати</button></div>') +
          '<div class="n">Сигнал приходить у ваш особистий чат із ботом після закриття бару.</div></div>'
      : "";
    return { btn: '<span class="iw-bell-wrap"><button type="button" class="iw-bell' + (on ? " on" : "") +
      '" data-bell="' + esc(sym) + '" title="' + esc(tip) + '">🔔' + (lbl ? "<small>" + esc(lbl) + "</small>" : "") + "</button></span>", menu: menu };
  }

  var TF_LAST_KEY = "vsv_tf_last";
  function tfLast(k) {
    var d = k === "ob_tf" ? "1h" : "5m", L = k === "ob_tf" ? OB_TFS : VOB_TFS;
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
      '<button type="button" data-layer="show_imb" class="ly-imb ' + (IW.show_imb ? "on" : "") + '" title="IMB">IMB</button>' +
      (sbrTfOk(IW.tf) ? '<button type="button" data-layer="show_sbr" class="ly-sbr ' + (IW.show_sbr ? "on" : "") + '" title="SBR">SBR</button>' : '<button type="button" disabled class="ly-sbr off" title="SBR — лише на 1H і 4H">SBR</button>') +
      '<button type="button" data-layer="show_shl" class="ly-shl ' + (IW.show_shl ? "on" : "") + '" title="High/Low">High/Low</button>';
    liqUi();
  }

  function applyMine(d) {
    if (!d) return;
    IW.wl = d.wl || []; IW.max = d.max || IW.max;
    var changed = iwKey({ sel: d.sel, tf: d.tf || IW.tf, ob_tf: d.ob_tf || IW.ob_tf, vob_tf: d.vob_tf || IW.vob_tf, poc_days: d.poc_days || IW.poc_days, poc_tf: d.poc_tf || IW.poc_tf }) !== iwKey();
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
    renderList(); renderTfBar();
    if (changed) { IW.sig = ""; IW.coin = null; loadCoin(true); }
  }

  function saveMine(body) {
    return apiPost("/api/me/watchlist", body).then(function (d) {
      if (d && d.error) iwMsg(d.error, true); else iwMsg("");
      applyMine(d); loadRows(); return d;
    }).catch(function () { iwMsg("не вдалося зберегти", true); });
  }

  function loadMine() { return api("/api/me/watchlist").then(function (d) { applyMine(d); return d; }); }

  function loadRows() {
    if (!IW.wl.length) return Promise.resolve();
    return api("/api/info/rows?symbols=" + encodeURIComponent(IW.wl.join(","))).then(function (d) {
      (d.rows || []).forEach(function (r) { IW.rows[r.symbol] = r; });
      renderList(); renderHead();
    }).catch(function () {});
  }

  function loadUniverse() {
    api("/api/info/universe").then(function (d) {
      IW.universe = d.symbols || [];
      var dl = $("#iw-universe");
      if (dl) dl.innerHTML = IW.universe.map(function (s) { return '<option value="' + esc(s.replace(/USDT$/, "")) + '">' + esc(s) + "</option>"; }).join("");
    }).catch(function () {});
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
    if (e.target.closest("[data-bellmenu]")) { e.stopPropagation(); return; }
    if (IW.bellOpen) { IW.bellOpen = ""; renderList(); }
  }, true);

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
      if (IW.coin) drawCoin(IW.coin, false);
      var b = {}; b[k] = IW[k]; saveMine(b); return;
    }
    var del = e.target.closest ? e.target.closest("[data-del]") : null;
    if (del) { e.stopPropagation(); saveMine({ remove: del.getAttribute("data-del") }); return; }
    var row = e.target.closest ? e.target.closest(".iw-row[data-sym]") : null;
    if (row) {
      var sym = row.getAttribute("data-sym");
      if (sym === IW.sel) return;
      IW.sel = sym; IW.sig = ""; IW.coin = null;
      renderList(); loadCoin(true); saveMine({ sel: sym });
    }
  });

  document.addEventListener("change", function (e) {
    var id = e.target && e.target.id;
    var MAP = { "iw-ob-tf": "ob_tf", "iw-vob-tf": "vob_tf", "iw-poc-days": "poc_days", "iw-poc-tf": "poc_tf" };
    var k = MAP[id]; if (!k) return;
    IW[k] = k === "poc_days" ? Number(e.target.value) : e.target.value; IW.sig = "";
    if (k === "ob_tf" || k === "vob_tf") tfLastSet(k, e.target.value);
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
        var css = Math.max(70, Math.min(190, (ts.width() || 600) * 0.2));
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
  function liqSrcLbl(d) { return d.src === "browser" ? '<span class="src" title="Дані з біржі взяв ваш браузер">🖥</span>' : '<span class="src" title="Дані взяв сервер">☁ ' + (Number(d.age) || 0) + "с</span>"; }
  function liqLoad() {
    var sym = IW.sel; if (!sym) return;
    var tok = ++LQ.seq;
    LQ.on = true; LQ.sym = sym; LQ.busy = tok; liqUi();
    liqStatus('<span class="wait">⏳ рахую ' + esc(sym) + "…</span>");
    var st = liqStepFromChart();
    liqFromBrowser(sym, st).then(null, function (e) { if (e && e.data) return { ok: false, reason: e.message }; return liqFromServer(sym, st); })
      .then(function (d) {
        if (LQ.busy !== tok || !LQ.on || LQ.sym !== IW.sel) return;
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

  function liqWidth(W, hr) { return Math.round(Math.max(70, Math.min(190, (W / hr) * 0.2)) * hr); }
  var LIQ_UP = "38,166,154", LIQ_DN = "239,83,80", LIQ_MAG = "245,158,11";
  function _liqPx(v) { var d = IW.prec != null ? IW.prec : 4; return Number(v).toFixed(d); }
  function _liqDraw(ctx, liq, sr, XR, W, H, hr, vr, font, dark, levels) {
    var rows = liq.rows, taken = [], px0 = XR + 3 * hr, px1 = W - 3 * hr, PW = px1 - px0, mx = 0;
    rows.forEach(function (r) { if (r.pct > mx) mx = r.pct; });
    if (mx <= 0 || PW <= 10 * hr) return taken;
    ctx.fillStyle = dark ? "rgba(13,17,26,.72)" : "rgba(248,250,252,.82)"; _rrect(ctx, px0, 2 * vr, PW, H - 4 * vr, 6 * vr); ctx.fill();
    ctx.strokeStyle = dark ? "rgba(56,189,248,.28)" : "rgba(14,116,144,.25)"; ctx.lineWidth = Math.max(1, Math.round(hr)); _rrect(ctx, px0, 2 * vr, PW, H - 4 * vr, 6 * vr); ctx.stroke();
    ctx.font = font(700, 10); ctx.textBaseline = "middle"; ctx.textAlign = "left"; ctx.fillStyle = dark ? "#7dd3fc" : "#0369a1";
    ctx.fillText("💧 " + String(liq.exchange || "").toUpperCase() + " · " + liqBarsLbl(liq.bars), px0 + 7 * hr, 12 * vr);
    var up = Number(liq.above_pct) || 0, dn = Number(liq.below_pct) || 0; ctx.textAlign = "right";
    var tDn = "▼" + dn.toFixed(0) + "%", wDn = ctx.measureText(tDn).width;
    ctx.fillStyle = "rgb(" + LIQ_DN + ")"; ctx.fillText(tDn, px1 - 6 * hr, 12 * vr);
    ctx.fillStyle = "rgb(" + LIQ_UP + ")"; ctx.fillText("▲" + up.toFixed(0) + "%", px1 - 6 * hr - wDn - 6 * hr, 12 * vr);
    var top = 22 * vr, bx0 = px0 + 6 * hr, bx1 = px1 - 6 * hr, BW = bx1 - bx0;
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
  BoxPrimitive.prototype.setLiq = function (liq) { this._liq = liq && liq.rows && liq.rows.length ? liq : null; if (this._req) this._req(); };
  BoxPrimitive.prototype.hitAt = function (x, y) {
    for (var i = this._hits.length - 1; i >= 0; i--) { var h = this._hits[i]; if (x >= h.x0 - 3 && x <= h.x1 + 3 && y >= h.y0 - 3 && y <= h.y1 + 3) return h.sig; }
    return null;
  };
  function _rrect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h); ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath(); }

  var IMB_SELL = "rgba(234,128,38,.92)", IMB_BUY = "rgba(38,166,154,.92)", IMB_MIN_PX = 4;
  function _imbBar(ctx, X1, w, y, bh, pct, color, hr, vr, font, dark) {
    if (pct == null || pct <= 0 || bh < 1) return;
    var sw = Math.max(2 * hr, w * Math.min(100, Math.max(0, pct)) / 100);
    ctx.fillStyle = color; ctx.fillRect(X1, y, sw, bh);
    ctx.strokeStyle = dark ? "rgba(0,0,0,.55)" : "rgba(0,0,0,.25)"; ctx.lineWidth = Math.max(1, Math.round(hr)); ctx.strokeRect(X1, y, sw, bh);
    if (bh < 9 * vr) return;
    var fp = Math.max(8, Math.min(32, bh / vr * 0.62)); ctx.font = font(600, fp); ctx.textBaseline = "middle"; ctx.fillStyle = dark ? "rgba(240,242,248,.96)" : "rgba(20,24,32,.9)";
    var t = pct + "%", tw = ctx.measureText(t).width;
    if (sw >= tw + 10 * hr) { ctx.textAlign = "right"; ctx.fillText(t, X1 + sw - 5 * hr, y + bh / 2); }
    else if (w - sw >= tw + 8 * hr) { ctx.textAlign = "left"; ctx.fillText(t, X1 + sw + 4 * hr, y + bh / 2); }
  }
  function _imbBody(ctx, b, X1, X2, yT, yB, hr, vr, font, dark, wms) {
    var w = X2 - X1, h = yB - yT;
    if (w > 4 * hr && (b.topPct != null || b.botPct != null)) {
      var bh = h / 2;
      _imbBar(ctx, X1, w, yT, bh, b.topPct, b.topCol || IMB_SELL, hr, vr, font, dark);
      _imbBar(ctx, X1, w, yT + bh, bh, b.botPct, b.botCol || IMB_BUY, hr, vr, font, dark);
    }
    if (b.volTxt && h >= 18 * vr && w > 40 * hr) {
      var wk = b.volTxt + "|" + Math.round(w) + "|" + Math.round(h), fv;
      if (b._wk === wk) { fv = b._fv; ctx.font = font(500, fv); }
      else { fv = Math.max(10, Math.min(64, h / vr * 0.82)); ctx.font = font(500, fv); while (fv > 10 && ctx.measureText(b.volTxt).width > w * 0.92) { fv -= 2; ctx.font = font(500, fv); } b._wk = wk; b._fv = fv; }
      var tw = ctx.measureText(b.volTxt).width, cx = (X1 + X2) / 2, cy = (yT + yB) / 2;
      var wm = { txt: b.volTxt, fv: fv, x: cx, y: cy + fv * vr * 0.04, r: { x0: cx - tw / 2, x1: cx + tw / 2, y0: cy - fv * vr * 0.45, y1: cy + fv * vr * 0.45 } };
      if (wms) wms.push(wm); else wmDraw(ctx, [wm], font, dark);
    }
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  }
  function wmDraw(ctx, wms, font, dark) {
    var done = [];
    for (var i = wms.length - 1; i >= 0; i--) {
      var w = wms[i], r = w.r;
      if (done.some(function (q) { return r.x0 < q.x1 && r.x1 > q.x0 && r.y0 < q.y1 && r.y1 > q.y0; })) continue;
      done.push(r); ctx.font = font(500, w.fv); ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = dark ? "rgba(4,6,12,.5)" : "rgba(0,0,0,.16)"; ctx.fillText(w.txt, w.x, w.y);
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
          placed.push({ b: b, X1: X1, X2: X2, yT: yT, yB: yBt });
        }
        var wmTaken = wmDraw(ctx, wms, font, dark);
        var bs = 18, marks = [];
        self._s.forEach(function (s) {
          var x = ts.timeToCoordinate(s.t);
          var y = sr.priceToCoordinate(s.side === "SHORT" ? s.hi : s.lo);
          if (x == null || y == null) return;
          var up = s.side === "SHORT";
          var cy = up ? y - 13 - bs / 2 : y + 13 + bs / 2;
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
          var s = m.s, x = m.x, cy = m.cy, x0 = m.x0, y0 = m.y0, col = s.side === "SHORT" ? "#f23645" : "#22c55e";
          _rrect(ctx, x0 * hr, y0 * vr, bs * hr, bs * vr, 4 * vr); ctx.fillStyle = col; ctx.fill();
          ctx.strokeStyle = dark ? "rgba(255,255,255,.35)" : "rgba(0,0,0,.25)"; ctx.lineWidth = Math.max(1, Math.round(hr)); ctx.stroke();
          ctx.beginPath();
          if (s.side === "SHORT") { ctx.moveTo((x - 4) * hr, (y0 + bs) * vr); ctx.lineTo((x + 4) * hr, (y0 + bs) * vr); ctx.lineTo(x * hr, (y0 + bs + 5) * vr); }
          else { ctx.moveTo((x - 4) * hr, y0 * vr); ctx.lineTo((x + 4) * hr, y0 * vr); ctx.lineTo(x * hr, (y0 - 5) * vr); }
          ctx.closePath(); ctx.fillStyle = col; ctx.fill();
          ctx.font = font(800, 11.5); ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
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
      var r = box.getBoundingClientRect(), psw = 0; try { psw = IW.chart.priceScale("right").width(); } catch (er) {}
      if (psw > 0 && e.clientX - r.left >= r.width - psw) { setAutoScale(true); }
      setTimeout(function () { var b = $("#iw-auto"); if (b) b.classList.toggle("on", autoScaleOn()); }, 0);
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

  function setFull(on) {
    var card = $("#iw-chart"), b = $("#iw-fs"); if (!card) return;
    on = !!on; card.classList.toggle("iw-full", on); document.body.classList.toggle("iw-full-on", on);
    if (b) { b.setAttribute("aria-pressed", on ? "true" : "false"); b.textContent = on ? "✕" : "⛶"; b.title = on ? "Згорнути (Esc)" : "На весь екран"; }
  }
  function isFull() { var c = $("#iw-chart"); return !!(c && c.classList.contains("iw-full")); }
  document.addEventListener("click", function (e) { var b = e.target.closest ? e.target.closest("#iw-fs") : null; if (b) { e.preventDefault(); setFull(!isFull()); } });
  document.addEventListener("keydown", function (e) { if ((e.key === "Escape" || e.key === "Esc") && isFull()) { setFull(false); } });

  function ensureChart() {
    if (IW.chart || !window.LightweightCharts) return !!IW.chart;
    var box = $("#iw-box"), dark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
    IW.chart = LightweightCharts.createChart(box, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: dark ? "#aab6c8" : "#3d4759", fontFamily: "-apple-system,Segoe UI,Roboto,sans-serif" },
      grid: { vertLines: { color: dark ? "rgba(38,50,74,.45)" : "rgba(211,218,230,.6)" }, horzLines: { color: dark ? "rgba(38,50,74,.45)" : "rgba(211,218,230,.6)" } },
      rightPriceScale: { borderColor: dark ? "#26324a" : "#d3dae6" },
      timeScale: { borderColor: dark ? "#26324a" : "#d3dae6", timeVisible: true, secondsVisible: false, rightOffset: 18, tickMarkFormatter: locTick },
      crosshair: { mode: 0 }, kineticScroll: { mouse: true, touch: true },
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
      handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: { time: true, price: true }, axisDoubleClickReset: { time: true, price: true } },
      localization: { locale: "uk-UA", timeFormatter: locFull },
    });
    IW.series = IW.chart.addCandlestickSeries({ upColor: "#c7ccd6", downColor: "#5b6475", borderVisible: true, borderUpColor: "#c7ccd6", borderDownColor: "#8b94a5", wickUpColor: "#c7ccd6", wickDownColor: "#8b94a5", priceFormat: { type: "price", precision: 6, minMove: 0.000001 } });
    if (!dark) IW.series.applyOptions({ upColor: "#26a69a", downColor: "#ef5350", borderUpColor: "#26a69a", borderDownColor: "#ef5350", wickUpColor: "#26a69a", wickDownColor: "#ef5350" });
    IW.prim = new BoxPrimitive(); IW.series.attachPrimitive(IW.prim); IW.pocLine = null;
    IW.chart.subscribeClick(function (p) { var s = p && p.point ? IW.prim.hitAt(p.point.x, p.point.y) : null; sigCard(s, p && p.point ? p.point.x : 0, p && p.point ? p.point.y : 0); });
    IW.chart.subscribeCrosshairMove(function (p) { var hit = p && p.point ? IW.prim.hitAt(p.point.x, p.point.y) : null; box.style.cursor = hit ? "pointer" : ""; });
    tvPanInit(box);

    if (window.VSV_PRO) { try { window.VSV_PRO.attach(IW); } catch (err) { console.error("[VSV_PRO]", err); } }
    return true;
  }

  function precisionOf(p) { p = Number(p) || 0; return p < 0.0001 ? 8 : p < 0.01 ? 6 : p < 1 ? 5 : p < 100 ? 4 : 2; }

  function boxesOf(c) {
    var o = c.ohlc || [], out = [];
    (c.vob || []).forEach(function (v) {
      var L = v.side === "LONG";
      out.push({ kind: "vob", t: snapSec(v.start, o), top: v.top, bottom: v.bottom, dashed: v.breaker,
        fill: L ? "rgba(34,171,94,.07)" : "rgba(242,54,69,.07)", stroke: L ? "rgba(34,197,94,.7)" : "rgba(242,54,69,.7)", text: L ? "#86efac" : "#fca5a5",
        volTxt: fmtVol(v.volume, 3), topPct: v.hi_pct, botPct: v.lo_pct,
        topCol: L ? "rgba(34,197,94,.45)" : "rgba(242,54,69,.45)", botCol: L ? "rgba(34,197,94,.18)" : "rgba(242,54,69,.18)",
        label: tfWord(v.tf || c.vob_tf) + " VOB" + (v.breaker ? " ✕" : ""),
        sub: [fmtVol(v.volume, 3), v.hi_pct != null ? Math.round(Math.max(v.hi_pct, v.lo_pct)) + "%" : (v.pct != null ? v.pct + "%" : "")].filter(Boolean).join(" · ") });
    });
    (c.ob || []).forEach(function (b) {
      var L = b.side === "LONG", sw = b.fam === "swing";
      out.push({ kind: "ob", t: snapSec(b.start, o), top: b.top, bottom: b.bottom, dashed: true,
        fill: L ? (sw ? "rgba(45,212,191,.12)" : "rgba(45,212,191,.06)") : (sw ? "rgba(251,146,60,.12)" : "rgba(251,146,60,.06)"),
        stroke: L ? "rgba(45,212,191,.8)" : "rgba(251,146,60,.8)", text: L ? "#5eead4" : "#fdba74",
        label: "OB " + tfLbl(c.ob_tf), sub: (sw ? "swing" : "int") + (b.tag ? " · " + b.tag : "") });
    });
    if (IW.show_imb) (c.imb || []).forEach(function (z) {
      var L = z.side === "LONG";
      out.push({ kind: "imb", thin: true, ext: 30, t: snapSec(z.start, o), top: z.top, bottom: z.bottom,
        fill: L ? "rgba(38,166,154,.22)" : "rgba(160,92,40,.30)", stroke: L ? "rgba(38,166,154,.6)" : "rgba(234,128,38,.6)", text: L ? "#5eead4" : "#fdba74",
        volTxt: fmtVol(z.vol, 3), topPct: z.sell_pct, botPct: z.buy_pct, topCol: IMB_SELL, botCol: IMB_BUY, label: "IMB" });
    });
    return out;
  }

  function sbrOf(c) {
    if (!IW.show_sbr || !sbrTfOk(c.tf || IW.tf)) return [];
    var o = c.ohlc || [], byT = {}, out = [];
    o.forEach(function (b) { byT[b.time] = b; });
    (c.sbr || []).forEach(function (s) { var bar = byT[s.t]; if (!bar) return; out.push(Object.assign({ hi: bar.high, lo: bar.low, close: bar.close }, s)); });
    return out;
  }

  function sigCard(s, x, y) {
    var box = $("#iw-box"), el = $("#iw-sig");
    if (!box) return;
    if (!el) { el = document.createElement("div"); el.id = "iw-sig"; el.className = "iw-sig"; box.appendChild(el); sigDragInit(el, box); }
    if (!s) { el.style.display = "none"; IW.sigSel = null; sigArrow(); return; }
    IW.sigSel = { t: s.t, side: s.side };
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
    el.classList.toggle("is-short", S); el.classList.toggle("is-long", !S); el.style.display = "block";
    var bw = box.clientWidth, bh = box.clientHeight, ew = el.offsetWidth, eh = el.offsetHeight;
    var left = Math.min(Math.max(8, x + 14), bw - ew - 8), top = Math.min(Math.max(8, y - eh / 2), bh - eh - 8);
    el.style.left = left + "px"; el.style.top = top + "px"; sigArrowLoop();
  }

  function sigAnchor() {
    var sel = IW.sigSel, hits = (IW.prim && IW.prim._hits) || []; if (!sel) return null;
    for (var i = 0; i < hits.length; i++) {
      var h = hits[i], g = h.sig || {};
      if (g.t === sel.t && g.side === sel.side) return { x: (h.x0 + h.x1) / 2, y: (h.y0 + h.y1) / 2, r: (h.x1 - h.x0) / 2 };
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
    var S = IW.sigSel && IW.sigSel.side === "SHORT"; svg.classList.toggle("short", S); svg.classList.toggle("long", !S);
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

  function structOf(c) {
    var o = c.ohlc || [], lines = [], marks = []; if (!o.length) return { lines: lines, marks: marks };
    var first = o[0].time;
    (c.events || []).forEach(function (e) {
      if (!e.to || e.to < first) return;
      var bull = e.dir === "bull", col = bull ? "#22c55e" : "#ef4444", t2 = snapSec(e.to, o);
      lines.push({ t1: snapSec(e.from, o), t2: t2, price: e.level, color: col, dashed: e.tag !== "BOS", text: e.tag, up: bull });
    });
    marks.sort(function (a, b) { return a.time - b.time; });
    return { lines: lines, marks: marks };
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

  function renderHead() {
    var h = $("#iw-head"); if (!h) return;
    var sym = IW.sel; if (!sym) { h.innerHTML = '<span class="iw-title">—</span>'; return; }
    var c = IW.coin || {}, px = livePx(sym);
    h.innerHTML = '<span class="iw-title' + mmCls(sym) + '" title="' + esc(mmTip(sym)) + '">' + tvSym(sym) + "</span>" +
      '<span class="iw-px' + dirCls(sym) + '">' + (px ? fmtPrice(px) : "—") + "</span>" + pdBadge(c.pd) + pocBadge(c.poc, c.poc_state) +
      sbrLastBadge("1h", (c.sbr_last || {})["1h"], px) + sbrLastBadge("4h", (c.sbr_last || {})["4h"], px) +
      (c.src === "ext" ? '<span class="bdg warn" title="Поза WATCHLIST бота">🛰 поза ботом</span>' : "") +
      '<span class="iw-legend"><span title="VOB"><i style="background:rgba(34,197,94,.55)"></i>VOB ' + tfLbl(IW.vob_tf) + "</span>" +
      '<span title="OB"><i style="background:rgba(45,212,191,.35);outline:1px dashed rgba(45,212,191,.9)"></i>OB ' + tfLbl(IW.ob_tf) + "</span>" +
      '<span title="CHoCH / BOS"><i style="background:#22c55e;height:2px"></i>CHoCH/BOS</span>' +
      '<span title="IMB"><i style="background:rgba(234,128,38,.6)"></i>IMB</span>' +
      '<span title="SBR"><i style="background:#f23645;width:9px"></i><i style="background:#22c55e;width:9px;margin-left:-3px"></i>SBR</span>' +
      '<span title="POC"><i style="background:#f5f5f5;height:2px"></i>POC</span>' +
      '<span title="High/Low"><i style="background:#f87171;height:2px;width:7px"></i><i style="background:#4ade80;height:2px;width:7px;margin-left:-3px"></i>High/Low</span></span>';
  }

  function setNote(t) { var n = $("#iw-note"); if (n) { n.textContent = t || ""; n.style.display = t ? "" : "none"; } }
  function setWait(c) {
    var w = $("#iw-tf-wait"); if (!w) return;
    var parts = []; if (c && c.ob_wait) parts.push("OB " + tfLbl(c.ob_tf)); if (c && c.vob_wait) parts.push("VOB " + tfLbl(c.vob_tf));
    w.innerHTML = parts.length ? '<span class="wait">⏳ рахується: ' + esc(parts.join(", ")) + "</span>" : "";
  }

  function drawCoin(c, fit) {
    if (!ensureChart()) { setNote("графічна бібліотека не завантажилась"); return; }
    var o = c.ohlc || [], key = c.symbol + "|" + c.tf, prec = precisionOf(o.length ? o[o.length - 1].close : 0);
    if (IW.prec !== prec) { IW.series.applyOptions({ priceFormat: { type: "price", precision: prec, minMove: Math.pow(10, -prec) } }); IW.prec = prec; }
    var ttl = String(c.symbol || "");
    if (IW.seriesTitle !== ttl) { IW.series.applyOptions({ title: ttl }); IW.seriesTitle = ttl; }
    var j = -1;
    if (!fit && IW.chartKey === key && IW.lastT != null && o.length) { for (var q = o.length - 1; q >= Math.max(0, o.length - 4); q--) { if (o[q].time === IW.lastT) { j = q; break; } } }
    if (j >= 0) { for (var u = j; u < o.length; u++) IW.series.update(o[u]); } else IW.series.setData(o);
    IW.lastT = o.length ? o[o.length - 1].time : null;
    var st = structOf(c); IW.series.setMarkers(st.marks);
    var lv = [];
    if (c.poc && c.poc.poc) lv.push({ price: Number(c.poc.poc), title: "POC", color: "#f5f5f5" });
    if (c.shl && IW.show_shl) ["high", "low"].forEach(function (k) { var v = c.shl[k]; if (v && v.price > 0) lv.push({ price: Number(v.price), title: (v.label || "") + " " + (c.shl.tf || ""), color: k === "high" ? "#f87171" : "#4ade80" }); });
    IW.prim.set(boxesOf(c), st.lines, sbrOf(c), lv); setPoc(c.poc); setShl(c.shl);
    if (fit || IW.chartKey !== key) { setAutoScale(true); IW.chart.timeScale().fitContent(); IW.chartKey = key; }
    setNote(o.length ? "" : (c.reason || "немає даних")); setWait(c);
    
    if (window.VSV_PRO) { try { window.VSV_PRO.onData(IW, c); } catch (err) { console.error("[VSV_PRO]", err); } }
  }

  var coinBusy = false;
  function loadCoin(force, keepBars) {
    var sym = IW.sel;
    if (!sym) { setNote("оберіть монету зі списку"); renderHead(); return; }
    if (coinBusy && !force && !keepBars) return;
    coinBusy = true;
    if (force) {
      setNote("завантаження " + sym + " · " + tfLbl(IW.tf) + "…");
      if (IW.series) { IW.series.setData([]); IW.series.setMarkers([]); IW.prim.set([], [], []); setPoc(null); setShl(null); IW.lastT = null; }
      sigCard(null); if (LQ && LQ.on && LQ.sym !== sym) liqOff();
    }
    var want = iwKey();
    var q = "/api/info/coin?symbol=" + encodeURIComponent(sym) + "&tf=" + IW.tf + "&ob_tf=" + IW.ob_tf + "&vob_tf=" + IW.vob_tf + "&poc_days=" + IW.poc_days + "&poc_tf=" + IW.poc_tf + (IW.sig && !force ? "&sig=" + encodeURIComponent(IW.sig) : "");
    api(q).then(function (c) {
      if (want !== iwKey() || c.same) return;
      IW.coin = c; IW.sig = c.sig || "";
      renderHead(); renderList(); drawCoin(c, force);
      if (!c.ok || c.ob_wait || c.vob_wait) setTimeout(function () { if (want === iwKey()) loadCoin(false, true); }, 4000);
    }).catch(function () { setNote("не вдалося завантажити графік"); })
      .then(function () { coinBusy = false; });
  }

  document.addEventListener("click", function (e) {
    if (!e.target.closest || !e.target.closest("#theme-btn") || !IW.chart) return;
    setTimeout(function () {
      try { IW.chart.remove(); } catch (x) {}
      IW.chart = null; IW.series = null; IW.prim = null; IW.pocLine = null; IW.chartKey = "";
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
  function ppUsd(v) { v = Number(v || 0); return ppAmt((v >= 0 ? "+" : "−") + "$" + Math.abs(v).toFixed(2)); }
  function ppMoney(v) { return ppAmt("$" + Number(v || 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })); }
  function ppCls(v) { v = Number(v || 0); return v > 0 ? "pp-up" : (v < 0 ? "pp-dn" : ""); }
  function ppPct(v, dp) { v = Number(v || 0); return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(dp == null ? 2 : dp) + "%"; }

  function ppSync() {
    var P = PP.period, today = tcalYmd(new Date());
    document.querySelectorAll("#pp-period button[data-p]").forEach(function (b) { b.classList.toggle("on", b.getAttribute("data-p") === P.mode); });
    $("#pp-day-wrap").style.display = P.mode === "day" ? "inline-flex" : "none";
    $("#pp-range-wrap").style.display = P.mode === "range" ? "inline-flex" : "none";
    [["#pp-day", P.day], ["#pp-from", P.from], ["#pp-to", P.to]].forEach(function (x) { var el = $(x[0]); if (el) { el.dataset.ymd = x[1] || today; el.value = tcalFmt(x[1] || today); } });
    var eb = $("#pp-amt"); if (eb) eb.textContent = PP.shown ? "🙈 Сховати суми" : "👁 Показати суми";
  }
  function ppSave() { try { localStorage.setItem(PP_KEY, JSON.stringify(PP.period)); } catch (e) {} }

  function ppCard(l, v, sub, vc, tip) { return '<div class="pp-kpi" title="' + esc(tip || "") + '"><div class="pp-kpi-l">' + l + '</div><div class="pp-kpi-v ' + (vc || "") + '">' + v + "</div>" + (sub ? '<div class="pp-kpi-s">' + sub + "</div>" : "") + "</div>"; }
  function ppNum(v) { var n = Number(v || 0); return n % 1 ? n.toFixed(2) : String(n); }
  function ppInlineHTML(a) {
    a = a || {};
    var usd = function (v) { return '<b class="ppi-usd">$' + ppNum(v) + "</b>"; }, pct = function (v) { return '<b class="ppi-pct">' + ppNum(v) + "%</b>"; };
    var deal = a.mode === "pct" && a.margin_pct != null ? pct(a.margin_pct) + ' <span class="ppi-l">від equity</span> ' + usd(a.equity != null ? a.equity : a.balance) + ' <span class="ppi-l">=</span> ' + usd(a.margin_usd) : usd(a.margin_usd);
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
  var PP_ALERT = { ok: ["✓ OK", "ok"], against: ["⚠ ПРОТИ", "against"], pre: ["⚑ ДО ВХОДУ", "pre"] };
  function ppTimer(t) { var s = Math.max(0, Math.floor(Date.now() / 1000 - t)), h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); var p = function (n) { return String(n).padStart(2, "0"); }; return (h >= 24 ? Math.floor(h / 24) + "д " + p(h % 24) : p(h)) + ":" + p(m) + ":" + p(s % 60); }
  function exhColor(x) { x = Math.max(0, Math.min(100, Number(x) || 0)); return x >= 100 ? "#f472b6" : (x >= 75 ? "#fb923c" : (x >= 50 ? "#facc15" : "#60a5fa")); }
  function ppExh(p) {
    var e = p.exh, f = p.spring_fire, fd = f && f.fired;
    if (!e) return '<span class="muted">—</span>';
    var pct = Math.max(0, Math.min(100, Number(e.pct) || 0)), col = exhColor(pct), sg = function (x) { return (x >= 0 ? "+" : "") + Number(x).toFixed(2) + "%"; };
    var lab = '<b style="color:' + col + '">' + Math.round(pct) + "%</b>" + (e.climax ? '<b style="color:#f472b6">💥</b>' : "") + (fd && fd.side !== p.side ? '<b class="pp-blink" style="color:#f87171">⚠</b>' : "");
    return '<span class="pp-exh" title=""><i><b style="width:' + pct + "%;background-size:" + (pct > 0 ? 10000 / pct : 100) + '% 100%"></b></i>' + lab + (p.exh_trail && p.exh_trail.stop_px != null ? ' <b class="pp-exh-st" title="🎯 Кульмінацію зловлено">🎯 ' + fmtPrice(p.exh_trail.stop_px).replace("$", "") + "</b>" : "") + "</span>";
  }
  function ppLvl(v, done) { return !v ? '<span class="muted">—</span>' : fmtPrice(v).replace("$", "") + (done ? ' <span class="pp-up">✓</span>' : ""); }
  function ppPnlCols(pc, usd) { return '<span class="pp-pnl"><span class="pp-pnl-p">' + ppPct(pc) + "</span>" + (usd != null ? '<span class="pp-pnl-s">·</span><span class="pp-pnl-u"><b class="pp-usd ' + (Number(usd) >= 0 ? "up" : "dn") + '">' + ppUsd(usd) + "</b></span>" : "") + "</span>"; }
  function ppRenderOpen(d) {
    var rows = d.positions || [], tb = $("#pp-open tbody"); $("#pp-open-n").textContent = rows.length;
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="11" class="muted">Відкритих угод немає</td></tr>'; return; }
    tb.innerHTML = rows.map(function (p) {
      var pc = Number(p.pnl_pct || 0), n = p.pnl_net, z = p.paper_size || {}, cl = (n ? n.net_usd : pc) >= 0 ? "pp-up" : "pp-dn";
      var a = p.mmd && p.mmd.alert, al = a && PP_ALERT[a.level];
      return "<tr><td class=\"sym\">" + tvSym(p.symbol) + '</td><td class="' + (p.side === "LONG" ? "pp-long" : "pp-short") + '">' + esc(p.side) + "</td>" +
        '<td class="r">' + fmtPrice(p.entry_price) + '</td><td class="r">' + fmtPrice(p.current_price) + "</td>" +
        '<td class="r ' + cl + '">' + ppPnlCols(pc, n ? n.net_usd : null) + "</td>" + "<td>" + ppExh(p) + "</td>" +
        "<td>" + (al ? '<span class="pp-al ' + al[1] + '">' + al[0] + (a.level === "against" && a.tf ? " " + esc(a.tf) : "") + "</span>" : '<span class="muted">—</span>') + "</td>" +
        '<td class="r">' + ppLvl(p.manual_sl) + '</td><td class="r">' + ppLvl(p.manual_tp1, p.tp1_done) + '</td><td class="r">' + ppLvl(p.manual_tp) + "</td>" +
        '<td class="r pp-timer" data-since="' + Math.floor(p.opened_at || 0) + '">' + (p.opened_at ? ppTimer(p.opened_at) : "—") + "</td></tr>";
    }).join("");
  }
  var PP_REASON = { manual: "✋ вручну", stop_loss: "🛡 Stop Loss", take_profit: "🎯 Take Profit", manual_sl: "🛡 Manual SL", manual_tp: "🎯 Manual TP-2", pilot_target: "🎯 автопілот", breakeven: "🟰 беззбиток", sbr_exit: "🔔 протилежний SBR", sbr_trail_exit: "🔔⏳ трейл SBR", opposite_ob_exit: "🧱 протилежний OB", mm_flat_exit: "🧮 МММ ⚖", mm_against_exit: "🧮 МММ проти" };
  function ppRenderClosed(d) {
    var rows = d.closed || [], tb = $("#pp-closed tbody"); $("#pp-closed-n").textContent = rows.length;
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="7" class="muted">Закритих угод немає</td></tr>'; return; }
    tb.innerHTML = rows.map(function (c) {
      var pc = Number(c.pnl_pct || 0), t = new Date((c.closed_at || 0) * 1000), p2 = function (n) { return String(n).padStart(2, "0"); };
      return "<tr><td>" + p2(t.getDate()) + "." + p2(t.getMonth() + 1) + " " + p2(t.getHours()) + ":" + p2(t.getMinutes()) + "</td>" +
        '<td class="sym">' + tvSym(c.symbol) + '</td><td class="' + (c.side === "LONG" ? "pp-long" : "pp-short") + '">' + esc(c.side) + "</td>" +
        '<td class="r">' + fmtPrice(c.entry_price) + '</td><td class="r">' + fmtPrice(c.exit_price) + "</td>" +
        '<td class="r ' + ppCls(c.net_usd != null ? c.net_usd : pc) + '">' + ppPnlCols(pc, c.net_usd) + "</td>" +
        '<td style="font-family:var(--font)">' + esc(PP_REASON[c.reason] || c.reason || "—") + "</td></tr>";
    }).join("");
  }
  function ppClosedApply() { var on = false; try { on = localStorage.getItem(PP_CL) === "1"; } catch (e) {} $("#pp-closed-body").style.display = on ? "" : "none"; $("#pp-closed-head").setAttribute("aria-expanded", on ? "true" : "false"); }
  function loadPaper() {
    var card = $("#pp-card"); if (!card || PP.busy) return;
    var r = ppRange(PP.period); if (!r) return;
    var q = r.from != null ? "?from=" + r.from + "&to=" + r.to : "";
    PP.busy = true;
    api("/api/info/paper" + q).then(function (d) {
      PP.busy = false; if (!d || !d.ok) { card.style.display = "none"; return; }
      card.style.display = ""; var cs = (d.closed || []).length + "|" + ((d.closed || [])[0] || {}).closed_at;
      if (cs !== PP.closedSig) { PP.closedSig = cs; PP.daysAt = 0; }
      PP.last = d; PP.label = r.label; ppSync(); ppClosedApply(); $("#pp-note").textContent = r.label;
      var sa = d.account || {}; $("#pp-inline").innerHTML = ppInlineHTML(sa); ppRenderKpis(d, r.label); ppRenderOpen(d); ppRenderClosed(d);
    }, function () { PP.busy = false; });
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

  var REC = { dir: "all", rows: [], meta: {} };
  try { var _rd = localStorage.getItem("vsv_rec_dir"); if (_rd) REC.dir = _rd; } catch (e) {}
  function recSide(r) { return r.side === "LONG" || r.side === "SHORT" ? r.side : ""; }
  function recPct(v) { v = Number(v); if (!isFinite(v)) return ""; return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2) + "%"; }
  function rs(w, html, cls) { return '<span class="rs ' + (cls || "") + '" style="width:' + w + 'px">' + html + "</span>"; }
  function recChCol(c) { return c >= 85 ? "#fb923c" : c >= 65 ? "#facc15" : "#a78bfa"; }
  function recPath(r, up) {
    var sl = Number(r.sl), tp = Number(r.tp), px = Number(r.price);
    if (!(sl > 0 && tp > 0 && px > 0)) return rs(120, '<span class="muted" title="' + esc(r.sl_label || "") + '">' + (r.sl_wait ? "⏳" : "—") + "</span>", "c") + rs(44, "");
    var lo = Math.min(sl, tp), hi = Math.max(sl, tp), pos = hi > lo ? (px - lo) / (hi - lo) * 100 : 50;
    pos = Math.max(0, Math.min(100, pos)); var p0 = pos.toFixed(0);
    var grad = up ? "linear-gradient(90deg,#ef4444,#f59e0b " + (pos / 2).toFixed(0) + "%,#64748b " + p0 + "%,#22c55e)" : "linear-gradient(90deg,#22c55e,#64748b " + p0 + "%,#f59e0b " + ((pos + 100) / 2).toFixed(0) + "%,#ef4444)";
    var rr = Number(r.rr), rrc = !isFinite(rr) ? "" : rr >= 1.5 ? "rr-good" : rr >= 1 ? "rr-mid" : "rr-bad";
    return '<span class="rec-path" title="Ризик ' + recPct(r.sl_pct) + " · хід " + recPct(r.tp_pct) + (isFinite(rr) ? " · R " + rr.toFixed(2) : "") + '"><span class="bar" style="background:' + grad + '">' +
      '<i style="left:' + pos.toFixed(1) + '%"></i></span></span>' + rs(44, isFinite(rr) ? '<b class="rr ' + rrc + '">' + rr.toFixed(2) + "R</b>" : "", "r");
  }
  function recRow(r) {
    var sd = recSide(r), up = sd === "LONG", dc = sd ? (up ? "up" : "dn") : "";
    var prob = r.prob_long != null ? (up ? Number(r.prob_long) : 100 - Number(r.prob_long)) : null;
    var ch = Math.max(0, Math.min(100, Math.round(Number(r.charge) || 0))), chIn = Number(r.charge_in), chD = isFinite(chIn) && chIn > 0 ? ch - Math.round(chIn) : null;
    var icon = r.src === "wl" ? '<span class="rec-ic" title="WATCHLIST бота">🌀</span>' : '<span class="rec-ic" title="Скан біржі">🔭</span>';
    var nb = r.bybit === false ? ' <span class="rec-nb" title="Монети немає на Bybit">⚠</span>' : "";
    var chCell = '<span class="rec-ch"><i style="width:' + ch + "%;background:linear-gradient(90deg,#a78bfa," + recChCol(ch) + ')"></i></span>' + rs(26, '<b style="color:' + recChCol(ch) + '">' + ch + "</b>", "r") + rs(30, chD ? '<small class="' + (chD > 0 ? "pp-up" : "pp-dn") + '">' + (chD > 0 ? "▲" : "▼") + Math.abs(chD) + "</small>" : "", "l");
    var pr = prob != null && isFinite(prob) ? prob : null;
    var prCell = pr == null ? rs(100, "—", "r") : '<span class="rec-prob ' + dc + '"><i style="width:' + Math.max(0, Math.min(100, pr)).toFixed(0) + '%"></i></span>' + rs(40, '<b class="rec-' + dc + '">' + pr.toFixed(0) + "%</b>", "r");
    var slCell = r.sl > 0 ? '<span class="rec-lv sl">' + rs(96, fmtPrice(r.sl), "r") + rs(64, "<small>" + recPct(r.sl_pct) + "</small>", "r") + rs(30, '<small class="rec-tf">' + esc(String(r.sl_tf || "").toUpperCase()) + "</small>", "r") + "</span>" : rs(190, "—", "r");
    var tpCell = r.tp > 0 ? '<span class="rec-lv tp">' + rs(96, fmtPrice(r.tp), "r") + rs(64, "<small>" + recPct(r.tp_pct) + "</small>", "r") + "</span>" : rs(160, "—", "r");
    return '<tr class="rec-row ' + dc + '"><td class="sym">' + icon + tvSym(r.symbol) + nb + "</td>" +
      '<td><span class="rec-dir ' + dc + '">' + (sd ? (up ? "▲ LONG" : "▼ SHORT") : "⚖") + "</span></td>" +
      '<td class="r">' + prCell + "</td>" + '<td class="r">' + slCell + "</td>" + '<td class="r">' + tpCell + "</td>" +
      '<td class="c">' + recPath(r, up) + "</td>" + '<td class="r">' + chCell + "</td></tr>";
  }
  function recRender() {
    var rows = REC.rows || [], n = { all: rows.length, LONG: 0, SHORT: 0 };
    rows.forEach(function (r) { var s = recSide(r); if (s) n[s]++; });
    var lbl = { all: "Всі", LONG: "🟢 LONG", SHORT: "🔴 SHORT" };
    document.querySelectorAll("#rec-tabs button[data-d]").forEach(function (b) { var d = b.getAttribute("data-d"); b.textContent = lbl[d] + " (" + n[d] + ")"; b.classList.toggle("on", d === REC.dir); });
    var vis = rows.filter(function (r) { return REC.dir === "all" || recSide(r) === REC.dir; }), tb = $("#rec-tbl tbody");
    if (tb) tb.innerHTML = vis.length ? vis.map(recRow).join("") : '<tr><td colspan="7" class="muted">' + (rows.length ? "Немає монет у цій вкладці" : "Зараз немає рекомендованих монет") + "</td></tr>";
    var rn = $("#rec-note"); if (rn) rn.textContent = new Date().toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" });
  }
  function loadRec() {
    var card = $("#rec-card"); if (!card) return;
    api("/api/info/recommended").then(function (d) { if (!d || !d.ok) { card.style.display = "none"; return; } card.style.display = ""; REC.rows = d.rows || []; REC.meta = d; recRender(); }, function () {});
  }
  document.addEventListener("click", function (e) {
    var b = e.target.closest ? e.target.closest("#rec-tabs button[data-d]") : null; if (!b) return;
    REC.dir = b.getAttribute("data-d"); try { localStorage.setItem("vsv_rec_dir", REC.dir); } catch (x) {} recRender();
  });

  var DENY_TXT = {
    blocked: "⛔ Ваш акаунт заблоковано адміністратором — доступ до кабінету закрито.",
    expired: "⏳ Строк вашого доступу завершився — зверніться до адміністратора для продовження.",
    not_active: "⏳ Акаунт очікує схвалення адміністратора."
  };
  function showData(ok, code) {
    document.body.classList.add("app-ready");
    var hero = $("#auth-hero"); if (hero) hero.style.display = ok ? "none" : "";
    var dn = $("#auth-deny"); if (dn) { dn.textContent = (!ok && DENY_TXT[code]) || ""; dn.style.display = (!ok && DENY_TXT[code]) ? "" : "none"; }
    var lay = $("#iw-layout"); if (lay) lay.style.display = ok ? "" : "none";
    if (!ok) { var n = $("#notify-card"); if (n) n.style.display = "none"; }
    if (!ok) { var pc = $("#pp-card"); if (pc) pc.style.display = "none"; }
    if (!ok) { var rc = $("#rec-card"); if (rc) rc.style.display = "none"; }
    var tb = $(".topbar"); if (tb) tb.style.display = ok ? "" : "none";
  }

  var started = false;
  function boot() {
    loadMine().then(function () {
      showData(true); setConn("ok", "онлайн");
      if (!started) {
        started = true; loadUniverse(); loadRows(); loadCoin(true); loadPaper(); loadRec();
        setInterval(function () { if (!document.hidden) loadRec(); }, 30000);
        setInterval(function () { if (!document.hidden) loadPaper(); }, 5000);
        document.addEventListener("visibilitychange", function () { if (!document.hidden) loadPaper(); });
        setInterval(function () { if (!document.hidden) loadRows(); }, ROWS_MS);
        setInterval(function () { if (!document.hidden) loadCoin(false); }, COIN_MS);
        setInterval(function () { if (!document.hidden) { refreshAuth(); refreshNotify(); } }, 60000);
      }
      var lu = $("#last-update"); if (lu) lu.textContent = "оновлено: " + new Date().toLocaleTimeString("uk-UA");
    }).catch(function (e) {
      if (e && (e.status === 401 || e.status === 403)) {
        var deny = e.status === 403 && DENY_TXT[e.code];
        if (getTok()) setTok("");
        showData(false, deny ? e.code : "");
        setConn(deny ? "err" : "wait", deny ? "доступ закрито" : "потрібен вхід");
      } else { showData(true); setConn("err", "бот недоступний"); setTimeout(boot, 10000); }
    });
    refreshAuth(); refreshNotify();
  }

  (function fixChrome() {
    var imgs = document.querySelectorAll("img.logo, img.hero-logo");
    for (var i = 0; i < imgs.length; i++) {
      (function (img) {
        img.onerror = function () {
          if (!img._triedBot && BASE) { img._triedBot = true; img.src = BASE + "/favicon.ico"; return; }
          img.onerror = null; img.style.display = "none";
          var d = img.nextElementSibling; if (d && d.classList && d.classList.contains("dot")) d.style.display = "inline-block";
        };
        img.src = "logo.png";
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
  window.VSV_HOOK = { get IW() { return IW; } };
})();