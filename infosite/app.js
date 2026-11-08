// ─────────────────────────────────────────────────────────────────────────
//  VSV — особистий кабінет користувача (інфосайт).
//  Кожен користувач бачить СВІЙ список монет (зберігається в БД бота на його
//  акаунт) і графік обраної монети зі структурою OB 1H та Volumized OB.
//
//  API бота:
//    GET  /api/me                     — хто увійшов
//    GET  /api/me/watchlist           — мій список монет + обрана монета
//    POST /api/me/watchlist           — {add|remove|sel} (лише свій акаунт)
//    GET  /api/info/universe          — монети WATCHLIST бота (підказки)
//    GET  /api/info/rows?symbols=…    — PD-зона + POC по моїх монетах
//    GET  /api/info/coin?symbol=&sig= — графік: бари + OB + VOB + PD + POC
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
  function logout() { setTok(""); window.location.reload(); }

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
        if (!r.ok) { var e = new Error("HTTP " + r.status); e.status = r.status; throw e; }
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
    body.innerHTML =
      '<div class="ntf-info">' +
        '<p>Ринкові сигнали (₿ BTCUSDT, 💰 Funding, 🎯 рекомендації, 🚀 аномальний ріст, ' +
        'угоди) приходять у <b>Telegram-групу</b>, а не в приватний чат.</p>' +
        '<p>Доступ до групи — <b>після реєстрації та схвалення адміністратором</b>; ' +
        'посилання на вхід прийде автоматично.</p>' +
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
  var IW = { wl: [], sel: "", rows: {}, max: 40, sig: "", coin: null, chart: null, series: null,
             prim: null, chartSym: "", universe: [] };

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
    var tip = "POC " + poc.days + "д · " + poc.tf + " · " + poc.market + " — ціна найбільшого обсягу. " +
      "Ціна ВИЩЕ POC → краще SHORT, НИЖЧЕ → краще LONG. " + hint + ".";
    return '<span class="bdg ' + cls + '" title="' + esc(tip + " Ціна↔POC: " + (d == null ? "—" : d + "%")) + '">POC ' +
      fmtPrice(poc.poc).replace("$", "") +
      (d == null ? "" : " " + (d >= 0 ? "+" : "") + Number(d).toFixed(2) + "%") + "</span>";
  }

  function iwMsg(t, err) {
    var el = $("#iw-msg");
    if (el) { el.textContent = t || ""; el.className = "iw-msg" + (err ? " err" : ""); }
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
        ? '<small title="Монети немає у WATCHLIST бота — сайт рахує її окремо">🛰</small>' : "";
      var sub = r.err
        ? '<span class="bdg warn" title="' + esc(r.err) + '">⚠ ' + esc(r.err).slice(0, 40) + "</span>"
        : pdBadge(r.pd) + pocBadge(r.poc, r.poc_state);
      return '<div class="iw-row' + (sym === IW.sel ? " sel" : "") + '" data-sym="' + esc(sym) + '">' +
        '<span class="iw-sym">' + esc(sym.replace(/USDT$/, "")) + '<small>USDT</small>' + ext + "</span>" +
        '<span class="iw-price">' + (r.price ? fmtPrice(r.price) : "—") + "</span>" +
        '<button class="iw-del" data-del="' + esc(sym) + '" title="Прибрати зі списку">×</button>' +
        '<div class="iw-sub">' + sub + "</div>" +
      "</div>";
    }).join("");
  }

  function applyMine(d) {
    if (!d) return;
    IW.wl = d.wl || [];
    IW.max = d.max || IW.max;
    var changed = (d.sel || "") !== IW.sel;
    IW.sel = d.sel || "";
    renderList();
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

  // ═══ 📈 ГРАФІК: бари + бокси OB 1H і Volumized OB ═════════════════════════
  // Примітив малює прямокутники від свічки блоку до правого краю (як LuxAlgo /
  // «Volumized Order Blocks» у TradingView).
  function BoxPrimitive() { this._b = []; this._chart = null; this._series = null; this._req = null; }
  BoxPrimitive.prototype.attached = function (p) { this._chart = p.chart; this._series = p.series; this._req = p.requestUpdate; };
  BoxPrimitive.prototype.detached = function () { this._chart = null; this._series = null; this._req = null; };
  BoxPrimitive.prototype.set = function (list) { this._b = list || []; if (this._req) this._req(); };
  BoxPrimitive.prototype.paneViews = function () {
    var self = this;
    return [{ renderer: function () { return { draw: function (target) {
      if (!self._chart || !self._series) return;
      for (var i = self._b.length - 1; i >= 0; i--) {
        var b = self._b[i];
        var x1 = self._chart.timeScale().timeToCoordinate(b.t);
        var yA = self._series.priceToCoordinate(b.top), yB = self._series.priceToCoordinate(b.bottom);
        if (x1 == null || yA == null || yB == null) continue;
        target.useBitmapCoordinateSpace(function (sc) {
          var ctx = sc.context, hr = sc.horizontalPixelRatio, vr = sc.verticalPixelRatio;
          var X1 = Math.max(0, x1 * hr), X2 = sc.bitmapSize.width;
          var yT = Math.min(yA, yB) * vr, yBt = Math.max(yA, yB) * vr;
          if (X2 <= X1 || yBt - yT <= 0) return;
          ctx.fillStyle = b.fill; ctx.fillRect(X1, yT, X2 - X1, yBt - yT);
          ctx.strokeStyle = b.stroke; ctx.lineWidth = Math.max(1, Math.round(1.2 * hr));
          if (b.dashed && ctx.setLineDash) ctx.setLineDash([6 * hr, 4 * hr]);
          ctx.strokeRect(X1, yT, X2 - X1, yBt - yT);
          if (ctx.setLineDash) ctx.setLineDash([]);
          if (b.label) {
            ctx.font = "700 " + Math.round(10.5 * vr) + "px -apple-system,Segoe UI,Roboto,sans-serif";
            ctx.fillStyle = b.text; ctx.textAlign = "right"; ctx.textBaseline = "middle";
            ctx.fillText(b.label, X2 - 10 * hr, (yT + yBt) / 2);
            ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
          }
        });
      }
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

  function cssVar(n, d) {
    try { return getComputedStyle(document.documentElement).getPropertyValue(n).trim() || d; } catch (e) { return d; }
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
      timeScale: { borderColor: dark ? "#26324a" : "#d3dae6", timeVisible: true, secondsVisible: false },
      crosshair: { mode: 0 },
      // Мова явно: деякі браузери віддають тег на кшталт «en-US@posix», на
      // якому бібліотека падає і графік лишається порожнім.
      localization: { locale: "uk-UA" },
    });
    IW.series = IW.chart.addCandlestickSeries({
      upColor: "#2fd18b", downColor: "#ff5d6c", borderVisible: false,
      wickUpColor: "#2fd18b", wickDownColor: "#ff5d6c",
      priceFormat: { type: "price", precision: 6, minMove: 0.000001 },
    });
    IW.prim = new BoxPrimitive();
    IW.series.attachPrimitive(IW.prim);
    return true;
  }

  function precisionOf(p) {
    p = Number(p) || 0;
    return p < 0.0001 ? 8 : p < 0.01 ? 6 : p < 1 ? 5 : p < 100 ? 4 : 2;
  }

  function boxesOf(c) {
    var o = c.ohlc || [], out = [];
    (c.vob || []).forEach(function (v) {
      var L = v.side === "LONG";
      out.push({ t: snapSec(v.start, o), top: v.top, bottom: v.bottom, dashed: v.breaker,
        fill: L ? "rgba(47,209,139,.16)" : "rgba(255,93,108,.16)",
        stroke: L ? "rgba(47,209,139,.85)" : "rgba(255,93,108,.85)",
        text: L ? "#6ee7b7" : "#ff9aa4",
        label: "VOB " + (v.tf || "") + (v.breaker ? " ✕" : "") + (v.pct != null ? " · " + v.pct + "%" : "") });
    });
    (c.ob || []).forEach(function (b) {
      var L = b.side === "LONG", sw = b.fam === "swing";
      out.push({ t: snapSec(b.start, o), top: b.top, bottom: b.bottom, dashed: true,
        fill: L ? (sw ? "rgba(45,212,191,.14)" : "rgba(45,212,191,.07)") : (sw ? "rgba(251,146,60,.14)" : "rgba(251,146,60,.07)"),
        stroke: L ? "rgba(45,212,191,.8)" : "rgba(251,146,60,.8)",
        text: L ? "#5eead4" : "#fdba74",
        label: "OB " + (c.ob_tf || "") + " " + (sw ? "swing" : "internal") + (b.tag ? " · " + b.tag : "") });
    });
    return out;
  }

  function renderHead() {
    var h = $("#iw-head");
    if (!h) return;
    var sym = IW.sel;
    if (!sym) { h.innerHTML = '<span class="iw-title">—</span>'; return; }
    var c = IW.coin || {}, r = IW.rows[sym] || {};
    var pd = c.pd || r.pd, poc = c.poc || r.poc;
    var o = c.ohlc || [], px = o.length ? o[o.length - 1].close : r.price;
    h.innerHTML =
      '<span class="iw-title">' + tvSym(sym) + "</span>" +
      (c.tf ? '<span class="iw-tf">' + esc(c.tf) + "</span>" : "") +
      '<span class="iw-px">' + (px ? fmtPrice(px) : "—") + "</span>" +
      pdBadge(pd) + pocBadge(poc, r.poc_state) +
      (c.src === "ext" ? '<span class="bdg warn" title="Монети немає у WATCHLIST бота — сайт рахує її окремо тими самими алгоритмами">🛰 поза ботом</span>' : "") +
      '<span class="iw-legend">' +
        '<span title="Volumized Order Block — суцільна рамка; пунктир із ✕ — блок уже пробитий (breaker)"><i style="background:rgba(47,209,139,.5)"></i>VOB</span>' +
        '<span title="Order Block ' + esc(c.ob_tf || "1H") + ' — пунктир; густіший — swing, світліший — internal"><i style="background:rgba(45,212,191,.35);outline:1px dashed rgba(45,212,191,.9)"></i>OB ' + esc(c.ob_tf || "1H") + "</span>" +
      "</span>";
  }

  function setNote(t) {
    var n = $("#iw-note");
    if (!n) return;
    n.textContent = t || "";
    n.style.display = t ? "" : "none";
  }

  function drawCoin(c, fit) {
    if (!ensureChart()) { setNote("графічна бібліотека не завантажилась"); return; }
    var o = c.ohlc || [];
    var prec = precisionOf(o.length ? o[o.length - 1].close : 0);
    IW.series.applyOptions({ priceFormat: { type: "price", precision: prec, minMove: Math.pow(10, -prec) } });
    IW.series.setData(o);
    IW.prim.set(boxesOf(c));
    if (fit || IW.chartSym !== c.symbol) {
      IW.chart.timeScale().fitContent();
      IW.chartSym = c.symbol;
    }
    setNote(o.length ? "" : (c.reason || "немає даних"));
  }

  var coinBusy = false;
  function loadCoin(force) {
    var sym = IW.sel;
    if (!sym) { setNote("оберіть монету зі списку"); renderHead(); return; }
    if (coinBusy && !force) return;
    coinBusy = true;
    if (force) { setNote("завантаження " + sym + "…"); if (IW.series) { IW.series.setData([]); IW.prim.set([]); } }
    var q = "/api/info/coin?symbol=" + encodeURIComponent(sym) + (IW.sig && !force ? "&sig=" + encodeURIComponent(IW.sig) : "");
    api(q).then(function (c) {
      if (sym !== IW.sel) return;              // поки чекали — обрали іншу монету
      if (c.same) {
        if (IW.coin) { IW.coin.pd = c.pd; IW.coin.poc = c.poc; }
        renderHead();
        return;
      }
      IW.coin = c; IW.sig = c.sig || "";
      renderHead();
      drawCoin(c, force);
      // монета ще рахується (поза ботом) — перепитуємо швидше
      if (!c.ok) setTimeout(function () { if (sym === IW.sel) loadCoin(false); }, 4000);
    }).catch(function () { setNote("не вдалося завантажити графік"); })
      .then(function () { coinBusy = false; });
  }

  // Тема змінилась → перемалювати кольори графіка.
  document.addEventListener("click", function (e) {
    if (!e.target.closest || !e.target.closest("#theme-btn") || !IW.chart) return;
    setTimeout(function () {
      try { IW.chart.remove(); } catch (x) {}
      IW.chart = null; IW.series = null; IW.prim = null; IW.chartSym = "";
      if (IW.coin) drawCoin(IW.coin, true);
    }, 30);
  });

  // ═══ показ / автооновлення ═════════════════════════════════════════════
  function showData(ok) {
    document.body.classList.add("app-ready");
    var hero = $("#auth-hero"); if (hero) hero.style.display = ok ? "none" : "";
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
        if (getTok()) setTok("");
        showData(false);
        setConn("wait", "потрібен вхід");
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
