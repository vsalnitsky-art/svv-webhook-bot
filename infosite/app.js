// ─────────────────────────────────────────────────────────────────────────
//  VSV Bot — інфо-панель. ТІЛЬКИ читає JSON API бота й малює дані.
//  Жодних дій, що змінюють стан бота (ніяких POST на керування).
//
//  API бота, які використовує сайт (усі — read-only GET):
//    /api/health             — живий бот чи ні
//    /api/stats              — головна інфопанель
//    /api/fuel-filter/state  — ₿ банер BTC + 💰 funding-таблиця + статус демона
//    /api/sm/auto-gate       — активний символ вердикту (sm_bias_symbol)
//    /api/sm/bias?symbol=…   — вікно потенціалу (verdict, forecast, move-potential)
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

  function dirCell(dir) {
    if (dir === "LONG") return '<span class="dir-long">🟢 LONG</span>';
    if (dir === "SHORT") return '<span class="dir-short">🔴 SHORT</span>';
    return '<span class="muted">⚪ WAIT</span>';
  }

  function fmtDur(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    if (h) return h + "г " + m + "хв";
    if (m) return m + "хв " + s + "с";
    return s + "с";
  }

  function hms(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    var p = function (n) { return (n < 10 ? "0" : "") + n; };
    return p(h) + ":" + p(m) + ":" + p(s);
  }

  function fmtCountdown(ms) {
    if (!ms) return '<span class="muted">—</span>';
    var diff = Math.floor((ms - Date.now()) / 1000);
    if (diff <= 0) return '<span class="muted">зараз</span>';
    return fmtDur(diff);
  }

  function fmtPct(v, digits) {
    if (v == null || isNaN(v)) return "—";
    return Number(v).toFixed(digits == null ? 1 : digits) + "%";
  }

  function fmtNum(v) {
    if (v == null || isNaN(v)) return "—";
    var n = Number(v);
    if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return String(n);
  }

  function fmtUsd(v, digits) {
    if (v == null || isNaN(v)) return "—";
    return (Number(v) >= 0 ? "" : "-") + "$" + Math.abs(Number(v)).toFixed(digits == null ? 2 : digits);
  }

  // Compact money: $9.77M / $1.2K — for 24h volume etc.
  function fmtUsdC(v) {
    if (v == null || isNaN(v)) return "—";
    return "$" + fmtNum(Math.abs(Number(v)));
  }

  // Price with thousands separators: $64,436.20
  function fmtPrice(v) {
    if (v == null || isNaN(v)) return "—";
    return "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function sideOf(v) { return v > 0 ? "LONG" : (v < 0 ? "SHORT" : "WAIT"); }

  function setConn(state, text) {
    var el = $("#conn");
    el.className = "conn conn--" + state;
    el.textContent = "● " + text;
  }

  // ── 🎯 Вікно потенціалу ────────────────────────────────────────────────────
  function fcLine(tf, side, conf) {
    if (!side) return '<div class="fc-line muted">Forecast ' + tf + ': —</div>';
    var cls = side === "LONG" ? "dir-long" : (side === "SHORT" ? "dir-short" : "muted");
    var dot = side === "LONG" ? "🟢" : (side === "SHORT" ? "🔴" : "⚪");
    var c = (conf != null && !isNaN(conf)) ? " · " + Math.round(conf) + "%" : "";
    return '<div class="fc-line">Forecast ' + tf + ': <span class="' + cls + '">' +
      dot + " " + esc(side) + c + "</span></div>";
  }

  function moveBlock(mv) {
    if (!mv || !mv.ok) return "";
    var vmap = {
      EXHAUSTED: ["🔴 ВИСНАЖЕНИЙ — запас малий", "exh-bad"],
      MATURE:    ["🟡 ЗРІЛИЙ — рухайся обережно", "exh-mid"],
      FRESH:     ["🟢 СВІЖИЙ — є запас ходу", "exh-good"]
    };
    var vd = vmap[mv.verdict] || ["—", "muted"];
    // exhaustion comes back on a 0..100 scale already.
    var ex = mv.exhaustion != null ? Math.max(0, Math.min(100, Number(mv.exhaustion))) : null;
    var exTxt = ex != null ? ex.toFixed(0) + "%" : "—";
    var exCls = ex == null ? "exh-mid" : (ex >= 70 ? "exh-bad" : (ex >= 40 ? "exh-mid" : "exh-good"));
    var dirWord = mv.side === "LONG" ? "↑ ВГОРУ (LONG)" : "↓ ВНИЗ (SHORT)";
    var notes = (mv.notes || []).map(function (n) { return "<li>" + esc(n) + "</li>"; }).join("");
    return '' +
      '<div class="pot-sub">' +
        '<div class="pot-sub-title">📊 Потенціал ' + esc(mv.side || "") + '</div>' +
        '<div class="verdict-badge ' + vd[1] + '">' + esc(vd[0]) + '</div>' +
      '</div>' +
      '<div class="pot-dir">' + esc(dirWord) + '</div>' +
      '<div class="metrics">' +
        metric("ATR / бар", fmtPct(mv.atr_pct, 2)) +
        metric("Розтяг від середн.", (mv.stretch_atr != null ? Number(mv.stretch_atr).toFixed(1) + " ATR" : "—")) +
        metric("Запас ходу", (mv.runway_pct != null ? fmtPct(mv.runway_pct, 2) : "—") +
          (mv.runway_atr != null ? ' <span class="muted">(' + Number(mv.runway_atr).toFixed(1) + " ATR)</span>" : "")) +
        metric("Хід за день / середній", (mv.adr_used_pct != null ? Number(mv.adr_used_pct).toFixed(0) + "%" : "—") +
          (mv.adr_pct != null ? ' <span class="muted">від сер. ' + Number(mv.adr_pct).toFixed(2) + "%/добу</span>" : "")) +
      '</div>' +
      '<div class="exh-row">' +
        '<span class="exh-lbl">Виснаженість</span>' +
        '<div class="exh-bar"><i class="' + exCls + '" style="width:' + (ex || 0) + '%"></i></div>' +
        '<span class="exh-val ' + exCls + '">' + exTxt + '</span>' +
      '</div>' +
      (notes ? '<ul class="notes">' + notes + "</ul>" : "");
  }

  function metric(label, valHtml) {
    return '<div class="metric"><div class="metric-lbl">' + esc(label) + '</div>' +
      '<div class="metric-val">' + valHtml + "</div></div>";
  }

  function renderPotential(b) {
    var body = $("#potential-body");
    if (!b || b.ok === false) {
      body.innerHTML = '<div class="muted">дані потенціалу недоступні</div>';
      return;
    }
    $("#pot-symbol").textContent = b.symbol || "—";
    var verdict = b.verdict || "WAIT";
    var vcls = verdict === "LONG" ? "dir-long" : (verdict === "SHORT" ? "dir-short" : "muted");
    var conf = Math.max(0, Math.min(100, Math.round(b.confidence || 0)));
    var comp = b.components || {};
    var fco = comp.forecast || {};
    var fc1s = fco.f1_side != null ? sideOf(fco.f1_side) : null;
    var fc4s = fco.f4_side != null ? sideOf(fco.f4_side) : null;
    // ММ (sentiment) line.
    var sm = comp.sentiment;
    var mmLine = "";
    if (sm && sm.long_pct != null) {
      var mb = sm.bias === "LONG" ? "dir-long" : (sm.bias === "SHORT" ? "dir-short" : "muted");
      mmLine = '<div class="fc-line">ММ: <span class="' + mb + '">' + esc(sm.bias || "—") +
        '</span> <span class="muted">L ' + Number(sm.long_pct).toFixed(0) + '% / S ' +
        Number(sm.short_pct).toFixed(0) + '%</span></div>';
    } else if (verdict === "WAIT") {
      mmLine = '<div class="fc-line accent">ММ збалансований — напрямку немає</div>';
    }
    var mv = (b.move_long && b.verdict === "LONG") ? b.move_long
           : (b.move_short && b.verdict === "SHORT") ? b.move_short
           : (b.move || b.move_long || b.move_short);

    body.innerHTML =
      '<div class="pot-head">' +
        '<div class="pot-verdict ' + vcls + '">' + dirCell(verdict) + '</div>' +
        '<div class="conf-wrap"><div class="conf-bar"><i style="width:' + conf + '%"></i></div>' +
          '<span class="conf-txt">впевненість ' + conf + '%</span></div>' +
      '</div>' +
      '<div class="price-row"><span>💲 Ціна ' + esc(b.symbol || "") + ' (ф’ючерс)</span><b>' +
        fmtPrice(b.price) + '</b></div>' +
      '<div class="fc-list">' + fcLine("1H", fc1s, fco.f1_conf) + fcLine("4H", fc4s, fco.f4_conf) + mmLine + '</div>' +
      moveBlock(mv);
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
    if (!n || n.ok === false) { card.style.display = "none"; return; }
    card.style.display = "";
    var linked = !!n.tg_linked;
    var warn = linked ? "" :
      '<div class="ntf-warn">⚠️ Прив’яжіть Telegram (напишіть боту <b>/start</b>), ' +
      'щоб отримувати сповіщення' +
      (BOT_LINK ? ' — <a href="' + esc(BOT_LINK) + '" target="_blank" rel="noopener">відкрити бота</a>' : "") +
      '.</div>';
    body.innerHTML = warn +
      toggleRow("notify_btc", "₿ BTCUSDT — START / STOP / PAUSE", n.notify_btc, !linked) +
      toggleRow("notify_funding", "💰 Funding — поява монети з ММ", n.notify_funding, !linked);
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

  // ── ₿ BTCUSDT бар + перекидний годинник ─────────────────────────────────────
  var btcState = { held: 0, at: 0, period: 300, dir: null, status: "STOP", paused: false, strength: 0 };

  function flipClock(sec) {
    var str = hms(sec);
    return '<div class="flip">' + str.split("").map(function (ch) {
      if (ch === ":") return '<span class="flip-sep">:</span>';
      return '<span class="flip-digit">' + ch + "</span>";
    }).join("") + "</div>";
  }

  function renderBtcStatic() {
    var b = btcState;
    var box = $("#btc-bar");
    var dirTxt = dirCell(b.dir);
    var statusCls, statusTxt;
    if (b.paused) { statusCls = "st-pause"; statusTxt = "⏸ ПАУЗА"; }
    else if (b.status === "START") { statusCls = "st-start"; statusTxt = "🟢 START TRADING"; }
    else if (b.status === "STOP") { statusCls = "st-stop"; statusTxt = "⛔ STOP TRADING"; }
    else {
      var pct = b.period > 0 ? Math.min(100, Math.floor((nowHeld()) / b.period * 100)) : 0;
      statusCls = "st-count"; statusTxt = "⏳ " + pct + "%";
    }
    var str = Math.max(0, Math.min(100, Number(b.strength || 0)));
    var scol = str >= 60 ? "band-strong" : (str >= 30 ? "band-mid" : (str >= 10 ? "band-weak" : "band-none"));
    box.innerHTML =
      '<div class="btc-left">' +
        '<span class="btc-sym">₿ BTCUSDT</span> ' + dirTxt +
      '</div>' +
      '<div class="btc-strength"><div class="sbar"><i class="' + scol + '" style="width:' + str + '%"></i></div>' +
        '<span class="sbar-lbl">' + Math.round(str) + '% сила</span></div>' +
      '<div id="btc-clock">' + flipClock(nowHeld()) + '</div>' +
      '<div class="btc-status ' + statusCls + '">' + statusTxt + '</div>';
  }

  function nowHeld() {
    if (!btcState.at) return btcState.held;
    return btcState.held + Math.floor((Date.now() - btcState.at) / 1000);
  }

  function setBtc(b) {
    b = b || {};
    btcState = {
      held: b.held_sec || 0, at: Date.now(),
      period: b.period_sec || 300, dir: b.dir || null,
      status: b.status || "STOP", paused: !!b.paused, strength: b.strength || 0
    };
    renderBtcStatic();
  }

  // Local 1s tick so the flip clock advances smoothly between polls.
  setInterval(function () {
    var c = $("#btc-clock");
    if (c && (btcState.dir || btcState.status !== "STOP")) c.innerHTML = flipClock(nowHeld());
  }, 1000);

  // ── 💰 Funding ───────────────────────────────────────────────────────────────
  function mmCell(dir, str) {
    var dot = dir === "LONG" ? "🟢" : (dir === "SHORT" ? "🔴" : "⚪");
    var s = (str != null && !isNaN(str)) ? Math.round(str) + "%" : "—";
    return dot + " " + s;
  }

  function fundingProgress(rate, threshold) {
    if (rate == null) return '<span class="muted">—</span>';
    var thr = threshold != null ? Math.abs(threshold) : 4;   // entry→-4% default
    var pct = Math.max(0, Math.min(100, Math.abs(rate) / thr * 100));
    return '<div class="fbar"><i style="width:' + pct.toFixed(0) + '%"></i></div>';
  }

  function renderFunding(rowsArr) {
    rowsArr = rowsArr || [];
    $("#funding-count").textContent = rowsArr.length;
    var tb = $("#funding-table tbody");
    if (!rowsArr.length) {
      tb.innerHTML = '<tr><td colspan="7" class="muted">немає funding-монет</td></tr>';
      return;
    }
    tb.innerHTML = rowsArr.map(function (a) {
      var trend = "";
      if (a.funding_rate != null && a.funding_prev_rate != null) {
        trend = (Math.abs(a.funding_rate) >= Math.abs(a.funding_prev_rate))
          ? ' <span class="dir-short small">→ збільш.</span>'
          : ' <span class="dir-long small">← зменш.</span>';
      }
      var paused = a.paused ? ' <span class="pill small">⏸</span>' : "";
      return "<tr>" +
        "<td><b>" + esc(a.symbol) + "</b>" + paused + "</td>" +
        "<td>" + dirCell(a.dir) + "</td>" +
        "<td>" + mmCell(a.mm, a.mm_str) + "</td>" +
        "<td class=\"mono\">" + hms(a.held_sec) + "</td>" +
        "<td>" + fundingProgress(a.funding_rate, a.entry_threshold) + "</td>" +
        "<td><b class=\"dir-short\">" + (a.funding_rate != null ? fmtPct(a.funding_rate, 3) : "—") + "</b>" +
          ' <span class="muted">· ⏱ ' + fmtCountdown(a.funding_next_ms) + "</span>" + trend + "</td>" +
        "<td>" + fmtUsdC(a.vol24h) + "</td>" +
      "</tr>";
    }).join("");
  }

  // ── 👤 auth links (вхід / реєстрація / кабінет) ─────────────────────────────
  function renderAuth(me) {
    var el = $("#auth-links");
    if (!el) return;
    if (me && me.ok) {
      el.innerHTML =
        '<span class="who">👤 ' + esc(me.email || "") + (me.is_admin ? " · адмін" : "") + '</span>' +
        '<a href="' + botUrl("/cabinet") + '" target="_blank" rel="noopener">Кабінет</a>' +
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

  // ── poll loop ──────────────────────────────────────────────────────────────
  var potSymbol = FORCED_SYMBOL || "BTCUSDT";

  function refreshPotential() {
    var chain = FORCED_SYMBOL
      ? Promise.resolve(FORCED_SYMBOL)
      : api("/api/sm/auto-gate").then(function (g) {
          return (g && g.symbol) ? String(g.symbol).toUpperCase() : "BTCUSDT";
        }).catch(function () { return potSymbol; });
    return chain.then(function (sym) {
      potSymbol = sym;
      return api("/api/sm/bias?symbol=" + encodeURIComponent(sym))
        .then(renderPotential)
        .catch(function () { renderPotential({ ok: false }); });
    });
  }

  // Show/hide the data cards + the "no access" hero. On the auth screen the
  // top-bar labels (login/register links + connection pill) are hidden so the
  // hero is the only call-to-action.
  function showData(ok) {
    var hero = $("#auth-hero");
    if (hero) hero.style.display = ok ? "none" : "";
    ["potential-card", "btc-card", "funding-card"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.style.display = ok ? "" : "none";
    });
    if (!ok) { var n = $("#notify-card"); if (n) n.style.display = "none"; }
    // On the auth screen hide the whole top bar — the hero is the only UI.
    var tb = $(".topbar"); if (tb) tb.style.display = ok ? "" : "none";
  }

  function refresh() {
    var authed = !!getTok() || !!API_KEY;   // have a login token or a public key?
    api("/api/fuel-filter/state").then(function (st) {
      showData(true);
      if (st) { setBtc(st.btc_start); renderFunding(st.anomalies); }
      setConn("ok", "онлайн");
      $("#last-update").textContent = "оновлено: " + new Date().toLocaleTimeString("uk-UA");
      refreshPotential();
    }).catch(function (e) {
      if (!authed) {
        // A guest (no token) always gets the login/registration screen —
        // never a scary «бот недоступний».
        showData(false);
        setConn("wait", "потрібен вхід");
      } else if (e && (e.status === 401 || e.status === 403)) {
        // Token expired / invalid → drop it and show the login screen.
        setTok("");
        showData(false);
        setConn("wait", "сесія завершена — увійдіть");
      } else {
        setConn("err", "бот недоступний");   // logged in, but bot/network down
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
  refresh();
  setInterval(refresh, REFRESH);
})();
