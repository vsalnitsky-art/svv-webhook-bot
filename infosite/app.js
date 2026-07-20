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

  function fmtDur(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    if (h) return h + "г " + m + "хв";
    if (m) return m + "хв " + s + "с";
    return s + "с";
  }

  function hms(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    var d = Math.floor(sec / 86400);
    var h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    var p = function (n) { return (n < 10 ? "0" : "") + n; };
    var t = p(h) + ":" + p(m) + ":" + p(s);
    return d > 0 ? d + "д " + t : t;   // >24 год → «1д 06:51:36», як у боті
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

  // ⚡ CTR-15M line — same logic/look as the bot's CTR badge.
  function ctrLine(ctr) {
    if (!ctr || ctr.stc == null) return "";
    var stc = Number(ctr.stc);
    var revBias, col;
    if (stc > 50) { revBias = "🔴 SHORT-нахил"; col = "#fca5a5"; }
    else if (stc < 50) { revBias = "🟢 LONG-нахил"; col = "#86efac"; }
    else { revBias = "⚪ нейтрально"; col = "#aaa"; }
    var leanPct = Math.round(Math.abs(stc - 50) / 50 * 100);
    var tf = (ctr.tf && ctr.tf !== "scanner") ? "·" + String(ctr.tf).toUpperCase() : "-15M";
    var cross = "";
    if (ctr.last_dir) {
      var age = ctr.last_signal_age_bars;
      var ageStr = age == null ? "" : (age === 0 ? "now" : age === 1 ? "1 bar ago" : age + " bars ago");
      var stale = (age != null && age > 6);
      var cc = stale ? "#9aa3b5" : (ctr.last_dir === "LONG" ? "#86efac" : "#fca5a5");
      var ic = ctr.last_dir === "LONG" ? "🟢" : "🔴";
      cross = ' <span style="color:' + cc + '">' + ic + " " + esc(ctr.last_dir) +
        (ageStr ? " (" + ageStr + ")" : "") + (stale ? " ⚠️застарілий" : "") + "</span> ·";
    }
    var h1 = "";
    if (ctr.stc_1h != null) {
      var s1 = Number(ctr.stc_1h);
      var i1, c1;
      if (s1 > 50) { i1 = "🔴"; c1 = "#fca5a5"; }
      else if (s1 < 50) { i1 = "🟢"; c1 = "#86efac"; }
      else { i1 = "⚪"; c1 = "#aaa"; }
      var p1 = Math.round(Math.abs(s1 - 50) / 50 * 100);
      h1 = ' <span title="CTR 1H · STC ' + s1.toFixed(0) + '/100" style="color:' + c1 + '">· 1H ' + i1 + " " + p1 + "%</span>";
    }
    return '<div class="banner-ctr">⚡ CTR' + tf + cross +
      ' <span style="color:' + col + '">' + revBias + " " + leanPct + "%</span>" + h1 + "</div>";
  }

  // Colour attention/lean words in a rationale string:
  //  Premium (дорого→SHORT) red · Discount (дешево→LONG) green · Equilibrium grey
  //  ⚠ обмежено — warning amber · перекупленість red · перепроданість green.
  function colorizePD(s) {
    return String(s || "")
      .replace(/Premium/g, '<span style="color:#f87171;font-weight:800">Premium</span>')
      .replace(/Discount/g, '<span style="color:#4ade80;font-weight:800">Discount</span>')
      .replace(/Equilibrium/g, '<span style="color:#9aa3b5;font-weight:800">Equilibrium</span>')
      .replace(/⚠ обмежено/g, '<span style="color:#fbbf24;font-weight:800">⚠ обмежено</span>')
      .replace(/перекупленість/g, '<span style="color:#f87171;font-weight:700">перекупленість</span>')
      .replace(/перепроданість/g, '<span style="color:#4ade80;font-weight:700">перепроданість</span>')
      // Напрямок: LONG зелений, SHORT червоний.
      .replace(/\bLONG\b/g, '<span style="color:#4ade80;font-weight:800">LONG</span>')
      .replace(/\bSHORT\b/g, '<span style="color:#f87171;font-weight:800">SHORT</span>');
  }

  // 🧠 Decision Center line — same data/labels as the bot.
  function decisionLine(dec) {
    if (!dec || !dec.headline) return "";
    var rec = (dec.recommended || "NEUTRAL");
    var vmap = { good: "СИЛЬНИЙ", marginal: "СЕРЕДНІЙ", poor: "СЛАБКИЙ" };
    var vcol = { good: "#4ade80", marginal: "#facc15", poor: "#f87171" };
    var vl = vmap[dec.verdict] || String(dec.verdict || "").toUpperCase();
    var hcol = rec === "LONG" ? "#4ade80" : rec === "SHORT" ? "#f87171" : "#cbd5e1";
    var pl = Math.round((dec.prob_long || 0) * 100), ps = Math.round((dec.prob_short || 0) * 100);
    return '<div class="banner-dec">' +
      '<div class="dec-head">' + (rec === "NEUTRAL" ? "⚖️" : EINSTEIN_ICON) +
        ' <b style="color:' + hcol + '">' + esc(dec.headline) + "</b>" +
        ' <span class="dec-badge" style="color:' + (vcol[dec.verdict] || "#cbd5e1") + '">' + esc(vl) + "</span></div>" +
      (dec.rationale ? '<div class="dec-rat">' + colorizePD(esc(dec.rationale)) + "</div>" : "") +
      '<div class="dec-bar"><i class="dl" style="width:' + pl + '%"></i><i class="ds" style="width:' + ps + '%"></i></div>' +
      '<div class="dec-pct"><span class="dir-long">LONG ' + pl + '%</span>' +
        '<span class="dir-short">' + ps + '% SHORT</span></div>' +
    "</div>";
  }

  function renderPotential(b) {
    var banner = $("#banner-body");
    var body = $("#potential-body");
    if (!b || b.ok === false) {
      if (banner) banner.innerHTML = '<div class="muted">дані недоступні</div>';
      if (body) body.innerHTML = '<div class="muted">дані потенціалу недоступні</div>';
      return;
    }
    if ($("#pot-symbol")) $("#pot-symbol").textContent = b.symbol || "—";
    var verdict = b.verdict || "WAIT";
    var vcls = verdict === "LONG" ? "dir-long" : (verdict === "SHORT" ? "dir-short" : "muted");
    var conf = Math.max(0, Math.min(100, Math.round(b.confidence || 0)));
    var pcol = verdict === "LONG" ? "#4ade80" : verdict === "SHORT" ? "#f87171" : "#dbeafe";
    // Reasons — EXACTLY as the bot: text is RAW HTML (contains coloured <span>),
    // so it must NOT be escaped (the bot inserts it verbatim).
    var icons = { ok: "✓", warn: "⚠", wait: "•" };
    var wlWords = ["OB-фільтр", "Volumized", "Watchlist", "Статус монет"];
    var isWl = function (t) { return wlWords.some(function (w) { return (t || "").indexOf(w) >= 0; }); };
    var reasonsHtml = (b.reasons || []).filter(function (r) { return !isWl(r[1]); }).map(function (r) {
      var kind = r[0], text = r[1], dir = r[2];
      if ((text || "").indexOf("ММ") >= 0) {
        var c = dir === "long" ? "#22ff88" : dir === "short" ? "#ff4d4d" : "#facc15";
        return '<div class="rsn" style="color:' + c + ';font-weight:800;text-shadow:0 0 8px ' + c + '66">' +
          (icons[kind] || "•") + " " + text + "</div>";
      }
      var st = dir === "long" ? ' style="color:#4ade80"' : dir === "short" ? ' style="color:#f87171"' : "";
      return '<div class="rsn"' + st + ">" + (icons[kind] || "•") + " " + text + "</div>";
    }).join("") || '<div class="rsn muted">• немає сигналів</div>';

    // ── TOP BANNER: verdict + confidence + price + forecast + CTR + Decision ──
    if (banner) {
      banner.innerHTML =
        '<div class="pot-head">' +
          '<div class="pot-verdict ' + vcls + '">' + dirCell(verdict) + '</div>' +
          '<div class="conf-wrap"><div class="conf-bar"><i style="width:' + conf + '%;background:' + dirGrad(verdict) + '"></i></div>' +
            '<span class="conf-txt">впевненість ' + conf + '%</span></div>' +
          '<span class="pill" style="margin-left:auto">' + esc(b.symbol || "—") + '</span>' +
        '</div>' +
        '<div class="price-row"><span>💲 Ціна ' + esc(b.symbol || "") + ' (ф’ючерс)</span>' +
          '<b style="color:' + pcol + '">' + fmtPrice(b.price) + '</b></div>' +
        '<div class="reasons-list">' + reasonsHtml + '</div>' +
        ctrLine(b.ctr) +
        decisionLine(b.decision);
    }

    // ── Потенціал (move-potential block) stays in its own card ──
    var mv = (b.move_long && b.verdict === "LONG") ? b.move_long
           : (b.move_short && b.verdict === "SHORT") ? b.move_short
           : (b.move || b.move_long || b.move_short);
    if (body) body.innerHTML = moveBlock(mv) || '<div class="muted">немає даних потенціалу</div>';
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

  // ── ₿ BTCUSDT бар + перекидний годинник ─────────────────────────────────────
  var btcState = { held: 0, at: 0, period: 300, dir: null, status: "STOP", paused: false, strength: 0 };

  // Directional progress-bar gradient — колір за напрямком, від СВІТЛОГО до
  // ТЕМНОГО: LONG=зелений, SHORT=червоний, WAIT=сірий. 1:1 з ботом.
  function dirGrad(dir) {
    if (dir === "LONG")  return "linear-gradient(90deg,#86efac,#15803d)";
    if (dir === "SHORT") return "linear-gradient(90deg,#fca5a5,#b91c1c)";
    return "linear-gradient(90deg,#cbd5e1,#475569)";
  }

  // 🧑‍🔬 Ейнштейн-іконка (замість 🧠) — посилання на SVG-спрайт з index.html.
  var EINSTEIN_ICON = '<svg class="ein-icon"><use href="#ein-icon"/></svg>';

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
    box.innerHTML =
      '<div class="btc-left">' +
        '<span class="btc-sym">₿ BTCUSDT</span> ' + dirTxt +
      '</div>' +
      '<div class="btc-strength"><div class="sbar"><i style="width:' + str + '%;background:' + dirGrad(b.dir) + '"></i></div>' +
        '<span class="sbar-lbl">' + Math.round(str) + '% сила</span></div>' +
      '<div id="btc-clock">' + flipClock(nowHeld()) + '</div>' +
      '<div class="btc-status ' + statusCls + '">' + statusTxt + '</div>';
  }

  function nowHeld() {
    if (!btcState.at) return btcState.held;
    return btcState.held + Math.floor((Date.now() - btcState.at) / 1000);
  }

  // 💸 Money-rain: гроші «сиплються» в момент активності (₿ START / сильний
  // тиск ММ). 1:1 з ботом. dir → відтінок: LONG=зелений, SHORT=червоний.
  function spawnMoneyRain(dir, label) {
    try {
      if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      if (window._ffRainBusy) return;
      window._ffRainBusy = true;
      var host = document.getElementById("ff-money-rain");
      if (!host) { host = document.createElement("div"); host.id = "ff-money-rain"; document.body.appendChild(host); }
      var glow = dir === "LONG" ? "rgba(34,255,136,0.9)" : (dir === "SHORT" ? "rgba(255,77,77,0.9)" : "rgba(247,147,26,0.9)");
      var emojis = dir === "SHORT" ? ["💸","💵","💰","🪙","📉"] : (dir === "LONG" ? ["💸","💵","💰","🪙","📈"] : ["💸","💵","💰","🪙","🤑"]);
      var N = 34, maxLife = 0;
      for (var i = 0; i < N; i++) {
        var el = document.createElement("span");
        el.className = "ff-money";
        el.textContent = emojis[(Math.random() * emojis.length) | 0];
        var dur = 2.4 + Math.random() * 2.6, delay = Math.random() * 0.9, size = 18 + Math.random() * 20;
        el.style.left = (Math.random() * 100) + "vw";
        el.style.fontSize = size.toFixed(0) + "px";
        el.style.animationName = "ff-money-fall, ff-money-sway";
        el.style.animationTimingFunction = "linear, ease-in-out";
        el.style.animationIterationCount = "1, " + (2 + ((Math.random() * 3) | 0));
        el.style.animationFillMode = "forwards";
        el.style.animationDelay = delay.toFixed(2) + "s";
        el.style.animationDuration = dur.toFixed(2) + "s, " + (0.8 + Math.random() * 0.7).toFixed(2) + "s";
        el.style.filter = "drop-shadow(0 0 6px " + glow + ")";
        host.appendChild(el);
        maxLife = Math.max(maxLife, dur + delay);
      }
      if (label) {
        var fl = document.createElement("div");
        fl.className = "ff-rain-flash";
        fl.textContent = label;
        fl.style.color = dir === "SHORT" ? "#ff4d4d" : (dir === "LONG" ? "#22ff88" : "#f7931a");
        fl.style.background = dir === "SHORT" ? "rgba(239,68,68,0.14)" : (dir === "LONG" ? "rgba(34,197,94,0.14)" : "rgba(247,147,26,0.14)");
        fl.style.border = "1px solid " + glow;
        fl.style.boxShadow = "0 0 22px " + glow;
        document.body.appendChild(fl);
        setTimeout(function () { try { fl.remove(); } catch (e) {} }, 2700);
      }
      setTimeout(function () { try { host.innerHTML = ""; } catch (e) {} window._ffRainBusy = false; }, (maxLife * 1000) + 400);
    } catch (e) { window._ffRainBusy = false; }
  }

  function ffCheckMoneyRain(isStart, dir, strength) {
    if (!window._ffRainPrimed) {
      var s0 = strength >= 60 && (dir === "LONG" || dir === "SHORT");
      window._ffRainPrev = { start: !!isStart, strong: s0, dir: dir || null };
      window._ffRainPrimed = true;
      return;
    }
    var prev = window._ffRainPrev || { start: false, strong: false, dir: null };
    if (isStart && !prev.start) {
      var side = dir === "LONG" ? "🟢 LONG" : (dir === "SHORT" ? "🔴 SHORT" : "сеанс");
      spawnMoneyRain(dir || null, "⚡ ₿ START TRADING · " + side);
    }
    var strongNow = strength >= 60 && (dir === "LONG" || dir === "SHORT");
    var dirChanged = prev.strong && dir && prev.dir && dir !== prev.dir;
    if ((strongNow && !prev.strong) || (strongNow && dirChanged)) {
      if (!(isStart && !prev.start)) {
        var pull = dir === "SHORT" ? "тягне ВНИЗ 🔴" : "тягне ВГОРУ 🟢";
        spawnMoneyRain(dir, "💰 ММ " + pull + " · сильний тиск " + Math.round(strength) + "%");
      }
    }
    window._ffRainPrev = { start: isStart, strong: strongNow ? true : (strength < 45 ? false : prev.strong), dir: dir || prev.dir };
  }

  function setBtc(b) {
    b = b || {};
    btcState = {
      held: b.held_sec || 0, at: Date.now(),
      period: b.period_sec || 300, dir: b.dir || null,
      status: b.status || "STOP", paused: !!b.paused, strength: b.strength || 0
    };
    renderBtcStatic();
    ffCheckMoneyRain(btcState.status === "START" && !btcState.paused, btcState.dir, Number(btcState.strength || 0));
  }

  // Local 1s tick so the flip clock advances smoothly between polls.
  setInterval(function () {
    var c = $("#btc-clock");
    if (c && (btcState.dir || btcState.status !== "STOP")) c.innerHTML = flipClock(nowHeld());
    // Живі таймери угод / «рекомендованої» — щосекунди, а не стрибками на 5с
    // полінгу. Кожна клітинка тримає epoch у data-ts.
    var nowT = Math.floor(Date.now() / 1000);
    var els = document.querySelectorAll(".live-timer[data-ts]");
    for (var i = 0; i < els.length; i++) {
      var ts = parseInt(els[i].getAttribute("data-ts"), 10);
      if (ts) els[i].textContent = hms(nowT - ts);
    }
  }, 1000);

  // ── 💰 Funding (1:1 з ботом) ─────────────────────────────────────────────────
  function fuelBand(now) {
    if (now == null) return { txt: "", col: "#8b93a7" };
    now = Number(now);
    if (now < 10) return { txt: "рівновага", col: "#8b93a7" };
    if (now < 30) return { txt: "легкий тиск", col: "#bef264" };
    if (now < 60) return { txt: "помірний тиск", col: "#84cc16" };
    if (now < 85) return { txt: "сильний тиск", col: "#22c55e" };
    return { txt: "потужний тиск", col: "#16a34a" };
  }

  // Shared human explanation of the ММ widget (hover on the cell) — 1:1 з ботом.
  var FUEL_TOOLTIP = "ММ — тиск маркетмейкера на ціну. Крапка = куди тисне: " +
    "🟢 вгору (лонг) · 🔴 вниз (шорт) · ⚪ рівновага. 0–100% — сила тиску " +
    "(наскільки односторонні ліквідації/паливо): рівновага <10 · легкий 10–30 · " +
    "помірний 30–60 · сильний 60–85 · потужний 85+. Стрілка ↑/↓ — тиск " +
    "міцніє/слабшає. Що сильніший тиск у бік крапки — то ймовірніший рух ціни туди.";

  // ММ (fuel) strength cell — exact copy of the bot's ffFuelCell.
  function ffFuelCell(dir, now, prev) {
    var dot = dir === "LONG" ? "🟢" : (dir === "SHORT" ? "🔴" : "⚪");
    var dotSeg = '<span style="display:inline-block;width:16px;text-align:center;flex:none">' + dot + "</span>";
    if (now == null) {
      return '<span title="' + FUEL_TOOLTIP + '" style="display:inline-flex;align-items:center;gap:6px;white-space:nowrap">' + dotSeg +
        '<span style="display:inline-block;width:40px;text-align:right;color:#8b93a7;flex:none">—</span></span>';
    }
    now = Number(now);
    var col = now >= 60 ? "#22c55e" : (now >= 30 ? "#84cc16" : (now >= 10 ? "#bef264" : "#8b93a7"));
    var arrowChar = "", arrowCol = "#8b93a7";
    if (prev != null) {
      if (now > prev + 1) { arrowChar = "↑"; arrowCol = "#4ade80"; }
      else if (now < prev - 1) { arrowChar = "↓"; arrowCol = "#f87171"; }
      else { arrowChar = "→"; arrowCol = "#8b93a7"; }
    }
    var w = Math.max(0, Math.min(100, now));
    var b = fuelBand(now);
    var pctSeg = '<b style="display:inline-block;width:40px;text-align:right;color:' + col + ';font-variant-numeric:tabular-nums;flex:none">' + Math.round(now) + "%</b>";
    var arrowSeg = '<span style="display:inline-block;width:12px;text-align:center;color:' + arrowCol + ';flex:none">' + arrowChar + "</span>";
    var barSeg = '<span style="display:inline-block;vertical-align:middle;height:6px;width:48px;border-radius:3px;background:rgba(255,255,255,0.08);overflow:hidden;flex:none"><span style="display:block;height:100%;width:' + w + '%;background:' + col + '"></span></span>';
    var bandSeg = '<span style="display:inline-block;width:88px;text-align:left;color:' + b.col + ';font-size:0.66rem;font-weight:600;flex:none">' + b.txt + "</span>";
    return '<span title="' + FUEL_TOOLTIP + '" style="display:inline-flex;align-items:center;gap:6px;white-space:nowrap">' + dotSeg + pctSeg + arrowSeg + barSeg + bandSeg + "</span>";
  }

  // SCORE label EN→UA — exact copy of the bot's SCORE_LABEL_UA/scoreLabelUA.
  var SCORE_LABEL_UA = {
    "STRONG HOLD": "ВАРТО ВІДКРИВАТИ",
    "HOLD": "МОЖНА ВІДКРИВАТИ",
    "NEUTRAL": "ЗАЧЕКАТИ",
    "WEAK": "НЕ ВАРТО",
    "EXHAUSTED": "НЕ ВІДКРИВАТИ",
  };
  function scoreLabelUA(lbl) { return SCORE_LABEL_UA[lbl] || lbl || ""; }

  // SCORE (hold quality) badge — exact copy of the bot's ffScoreBadgeHTML.
  function ffScoreBadgeHTML(sc) {
    if (!sc || !sc.label) return '<span style="color:#555">—</span>';
    var scd = sc.dir || null;
    var scDir = scd === "LONG"
      ? '<span style="font-size:0.78rem;margin-right:5px">🟢<span style="color:#22c55e;font-weight:900">▲</span></span>'
      : (scd === "SHORT" ? '<span style="font-size:0.78rem;margin-right:5px">🔴<span style="color:#ef4444;font-weight:900">▼</span></span>' : "");
    var scWarn = sc.conflict
      ? '<span title="Конфлікт: бабло й рух ціни в різні боки" style="margin-left:4px">⚠️</span>'
      : "";
    return scDir + '<span title="Оцінка утримання ' + sc.score + '/100 у напрямку ' + (sc.dir || "—") + (sc.conflict ? " · ⚠ бабло проти ціни" : "") +
      '" style="display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:6px;background:' + sc.color + '22;border:1px solid ' + sc.color +
      ';font-weight:800;font-size:0.74rem;color:' + sc.color + '">' + sc.score +
      '<span style="font-size:0.6rem;letter-spacing:0.3px">' + scoreLabelUA(sc.label) + "</span></span>" + scWarn;
  }

  // F-Trend badge — funding напрямок за ~30 хв (Dashboard-метрика) — 1:1 з ботом.
  function fTrendBadge(t) {
    if (t == null) return "";
    if (t < 0) return ' <span title="F-Trend: funding поглиблюється (глибше в мінус) за ~30 хв" style="color:#ff5252;font-size:0.64rem;font-weight:700">F🔻</span>';
    if (t > 0) return ' <span title="F-Trend: funding послаблюється (до нуля) за ~30 хв" style="color:#4caf50;font-size:0.64rem;font-weight:700">F🔺</span>';
    return ' <span title="F-Trend: без чіткого тренду за ~30 хв" style="color:#888;font-size:0.64rem">F➖</span>';
  }

  // Vol 24h trend arrow vs ~2-min baseline (0.5% dead-zone) — 1:1 з ботом.
  function volTrendArrow(v, prev) {
    if (v == null || prev == null || !(prev > 0)) return "";
    if (v > prev * 1.005) return ' <span title="Обсяг 24h зростає" style="color:#4ade80;font-size:0.72rem">▲</span>';
    if (v < prev * 0.995) return ' <span title="Обсяг 24h спадає" style="color:#f87171;font-size:0.72rem">▼</span>';
    return ' <span title="Обсяг 24h без змін" style="color:#8b93a7;font-size:0.72rem">→</span>';
  }

  function renderFunding(rowsArr) {
    rowsArr = rowsArr || [];
    $("#funding-count").textContent = rowsArr.length;
    var tb = $("#funding-table tbody");
    if (!rowsArr.length) {
      tb.innerHTML = '<tr><td colspan="8" class="tm-empty-msg" style="color:#8b93a7">Немає монет з ММ із 💰 Funding Rate Scanner</td></tr>';
      return;
    }
    tb.innerHTML = rowsArr.map(function (a) {
      var dirHtml = a.dir === "LONG"
        ? '<span style="color:#22c55e">🟢 LONG</span>'
        : (a.dir === "SHORT" ? '<span style="color:#ef4444">🔴 SHORT</span>'
          : '<span style="color:#8b93a7">⚪ —</span>');
      var heldCell = '<span class="mono" style="color:#cbd5e1;font-weight:600">' + hms(a.held_sec) + "</span>";
      var entry = (a.entry_threshold != null) ? a.entry_threshold : -1;
      var END = -4;
      var cur = (a.funding_rate != null) ? a.funding_rate : entry;
      var fill = (entry - END) !== 0 ? (cur - entry) / (END - entry) : 0;
      fill = Math.max(0, Math.min(1, fill));
      var prev = a.funding_prev_rate;
      // Funding % trend colour — для «зараз» у прогресі + колонки Funding.
      var _rateCol = "#cbd5e1", _rateTip = "без даних тренду", _rateExtra = "", _rateMark = "";
      if (prev != null && a.funding_rate != null) {
        if (a.funding_rate < prev - 0.0005) { _rateCol = "#f87171"; _rateTip = "зменшується (глибше в мінус)"; }
        else if (a.funding_rate > prev + 0.0005) { _rateCol = "#4ade80"; _rateTip = "збільшується (до нуля)"; }
        else {
          var _av = Math.abs(a.funding_rate);
          var _step = Math.round(_av / 0.5) * 0.5;
          if (Math.abs(_av - _step) < 0.01 && _step >= 0.5 && _step <= 4) {
            _rateCol = "#fbbf24";
            _rateTip = "⚡ цікаве значення: стабільний funding " + _step + "% (чітке число)";
            _rateExtra = ";text-shadow:0 0 9px rgba(251,191,36,0.9);font-weight:800";
            _rateMark = "✦";
          } else {
            _rateCol = "#7dd3fc";
            _rateTip = "без змін (стабільний)";
          }
        }
      }
      // ABSOLUTE scale Entry ≤ (0%) → −4% (100%): FILL = поточний рівень (−0.9%
      // помітно довший за −0.1%), amber TICK = пік («Goal») — 1:1 з ботом.
      // (END = -4 вже оголошено вище.)
      var entryThr = (a.entry_threshold != null) ? a.entry_threshold : -0.1;
      var peakRate = (a.funding_rate_min != null) ? a.funding_rate_min : cur;
      var _rng = (END - entryThr) || 1;
      var fillS = (Math.max(0, Math.min(1, (cur - entryThr) / _rng)) * 100).toFixed(1);
      var peakS = (Math.max(0, Math.min(1, (peakRate - entryThr) / _rng)) * 100).toFixed(1);
      var progHtml = '<div style="display:flex;align-items:center;gap:6px">' +
        '<span title="Entry ≤ (старт шкали)" style="font-size:0.6rem;color:#8b93a7;white-space:nowrap">' + Number(entryThr).toFixed(2) + "%</span>" +
        '<div style="width:90px;flex:0 0 90px;height:12px;border-radius:6px;background:rgba(255,255,255,0.07);position:relative" title="Поточний ' + Number(cur).toFixed(3) + '% · шкала Entry ' + Number(entryThr).toFixed(2) + '% → −4%">' +
        '<div style="height:100%;width:' + fillS + '%;background:linear-gradient(90deg,#34d399,#22c55e);border-radius:6px"></div>' +
        '<span style="position:absolute;left:' + peakS + '%;top:-3px;bottom:-3px;width:3px;margin-left:-1.5px;background:#fbbf24;box-shadow:0 0 5px #fbbf24;border-radius:2px"></span>' +
        '</div>' +
        '<span title="пік (найглибший)" style="font-size:0.6rem;color:#fbbf24;white-space:nowrap">' + Number(peakRate).toFixed(3) + "%</span>" +
        '</div>';
      // Fixed-width ✦ slot (ALWAYS present) so the gold mark never shifts the
      // numbers — Funding column stays aligned across rows (1:1 з ботом).
      var markSlot = '<span style="display:inline-block;width:0.95em;text-align:center;color:#fbbf24' + (_rateMark ? ";text-shadow:0 0 9px rgba(251,191,36,0.9)" : "") + '">' + (_rateMark || "") + "</span>";
      var rateTxt = (a.funding_stale || a.funding_rate == null)
        ? markSlot + '<span style="color:#667">—</span>'
        : markSlot + '<span title="Funding ' + _rateTip + '" style="font-weight:700;font-variant-numeric:tabular-nums;color:' + _rateCol + _rateExtra + '">' + (a.funding_rate >= 0 ? "+" : "") + Number(a.funding_rate).toFixed(3) + "%</span>";
      var cdTxt = a.funding_next_ms
        ? '<span style="font-size:0.66rem;color:#9aa3b5">⏳ ' + fmtCountdown(a.funding_next_ms) + "</span>"
        : (a.funding_stale
            ? '<span title="Вийшла з Funding Rate Scanner — тримається лише на ММ" style="font-size:0.62rem;color:#8b93a7">· норм. (на ММ)</span>'
            : "");
      var v = a.vol24h;
      var _volRange = (a.vol24h_min != null && a.vol24h_max != null)
        ? '<span title="Діапазон 24h-обсягу за час стеження (мін–макс)" style="font-size:0.6rem;color:#8b93a7">(' + fmtUsdC(a.vol24h_min) + "–" + fmtUsdC(a.vol24h_max) + ")</span>"
        : "";
      var volTxt = (v != null && v > 0)
        ? ('<span style="display:inline-flex;align-items:center;gap:5px"><b style="display:inline-block;width:52px;text-align:right;font-variant-numeric:tabular-nums">' + fmtUsdC(v) + "</b><span style=\"display:inline-block;width:12px;text-align:center\">" + volTrendArrow(v, a.vol24h_prev) + "</span>" + _volRange + "</span>")
        : '<span style="color:#667">—</span>';
      var paused = a.paused ? ' <span title="Сесія на паузі (WAIT)" style="color:#fbbf24;font-weight:700">⏸</span>' : "";
      var _rowBg = a.opportunity_hot
        ? "background:rgba(74,222,128,0.13);box-shadow:inset 3px 0 0 #4ade80"
        : "background:rgba(16,185,129,0.06)";
      return '<tr style="' + _rowBg + '">' +
        '<td style="font-weight:600">' + (a.spike ? '<span title="Аномальний ріст: різкий рух ціни ' + (a.spike_move != null ? ((a.spike_move >= 0 ? "+" : "") + a.spike_move + "% ") : "") + '+ зростання обсягу — варто розглянути" style="margin-right:4px">🚀</span>' : "") + '<span style="color:#34d399;margin-right:4px;font-size:0.7rem">💰</span>' + tvSym(a.symbol) + "</td>" +
        "<td>" + dirHtml + "</td>" +
        '<td style="font-size:0.72rem">' + ffFuelCell(a.mm, a.mm_str, a.mm_str_prev) + paused + "</td>" +
        '<td style="white-space:nowrap">' + ffScoreBadgeHTML(a.score) + "</td>" +
        '<td style="font-size:0.78rem">' + heldCell + "</td>" +
        '<td>' + progHtml + "</td>" +
        '<td style="font-size:0.72rem;white-space:nowrap">' + rateTxt + " " + cdTxt + fTrendBadge(a.f_trend) + "</td>" +
        '<td style="font-size:0.72rem;color:#cbd5e1;white-space:nowrap">' + volTxt + "</td>" +
      "</tr>" + oppStatsRow(a);
    }).join("");
  }

  // Compact дата-час.
  function fmtDT(unix) {
    if (!unix) return "—";
    var d = new Date(unix * 1000);
    var p = function (n) { return (n < 10 ? "0" : "") + n; };
    return p(d.getDate()) + "." + p(d.getMonth() + 1) + " " + p(d.getHours()) + ":" + p(d.getMinutes());
  }

  // 🎯 Другий рядок «рекомендована ботом» = ВІДКРИТА рекомендована угода
  // (Черга-2 її відкрила): таймер угоди + статистика (1:1 з ботом).
  function oppStatsRow(a) {
    if (!a.opportunity_hot) return "";
    var st = a.opp_stats || {};
    var nowS = Math.floor(Date.now() / 1000);
    var timer = st.since
      ? '<span class="live-timer" data-ts="' + Math.floor(st.since) + '">' + hms(nowS - Math.floor(st.since)) + '</span>'
      : "—";
    var first = st.first_at ? fmtDT(Math.floor(st.first_at)) : "—";
    var entryPx = (st.start_price != null) ? fmtPrice(st.start_price) : "—";
    // Live PnL % of the open recommended trade (замість сер. тривалості).
    var pnl = (st.pnl_pct != null)
      ? '<b style="color:' + (st.pnl_pct >= 0 ? "#4ade80" : "#f87171") + '">' + (st.pnl_pct >= 0 ? "+" : "") + Number(st.pnl_pct).toFixed(2) + "%</b>"
      : "<b>—</b>";
    var _rec = (st.recent || []).map(function (e) {
      return fmtPrice(e.start_price) + "→" + fmtPrice(e.end_price) + " · " + hms(e.dur_sec);
    }).join("\n");
    var _tip = (_rec ? "Останні епізоди (вхід→вихід · тривалість):\n" + _rec : "Ще не було завершених епізодів").replace(/"/g, "&quot;");
    return '<tr style="background:rgba(74,222,128,0.07)"><td colspan="8" title="' + _tip + '" style="padding:1px 8px 5px 30px;font-size:0.68rem;color:#a7f3d0;white-space:nowrap;cursor:help">' +
      '🎯 <b style="color:#4ade80">рекомендована ботом</b> · в угоді <b>' + timer + "</b> · вхід @ <b>" + entryPx + "</b> · PnL: " + pnl + " · епізодів: <b>" + (st.count || 0) + "</b> · вперше: " + first +
      "</td></tr>";
  }

  // ── 💼 Відкриті угоди ────────────────────────────────────────────────────────
  function pnlCell(p) {
    if (p == null || isNaN(p)) return '<span class="muted">—</span>';
    var cls = Number(p) >= 0 ? "pnl-pos" : "pnl-neg";
    return '<span class="' + cls + '">' + (Number(p) >= 0 ? "+" : "") + Number(p).toFixed(2) + "%</span>";
  }
  function exhCell(e) {
    if (e == null || isNaN(e)) return '<span class="muted">—</span>';
    var v = Math.max(0, Math.min(100, Number(e)));
    var cls = v >= 70 ? "exh-bad" : (v >= 40 ? "exh-mid" : "exh-good");
    return '<span class="' + cls + '">' + v.toFixed(0) + "%</span>";
  }
  // 🔄 reversal-readiness cell — same look as the bot's ffRevCell.
  function revCell(pct) {
    if (pct == null || isNaN(pct)) return '<span class="muted">—</span>';
    var v = Math.max(0, Math.min(100, Math.round(pct)));
    var col = v >= 80 ? "#f87171" : (v >= 50 ? "#fbbf24" : "#4ade80");
    return '<span style="color:' + col + ';font-weight:700">🔄 ' + v + "%</span>";
  }
  function priceCell(v) {
    return (v != null && Number(v) > 0) ? '<span class="mono">' + fmtPrice(v) + "</span>" : '<span class="muted">—</span>';
  }
  // Manual SL/TP by magnitude: 2dp only for ≥$100, 4dp for $1–100 (keeps 2.0804),
  // more for sub-$1; trailing zeros stripped — matches the bot's roundSlTp/fmtSlTp.
  function sltpCell(v) {
    if (v == null || isNaN(v) || Number(v) <= 0) return '<span class="muted">—</span>';
    var n = Number(v);
    var dp = n < 0.0001 ? 8 : (n < 0.01 ? 6 : (n < 1 ? 5 : (n < 100 ? 4 : 2)));
    return '<span class="mono">$' + String(Number(n.toFixed(dp))) + "</span>";
  }
  function renderTrades(state) {
    var tb = $("#trades-table tbody");
    var real = (state && state.positions) || [];
    var paper = (state && state.shadow_positions) || [];
    var rows = real.map(function (p) { return { p: p, m: "real" }; })
      .concat(paper.map(function (p) { return { p: p, m: "paper" }; }));
    $("#trades-count").textContent = rows.length;
    if (!rows.length) {
      tb.innerHTML = '<tr><td colspan="12" class="muted">немає відкритих угод</td></tr>';
      return;
    }
    var now = Math.floor(Date.now() / 1000);
    tb.innerHTML = rows.map(function (r) {
      var p = r.p;
      var mk = r.m === "real"
        ? '<span class="tag-real">● LIVE</span>'
        : '<span class="tag-paper">◌ PAPER</span>';
      var timer = p.opened_at ? '<span class="mono live-timer" data-ts="' + Math.floor(p.opened_at) + '">' + hms(now - Math.floor(p.opened_at)) + "</span>" : '<span class="muted">—</span>';
      var mMark = p.manual_mode ? '<span title="Ручне керування (Manual mode ON)" style="color:#fbbf24;margin-right:3px">✋</span>' : '';
      return "<tr>" +
        "<td>" + mMark + "<b>" + tvSym(p.symbol) + "</b></td>" +
        "<td>" + dirCell(p.side) + "</td>" +
        "<td>" + mk + "</td>" +
        "<td>" + priceCell(p.entry_price) + "</td>" +
        "<td>" + priceCell(p.current_price) + "</td>" +
        "<td>" + pnlCell(p.pnl_pct) + "</td>" +
        '<td style="font-size:0.72rem">' + ffFuelCell(p.fuel_dir || p.side, p.fuel_str, p.fuel_str_prev) + "</td>" +
        "<td>" + exhCell(p.exhaustion) + "</td>" +
        "<td>" + revCell(p.ctr_rev_pct) + "</td>" +
        "<td>" + sltpCell(p.manual_sl) + "</td>" +
        "<td>" + sltpCell(p.manual_tp) + "</td>" +
        "<td>" + timer + "</td>" +
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
    ["banner-card", "potential-card", "btc-card", "funding-card", "trades-card"].forEach(function (id) {
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
      api("/api/tm/state").then(renderTrades).catch(function () { renderTrades(null); });
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
