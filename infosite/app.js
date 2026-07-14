// ─────────────────────────────────────────────────────────────────────────
//  SVV Bot — інфо-панель. ТІЛЬКИ читає JSON API бота й малює дані.
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

  // ── helpers ──────────────────────────────────────────────────────────────
  function $(sel) { return document.querySelector(sel); }

  function api(path) {
    return fetch(BASE + path, { headers: { "Accept": "application/json" }, credentials: "same-origin" })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); });
  }

  function apiPost(path, body) {
    return fetch(BASE + path, {
      method: "POST", credentials: "same-origin",
      headers: { "Accept": "application/json", "Content-Type": "application/json" },
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

  function setConn(state, text) {
    var el = $("#conn");
    el.className = "conn conn--" + state;
    el.textContent = "● " + text;
  }

  // ── 🎯 Вікно потенціалу ────────────────────────────────────────────────────
  function fcLine(tf, fc) {
    if (!fc || !fc.side) return '<div class="fc-line muted">Forecast ' + tf + ': —</div>';
    var cls = fc.side === "LONG" ? "dir-long" : (fc.side === "SHORT" ? "dir-short" : "muted");
    var conf = fc.confidence != null ? ' <span class="muted">(впевненість ' + Math.round(fc.confidence) + '%)</span>' : "";
    return '<div class="fc-line">✓ Forecast ' + tf + ': <span class="' + cls + '">' +
      esc(fc.side) + " " + Math.round(fc.pct || 0) + "%</span>" + conf + "</div>";
  }

  function moveBlock(mv) {
    if (!mv || !mv.ok) return "";
    var vmap = {
      EXHAUSTED: ["🔴 ВИСНАЖЕНИЙ — запас малий", "exh-bad"],
      MATURE:    ["🟡 ЗРІЛИЙ — рухайся обережно", "exh-mid"],
      FRESH:     ["🟢 СВІЖИЙ — є запас ходу", "exh-good"]
    };
    var vd = vmap[mv.verdict] || ["—", "muted"];
    var ex = mv.exhaustion != null ? Math.round(mv.exhaustion * 1000) / 10 : null;
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
        metric("ATR / бар", fmtPct(mv.atr_pct, 3)) +
        metric("Розтяг від середн.", (mv.stretch_atr != null ? mv.stretch_atr + " ATR" : "—")) +
        metric("Запас ходу", (mv.runway_pct != null ? fmtPct(mv.runway_pct, 3) : "—") +
          (mv.runway_atr != null ? ' <span class="muted">(' + mv.runway_atr + " ATR)</span>" : "")) +
        metric("Денний хід (ADR)", (mv.adr_used_pct != null ? mv.adr_used_pct + "%" : "—") +
          (mv.adr_pct != null ? ' <span class="muted">з ' + mv.adr_pct + "%</span>" : "")) +
      '</div>' +
      '<div class="exh-row">' +
        '<span class="exh-lbl">Виснаженість</span>' +
        '<div class="exh-bar"><i class="' + exCls + '" style="width:' + (ex || 0) + '%"></i></div>' +
        '<span class="exh-val ' + exCls + '">' + (ex != null ? ex : "—") + '</span>' +
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
    var fc1 = b.forecast_1h || (comp.forecast && comp.forecast.f1) || comp.forecast_1h;
    var fc4 = b.forecast_4h || (comp.forecast && comp.forecast.f4) || comp.forecast_4h;
    // Fall back to reasons list for the ММ line if present.
    var mmLine = "";
    var sm = comp.sentiment;
    if (verdict === "WAIT") mmLine = '<div class="fc-line accent">• ММ збалансований — напрямку немає</div>';
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
        (b.price != null ? fmtUsd(b.price, 2) : "—") + '</b></div>' +
      '<div class="fc-list">' + fcLine("1H", fc1) + fcLine("4H", fc4) + mmLine + '</div>' +
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
      toggleRow("notify_btc", "₿ BTCUSDT — СТАРТ / СТОП / ПАУЗА", n.notify_btc, !linked) +
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
    else if (b.status === "START") { statusCls = "st-start"; statusTxt = "🟢 СТАРТ"; }
    else if (b.status === "STOP") { statusCls = "st-stop"; statusTxt = "⛔ СТОП"; }
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
        "<td>" + fmtUsd(a.vol24h, 1) + "</td>" +
      "</tr>";
    }).join("");
  }

  // ── 🤖 головна інфопанель ────────────────────────────────────────────────────
  function tile(label, valueHtml, cls) {
    return '<div class="tile"><div class="tile-val ' + (cls || "") + '">' + valueHtml +
      "</div><div class=\"tile-lbl\">" + esc(label) + "</div></div>";
  }

  function renderOverview(stats, health, ffRunning, lastTick) {
    var d = (stats && stats.data) || {};
    var ts = d.trade_stats || {};
    var alive = !!(health && health.status === "ok");
    var pnl = ts.total_pnl;
    var pnlCls = (pnl == null) ? "" : (Number(pnl) >= 0 ? "pnl-pos" : "pnl-neg");
    var tickTxt = "—";
    if (lastTick) {
      var ago = Math.floor(Date.now() / 1000 - lastTick);
      tickTxt = (ago >= 0 && ago < 86400) ? fmtDur(ago) + " тому" : "—";
    }
    $("#overview").innerHTML =
      tile("Бот", alive ? '<span class="pnl-pos">● онлайн</span>' : '<span class="pnl-neg">● офлайн</span>') +
      tile("Стратегія FF", ffRunning ? '<span class="pnl-pos">працює</span>' : '<span class="muted">стоп</span>') +
      tile("Останній цикл", '<span class="small">' + esc(tickTxt) + "</span>") +
      tile("Баланс (paper)", fmtUsd(d.paper_balance)) +
      tile("Win rate", fmtPct(ts.win_rate)) +
      tile("Total PnL", fmtUsd(pnl), pnlCls) +
      tile("Profit factor", ts.profit_factor != null ? Number(ts.profit_factor).toFixed(2) : "—") +
      tile("Угод (30д)", (ts.total_trades != null ? ts.total_trades : "—") +
        ' <span class="small muted">(' + (ts.winning_trades || 0) + "W/" + (ts.losing_trades || 0) + "L)</span>") +
      tile("Відкрито зараз", d.open_trades != null ? d.open_trades : "—") +
      tile("Sleepers", (d.ready_sleepers != null ? d.ready_sleepers : "—") +
        ' <span class="small muted">/ ' + (d.sleeper_count != null ? d.sleeper_count : "—") + " готових</span>");
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

  function refresh() {
    var ff = api("/api/fuel-filter/state").catch(function () { return null; });
    var stats = api("/api/stats").catch(function () { return null; });
    var health = api("/api/health").catch(function () { return null; });

    Promise.all([ff, stats, health]).then(function (res) {
      var st = res[0], stat = res[1], hp = res[2];
      if (st) { setBtc(st.btc_start); renderFunding(st.anomalies); }
      renderOverview(stat, hp, st ? st.running : false, st ? st.last_tick_ts : null);
      var anyOk = st || stat || hp;
      if (anyOk) {
        setConn("ok", "онлайн");
        $("#last-update").textContent = "оновлено: " + new Date().toLocaleTimeString("uk-UA");
      } else {
        setConn("err", "бот недоступний");
      }
    });
    refreshPotential();
    refreshNotify();
  }

  setConn("wait", "підключення…");
  refresh();
  setInterval(refresh, REFRESH);
})();
