// ─────────────────────────────────────────────────────────────────────────
//  SVV Bot — інфо-панель. ТІЛЬКИ читає JSON API бота й малює таблиці.
//  Жодних дій, що змінюють стан бота (ніяких POST).
// ─────────────────────────────────────────────────────────────────────────

(function () {
  "use strict";

  var CFG = window.INFOSITE_CONFIG || {};
  var BASE = (CFG.BOT_API_BASE || "").replace(/\/+$/, "");
  var REFRESH = CFG.REFRESH_MS || 5000;

  // ── helpers ──────────────────────────────────────────────────────────────
  function $(sel) { return document.querySelector(sel); }

  function api(path) {
    return fetch(BASE + path, { headers: { "Accept": "application/json" } })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function dirCell(dir) {
    if (dir === "LONG") return '<span class="dir-long">▲ LONG</span>';
    if (dir === "SHORT") return '<span class="dir-short">▼ SHORT</span>';
    return '<span class="muted">—</span>';
  }

  function fmtDur(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    if (h) return h + "г " + m + "хв";
    if (m) return m + "хв " + s + "с";
    return s + "с";
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

  function pnlCell(pct, usdt) {
    if (pct == null && usdt == null) return '<span class="muted">—</span>';
    var cls = (Number(pct) >= 0) ? "pnl-pos" : "pnl-neg";
    var txt = (pct != null ? fmtPct(pct, 2) : "");
    if (usdt != null) txt += " (" + (Number(usdt) >= 0 ? "+" : "") + Number(usdt).toFixed(2) + "$)";
    return '<span class="' + cls + '">' + esc(txt) + "</span>";
  }

  function fmtPrice(v) {
    if (v == null || isNaN(v)) return "—";
    var n = Number(v);
    return n >= 100 ? n.toFixed(2) : n.toPrecision(5);
  }

  function rows(tbodySel, html, colspan, emptyText) {
    var tb = $(tbodySel + " tbody");
    if (!tb) return;
    tb.innerHTML = html || ('<tr><td colspan="' + colspan + '" class="muted">' + esc(emptyText) + "</td></tr>");
  }

  function setConn(state, text) {
    var el = $("#conn");
    el.className = "conn conn--" + state;
    el.textContent = "● " + text;
  }

  // ── renderers ──────────────────────────────────────────────────────────────
  function renderBtc(b) {
    b = b || {};
    var cls = "btc-count", label = "COUNTING";
    if (b.status === "START") { cls = "btc-start"; label = "START"; }
    else if (b.status === "STOP") { cls = "btc-stop"; label = "STOP"; }
    var dirTxt = b.dir ? dirCell(b.dir) : '<span class="muted">WAIT</span>';
    var prog = Math.max(0, Math.min(100, Number(b.progress || 0)));
    $("#btc-body").innerHTML =
      '<div class="btc-badge ' + cls + '">₿ ' + esc(label) + "</div>" +
      '<div class="btc-meta">' +
        "<div>ММ-напрямок: " + dirTxt + "</div>" +
        "<div>тримається: <b>" + fmtDur(b.held_sec) + "</b></div>" +
        '<div class="bar"><i style="width:' + prog + '%"></i></div>' +
      "</div>";
  }

  function renderQueue(timers) {
    timers = timers || [];
    $("#queue-count").textContent = timers.length;
    var html = timers.map(function (t) {
      return "<tr>" +
        "<td><b>" + esc(t.symbol) + "</b></td>" +
        "<td>" + dirCell(t.dir) + "</td>" +
        "<td>" + (t.exhaustion != null ? fmtPct(t.exhaustion) : '<span class="muted">—</span>') + "</td>" +
        "<td>" + fmtDur(t.held_sec) + "</td>" +
        "<td>" + (t.engine_attempts || 0) + "</td>" +
      "</tr>";
    }).join("");
    rows("#queue-table", html, 5, "черга порожня");
  }

  function renderFunding(rowsArr) {
    rowsArr = rowsArr || [];
    $("#funding-count").textContent = rowsArr.length;
    var html = rowsArr.map(function (a) {
      var trend = '<span class="muted">—</span>';
      if (a.funding_rate != null && a.funding_prev_rate != null) {
        trend = (a.funding_rate >= a.funding_prev_rate)
          ? '<span class="dir-short">→ збільшується</span>'
          : '<span class="dir-long">← зменшується</span>';
      }
      return "<tr>" +
        "<td><b>" + esc(a.symbol) + "</b></td>" +
        "<td>" + dirCell(a.dir) + "</td>" +
        "<td>" + (a.funding_rate != null ? fmtPct(a.funding_rate, 4) : "—") + "</td>" +
        "<td>" + fmtCountdown(a.funding_next_ms) + "</td>" +
        "<td>" + trend + "</td>" +
        "<td>" + fmtNum(a.vol24h) + "</td>" +
        "<td>" + fmtDur(a.held_sec) + "</td>" +
      "</tr>";
    }).join("");
    rows("#funding-table", html, 7, "немає funding-монет");
  }

  function renderOpen(trades) {
    trades = trades || [];
    $("#open-count").textContent = trades.length;
    var now = Date.now();
    var html = trades.map(function (t) {
      var held = t.entry_time ? (now - Date.parse(t.entry_time)) / 1000 : 0;
      var tag = t.is_paper ? '<span class="tag-paper">paper</span>' : '<span class="tag-real">real</span>';
      return "<tr>" +
        "<td><b>" + esc(t.symbol) + "</b></td>" +
        "<td>" + dirCell(t.direction) + "</td>" +
        "<td>" + fmtPrice(t.entry_price) + "</td>" +
        "<td>" + fmtDur(held) + "</td>" +
        "<td>" + pnlCell(t.pnl_percent, t.pnl_usdt) + "</td>" +
        "<td>" + tag + "</td>" +
      "</tr>";
    }).join("");
    rows("#open-table", html, 6, "немає відкритих угод");
  }

  function renderClosed(trades) {
    trades = trades || [];
    var html = trades.map(function (t) {
      return "<tr>" +
        "<td><b>" + esc(t.symbol) + "</b></td>" +
        "<td>" + dirCell(t.direction) + "</td>" +
        "<td>" + fmtPrice(t.entry_price) + "</td>" +
        "<td>" + fmtPrice(t.exit_price) + "</td>" +
        "<td>" + pnlCell(t.pnl_percent, t.pnl_usdt) + "</td>" +
        '<td class="muted">' + esc(t.exit_reason || "—") + "</td>" +
      "</tr>";
    }).join("");
    rows("#closed-table", html, 6, "немає закритих угод");
  }

  // ── poll loop ──────────────────────────────────────────────────────────────
  function refresh() {
    var jobs = [
      api("/api/fuel-filter/state").then(function (st) {
        renderBtc(st.btc_start);
        renderQueue(st.timers);
        renderFunding(st.anomalies);
        return true;
      }),
      api("/api/trades?status=OPEN&limit=50").then(function (r) {
        renderOpen((r && r.data) || []);
        return true;
      }),
      api("/api/trades?status=CLOSED&limit=20").then(function (r) {
        renderClosed((r && r.data) || []);
        return true;
      }),
    ];

    Promise.allSettled(jobs).then(function (res) {
      var ok = res.some(function (x) { return x.status === "fulfilled"; });
      if (ok) {
        setConn("ok", "онлайн");
        $("#last-update").textContent = "оновлено: " + new Date().toLocaleTimeString("uk-UA");
      } else {
        setConn("err", "бот недоступний");
      }
    });
  }

  setConn("wait", "підключення…");
  refresh();
  setInterval(refresh, REFRESH);
})();
