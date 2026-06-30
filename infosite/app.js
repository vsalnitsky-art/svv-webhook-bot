// ─────────────────────────────────────────────────────────────────────────
//  SVV Bot — інфо-панель. ТІЛЬКИ читає JSON API бота й малює дані.
//  Жодних дій, що змінюють стан бота (ніяких POST).
//
//  Адреси API бота, які використовує сайт (усі — read-only GET):
//    /api/health             — живий бот чи ні
//    /api/stats              — головна інфопанель (угоди, баланс, win rate)
//    /api/fuel-filter/state  — банер ₿ BTC + 💰 funding-таблиця + статус демона
// ─────────────────────────────────────────────────────────────────────────

(function () {
  "use strict";

  var CFG = window.INFOSITE_CONFIG || {};
  // Порожній BOT_API_BASE → той самий домен, з якого відкрито сайт.
  var BASE = (CFG.BOT_API_BASE || "").replace(/\/+$/, "") || window.location.origin;
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

  function fmtUsd(v, digits) {
    if (v == null || isNaN(v)) return "—";
    return (Number(v) >= 0 ? "" : "-") + "$" + Math.abs(Number(v)).toFixed(digits == null ? 2 : digits);
  }

  function setConn(state, text) {
    var el = $("#conn");
    el.className = "conn conn--" + state;
    el.textContent = "● " + text;
  }

  // ── головна інфопанель ─────────────────────────────────────────────────────
  function tile(label, valueHtml, cls) {
    return '<div class="tile">' +
      '<div class="tile-val ' + (cls || "") + '">' + valueHtml + "</div>" +
      '<div class="tile-lbl">' + esc(label) + "</div>" +
    "</div>";
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

    var html =
      tile("Бот", alive
        ? '<span class="pnl-pos">● онлайн</span>'
        : '<span class="pnl-neg">● офлайн</span>') +
      tile("Стратегія FF", ffRunning
        ? '<span class="pnl-pos">працює</span>'
        : '<span class="muted">стоп</span>') +
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

    $("#overview").innerHTML = html;
  }

  // ── ₿ BTC / двигун ─────────────────────────────────────────────────────────
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

  // ── 💰 Funding ───────────────────────────────────────────────────────────────
  function renderFunding(rowsArr) {
    rowsArr = rowsArr || [];
    $("#funding-count").textContent = rowsArr.length;
    var tb = $("#funding-table tbody");
    if (!rowsArr.length) {
      tb.innerHTML = '<tr><td colspan="7" class="muted">немає funding-монет</td></tr>';
      return;
    }
    tb.innerHTML = rowsArr.map(function (a) {
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
  }

  // ── poll loop ──────────────────────────────────────────────────────────────
  function refresh() {
    var ff = api("/api/fuel-filter/state").catch(function () { return null; });
    var stats = api("/api/stats").catch(function () { return null; });
    var health = api("/api/health").catch(function () { return null; });

    Promise.all([ff, stats, health]).then(function (res) {
      var st = res[0], stat = res[1], hp = res[2];

      if (st) {
        renderBtc(st.btc_start);
        renderFunding(st.anomalies);
      }
      renderOverview(stat, hp, st ? st.running : false, st ? st.last_tick_ts : null);

      var anyOk = st || stat || hp;
      if (anyOk) {
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
