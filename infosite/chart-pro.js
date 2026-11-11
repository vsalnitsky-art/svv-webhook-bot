// ═══════════════════════════════════════════════════════════════════════
//  chart-pro.js — TradingView-подібні функції для lightweight-charts v4
//  Підключається ПІСЛЯ app.js. Використовує window.VSV_HOOK та VSV_PRO.
// ═══════════════════════════════════════════════════════════════════════
(function () {
  "use strict";

  var PRO = {
    type: "candles",          // "candles" | "ha"
    log: false,
    measure: { on: false, start: null, end: null },
    lastHA: null,
    bound: false,
  };

  // ── helpers ────────────────────────────────────────────────────────────
  function px(v) { return v == null ? "—" : String(v); }

  function fmtBar(v, IW) {
    if (v == null) return "—";
    var d = IW && IW.prec != null ? IW.prec : 4;
    return Number(v).toFixed(d);
  }

  function toHA(data) {
    var out = [], prevO = null, prevC = null;
    for (var i = 0; i < data.length; i++) {
      var d = data[i];
      var c = (d.open + d.high + d.low + d.close) / 4;
      var o = prevO == null ? (d.open + d.close) / 2 : (prevO + prevC) / 2;
      var h = Math.max(d.high, o, c);
      var l = Math.min(d.low, o, c);
      out.push({ time: d.time, open: o, high: h, low: l, close: c });
      prevO = o; prevC = c;
    }
    return out;
  }

  // ── HOOK: викликається з app.js ПІСЛЯ створення графіка ────────────────
  function attach(IW) {
    if (!IW || !IW.chart || !IW.series) return;
    var box = document.getElementById("iw-box");
    if (!box) return;

    // Очищення старих overlay-елементів (після зміни теми графік пересоздається)
    box.querySelectorAll(".tv-pro").forEach(function (el) { el.remove(); });

    // ── 1. OHLC-легенда (як TradingView) ────────────────────────────────
    var legend = document.createElement("div");
    legend.className = "tv-pro tv-ohlc";
    legend.innerHTML = '<div class="ohlc-legend"></div>';
    box.appendChild(legend);

    // ── 2. Overlay для лінійки (щоб не заважати скролу) ─────────────────
    var ov = document.createElement("div");
    ov.className = "tv-pro tv-measure-overlay";
    box.appendChild(ov);

    // ── 3. SVG для візуалізації вимірювання ─────────────────────────────
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "tv-pro tv-measure");
    box.appendChild(svg);

    // ── 4. Лічильник барів ───────────────────────────────────────────────
    var bars = document.createElement("div");
    bars.className = "tv-pro tv-bars";
    bars.id = "tv-bars";
    box.appendChild(bars);

    // ── 5. Підписка на crosshair (легенда слідує за мишкою) ─────────────
    IW.chart.subscribeCrosshairMove(function (param) {
      if (!param || !param.time) {
        renderLegendLast(IW);
        return;
      }
      var data = param.seriesData && param.seriesData.get(IW.series);
      if (!data || data.open == null) {
        renderLegendLast(IW);
      } else {
        renderLegend(IW, data, param.time);
      }
    });

    // ── 6. Підписка на видимий діапазон (лічильник барів) ───────────────
    IW.chart.timeScale().subscribeVisibleLogicalRangeChange(function (r) {
      var el = document.getElementById("tv-bars");
      if (!el || !r) return;
      el.textContent = "📊 " + Math.round(r.to - r.from) + " бар";
    });

    // ── 7. Обгортка setData/update для Heikin Ashi ──────────────────────
    if (!IW.series.__proWrapped) {
      var origSetData = IW.series.setData.bind(IW.series);
      var origUpdate = IW.series.update.bind(IW.series);

      IW.series.setData = function (data) {
        if (PRO.type === "ha") {
          var ha = toHA(data || []);
          PRO.lastHA = ha.length ? ha[ha.length - 1] : null;
          return origSetData(ha);
        }
        PRO.lastHA = null;
        return origSetData(data || []);
      };
      IW.series.update = function (bar) {
        if (PRO.type === "ha" && bar) {
          var prevO = PRO.lastHA && PRO.lastHA.open, prevC = PRO.lastHA && PRO.lastHA.close;
          var c = (bar.open + bar.high + bar.low + bar.close) / 4;
          var o = prevO == null ? (bar.open + bar.close) / 2 : (prevO + prevC) / 2;
          var h = Math.max(bar.high, o, c);
          var l = Math.min(bar.low, o, c);
          var ha = { time: bar.time, open: o, high: h, low: l, close: c };
          PRO.lastHA = ha;
          return origUpdate(ha);
        }
        return origUpdate(bar);
      };
      IW.series.__proWrapped = true;
    }

    // ── 8. Прив'язка подій лінійки (один раз на #iw-box) ────────────────
    bindMeasure(IW, box, ov, svg);

    // ── 9. Відновлення налаштувань (після зміни теми) ────────────────────
    if (PRO.log) applyLog(IW);
    updateTypeBtn();
    renderLegendLast(IW);
  }

  // ── HOOK: викликається з app.js ПІСЛЯ оновлення даних ─────────────────
  function onData(IW, c) {
    if (!IW || !c) return;
    renderLegendLast(IW);
    // Якщо HA — переприкладаємо дані (бо drawCoin ставить сирі)
    if (PRO.type === "ha" && IW.series && c.ohlc) {
      var range = IW.chart && IW.chart.timeScale().getVisibleLogicalRange();
      IW.series.setData(c.ohlc);
      if (range && IW.chart) IW.chart.timeScale().setVisibleLogicalRange(range);
    }
  }

  // ═══ OHLC-легенда ══════════════════════════════════════════════════════
  function renderLegend(IW, bar, time) {
    var el = document.querySelector(".tv-ohlc .ohlc-legend");
    if (!el || !bar) return;
    var o = bar.open, h = bar.high, l = bar.low, c = bar.close;
    var chg = o ? ((c - o) / o) * 100 : 0;
    var cls = chg >= 0 ? "up" : "dn";
    var f = function (v) { return fmtBar(v, IW); };
    var t = "";
    if (time) {
      var d = new Date((typeof time === "number" ? time : time.time) * 1000);
      var p2 = function (n) { return ("0" + n).slice(-2); };
      t = '<span class="t">' + p2(d.getDate()) + "." + p2(d.getMonth() + 1) + " " + p2(d.getHours()) + ":" + p2(d.getMinutes()) + "</span>";
    }
    el.innerHTML =
      '<span class="sym">' + (IW.sel || "") + "</span>" +
      t +
      '<span class="k">O</span><span class="v">' + f(o) + "</span>" +
      '<span class="k">H</span><span class="v">' + f(h) + "</span>" +
      '<span class="k">L</span><span class="v">' + f(l) + "</span>" +
      '<span class="k">C</span><span class="v ' + cls + '">' + f(c) + "</span>" +
      '<span class="chg ' + cls + '">' + (chg >= 0 ? "+" : "") + chg.toFixed(2) + "%</span>";
  }

  function renderLegendLast(IW) {
    if (!IW || !IW.coin) return;
    var ohlc = IW.coin.ohlc || [];
    if (!ohlc.length) return;
    renderLegend(IW, ohlc[ohlc.length - 1], ohlc[ohlc.length - 1].time);
  }

  // ═══ Heikin Ashi ════════════════════════════════════════════════════════
  function toggleType() {
    var IW = window.VSV_HOOK && window.VSV_HOOK.IW;
    if (!IW || !IW.series || !IW.coin) return;
    PRO.type = PRO.type === "candles" ? "ha" : "candles";
    // Переприкладаємо дані через обгортку (вона сама вирішить HA чи ні)
    var range = IW.chart.timeScale().getVisibleLogicalRange();
    IW.series.setData(IW.coin.ohlc || []);
    if (range) IW.chart.timeScale().setVisibleLogicalRange(range);
    updateTypeBtn();
  }

  function updateTypeBtn() {
    var b = document.getElementById("tv-type");
    if (!b) return;
    b.textContent = PRO.type === "ha" ? "HA" : "🕯";
    b.title = PRO.type === "ha" ? "Heikin Ashi — натисніть для свічок (H)" : "Свічки — натисніть для Heikin Ashi (H)";
    b.classList.toggle("on", PRO.type === "ha");
  }

  // ═══ Логарифмічна шкала ═════════════════════════════════════════════════
  function toggleLog() {
    var IW = window.VSV_HOOK && window.VSV_HOOK.IW;
    if (!IW || !IW.chart) return;
    PRO.log = !PRO.log;
    applyLog(IW);
    var b = document.getElementById("tv-log");
    if (b) b.classList.toggle("on", PRO.log);
  }

  function applyLog(IW) {
    if (!IW.chart) return;
    IW.chart.priceScale("right").applyOptions({
      mode: PRO.log
        ? LightweightCharts.PriceScaleMode.Logarithmic
        : LightweightCharts.PriceScaleMode.Normal,
    });
  }

  // ═══ Скріншот ═══════════════════════════════════════════════════════════
  function screenshot() {
    var IW = window.VSV_HOOK && window.VSV_HOOK.IW;
    if (!IW || !IW.chart) return;
    try {
      var canvas = IW.chart.takeScreenshot();
      canvas.toBlob(function (blob) {
        if (!blob) return;
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = "vsv-" + (IW.sel || "chart") + "-" + Date.now() + ".png";
        a.click();
        setTimeout(function () { URL.revokeObjectURL(url); }, 1500);
      });
    } catch (e) { console.error("screenshot:", e); }
  }

  // ═══ Лінійка (Measure) ══════════════════════════════════════════════════
  function toggleMeasure() {
    PRO.measure.on = !PRO.measure.on;
    var box = document.getElementById("iw-box");
    if (box) box.classList.toggle("measuring", PRO.measure.on);
    var b = document.getElementById("tv-measure");
    if (b) b.classList.toggle("on", PRO.measure.on);
    if (!PRO.measure.on) clearMeasure();
  }

  function clearMeasure() {
    PRO.measure.start = null;
    PRO.measure.end = null;
    var svg = document.querySelector(".tv-measure");
    if (svg) { svg.style.display = "none"; svg.innerHTML = ""; }
  }

  function bindMeasure(IW, box, ov, svg) {
    if (box.__proMeasureBound) return;
    box.__proMeasureBound = true;

    ov.addEventListener("mousedown", function (e) {
      if (!PRO.measure.on || e.button !== 0) return;
      var r = box.getBoundingClientRect();
      PRO.measure.start = { x: e.clientX - r.left, y: e.clientY - r.top };
      PRO.measure.end = { x: PRO.measure.start.x, y: PRO.measure.start.y };
      e.preventDefault(); e.stopPropagation();
    });

    ov.addEventListener("mousemove", function (e) {
      if (!PRO.measure.on || !PRO.measure.start) return;
      var r = box.getBoundingClientRect();
      PRO.measure.end = { x: e.clientX - r.left, y: e.clientY - r.top };
      drawMeasure(IW, svg);
    });

    ov.addEventListener("mouseup", function () {
      if (PRO.measure.on && PRO.measure.start) drawMeasure(IW, svg);
    });
  }

  function drawMeasure(IW, svg) {
    if (!IW || !IW.chart || !IW.series) return;
    var s = PRO.measure.start, e = PRO.measure.end;
    if (!s || !e) return;
    svg.style.display = "block";
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.setAttribute("viewBox", "0 0 " + document.getElementById("iw-box").clientWidth + " " + document.getElementById("iw-box").clientHeight);

    var x = Math.min(s.x, e.x), y = Math.min(s.y, e.y);
    var w = Math.abs(e.x - s.x), h = Math.abs(e.y - s.y);

    var p1 = IW.series.coordinateToPrice(s.y);
    var p2 = IW.series.coordinateToPrice(e.y);
    var chg = p1 && p2 ? p2 - p1 : 0;
    var pct = p1 && p2 ? (p2 - p1) / p1 * 100 : 0;
    var cls = chg >= 0 ? "up" : "dn";

    // Кількість барів
    var ohlc = IW.coin && IW.coin.ohlc;
    var tfSec = 60;
    if (ohlc && ohlc.length > 1) tfSec = ohlc[1].time - ohlc[0].time;
    var bars = Math.round(Math.abs(e.x - s.x) / Math.max(1, (IW.chart.timeScale().options().barSpacing || 6))) || 0;

    // Значення для підпису
    var df = IW.prec != null ? IW.prec : 4;
    var lbl = (chg >= 0 ? "+" : "") + Number(chg).toFixed(df) +
              " (" + (pct >= 0 ? "+" : "") + pct.toFixed(2) + "%) · " + bars + " бар";

    // Позиція підпису (над прямокутником, або під ним якщо мало місця)
    var lblY = y > 36 ? y - 10 : y + h + 22;
    var lblX = Math.max(70, Math.min(x + w / 2, document.getElementById("iw-box").clientWidth - 70));

    svg.innerHTML =
      '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" class="meas-box ' + cls + '"/>' +
      '<line x1="' + s.x + '" y1="' + s.y + '" x2="' + e.x + '" y2="' + e.y + '" class="meas-line ' + cls + '"/>' +
      '<rect x="' + (lblX - 85) + '" y="' + (lblY - 9) + '" width="170" height="22" rx="4" class="meas-lbl"/>' +
      '<text x="' + lblX + '" y="' + (lblY + 4) + '" class="meas-txt ' + cls + '" text-anchor="middle">' + lbl + "</text>" +
      '<circle cx="' + s.x + '" cy="' + s.y + '" r="3" class="meas-dot"/>' +
      '<circle cx="' + e.x + '" cy="' + e.y + '" r="3" class="meas-dot"/>';
  }

  // ═══ Клавіатура ═════════════════════════════════════════════════════════
  function onKey(e) {
    var t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
    var IW = window.VSV_HOOK && window.VSV_HOOK.IW;
    if (!IW || !IW.chart) return;
    var ts = IW.chart.timeScale();
    var range = ts.getVisibleLogicalRange();
    if (!range) return;
    var step = e.shiftKey ? 10 : 1;

    switch (e.key) {
      case "ArrowLeft":
        ts.setVisibleLogicalRange({ from: range.from - step, to: range.to - step });
        e.preventDefault(); break;
      case "ArrowRight":
        ts.setVisibleLogicalRange({ from: range.from + step, to: range.to + step });
        e.preventDefault(); break;
      case "Home":
        ts.fitContent(); e.preventDefault(); break;
      case "End":
        ts.scrollToRealTime(); e.preventDefault(); break;
      case "+": case "=": {
        var mid = (range.from + range.to) / 2, w = (range.to - range.from) * 0.85;
        ts.setVisibleLogicalRange({ from: mid - w / 2, to: mid + w / 2 });
        e.preventDefault(); break;
      }
      case "-": case "_": {
        var mid2 = (range.from + range.to) / 2, w2 = (range.to - range.from) * 1.15;
        ts.setVisibleLogicalRange({ from: mid2 - w2 / 2, to: mid2 + w2 / 2 });
        e.preventDefault(); break;
      }
      case "l": case "L": toggleLog(); e.preventDefault(); break;
      case "h": case "H": toggleType(); e.preventDefault(); break;
      case "f": case "F":
        var fs = document.getElementById("iw-fs"); if (fs) fs.click();
        e.preventDefault(); break;
      case "m": case "M": toggleMeasure(); e.preventDefault(); break;
      case "s": case "S": screenshot(); e.preventDefault(); break;
      case "Escape":
        if (PRO.measure.on) toggleMeasure();
        break;
    }
  }

  // ═══ Init ═══════════════════════════════════════════════════════════════
  function init() {
    if (window.VSV_PRO) return;   // вже
    window.VSV_PRO = { attach: attach, onData: onData };
    document.addEventListener("keydown", onKey);

    var bType = document.getElementById("tv-type");
    if (bType) bType.addEventListener("click", toggleType);
    var bLog = document.getElementById("tv-log");
    if (bLog) bLog.addEventListener("click", toggleLog);
    var bMeas = document.getElementById("tv-measure");
    if (bMeas) bMeas.addEventListener("click", toggleMeasure);
    var bShot = document.getElementById("tv-shot");
    if (bShot) bShot.addEventListener("click", screenshot);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();