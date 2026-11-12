// ═══════════════════════════════════════════════════════════════════════
//  chart-pro.js — TradingView-подібні функції для lightweight-charts v4
// ═══════════════════════════════════════════════════════════════════════
(function () {
  "use strict";

  var PRO = {
    type: "candles",
    log: false,
    measure: { on: false, start: null, end: null },
    lastHA: null,
  };

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

  // ═══════════════════════════════════════════════════════════════════════
  //  ПОЗИЦІЇ (Long/Short Tools)
  // ═══════════════════════════════════════════════════════════════════════
  var posList = [];
  var pendingPos = null;
  var posLayer = null;

  function initPosLayer() {
    if (posLayer) return;
    var box = document.getElementById("iw-box");
    if (!box) return;
    posLayer = document.createElement("div");
    posLayer.className = "pos-layer";
    box.appendChild(posLayer);
  }

  function addPos(type, entryPrice, clickX) {
    var IW = window.VSV_HOOK && window.VSV_HOOK.IW;
    if (!IW) return;
    
    var time = IW.chart.timeScale().coordinateToTime(clickX);
    if (time === null) {
      var oc = IW.coin && IW.coin.ohlc;
      if (!oc || !oc.length) return;
      time = oc[oc.length - 1].time;
    }

    var id = "pos_" + Date.now();
    var wrap = document.createElement("div");
    wrap.className = "pos-wrap";
    wrap.innerHTML = 
      '<div class="pos-box profit"></div>' +
      '<div class="pos-box loss"></div>' +
      '<div class="pos-line entry"><div class="pos-label">Відкр. поз.</div></div>' +
      '<div class="pos-line profit"><div class="pos-label">Ціль</div></div>' +
      '<div class="pos-line loss"><div class="pos-label">Стоп</div></div>' +
      '<div class="pos-resize" title="Змінити ширину"></div>' +
      '<button class="pos-del" title="Видалити">×</button>';
    
    posLayer.appendChild(wrap);

    var tpPct = type === "long" ? 0.05 : -0.05;
    var slPct = type === "long" ? -0.02 : 0.02;

    var pos = {
      id: id,
      type: type,
      entry: entryPrice,
      target: entryPrice * (1 + tpPct),
      stop: entryPrice * (1 + slPct),
      time: time,
      symbol: IW.sel || null,
      width: 250,
      dom: {
        wrap: wrap,
        boxProfit: wrap.querySelector(".profit.pos-box"),
        boxLoss: wrap.querySelector(".loss.pos-box"),
        lineEntry: wrap.querySelector(".pos-line.entry"),
        lineTarget: wrap.querySelector(".pos-line.profit"),
        lineStop: wrap.querySelector(".pos-line.loss"),
        labelEntry: wrap.querySelector(".pos-line.entry .pos-label"),
        labelTarget: wrap.querySelector(".pos-line.profit .pos-label"),
        labelStop: wrap.querySelector(".pos-line.loss .pos-label"),
        resizeHandle: wrap.querySelector(".pos-resize"),
        delBtn: wrap.querySelector(".pos-del")
      }
    };

    setupDrag(pos.dom.lineTarget, function(dy) {
      var y = IW.series.priceToCoordinate(pos.target) + dy;
      pos.target = IW.series.coordinateToPrice(y);
    });

    setupDrag(pos.dom.lineStop, function(dy) {
      var y = IW.series.priceToCoordinate(pos.stop) + dy;
      pos.stop = IW.series.coordinateToPrice(y);
    });

    setupDrag(pos.dom.lineEntry, function(dy) {
      var yEntry = IW.series.priceToCoordinate(pos.entry) + dy;
      var yTgt = IW.series.priceToCoordinate(pos.target) + dy;
      var yStop = IW.series.priceToCoordinate(pos.stop) + dy;
      
      pos.entry = IW.series.coordinateToPrice(yEntry);
      pos.target = IW.series.coordinateToPrice(yTgt);
      pos.stop = IW.series.coordinateToPrice(yStop);
    });

    setupDragX(pos.dom.resizeHandle, function(dx) {
      pos.width = Math.max(60, pos.width + dx);
    });

    pos.dom.delBtn.addEventListener("pointerdown", function(e) {
      e.stopPropagation();
      removePos(pos);
    });

    posList.push(pos);
    renderPositions(IW);
  }

  function removePos(pos) {
    pos.dom.wrap.remove();
    posList = posList.filter(function(p) { return p.id !== pos.id; });
  }

  function setupDrag(el, onMove) {
    var dragging = false;
    var startY = 0;

    el.addEventListener("pointerdown", function(e) {
      if (e.button !== 0) return;
      dragging = true;
      startY = e.clientY;
      el.setPointerCapture(e.pointerId);
      e.stopPropagation(); 
    });

    el.addEventListener("pointermove", function(e) {
      if (!dragging) return;
      var dy = e.clientY - startY;
      onMove(dy);
      startY = e.clientY;
      
      var IW = window.VSV_HOOK && window.VSV_HOOK.IW;
      if (IW) renderPositions(IW);
    });

    el.addEventListener("pointerup", function(e) {
      if (!dragging) return;
      dragging = false;
      el.releasePointerCapture(e.pointerId);
    });
  }

  function setupDragX(el, onMove) {
    var dragging = false;
    var startX = 0;

    el.addEventListener("pointerdown", function(e) {
      if (e.button !== 0) return;
      dragging = true;
      startX = e.clientX;
      el.setPointerCapture(e.pointerId);
      e.stopPropagation(); 
    });

    el.addEventListener("pointermove", function(e) {
      if (!dragging) return;
      var dx = e.clientX - startX;
      onMove(dx);
      startX = e.clientX;
      
      var IW = window.VSV_HOOK && window.VSV_HOOK.IW;
      if (IW) renderPositions(IW);
    });

    el.addEventListener("pointerup", function(e) {
      if (!dragging) return;
      dragging = false;
      el.releasePointerCapture(e.pointerId);
    });
  }

  function renderPositions(IW) {
    if (!IW || !IW.chart || !IW.series) return;
    var ts = IW.chart.timeScale();
    var chartWidth = document.getElementById("iw-box").clientWidth;
    var psWidth = 60; 

    posList.forEach(function(pos) {
      if (pos.symbol && IW.sel && pos.symbol !== IW.sel) { pos.dom.wrap.style.display = "none"; return; }
      var x1 = ts.timeToCoordinate(pos.time);
      if (x1 === null) {
        var range = ts.getVisibleLogicalRange();
        var first = IW.coin && IW.coin.ohlc && IW.coin.ohlc[0];
        if (range && first && pos.time < first.time) x1 = 0;
        else x1 = chartWidth - psWidth; 
      }
      
      var yEntry = IW.series.priceToCoordinate(pos.entry);
      var yTarget = IW.series.priceToCoordinate(pos.target);
      var yStop = IW.series.priceToCoordinate(pos.stop);

      if (yEntry === null || yTarget === null || yStop === null) {
        pos.dom.wrap.style.display = "none";
        return;
      }
      pos.dom.wrap.style.display = "block";

      // Динамічний розрахунок розмірів контейнера позиції
      var pTop = Math.min(yEntry, yTarget, yStop);
      var pBottom = Math.max(yEntry, yTarget, yStop);
      var pHeightTotal = Math.max(10, pBottom - pTop);

      pos.dom.wrap.style.left = x1 + "px";
      pos.dom.wrap.style.top = pTop + "px";
      pos.dom.wrap.style.width = pos.width + "px";
      pos.dom.wrap.style.height = pHeightTotal + "px";

      // Блок прибутку (відносно контейнера)
      var pTopProfit = Math.min(yEntry, yTarget);
      var pHeightProfit = Math.abs(yEntry - yTarget);
      pos.dom.boxProfit.style.top = (pTopProfit - pTop) + "px";
      pos.dom.boxProfit.style.height = pHeightProfit + "px";
      pos.dom.boxProfit.style.left = "0px";
      pos.dom.boxProfit.style.width = "100%";

      // Блок збитку (відносно контейнера)
      var lTop = Math.min(yEntry, yStop);
      var lHeight = Math.abs(yEntry - yStop);
      pos.dom.boxLoss.style.top = (lTop - pTop) + "px";
      pos.dom.boxLoss.style.height = lHeight + "px";
      pos.dom.boxLoss.style.left = "0px";
      pos.dom.boxLoss.style.width = "100%";

      // Лінії (відносно контейнера)
      pos.dom.lineEntry.style.top = (yEntry - pTop) + "px";
      pos.dom.lineTarget.style.top = (yTarget - pTop) + "px";
      pos.dom.lineStop.style.top = (yStop - pTop) + "px";

      var isLong = pos.type === "long";
      var targetPct = Math.abs((pos.target - pos.entry) / pos.entry) * 100;
      var stopPct = Math.abs((pos.stop - pos.entry) / pos.entry) * 100;
      var rr = targetPct / (stopPct || 1);

      pos.dom.labelEntry.innerHTML = "Відкр. поз: " + fmtBar(pos.entry, IW) + 
        '<br><span style="font-size:10px;opacity:0.8">Ризик/Прибуток: ' + rr.toFixed(2) + '</span>';
      
      pos.dom.labelTarget.innerHTML = "Ціль: " + fmtBar(pos.target, IW) + " (" + targetPct.toFixed(2) + "%)";
      pos.dom.labelStop.innerHTML = "Стоп: " + fmtBar(pos.stop, IW) + " (" + stopPct.toFixed(2) + "%)";
    });
  }

  function attach(IW) {
    if (!IW || !IW.chart || !IW.series) return;
    var box = document.getElementById("iw-box");
    if (!box) return;

    // Повторний attach (перестворення графіка, зміна теми): прибираємо старі
    // оверлеї, щоб не накопичувались дублікати, і дозволяємо переприв'язку лінійки.
    box.querySelectorAll(".tv-pro").forEach(function (el) { el.remove(); });
    box.__proMeasureBound = false;

    initPosLayer();

    var ov = document.createElement("div");
    ov.className = "tv-pro tv-measure-overlay";
    box.appendChild(ov);

    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "tv-pro tv-measure");
    box.appendChild(svg);

    var bars = document.createElement("div");
    bars.className = "tv-pro tv-bars";
    bars.id = "tv-bars";
    box.appendChild(bars);

    IW.chart.subscribeCrosshairMove(function (param) {
      if (!param || !param.time) { renderLegendLast(IW); return; }
      var data = param.seriesData && param.seriesData.get(IW.series);
      if (!data || data.open == null) renderLegendLast(IW);
      else renderLegend(IW, data, param.time);
    });

    IW.chart.timeScale().subscribeVisibleLogicalRangeChange(function (r) {
      var el = document.getElementById("tv-bars");
      if (el && r) el.textContent = "📊 " + Math.round(r.to - r.from) + " бар";
      renderPositions(IW);
    });

    IW.chart.subscribeCrosshairMove(function() {
      if (posList.length) renderPositions(IW);
    });

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

    bindMeasure(IW, box, ov, svg);

    if (PRO.log) applyLog(IW);
    updateTypeBtn();
    renderLegendLast(IW);

    IW.chart.subscribeClick(function (param) {
      if (!pendingPos) return;
      if (!param.point) return;
      var price = IW.series.coordinateToPrice(param.point.y);
      if (price) {
        addPos(pendingPos, price, param.point.x);
        pendingPos = null;
        document.getElementById("iw-box").style.cursor = "";
        document.getElementById("tv-long").classList.remove("on");
        document.getElementById("tv-short").classList.remove("on");
      }
    });

    document.getElementById("tv-long").onclick = function() {
      pendingPos = "long";
      document.getElementById("iw-box").style.cursor = "crosshair";
      document.getElementById("tv-long").classList.add("on");
      document.getElementById("tv-short").classList.remove("on");
    };
    document.getElementById("tv-short").onclick = function() {
      pendingPos = "short";
      document.getElementById("iw-box").style.cursor = "crosshair";
      document.getElementById("tv-short").classList.add("on");
      document.getElementById("tv-long").classList.remove("on");
    };
    document.getElementById("tv-clear-pos").onclick = function() {
      posList.forEach(removePos);
      posList = [];
    };
  }

  function onData(IW, c) {
    if (!IW || !c) return;
    renderLegendLast(IW);
    renderPositions(IW);
    if (PRO.type === "ha" && IW.series && c.ohlc) {
      var range = IW.chart && IW.chart.timeScale().getVisibleLogicalRange();
      IW.series.setData(c.ohlc);
      if (range && IW.chart) IW.chart.timeScale().setVisibleLogicalRange(range);
    }
  }

  function renderLegend(IW, bar, time) {
    var el = document.querySelector("#tv-ohlc .ohlc-legend");
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
      '<span class="sym">' + (IW.sel || "") + "</span>" + t +
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

  function toggleType() {
    var IW = window.VSV_HOOK && window.VSV_HOOK.IW;
    if (!IW || !IW.series || !IW.coin) return;
    PRO.type = PRO.type === "candles" ? "ha" : "candles";
    var range = IW.chart.timeScale().getVisibleLogicalRange();
    IW.series.setData(IW.coin.ohlc || []);
    if (range) IW.chart.timeScale().setVisibleLogicalRange(range);
    updateTypeBtn();
  }

  function updateTypeBtn() {
    var b = document.getElementById("tv-type");
    if (!b) return;
    b.innerHTML = '<svg><use href="#i-candles"/></svg>' + (PRO.type === "ha" ? '<span>HA</span>' : "");
    b.title = PRO.type === "ha" ? "Heikin Ashi — повернути звичайні свічки (H)" : "Звичайні свічки — увімкнути Heikin Ashi (H)";
    b.classList.toggle("on", PRO.type === "ha");
  }

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
      mode: PRO.log ? LightweightCharts.PriceScaleMode.Logarithmic : LightweightCharts.PriceScaleMode.Normal,
    });
  }

  function screenshot() {
    var IW = window.VSV_HOOK && window.VSV_HOOK.IW;
    if (!IW || !IW.chart) return;
    try {
      var canvas = IW.chart.takeScreenshot();
      canvas.toBlob(function (blob) {
        if (!blob) return;
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url; a.download = "vsv-" + (IW.sel || "chart") + "-" + Date.now() + ".png";
        a.click();
        setTimeout(function () { URL.revokeObjectURL(url); }, 1500);
      });
    } catch (e) { console.error("screenshot:", e); }
  }

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
    svg.setAttribute("width", "100%"); svg.setAttribute("height", "100%");
    svg.setAttribute("viewBox", "0 0 " + document.getElementById("iw-box").clientWidth + " " + document.getElementById("iw-box").clientHeight);

    var x = Math.min(s.x, e.x), y = Math.min(s.y, e.y);
    var w = Math.abs(e.x - s.x), h = Math.abs(e.y - s.y);

    var p1 = IW.series.coordinateToPrice(s.y);
    var p2 = IW.series.coordinateToPrice(e.y);
    var chg = p1 && p2 ? p2 - p1 : 0;
    var pct = p1 && p2 ? (p2 - p1) / p1 * 100 : 0;
    var cls = chg >= 0 ? "up" : "dn";

    var bars = Math.round(Math.abs(e.x - s.x) / Math.max(1, (IW.chart.timeScale().options().barSpacing || 6))) || 0;
    var df = IW.prec != null ? IW.prec : 4;
    var lbl = (chg >= 0 ? "+" : "") + Number(chg).toFixed(df) + " (" + (pct >= 0 ? "+" : "") + pct.toFixed(2) + "%) · " + bars + " бар";

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
      case "ArrowLeft": ts.setVisibleLogicalRange({ from: range.from - step, to: range.to - step }); e.preventDefault(); break;
      case "ArrowRight": ts.setVisibleLogicalRange({ from: range.from + step, to: range.to + step }); e.preventDefault(); break;
      case "Home": ts.fitContent(); e.preventDefault(); break;
      case "End": ts.scrollToRealTime(); e.preventDefault(); break;
      case "+": case "=": {
        var mid = (range.from + range.to) / 2, w = (range.to - range.from) * 0.85;
        ts.setVisibleLogicalRange({ from: mid - w / 2, to: mid + w / 2 }); e.preventDefault(); break;
      }
      case "-": case "_": {
        var mid2 = (range.from + range.to) / 2, w2 = (range.to - range.from) * 1.15;
        ts.setVisibleLogicalRange({ from: mid2 - w2 / 2, to: mid2 + w2 / 2 }); e.preventDefault(); break;
      }
      case "l": case "L": toggleLog(); e.preventDefault(); break;
      case "h": case "H": toggleType(); e.preventDefault(); break;
      case "f": case "F": var fs = document.getElementById("iw-fs"); if (fs) fs.click(); e.preventDefault(); break;
      case "m": case "M": toggleMeasure(); e.preventDefault(); break;
      case "s": case "S": screenshot(); e.preventDefault(); break;
      case "Escape": if (PRO.measure.on) toggleMeasure(); break;
    }
  }

  function init() {
    if (window.VSV_PRO) return;
    window.VSV_PRO = { attach: attach, onData: onData };
    document.addEventListener("keydown", onKey);

    var bType = document.getElementById("tv-type"); if (bType) bType.addEventListener("click", toggleType);
    var bLog = document.getElementById("tv-log"); if (bLog) bLog.addEventListener("click", toggleLog);
    var bMeas = document.getElementById("tv-measure"); if (bMeas) bMeas.addEventListener("click", toggleMeasure);
    var bShot = document.getElementById("tv-shot"); if (bShot) bShot.addEventListener("click", screenshot);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();