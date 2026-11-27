/* VSV client/server version barrier and post-deploy refresh. */
(function () {
  "use strict";

  var CFG = window.INFOSITE_CONFIG || {};
  var BASE = String(CFG.BOT_API_BASE || "").replace(/\/+$/, "") || location.origin;
  var scriptUrl = document.currentScript && document.currentScript.src || "";
  // Сайт, відданий самим ботом (/info/ або сторінки бота), має той самий origin,
  // що й API бота; окремо розгорнутий інфосайт ходить на BOT_API_BASE.
  var botServed = BASE === location.origin || /\/info\/js\/version-sync\.js(?:\?|$)/.test(scriptUrl);
  var STORE_KEY = botServed ? "vsv-bot-client-version-v1" : "vsv-info-client-version-v1";
  var CHECK_MS = 45000;
  var TIMEOUT_MS = 8000;
  var inFlight = null;
  var reloading = false;
  var baseline = null;

  try { baseline = localStorage.getItem(STORE_KEY) || null; } catch (e) {}

  function requestVersion(url) {
    var ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
    var timer = ctl ? setTimeout(function () { ctl.abort(); }, TIMEOUT_MS) : null;
    var freshUrl = url + (url.indexOf("?") >= 0 ? "&" : "?") + "_vsv=" + Date.now();
    return fetch(freshUrl, {
      method: "GET", headers: { "Accept": "application/json" },
      credentials: "omit", cache: "no-store", signal: ctl ? ctl.signal : undefined
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    }).then(function (data) {
      var value = data && String(data.version || "").trim();
      if (!value) throw new Error("Missing client version");
      return value;
    }).finally(function () { if (timer) clearTimeout(timer); });
  }

  function currentVersion() {
    var requests = [requestVersion(BASE + "/api/version").then(function (value) {
      return "bot=" + value;
    })];
    // The separately hosted info-site has its own deploy lifecycle.
    if (!botServed && location.protocol !== "file:") {
      requests.push(requestVersion(location.origin + "/version.json").then(function (value) {
        return "site=" + value;
      }));
    }
    return Promise.all(requests).then(function (parts) { return parts.join("|"); });
  }

  function store(version) {
    baseline = version;
    try { localStorage.setItem(STORE_KEY, version); } catch (e) {}
  }

  function check() {
    if (reloading) return Promise.resolve(null);
    if (inFlight) return inFlight;
    inFlight = currentVersion().then(function (version) {
      // A partial response during a network failure must never become a baseline.
      if (!baseline) {
        store(version);
        return { version: version, changed: false, reloading: false };
      }
      if (version === baseline) return { version: version, changed: false, reloading: false };

      // Save first: even a cached HTML response cannot cause a reload loop.
      store(version);
      reloading = true;
      document.dispatchEvent(new CustomEvent("vsv:version-change", {
        detail: { version: version, changed: true, reloading: true }
      }));
      location.reload();
      return { version: version, changed: true, reloading: true };
    }).catch(function () {
      // Version checks are advisory during an outage; live-data sync owns retries.
      return null;
    }).finally(function () { inFlight = null; });
    return inFlight;
  }

  var ready = check();
  window.VSV_VERSION_SYNC = { ready: ready, check: check };
  setInterval(function () { if (!document.hidden) check(); }, CHECK_MS);
  document.addEventListener("visibilitychange", function () { if (!document.hidden) check(); });
  window.addEventListener("online", check);
})();
