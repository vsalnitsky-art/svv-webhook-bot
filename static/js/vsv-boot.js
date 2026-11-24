/* VSV full-view loading barrier. No fixed timer: every registered view must finish. */
(function () {
  "use strict";
  var required = String(document.documentElement.getAttribute("data-boot-required") || "page")
    .split(",").map(function (x) { return x.trim(); }).filter(Boolean);
  var done = Object.create(null), revealed = false;
  function revealIfReady() {
    if (revealed || !required.every(function (name) { return done[name]; })) return;
    revealed = true;
    document.body.classList.add("vsv-app-ready");
    document.dispatchEvent(new CustomEvent("vsv:bot-ready"));
  }
  function ready(name) { done[name || "page"] = true; revealIfReady(); }
  window.VSV_BOOT = { ready: ready, required: required.slice() };
  ["terminal", "liqmap"].forEach(function (name) {
    document.addEventListener("vsv:" + name + "-ready", function () { ready(name); });
  });
  function pageReady() {
    var fonts = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
    Promise.resolve(fonts).catch(function () {}).then(function () { ready("page"); });
  }
  if (document.readyState === "complete") pageReady();
  else window.addEventListener("load", pageReady, { once: true });
})();
