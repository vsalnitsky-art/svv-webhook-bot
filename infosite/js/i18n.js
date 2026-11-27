/* VSV interface i18n — DOM-safe, database-synchronised, trading terminology. */
(function () {
  "use strict";

  var LANGS = [
    { code: "uk", native: "Українська", english: "Ukrainian", locale: "uk-UA" },
    { code: "en", native: "English", english: "English", locale: "en-GB" },
    { code: "ru", native: "Русский", english: "Russian", locale: "ru-RU" },
    { code: "pl", native: "Polski", english: "Polish", locale: "pl-PL" },
    { code: "de", native: "Deutsch", english: "German", locale: "de-DE" },
    { code: "es", native: "Español", english: "Spanish", locale: "es-ES" }
  ];
  var STORE_KEY = "vsv-info-language";
  var dictionaries = window.VSV_I18N_DICTIONARIES || {};
  var patterns = window.VSV_I18N_PATTERNS || {};
  var textRecords = new WeakMap();
  var attrRecords = new WeakMap();
  var observer = null;
  var current = "uk";
  var selectedAt = 0;
  var mounted = [];
  var readyResolve;
  var ready = new Promise(function (resolve) { readyResolve = resolve; });
  var CFG = window.INFOSITE_CONFIG || {};
  var BASE = window.VSV_DEMO ? location.origin : (String(CFG.BOT_API_BASE || "").replace(/\/+$/, "") || location.origin);
  var CROSS = BASE !== location.origin;

  function meta(code) {
    for (var i = 0; i < LANGS.length; i++) if (LANGS[i].code === code) return LANGS[i];
    return LANGS[0];
  }
  function valid(code) { return LANGS.some(function (x) { return x.code === code; }); }
  function normal(code) {
    code = String(code || "").toLowerCase().replace("_", "-").split("-")[0];
    return valid(code) ? code : "uk";
  }
  function savedLanguage() {
    try { var v = localStorage.getItem(STORE_KEY); return valid(v) ? v : ""; } catch (e) { return ""; }
  }
  function saveLanguage(code) { try { localStorage.setItem(STORE_KEY, code); } catch (e) {} }
  function clean(s) { return String(s == null ? "" : s).replace(/\s+/g, " ").trim(); }
  function dict(code) { return dictionaries[code] || {}; }
  function lookup(source, code) {
    var d = dict(code), key = clean(source);
    if (Object.prototype.hasOwnProperty.call(d, key)) return d[key];
    if (code !== "en" && Object.prototype.hasOwnProperty.call(dict("en"), key)) return dict("en")[key];
    return null;
  }
  function signature(source) {
    var values = [];
    var key = clean(source).replace(/\b[A-Z]{3,12}:[A-Z0-9]{1,24}(?:\.P)?\b/g, function (value) {
      values.push(value); return "{}";
    }).replace(/\b[A-Z][A-Z0-9]{1,15}USDT\b/g, function (value) {
      values.push(value); return "{}";
    }).replace(/\d{1,3}(?:[ ,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?/g, function (value) {
      values.push(value); return "{}";
    });
    return { key: key, values: values };
  }
  function patternLookup(source, code) {
    var sig = signature(source), table = patterns[code] || {};
    var template = table[sig.key];
    if (template == null && code !== "en") template = (patterns.en || {})[sig.key];
    if (template == null) return null;
    var index = 0;
    return String(template).replace(/\{\}/g, function () {
      return index < sig.values.length ? sig.values[index++] : "{}";
    });
  }
  function replacePhrase(out, key, value) {
    var escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    try {
      var re = new RegExp("(^|[^\\p{L}\\p{N}_])" + escaped + "(?=$|[^\\p{L}\\p{N}_])", "gu");
      return out.replace(re, function (_, prefix) { return prefix + value; });
    } catch (e) {
      return out.split(key).join(value);
    }
  }
  function phraseEntries(code) {
    /* Таблиці складають текст із окремих фраз, чисел і знаків. Використовуємо
       весь словник, а не короткий _phrases: інакше кожна нова комбінація
       залишалася українською. Кеш залежить від версії обох словників, тож
       віддалені й додаткові переклади одразу потрапляють до складених рядків. */
    var cache = phraseEntries.cache || (phraseEntries.cache = {}), key = normal(code);
    var local = dict(key), english = dict("en"), cached = cache[key];
    if (cached && cached.local === local && cached.en === english) return cached.list;
    var seen = {}, out = [];
    [local, english].forEach(function (d) {
      Object.keys(d).forEach(function (source) {
        if (source === "_phrases" || seen[source] || !/[А-Яа-яЇїІіЄєҐґ]/.test(source)) return;
        seen[source] = true; out.push(source);
      });
    });
    out.sort(function (a, b) { return b.length - a.length; });
    cache[key] = { local: local, en: english, list: out };
    return out;
  }
  function translate(source, code) {
    code = code || current;
    if (code === "uk") return String(source == null ? "" : source);
    var raw = String(source == null ? "" : source), direct = lookup(raw, code);
    if (direct != null) return direct;
    var dynamic = patternLookup(raw, code);
    if (dynamic != null) return dynamic;
    /* Підказки й коментарі таблиць складаються з незалежних фрагментів.
       Спочатку рядки, далі назва поля перед двокрапкою, потім роздільник.
       Так динамічні значення й числа не губляться у складеному title. */
    if (/\r?\n/.test(raw)) {
      return raw.split(/(\r?\n)/).map(function (line) {
        return /^\r?\n$/.test(line) || !line ? line : translate(line, code);
      }).join("");
    }
    if (raw.indexOf("→ ") === 0) return "→ " + translate(raw.slice(2), code);
    var metrics = raw.match(/^(.*) \(([▲▼·].*)\)$/);
    if (metrics) {
      return translate(metrics[1], code) + " (" + translate(metrics[2], code) + ")";
    }
    var colon = raw.indexOf(": ");
    if (colon > 0 && lookup(raw.slice(0, colon), code) != null) {
      return translate(raw.slice(0, colon), code) + ": " + translate(raw.slice(colon + 2), code);
    }
    if (raw.indexOf(" · ") >= 0) {
      return raw.split(" · ").map(function (part) { return translate(part, code); }).join(" · ");
    }
    if (!/[А-Яа-яЇїІіЄєҐґ]/.test(raw)) return raw;
    var phrases = phraseEntries(code), out = raw;
    for (var i = 0; i < phrases.length; i++) {
      var key = phrases[i], value = lookup(key, code);
      if (value && out.indexOf(key) >= 0) out = replacePhrase(out, key, value);
    }
    return out;
  }
  function t(key, vars) {
    var out = lookup(key, current);
    if (out == null) out = translate(key, current);
    if (vars) Object.keys(vars).forEach(function (name) {
      out = out.split("{{" + name + "}}").join(String(vars[name]));
    });
    return out;
  }
  function ignored(el) {
    return !!(el && el.closest && el.closest("script,style,code,pre,[data-i18n-ignore],.boot-loader,.chart-loading"));
  }
  function translateTextNode(node) {
    var p = node.parentElement;
    if (!p || ignored(p)) return;
    var old = textRecords.get(node), raw = node.nodeValue;
    if (old && raw === old.output && old.lang === current) return;
    var source = old && raw === old.output ? old.source : raw;
    if (!clean(source)) { textRecords.set(node, { source: source, output: raw, lang: current }); return; }
    var translated = translate(clean(source), current);
    var lead = source.match(/^\s*/)[0], tail = source.match(/\s*$/)[0];
    var output = translated === clean(source) && current !== "uk" ? source : lead + translated + tail;
    node.nodeValue = output;
    textRecords.set(node, { source: source, output: output, lang: current });
  }
  function translateAttr(el, name) {
    if (ignored(el)) return;
    var rec = attrRecords.get(el) || {};
    var old = rec[name], raw = el.getAttribute(name);
    if (raw == null || (old && raw === old.output && old.lang === current)) return;
    var source = old && raw === old.output ? old.source : raw;
    if (!clean(source)) return;
    var translated = translate(source, current);
    var output = translated === source && current !== "uk" ? source : translated;
    if (raw !== output) el.setAttribute(name, output);
    rec[name] = { source: source, output: output, lang: current };
    attrRecords.set(el, rec);
  }
  function scan(root) {
    if (!root) return;
    if (root.nodeType === Node.TEXT_NODE) { translateTextNode(root); return; }
    if (root.nodeType !== Node.ELEMENT_NODE && root.nodeType !== Node.DOCUMENT_NODE) return;
    if (root.nodeType === Node.ELEMENT_NODE && ignored(root)) return;
    if (root.nodeType === Node.ELEMENT_NODE) {
      ["title", "placeholder", "aria-label"].forEach(function (a) { translateAttr(root, a); });
    }
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        if (node.nodeType === Node.ELEMENT_NODE) return NodeFilter.FILTER_ACCEPT;
        if (!node.parentElement || ignored(node.parentElement)) return NodeFilter.FILTER_REJECT;
        return clean(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });
    var nodes = [], n;
    while ((n = walker.nextNode())) nodes.push(n);
    nodes.forEach(function (node) {
      if (node.nodeType === Node.TEXT_NODE) translateTextNode(node);
      else ["title", "placeholder", "aria-label"].forEach(function (a) { translateAttr(node, a); });
    });
  }
  function startObserver() {
    if (observer) observer.disconnect();
    observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (m) {
        if (m.type === "characterData") translateTextNode(m.target);
        else if (m.type === "childList") m.addedNodes.forEach(scan);
        else if (m.type === "attributes") translateAttr(m.target, m.attributeName);
      });
    });
    observer.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ["title", "placeholder", "aria-label"] });
  }

  function globeSVG() {
    return '<svg class="lang-globe" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18"/><path class="lang-orbit" d="M4 7c4-3 12-3 16 0"/></svg>';
  }
  function closeMenus(except) {
    mounted.forEach(function (m) {
      if (m === except) return;
      m.menu.hidden = true;
      m.button.setAttribute("aria-expanded", "false");
    });
  }
  function focusOption(m, index) {
    var opts = Array.from(m.menu.querySelectorAll(".lang-option"));
    if (!opts.length) return;
    index = (index + opts.length) % opts.length;
    opts[index].focus();
  }
  function mount(slot) {
    if (!slot || slot.dataset.langMounted) return;
    slot.dataset.langMounted = "1";
    slot.innerHTML = '<button class="lang-control" type="button" aria-haspopup="listbox" aria-expanded="false" aria-label="Language" title="Language">' + globeSVG() + '<span class="lang-label">Language</span><b class="lang-current"></b><svg class="lang-chev" viewBox="0 0 24 24" aria-hidden="true"><path d="m7 10 5 5 5-5"/></svg></button><div class="lang-menu" role="listbox" aria-label="Language" hidden></div>';
    var m = { slot: slot, button: slot.querySelector(".lang-control"), menu: slot.querySelector(".lang-menu") };
    m.menu.innerHTML = LANGS.map(function (l) {
      return '<button type="button" class="lang-option" role="option" data-lang="' + l.code + '" aria-checked="false" lang="' + l.code + '"><span class="lang-code">' + l.code + '</span><span class="lang-native" data-i18n-ignore>' + l.native + '</span><span class="lang-check" aria-hidden="true">✓</span></button>';
    }).join("");
    m.button.addEventListener("click", function () {
      var open = m.menu.hidden;
      closeMenus(m);
      m.menu.hidden = !open;
      m.button.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) {
        var active = m.menu.querySelector('[data-lang="' + current + '"]');
        (active || m.menu.querySelector(".lang-option")).focus();
      }
    });
    m.menu.addEventListener("click", function (e) {
      var option = e.target.closest("[data-lang]");
      if (!option) return;
      setLanguage(option.dataset.lang, { sync: true });
      closeMenus();
      m.button.focus();
    });
    m.menu.addEventListener("keydown", function (e) {
      var opts = Array.from(m.menu.querySelectorAll(".lang-option"));
      var at = opts.indexOf(document.activeElement);
      if (e.key === "ArrowDown") { e.preventDefault(); focusOption(m, at + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); focusOption(m, at - 1); }
      else if (e.key === "Home") { e.preventDefault(); focusOption(m, 0); }
      else if (e.key === "End") { e.preventDefault(); focusOption(m, opts.length - 1); }
      else if (e.key === "Escape") { e.preventDefault(); closeMenus(); m.button.focus(); }
      else if (e.key === "Tab") closeMenus();
    });
    mounted.push(m);
  }
  function syncControls() {
    mounted.forEach(function (m) {
      m.button.querySelector(".lang-current").textContent = current;
      m.button.setAttribute("aria-label", "Language — " + meta(current).english);
      m.menu.querySelectorAll(".lang-option").forEach(function (o) {
        var on = o.dataset.lang === current;
        o.setAttribute("aria-checked", on ? "true" : "false");
        o.querySelector(".lang-check").style.visibility = on ? "visible" : "hidden";
      });
    });
  }
  function request(path, options) {
    options = options || {};
    var headers = { Accept: "application/json" };
    var token = "";
    try { token = localStorage.getItem("vsv_info_token") || ""; } catch (e) {}
    if (token) headers["X-Info-Token"] = token;
    if (CFG.INFO_API_KEY) headers["X-API-Key"] = CFG.INFO_API_KEY;
    if (options.body) headers["Content-Type"] = "application/json";
    return fetch(BASE + path, {
      method: options.method || "GET", headers: headers, credentials: CROSS ? "omit" : "same-origin",
      cache: "no-store", body: options.body ? JSON.stringify(options.body) : undefined,
      signal: options.signal
    }).then(function (r) {
      if (!r.ok) { var e = new Error("HTTP " + r.status); e.status = r.status; throw e; }
      return r.json();
    });
  }
  function syncPreference(code) {
    return request("/api/me/language", { method: "POST", body: { language: code } }).catch(function () { return null; });
  }
  function setLanguage(code, options) {
    options = options || {};
    code = normal(code);
    current = code;
    saveLanguage(code);
    document.documentElement.lang = code;
    document.documentElement.dir = "ltr";
    syncControls();
    scan(document.body);
    startObserver();
    if (options.sync) {
      selectedAt = Date.now();
      loadRemoteDictionary(code);
      syncPreference(code);
    }
    document.dispatchEvent(new CustomEvent("vsv:language", { detail: { language: code, locale: meta(code).locale } }));
    return code;
  }
  function loadRemoteDictionary(code, signal) {
    return request("/api/i18n/" + code, { signal: signal }).then(function (d) {
      var strings = d && (d.strings || d.dictionary);
      if (strings && typeof strings === "object") {
        dictionaries[code] = Object.assign({}, dict(code), strings);
        if (code === current) { scan(document.body); startObserver(); }
      }
      return d;
    }).catch(function () { return null; });
  }
  function init() {
    var initial = savedLanguage();
    if (!initial && navigator.language) initial = normal(navigator.language);
    current = initial || "uk";
    mount(document.getElementById("auth-language"));
    mount(document.getElementById("top-language"));
    setLanguage(current);
    document.addEventListener("click", function (e) { if (!e.target.closest(".language-slot")) closeMenus(); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeMenus(); });
    var ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
    var timer = ctl ? setTimeout(function () { ctl.abort(); }, 5000) : null;
    var signal = ctl ? ctl.signal : undefined;
    var dictPromise = loadRemoteDictionary(current, signal);
    var prefPromise = request("/api/me/language", { signal: signal }).then(function (d) {
      if (d && valid(d.language) && Date.now() - selectedAt > 10000 && d.language !== current) {
        setLanguage(d.language);
        return loadRemoteDictionary(d.language, signal);
      }
      return null;
    }).catch(function () { return null; });
    Promise.all([dictPromise, prefPromise]).then(function () {
      if (timer) clearTimeout(timer);
      readyResolve(window.VSV_I18N);
    });
  }

  window.VSV_I18N = {
    languages: LANGS, ready: ready, t: t, translate: translate, scan: scan,
    setLanguage: setLanguage, getLanguage: function () { return current; },
    locale: function () { return meta(current).locale; },
    addDictionary: function (code, values) {
      if (!valid(code) || !values) return;
      dictionaries[code] = Object.assign({}, dict(code), values);
      if (code === current) { scan(document.body); startObserver(); }
    }
  };
  init();
})();
