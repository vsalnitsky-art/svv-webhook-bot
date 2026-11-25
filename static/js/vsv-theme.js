/* VSV / InfoSite shared SVG iconography. No dependencies, no data mutations. */
(function () {
  'use strict';

  var PATHS = {
    target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3"/>',
    radar: '<circle cx="12" cy="12" r="8"/><path d="M12 12 18 6M12 4v2M4 12h2m12 0h2M12 18v2"/><circle cx="12" cy="12" r="1"/>',
    chart: '<path d="M3 3v18h18M6 15l4-5 4 3 6-8"/>',
    wallet: '<path d="M3 6h15a2 2 0 0 1 2 2v11H5a2 2 0 0 1-2-2V6Zm0 0V5a2 2 0 0 1 2-2h12v3M15 11h5v5h-5a2 2 0 0 1 0-4Z"/>',
    search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    refresh: '<path d="M20 7v5h-5M4 17v-5h5M6.1 8a7 7 0 0 1 11.8-1L20 12M4 12l2.1 5a7 7 0 0 0 11.8-1"/>',
    settings: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3M5 5l2 2m10 10 2 2M5 19l2-2M17 7l2-2"/>',
    database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 2 4 3 8 3s8-1 8-3V5M4 12v7c0 2 4 3 8 3s8-1 8-3v-7"/>',
    shield: '<path d="M12 2 20 5v6c0 5-3 9-8 11-5-2-8-6-8-11V5l8-3Z"/><path d="m8 12 3 3 5-6"/>',
    user: '<circle cx="12" cy="8" r="4"/><path d="M4 21c1-5 4-7 8-7s7 2 8 7"/>',
    flask: '<path d="M9 3h6M10 3v6l-6 9a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-6-9V3M8 15h8"/>',
    alert: '<path d="M12 3 2 21h20L12 3Z"/><path d="M12 9v5m0 3v.1"/>',
    check: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    bell: '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/>',
    trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
    droplet: '<path d="M12 2S5 10 5 15a7 7 0 0 0 14 0c0-5-7-13-7-13Z"/><path d="M9 16a3 3 0 0 0 3 3"/>',
    bot: '<rect x="4" y="7" width="16" height="13" rx="3"/><path d="M12 3v4M8 12v2m8-2v2M8 17h8"/>',
    lock: '<rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3"/>',
    key: '<circle cx="8" cy="15" r="4"/><path d="m11 12 9-9m-4 4 3 3m-6 0 3 3"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    bolt: '<path d="m13 2-9 12h7l-1 8 9-12h-7l1-8Z"/>',
    box: '<path d="m12 2 9 5-9 5-9-5 9-5Zm-9 5v10l9 5 9-5V7M12 12v10"/>',
    clipboard: '<rect x="5" y="4" width="14" height="18" rx="2"/><path d="M9 4V2h6v2M9 10h6m-6 4h6m-6 4h4"/>',
    save: '<path d="M4 2h14l2 2v18H4V2Zm4 0v7h8V2M8 22v-8h8v8"/>',
    plug: '<path d="M8 2v6m8-6v6M6 8h12v3a6 6 0 0 1-6 6v5M9 22h6"/>',
    mail: '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m3 6 9 7 9-7"/>',
    plus: '<circle cx="12" cy="12" r="9"/><path d="M12 8v8m-4-4h8"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/>',
    bank: '<path d="m3 9 9-6 9 6M4 21h16M6 9v9m4-9v9m4-9v9m4-9v9M2 18h20"/>',
    broom: '<path d="m15 3 6 6M17 5 8 14M3 21c4 0 8-2 10-6l-4-4c-4 2-6 6-6 10Z"/>'
  };
  var EMOJI = {
    '🎯':'target','🛰':'radar','📡':'radar','📊':'chart','📈':'chart','📉':'chart','💼':'wallet',
    '🔍':'search','🔎':'search','🔄':'refresh','♻️':'refresh','⚙️':'settings','⚙':'settings',
    '🗄️':'database','🗄':'database','🛡️':'shield','🛡':'shield','👤':'user','🧪':'flask',
    '⚠️':'alert','⚠':'alert','✅':'check','✓':'check','🔔':'bell','🗑️':'trash','🗑':'trash',
    '💧':'droplet','🤖':'bot','🔒':'lock','🔐':'lock','🔑':'key','⏰':'clock','⏳':'clock',
    '⚡':'bolt','📦':'box','🧾':'clipboard','💾':'save','🔌':'plug','✉️':'mail','✉':'mail',
    '➕':'plus','＋':'plus','ℹ️':'info','🏦':'bank','🧹':'broom','🆕':'plus','🚦':'alert'
  };
  var SELECTOR = 'h1,h2,h3,h4,.card-h,.section-header,.panel-header,.panel-heading,.page-h,.db-header,.db-ic,.stat-icon,.empty-icon,.sm-side-icon,.sm-alarm-on,button,.btn,.nav a,.nav-link,.badge,.seclbl';
  var SVG_NS = 'http://www.w3.org/2000/svg';

  function svgFor(name) {
    var svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('class', 'vsv-icon vsv-icon-' + name);
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    svg.innerHTML = PATHS[name] || PATHS.info;
    return svg;
  }

  function replaceEmoji(root) {
    if (!root || root.nodeType !== 1 || root.namespaceURI !== SVG_NS && root.tagName === 'svg') return;
    var nodes = [];
    if (root.matches && root.matches(SELECTOR)) nodes.push(root);
    if (root.querySelectorAll) Array.prototype.push.apply(nodes, root.querySelectorAll(SELECTOR));
    nodes.forEach(function (el) {
      if (el.closest('table') && !el.matches('button,.btn')) return;
      var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
        acceptNode: function (node) {
          if (!node.nodeValue || !/[^\s]/.test(node.nodeValue)) return NodeFilter.FILTER_REJECT;
          var parent = node.parentElement;
          if (!parent || parent.closest('script,style,input,textarea,option,svg')) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }
      });
      var textNodes = [];
      while (walker.nextNode()) textNodes.push(walker.currentNode);
      textNodes.forEach(function (node) {
        var value = node.nodeValue;
        var match = /^(?:\s*)(🎯|🛰|📡|📊|📈|📉|💼|🔍|🔎|🔄|♻️|⚙️|⚙|🗄️|🗄|🛡️|🛡|👤|🧪|⚠️|⚠|✅|✓|🔔|🗑️|🗑|💧|🤖|🔒|🔐|🔑|⏰|⏳|⚡|📦|🧾|💾|🔌|✉️|✉|➕|＋|ℹ️|🏦|🧹|🆕|🚦)/.exec(value);
        if (!match) return;
        var key = match[1];
        var fragment = document.createDocumentFragment();
        fragment.appendChild(svgFor(EMOJI[key]));
        fragment.appendChild(document.createTextNode(value.slice(match[0].length) ? ' ' + value.slice(match[0].length).replace(/^\s+/, '') : ''));
        node.parentNode.replaceChild(fragment, node);
      });
    });
  }

  function start() {
    replaceEmoji(document.body);
    var observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        mutation.addedNodes.forEach(function (node) {
          if (node.nodeType === 1 && !node.classList.contains('vsv-icon')) replaceEmoji(node);
        });
      });
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
  else start();
})();
