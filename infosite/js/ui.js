/* Workspace-level navigation and keyboard accessibility. */
(function(){
  "use strict";
  var search=document.getElementById("site-search");
  if(search)search.addEventListener("click",function(){var input=document.getElementById("iw-add-in");if(!input)return;input.scrollIntoView({behavior:"smooth",block:"center"});setTimeout(function(){input.focus()},220)});
  document.querySelectorAll(".main-nav a").forEach(function(a){a.addEventListener("click",function(){document.querySelectorAll(".main-nav a").forEach(function(x){x.classList.remove("active")});a.classList.add("active")})});
  ["pp-closed-head","notify-head"].forEach(function(id){var el=document.getElementById(id);if(!el)return;el.addEventListener("keydown",function(e){if(e.key==="Enter"||e.key===" "){e.preventDefault();el.click()}})});
  if(window.VSV_DEMO){setTimeout(function(){var c=document.getElementById("conn");if(c){c.className="conn conn--wait";c.textContent="● демо-дані";c.title="Демонстраційні дані. Для живого API приберіть ?demo=1 з адреси."}},180)}
})();
