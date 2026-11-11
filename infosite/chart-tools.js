/* TradingView-style indicators, drawing tools and chart navigation for VSV. */
(function(){
  "use strict";
  var T={tool:"cursor",drawings:[],draft:null,svg:null,hint:null,iw:null,ind:{ema20:false,ema50:false,ema200:false,volume:true},series:{}};
  try{var saved=JSON.parse(localStorage.getItem("vsv_chart_indicators")||"null");if(saved)Object.assign(T.ind,saved)}catch(e){}
  function esc(s){return String(s).replace(/[&<>"']/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
  function currentIW(){return window.VSV_HOOK&&window.VSV_HOOK.IW}
  function priceFmt(v){var iw=currentIW(),d=iw&&iw.prec!=null?iw.prec:4;return Number(v).toFixed(Math.min(d,6))}
  function point(iw,e){var r=T.svg.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,t=iw.chart.timeScale().coordinateToTime(x),p=iw.series.coordinateToPrice(y);if(t==null&&iw.coin&&iw.coin.ohlc.length){var last=iw.coin.ohlc[iw.coin.ohlc.length-1].time,spacing=iw.chart.timeScale().options().barSpacing||6;t=last+Math.round((x-iw.chart.timeScale().width())/spacing)*900}return{x:x,y:y,t:t,p:p}}
  function activate(tool){T.tool=tool||"cursor";T.draft=null;if(T.svg){T.svg.classList.toggle("active",T.tool!=="cursor");T.svg.style.pointerEvents=T.tool!=="cursor"?"auto":"none"}document.querySelectorAll("[data-draw]").forEach(function(b){b.classList.toggle("on",b.getAttribute("data-draw")===T.tool)});if(T.hint){var names={trend:"Трендова лінія: потягніть між двома точками",hline:"Горизонтальна лінія: клацніть на ціні",rect:"Прямокутник: потягніть по області",fib:"Фібоначчі: потягніть між двома цінами"};T.hint.textContent=names[T.tool]||"";T.hint.style.display=names[T.tool]?"block":"none"}render()}
  function drawingHTML(d,iw,draft){
    if(!d||d.symbol!==iw.sel||d.t1==null||d.p1==null)return"";var ts=iw.chart.timeScale(),x1=d.type==="hline"?0:ts.timeToCoordinate(d.t1),x2=d.type==="hline"?T.svg.clientWidth:ts.timeToCoordinate(d.t2),y1=iw.series.priceToCoordinate(d.p1),y2=iw.series.priceToCoordinate(d.p2);if(y1==null)return"";
    var cls="drawing"+(draft?" draft":""),out="";
    if(d.type==="trend"){if(x1==null||x2==null||y2==null)return"";out='<g class="'+cls+'"><line class="draw-line" x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'"/><circle class="draw-handle" cx="'+x1+'" cy="'+y1+'" r="3"/><circle class="draw-handle" cx="'+x2+'" cy="'+y2+'" r="3"/></g>'}
    else if(d.type==="hline"){out='<g class="'+cls+'"><line class="draw-line" x1="0" y1="'+y1+'" x2="'+T.svg.clientWidth+'" y2="'+y1+'"/><text class="draw-label" x="8" y="'+(y1-7)+'">'+priceFmt(d.p1)+'</text></g>'}
    else if(d.type==="rect"){if(x1==null||x2==null||y2==null)return"";var x=Math.min(x1,x2),y=Math.min(y1,y2),w=Math.abs(x2-x1),h=Math.abs(y2-y1);out='<g class="'+cls+'"><rect class="draw-rect" x="'+x+'" y="'+y+'" width="'+Math.max(1,w)+'" height="'+Math.max(1,h)+'"/><text class="draw-label" x="'+(x+6)+'" y="'+(y+14)+'">'+priceFmt(Math.max(d.p1,d.p2))+'</text></g>'}
    else if(d.type==="fib"){if(x1==null||x2==null||y2==null)return"";var left=Math.min(x1,x2),right=Math.max(x1,x2),levels=[0,.236,.382,.5,.618,.786,1];out='<g class="'+cls+'">';levels.forEach(function(l){var y=y1+(y2-y1)*l,p=d.p1+(d.p2-d.p1)*l;out+='<line class="fib-line" x1="'+left+'" y1="'+y+'" x2="'+right+'" y2="'+y+'"/><text class="fib-label" x="'+(left+5)+'" y="'+(y-4)+'">'+l.toFixed(3)+' · '+priceFmt(p)+'</text>'});out+='<line class="draw-line" x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke-dasharray="3 4"/></g>'}
    return out;
  }
  function render(){var iw=currentIW();if(!T.svg||!iw||!iw.chart||!iw.series)return;var html=T.drawings.map(function(d){return drawingHTML(d,iw,false)}).join("");if(T.draft)html+=drawingHTML(T.draft,iw,true);T.svg.innerHTML=html;T.svg.setAttribute("viewBox","0 0 "+T.svg.clientWidth+" "+T.svg.clientHeight)}
  function finishDraft(){if(!T.draft)return;var d=T.draft;if(d.type!=="hline"&&Math.abs(d.x1-d.x2)<4&&Math.abs(d.y1-d.y2)<4){T.draft=null;render();return}delete d.x1;delete d.x2;delete d.y1;delete d.y2;T.drawings.push(d);T.draft=null;activate("cursor")}
  function initDrawings(iw){
    var box=document.getElementById("iw-box");if(!box)return;
    T.svg=document.createElementNS("http://www.w3.org/2000/svg","svg");T.svg.setAttribute("class","tv-drawings");T.svg.setAttribute("aria-label","Шар малювання");box.appendChild(T.svg);
    T.hint=document.createElement("div");T.hint.className="drawing-hint";T.hint.style.display="none";box.appendChild(T.hint);
    T.svg.addEventListener("pointerdown",function(e){if(T.tool==="cursor"||e.button!==0)return;var p=point(iw,e);if(p.p==null)return;e.preventDefault();e.stopPropagation();try{T.svg.setPointerCapture(e.pointerId)}catch(x){}var d={type:T.tool,symbol:iw.sel,t1:p.t,p1:p.p,t2:p.t,p2:p.p,x1:p.x,x2:p.x,y1:p.y,y2:p.y};if(T.tool==="hline"){delete d.x1;delete d.x2;delete d.y1;delete d.y2;T.drawings.push(d);activate("cursor")}else{T.draft=d;render()}});
    T.svg.addEventListener("pointermove",function(e){if(!T.draft)return;var p=point(iw,e);if(p.p==null)return;T.draft.t2=p.t;T.draft.p2=p.p;T.draft.x2=p.x;T.draft.y2=p.y;e.preventDefault();render()});
    T.svg.addEventListener("pointerup",function(e){if(!T.draft)return;e.preventDefault();finishDraft()});T.svg.addEventListener("pointercancel",function(){T.draft=null;render()});
    iw.chart.timeScale().subscribeVisibleLogicalRangeChange(render);activate(T.tool);render();
  }
  function ema(data,n){var out=[],k=2/(n+1),prev=null;data.forEach(function(b,i){var v=i?b.close*k+prev*(1-k):b.close;prev=v;if(i>=n-1)out.push({time:b.time,value:v})});return out}
  function saveInd(){try{localStorage.setItem("vsv_chart_indicators",JSON.stringify(T.ind))}catch(e){}}
  function ensureIndicatorSeries(iw){
    if(T.seriesIW===iw.chart){Object.keys(T.series).forEach(function(k){try{iw.chart.removeSeries(T.series[k])}catch(e){}})}
    T.seriesIW=iw.chart;T.series={};var opts={lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false,lineWidth:1};
    if(T.ind.ema20)T.series.ema20=iw.chart.addLineSeries(Object.assign({color:"#f0b90b",title:"EMA 20"},opts));
    if(T.ind.ema50)T.series.ema50=iw.chart.addLineSeries(Object.assign({color:"#2962ff",title:"EMA 50"},opts));
    if(T.ind.ema200)T.series.ema200=iw.chart.addLineSeries(Object.assign({color:"#ec4899",title:"EMA 200"},opts));
    if(T.ind.volume){T.series.volume=iw.chart.addHistogramSeries({priceScaleId:"volume",priceFormat:{type:"volume"},lastValueVisible:false,priceLineVisible:false});iw.chart.priceScale("volume").applyOptions({scaleMargins:{top:.82,bottom:0},visible:false})}
  }
  function updateIndicators(iw,c){if(!iw||!c)return;var data=c.ohlc||[];[20,50,200].forEach(function(n){var s=T.series["ema"+n];if(s)s.setData(ema(data,n))});if(T.series.volume)T.series.volume.setData(data.map(function(b){return{time:b.time,value:Number(b.volume)||0,color:b.close>=b.open?"rgba(0,187,127,.32)":"rgba(251,44,54,.30)"}}))}
  function syncIndUI(){document.querySelectorAll("[data-indicator]").forEach(function(c){c.checked=!!T.ind[c.getAttribute("data-indicator")]})}
  function zoom(k){var iw=currentIW();if(!iw||!iw.chart)return;var ts=iw.chart.timeScale(),r=ts.getVisibleLogicalRange();if(!r)return;var mid=(r.from+r.to)/2,w=(r.to-r.from)*k;ts.setVisibleLogicalRange({from:mid-w/2,to:mid+w/2})}
  function closeMenu(){var m=document.getElementById("tv-indicators-menu"),b=document.getElementById("tv-indicators");if(m)m.hidden=true;if(b)b.setAttribute("aria-expanded","false")}
  function bindUI(){
    document.querySelectorAll("[data-draw]").forEach(function(b){b.addEventListener("click",function(){activate(b.getAttribute("data-draw"))})});
    var undo=document.getElementById("tv-undo-draw");if(undo)undo.addEventListener("click",function(){var iw=currentIW();for(var i=T.drawings.length-1;i>=0;i--){if(!iw||T.drawings[i].symbol===iw.sel){T.drawings.splice(i,1);break}}render()});
    var clear=document.getElementById("tv-clear-draw");if(clear)clear.addEventListener("click",function(){var iw=currentIW();T.drawings=iw?T.drawings.filter(function(d){return d.symbol!==iw.sel}):[];render()});
    var zi=document.getElementById("tv-zoom-in"),zo=document.getElementById("tv-zoom-out"),rs=document.getElementById("tv-reset");if(zi)zi.addEventListener("click",function(){zoom(.8)});if(zo)zo.addEventListener("click",function(){zoom(1.25)});if(rs)rs.addEventListener("click",function(){var iw=currentIW();if(iw&&iw.chart)iw.chart.timeScale().fitContent()});
    var ib=document.getElementById("tv-indicators");if(ib)ib.addEventListener("click",function(e){e.stopPropagation();var m=document.getElementById("tv-indicators-menu"),open=m.hidden;m.hidden=!open;ib.setAttribute("aria-expanded",open?"true":"false")});
    document.querySelectorAll("[data-indicator]").forEach(function(c){c.addEventListener("change",function(){T.ind[c.getAttribute("data-indicator")]=c.checked;saveInd();var iw=currentIW();if(iw&&iw.chart){ensureIndicatorSeries(iw);updateIndicators(iw,iw.coin)}})});
    document.addEventListener("click",function(e){if(!e.target.closest(".tv-ind-wrap"))closeMenu()});document.addEventListener("keydown",function(e){if(e.key==="Escape"){closeMenu();if(T.tool!=="cursor")activate("cursor")}});syncIndUI();
  }
  function install(){if(!window.VSV_PRO)return;var oldAttach=window.VSV_PRO.attach,oldData=window.VSV_PRO.onData;window.VSV_PRO.attach=function(iw){oldAttach(iw);T.iw=iw;initDrawings(iw);ensureIndicatorSeries(iw);updateIndicators(iw,iw.coin)};window.VSV_PRO.onData=function(iw,c){oldData(iw,c);updateIndicators(iw,c);render()};bindUI()}
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",install);else install();
})();
