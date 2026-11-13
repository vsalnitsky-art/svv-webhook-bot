/* TradingView-style indicators and fully editable chart drawings for VSV. */
(function(){
  "use strict";
  var T={tool:"cursor",drawings:[],draft:null,svg:null,hint:null,selected:null,edit:null,ind:{ema20:false,ema50:false,ema200:false,volume:true},series:{}};
  var STORE="vsv_chart_drawings";
  function clone(v){return JSON.parse(JSON.stringify(v))}
  function esc(s){return String(s).replace(/[&<>"']/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
  function uid(){return "d"+Date.now().toString(36)+Math.random().toString(36).slice(2,7)}
  function currentIW(){return window.VSV_HOOK&&window.VSV_HOOK.IW}
  function save(){try{localStorage.setItem(STORE,JSON.stringify(T.drawings))}catch(e){}}
  function load(){try{var d=JSON.parse(localStorage.getItem(STORE)||"[]");if(Array.isArray(d))T.drawings=d.filter(function(x){return x&&x.id&&x.symbol&&x.type})}catch(e){}}
  try{var saved=JSON.parse(localStorage.getItem("vsv_chart_indicators")||"null");if(saved)Object.assign(T.ind,saved)}catch(e){}
  load();
  function priceFmt(v){var iw=currentIW(),d=iw&&iw.prec!=null?iw.prec:4;return Number(v).toFixed(Math.min(d,6))}
  function point(iw,e){
    var r=T.svg.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,ts=iw.chart.timeScale();
    var t=ts.coordinateToTime(x),p=iw.series.coordinateToPrice(y),l=ts.coordinateToLogical?ts.coordinateToLogical(x):null;
    /* Права область за останнім баром: час обчислюємо через логічний індекс,
       щоб об’єкти можна було малювати де завгодно, без «прилипання». */
    var ohlc=iw.coin&&iw.coin.ohlc;
    if(ohlc&&ohlc.length){
      var last=ohlc.length-1,step=ohlc.length>1?Number(ohlc[last].time)-Number(ohlc[last-1].time):0;
      if(t==null&&l!=null&&step>0)t=Number(ohlc[last].time)+Math.round(l-last)*step;
      if(t==null)t=Number(ohlc[last].time);
    }
    return{x:x,y:y,t:t,p:p,l:l};
  }
  function select(id){T.selected=id||null;var del=document.getElementById("tv-delete-draw");if(del)del.disabled=!T.selected;render()}
  function activate(tool){
    T.tool=tool||"cursor";T.draft=null;T.edit=null;
    if(T.svg){T.svg.classList.toggle("active",T.tool!=="cursor");T.svg.style.pointerEvents=T.tool!=="cursor"?"auto":"none"}
    document.querySelectorAll("[data-draw]").forEach(function(b){b.classList.toggle("on",b.getAttribute("data-draw")===T.tool)});
    if(T.hint){var names={trend:"Трендова лінія: потягніть між двома точками",hline:"Горизонтальна лінія: клацніть на ціні",rect:"Прямокутник: потягніть по області",fib:"Фібоначчі: потягніть між двома цінами",pencil:"Олівець: малюйте, не відриваючи курсор"};T.hint.textContent=names[T.tool]||(T.selected?"Перетягуйте об’єкт або його точки · Delete — видалити":"");T.hint.style.display=T.hint.textContent?"block":"none"}
    render();
  }
  function handle(x,y,n){return '<circle class="draw-handle" data-handle="'+n+'" cx="'+x+'" cy="'+y+'" r="4"/>'}
  function drawingHTML(d,iw,draft){
    if(!d||d.symbol!==iw.sel)return"";
    var ts=iw.chart.timeScale(),sr=iw.series,cls="drawing"+(draft?" draft":"")+(T.selected===d.id?" selected":""),out="";
    if(d.type==="pencil"){
      if(!d.points||d.points.length<2)return"";
      var path=d.points.map(function(p,i){var x=ts.logicalToCoordinate?ts.logicalToCoordinate(p.l):ts.timeToCoordinate(p.t),y=sr.priceToCoordinate(p.p);return x==null||y==null?null:(i?"L":"M")+x.toFixed(2)+" "+y.toFixed(2)}).filter(Boolean).join(" ");
      if(!path)return"";
      return '<g class="'+cls+'" data-id="'+esc(d.id)+'"><path class="draw-hit" d="'+path+'"/><path class="pencil-line" d="'+path+'"/></g>';
    }
    if(d.t1==null||d.p1==null)return"";
    /* Спочатку логічні координати (працюють і вільній правій області),
       часові — як запасний варіант для старих збережених об’єктів. */
    var xOf=function(tm,lg){return lg!=null&&ts.logicalToCoordinate?ts.logicalToCoordinate(lg):ts.timeToCoordinate(tm)};
    var x1=d.type==="hline"?0:xOf(d.t1,d.l1),x2=d.type==="hline"?T.svg.clientWidth:xOf(d.t2,d.l2),y1=sr.priceToCoordinate(d.p1),y2=sr.priceToCoordinate(d.p2);
    if(y1==null)return"";
    if(d.type==="trend"){
      if(x1==null||x2==null||y2==null)return"";
      out='<line class="draw-hit" x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'"/><line class="draw-line" x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'"/>'+handle(x1,y1,0)+handle(x2,y2,1);
    }else if(d.type==="hline"){
      out='<line class="draw-hit" x1="0" y1="'+y1+'" x2="'+T.svg.clientWidth+'" y2="'+y1+'"/><line class="draw-line" x1="0" y1="'+y1+'" x2="'+T.svg.clientWidth+'" y2="'+y1+'"/><text class="draw-label" x="8" y="'+(y1-7)+'">'+priceFmt(d.p1)+'</text>'+handle(T.svg.clientWidth/2,y1,0);
    }else if(d.type==="rect"){
      if(x1==null||x2==null||y2==null)return"";
      var x=Math.min(x1,x2),y=Math.min(y1,y2),w=Math.abs(x2-x1),h=Math.abs(y2-y1);
      out='<rect class="draw-rect-hit" x="'+x+'" y="'+y+'" width="'+Math.max(1,w)+'" height="'+Math.max(1,h)+'"/><rect class="draw-rect" x="'+x+'" y="'+y+'" width="'+Math.max(1,w)+'" height="'+Math.max(1,h)+'"/><text class="draw-label" x="'+(x+6)+'" y="'+(y+14)+'">'+priceFmt(Math.max(d.p1,d.p2))+'</text>'+handle(x1,y1,0)+handle(x2,y2,1);
    }else if(d.type==="fib"){
      if(x1==null||x2==null||y2==null)return"";
      var left=Math.min(x1,x2),right=Math.max(x1,x2),levels=[0,.236,.382,.5,.618,.786,1];
      out='<line class="draw-hit" x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'"/>';
      levels.forEach(function(l){var y=y1+(y2-y1)*l,p=d.p1+(d.p2-d.p1)*l;out+='<line class="fib-line" x1="'+left+'" y1="'+y+'" x2="'+right+'" y2="'+y+'"/><text class="fib-label" x="'+(left+5)+'" y="'+(y-4)+'">'+l.toFixed(3)+' · '+priceFmt(p)+'</text>'});
      out+='<line class="draw-line" x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke-dasharray="3 4"/>'+handle(x1,y1,0)+handle(x2,y2,1);
    }else if(d.type==="measure"){
      if(x1==null||x2==null||y2==null)return"";
      var mx=Math.min(x1,x2),my=Math.min(y1,y2),mw=Math.abs(x2-x1),mh=Math.abs(y2-y1);
      var chg=d.p2-d.p1,pct=d.p1?(chg/d.p1*100):0,mcls=chg>=0?"up":"dn";
      var bars=Math.round(Math.abs(x2-x1)/Math.max(1,(ts.options().barSpacing||6)))||0;
      var mdur="";
      var sec=Math.abs(Number(d.t2)-Number(d.t1));
      if(sec>0){var dd=Math.floor(sec/86400),hh=Math.floor(sec%86400/3600),mm2=Math.floor(sec%3600/60);if(dd||hh||mm2)mdur=" · "+(dd?dd+"д ":"")+(hh?hh+"г ":"")+(!dd&&mm2?mm2+"хв":"");mdur=mdur.replace(/[  ]+$/,"")}
      var df=iw.prec!=null?iw.prec:4;
      var lbl=(chg>=0?"+":"")+Number(chg).toFixed(df)+" ("+(pct>=0?"+":"")+pct.toFixed(2)+"%) · "+bars+" бар"+mdur;
      var lblW=Math.max(170,Math.round(lbl.length*6.4)+16);
      var lblY=my>36?my-10:my+mh+22;
      var lblX=Math.max(lblW/2+4,Math.min(mx+mw/2,T.svg.clientWidth-lblW/2-4));
      out='<rect class="draw-rect-hit" x="'+mx+'" y="'+my+'" width="'+Math.max(1,mw)+'" height="'+Math.max(1,mh)+'"/>'+
        '<rect class="meas-box '+mcls+'" x="'+mx+'" y="'+my+'" width="'+Math.max(1,mw)+'" height="'+Math.max(1,mh)+'"/>'+
        '<line class="draw-hit" x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'"/>'+
        '<line class="meas-line" x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'"/>'+
        '<rect class="meas-lbl" x="'+(lblX-lblW/2)+'" y="'+(lblY-10)+'" width="'+lblW+'" height="22" rx="4"/>'+
        '<text class="meas-txt '+mcls+'" x="'+lblX+'" y="'+(lblY+4)+'" text-anchor="middle">'+esc(lbl)+'</text>'+
        handle(x1,y1,0)+handle(x2,y2,1);
    }
    return '<g class="'+cls+'" data-id="'+esc(d.id)+'">'+out+'</g>';
  }
  function render(){
    var iw=currentIW();if(!T.svg||!iw||!iw.chart||!iw.series)return;
    var html=T.drawings.map(function(d){return drawingHTML(d,iw,false)}).join("");
    if(T.draft)html+=drawingHTML(T.draft,iw,true);
    T.svg.innerHTML=html;T.svg.setAttribute("viewBox","0 0 "+T.svg.clientWidth+" "+T.svg.clientHeight);
    var del=document.getElementById("tv-delete-draw");if(del)del.disabled=!T.selected;
  }
  function finishDraft(){
    if(!T.draft)return;
    var d=T.draft;
    if(d.type==="pencil"&&(!d.points||d.points.length<2)){T.draft=null;render();return}
    if(d.type!=="hline"&&d.type!=="pencil"&&Math.abs(d.x1-d.x2)<4&&Math.abs(d.y1-d.y2)<4){T.draft=null;render();return}
    delete d.x1;delete d.x2;delete d.y1;delete d.y2;
    T.drawings.push(d);T.draft=null;save();select(d.id);activate("cursor");
  }
  function removeSelected(){
    if(!T.selected)return;T.drawings=T.drawings.filter(function(d){return d.id!==T.selected});T.selected=null;save();activate("cursor");
  }
  function beginEdit(d,handleIndex,e){
    T.selected=d.id;T.edit={id:d.id,handle:handleIndex==null?null:Number(handleIndex),start:point(currentIW(),e),original:clone(d)};
    try{T.svg.setPointerCapture(e.pointerId)}catch(x){}render();
  }
  function moveEdit(iw,e){
    var ed=T.edit,d=T.drawings.find(function(x){return x.id===ed.id});if(!d)return;
    var p=point(iw,e),s=ed.start,dt=p.t!=null&&s.t!=null?p.t-s.t:0,dp=p.p-s.p,dl=p.l!=null&&s.l!=null?p.l-s.l:0;
    if(d.type==="pencil"){
      d.points=ed.original.points.map(function(q){return{l:q.l+dl,t:q.t,p:q.p+dp}});render();return;
    }
    var ol1=ed.original.l1,ol2=ed.original.l2;
    if(ed.handle===0){d.t1=ed.original.t1+dt;d.p1=ed.original.p1+dp;if(ol1!=null)d.l1=ol1+dl;if(d.type==="hline"){d.t2=d.t1;d.p2=d.p1;d.l2=d.l1}}
    else if(ed.handle===1){d.t2=ed.original.t2+dt;d.p2=ed.original.p2+dp;if(ol2!=null)d.l2=ol2+dl}
    else{d.t1=ed.original.t1+dt;d.p1=ed.original.p1+dp;d.t2=ed.original.t2+dt;d.p2=ed.original.p2+dp;if(ol1!=null)d.l1=ol1+dl;if(ol2!=null)d.l2=ol2+dl}
    render();
  }
  function initDrawings(iw){
    var box=document.getElementById("iw-box");if(!box)return;
    box.querySelectorAll(".tv-drawings,.drawing-hint").forEach(function(el){el.remove()});
    T.svg=null;T.hint=null;
    T.svg=document.createElementNS("http://www.w3.org/2000/svg","svg");T.svg.setAttribute("class","tv-drawings");T.svg.setAttribute("aria-label","Шар малювання");box.appendChild(T.svg);
    T.hint=document.createElement("div");T.hint.className="drawing-hint";T.hint.style.display="none";box.appendChild(T.hint);
    T.svg.addEventListener("pointerdown",function(e){
      if(e.button!==0)return;var node=e.target.closest?e.target.closest(".drawing"):null;
      if(T.tool==="cursor"){
        if(!node)return;var d=T.drawings.find(function(x){return x.id===node.getAttribute("data-id")});if(!d)return;
        e.preventDefault();e.stopPropagation();var h=e.target.getAttribute&&e.target.getAttribute("data-handle");beginEdit(d,h,e);return;
      }
      var p=point(iw,e);if(p.p==null)return;e.preventDefault();e.stopPropagation();try{T.svg.setPointerCapture(e.pointerId)}catch(x){}
      var d={id:uid(),type:T.tool,symbol:iw.sel,t1:p.t,p1:p.p,t2:p.t,p2:p.p,l1:p.l,l2:p.l,x1:p.x,x2:p.x,y1:p.y,y2:p.y};
      if(T.tool==="pencil")d.points=[{l:p.l,t:p.t,p:p.p,x:p.x,y:p.y}];
      if(T.tool==="hline"){delete d.x1;delete d.x2;delete d.y1;delete d.y2;T.drawings.push(d);save();select(d.id);activate("cursor")}else{T.draft=d;render()}
    });
    T.svg.addEventListener("pointermove",function(e){
      try{var r0=T.svg.getBoundingClientRect(),pr=iw.series.coordinateToPrice(e.clientY-r0.top),tm=iw.chart.timeScale().coordinateToTime(e.clientX-r0.left);if(pr!=null&&tm!=null&&iw.chart.setCrosshairPosition)iw.chart.setCrosshairPosition(pr,tm,iw.series)}catch(x){}
      if(T.edit){e.preventDefault();moveEdit(iw,e);return}
      if(!T.draft)return;var p=point(iw,e);if(p.p==null)return;
      if(T.draft.type==="pencil"){var pts=T.draft.points,last=pts[pts.length-1];if(!last||Math.abs(last.x-p.x)+Math.abs(last.y-p.y)>2){pts.push({l:p.l,t:p.t,p:p.p,x:p.x,y:p.y})}if(pts.length>1200)pts.splice(1,1)}else{T.draft.t2=p.t;T.draft.p2=p.p;T.draft.l2=p.l;T.draft.x2=p.x;T.draft.y2=p.y}
      e.preventDefault();render();
    });
    function up(e){if(T.edit){T.edit=null;save();activate("cursor");e.preventDefault();return}if(!T.draft)return;e.preventDefault();finishDraft()}
    T.svg.addEventListener("pointerup",up);T.svg.addEventListener("pointercancel",function(){if(T.edit){var d=T.drawings.find(function(x){return x.id===T.edit.id});if(d)Object.assign(d,T.edit.original);T.edit=null}T.draft=null;render()});
    T.svg.addEventListener("pointerleave",function(){try{if(iw.chart.clearCrosshairPosition)iw.chart.clearCrosshairPosition()}catch(x){}});
    iw.chart.timeScale().subscribeVisibleLogicalRangeChange(render);iw.chart.subscribeCrosshairMove(render);
    if(window.ResizeObserver){var ro=new ResizeObserver(render);ro.observe(box)}
    T.selected=null;activate(T.tool);render();
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
  function syncIndUI(){document.querySelectorAll("[data-indicator]").forEach(function(c){c.checked=!!T.ind[c.getAttribute("data-indicator")]});var sc=document.getElementById("tv-standard-candles");if(sc)sc.checked=localStorage.getItem("vsv_candle_standard")==="1"}
  function zoom(k){var iw=currentIW();if(!iw||!iw.chart)return;var ts=iw.chart.timeScale(),r=ts.getVisibleLogicalRange();if(!r)return;var mid=(r.from+r.to)/2,w=(r.to-r.from)*k;ts.setVisibleLogicalRange({from:mid-w/2,to:mid+w/2})}
  function closeMenu(){var m=document.getElementById("tv-indicators-menu"),b=document.getElementById("tv-indicators");if(m)m.hidden=true;if(b)b.setAttribute("aria-expanded","false")}
  function bindUI(){
    document.querySelectorAll("[data-draw]").forEach(function(b){b.addEventListener("click",function(){activate(b.getAttribute("data-draw"))})});
    var undo=document.getElementById("tv-undo-draw");if(undo)undo.addEventListener("click",function(){var iw=currentIW();for(var i=T.drawings.length-1;i>=0;i--){if(!iw||T.drawings[i].symbol===iw.sel){T.drawings.splice(i,1);break}}T.selected=null;save();render()});
    var del=document.getElementById("tv-delete-draw");if(del)del.addEventListener("click",removeSelected);
    var clear=document.getElementById("tv-clear-draw");if(clear)clear.addEventListener("click",function(){var iw=currentIW();T.drawings=iw?T.drawings.filter(function(d){return d.symbol!==iw.sel}):[];T.selected=null;save();activate("cursor")});
    var zi=document.getElementById("tv-zoom-in"),zo=document.getElementById("tv-zoom-out"),rs=document.getElementById("tv-reset");if(zi)zi.addEventListener("click",function(){zoom(.8)});if(zo)zo.addEventListener("click",function(){zoom(1.25)});if(rs)rs.addEventListener("click",function(){var iw=currentIW();if(iw&&iw.chart)iw.chart.timeScale().fitContent()});
    var ib=document.getElementById("tv-indicators");if(ib)ib.addEventListener("click",function(e){e.stopPropagation();var m=document.getElementById("tv-indicators-menu"),open=m.hidden;m.hidden=!open;ib.setAttribute("aria-expanded",open?"true":"false")});
    document.querySelectorAll("[data-indicator]").forEach(function(c){c.addEventListener("change",function(){T.ind[c.getAttribute("data-indicator")]=c.checked;saveInd();var iw=currentIW();if(iw&&iw.chart){ensureIndicatorSeries(iw);updateIndicators(iw,iw.coin)}})});
    var sc=document.getElementById("tv-standard-candles");if(sc)sc.addEventListener("change",function(){try{localStorage.setItem("vsv_candle_standard",sc.checked?"1":"0")}catch(e){}if(window.VSV_HOOK&&window.VSV_HOOK.applyCandles)window.VSV_HOOK.applyCandles()});
    document.addEventListener("click",function(e){if(!e.target.closest(".tv-ind-wrap"))closeMenu()});
    document.addEventListener("keydown",function(e){var tag=e.target&&e.target.tagName;if(tag==="INPUT"||tag==="TEXTAREA"||tag==="SELECT")return;if((e.key==="Delete"||e.key==="Backspace")&&T.selected){e.preventDefault();removeSelected();return}if(e.key==="Escape"){closeMenu();if(T.edit){var d=T.drawings.find(function(x){return x.id===T.edit.id});if(d)Object.assign(d,T.edit.original);T.edit=null;save()}if(T.draft)T.draft=null;if(T.tool!=="cursor")activate("cursor");else select(null)}});
    syncIndUI();
  }
  /* Зовнішній API: лінійка з chart-pro завершує малювання тут —
     об’єкт стає звичайним персистентним малюнком (виділення, Delete, localStorage). */
  window.VSV_TOOLS={addMeasure:function(m){
    var iw=currentIW();if(!iw||!m)return;
    var d={id:uid(),type:"measure",symbol:iw.sel,t1:Number(m.t1),p1:Number(m.p1),t2:Number(m.t2),p2:Number(m.p2),
      l1:m.l1!=null?Number(m.l1):null,l2:m.l2!=null?Number(m.l2):null};
    if(!isFinite(d.t1)||!isFinite(d.p1)||!isFinite(d.t2)||!isFinite(d.p2))return;
    T.drawings.push(d);save();select(d.id);render();
  }};
  function install(){if(!window.VSV_PRO)return;var oldAttach=window.VSV_PRO.attach,oldData=window.VSV_PRO.onData;window.VSV_PRO.attach=function(iw){oldAttach(iw);initDrawings(iw);ensureIndicatorSeries(iw);updateIndicators(iw,iw.coin)};window.VSV_PRO.onData=function(iw,c){oldData(iw,c);updateIndicators(iw,c);render()};bindUI()}
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",install);else install();
})();
