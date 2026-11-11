/* Explicit interface preview: activated only by ?demo=1; never replaces live API data. */
(function(){
  "use strict";
  var on=new URLSearchParams(location.search).get("demo")==="1";
  if(!on)return;
  window.VSV_DEMO=true;document.documentElement.classList.add("demo-mode");
  var previewDirection=new URLSearchParams(location.search).get("direction")==="long"?"LONG":"SHORT";
  try{localStorage.setItem("vsv_pp_amt","1")}catch(e){}
  var nativeFetch=window.fetch.bind(window),now=function(){return Math.floor(Date.now()/1000)};
  var state={wl:["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT"],max:20,sel:"BTCUSDT",tf:"15m",ob_tf:"1h",vob_tf:"5m",poc_days:7,poc_tf:"1h",show_imb:true,show_sbr:true,show_shl:true,sbr:{BTCUSDT:["1h","4h"]},tg_linked:true};
  var market={BTCUSDT:[104382.65,1.84],ETHUSDT:[3521.42,2.37],SOLUSDT:[168.74,-1.26],XRPUSDT:[2.1842,.64],DOGEUSDT:[.16842,-2.13]};
  function hash(s){var h=2166136261;for(var i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return Math.abs(h)}
  function candles(sym,tf){
    var step={"15m":900,"30m":1800,"1h":3600,"4h":14400,"1d":86400}[tf]||900,n=240,seed=hash(sym+tf),base=(market[sym]||market.BTCUSDT)[0],out=[],prev=base*.965;
    var end=Math.floor(now()/step)*step;
    function rnd(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296}
    for(var i=0;i<n;i++){
      var wave=Math.sin(i/13)*.004+Math.sin(i/37)*.006,drift=(i/n)*.038,change=(rnd()-.485)*.012+wave*.28;
      var open=prev,close=Math.max(base*.7,base*(.94+drift)+open*change),high=Math.max(open,close)*(1+rnd()*.0045),low=Math.min(open,close)*(1-rnd()*.0045);
      out.push({time:end-(n-1-i)*step,open:open,high:high,low:low,close:close,volume:800+rnd()*3200+Math.abs(close-open)/open*140000});prev=close;
    }
    var target=base,ratio=target/out[out.length-1].close;out=out.map(function(b,i){var k=1+(ratio-1)*i/(out.length-1);return{time:b.time,open:b.open*k,high:b.high*k,low:b.low*k,close:b.close*k,volume:b.volume}});
    return out;
  }
  function coin(q){
    var sym=q.get("symbol")||"BTCUSDT",tf=q.get("tf")||"15m",o=candles(sym,tf),last=o[o.length-1].close,step={"15m":900,"30m":1800,"1h":3600,"4h":14400,"1d":86400}[tf]||900;
    return{ok:true,symbol:sym,tf:tf,ohlc:o,sig:"demo-"+sym+tf,pd:{zone:"discount",pct:38,tf:"4H"},poc:{poc:last*.985,dist_pct:1.52,rel:"below",days:Number(q.get("poc_days"))||7,tf:q.get("poc_tf")||"1h"},
      vob:[{start:o[62].time,top:last*1.008,bottom:last*.994,side:"LONG",breaker:false,volume:1845230,hi_pct:64,lo_pct:36,tf:q.get("vob_tf")||"5m"},{start:o[139].time,top:last*1.023,bottom:last*1.014,side:"SHORT",breaker:false,volume:982400,hi_pct:71,lo_pct:29,tf:q.get("vob_tf")||"5m"}],
      ob:[{start:o[88].time,top:last*.997,bottom:last*.988,side:"LONG",fam:"swing",tag:"confirmed"},{start:o[170].time,top:last*1.017,bottom:last*1.009,side:"SHORT",fam:"int",tag:"fresh"}],
      imb:[{start:o[185].time,top:last*1.005,bottom:last*1.001,side:"LONG",vol:734200,sell_pct:38,buy_pct:62}],events:[{from:o[105].time,to:o[148].time,level:last*1.012,dir:"bear",tag:"CHoCH"},{from:o[156].time,to:o[204].time,level:last*.997,dir:"bull",tag:"BOS"}],
      sbr:[{t:o[196].time,side:"LONG",n:18,tf:"1H",trigger:last*.992,sl:last*.974,score:87.4,risk:"Помірний",ct:o[196].time}],sbr_last:{"1h":{side:"LONG",n:18,trigger:last*.992,score:87.4,ct:o[196].time},"4h":{side:"LONG",n:7,trigger:last*.981,score:79.2,ct:o[126].time}},shl:{tf:"4H",high:{price:last*1.028,label:"High"},low:{price:last*.968,label:"Low"}}};
  }
  function paper(){
    var t=now();return{ok:true,account:{equity:1038.90,balance:1038.90,closed_balance:1038.90,deposit:1000,available:906.32,margin_usd:125.50,margin_pct:12,leverage:5,fee_pct:.055,mode:"pct"},stats:{total_pnl_usd:3.01,pnl_since_from_usd:3.01,gross_profit_usd:3.01,gross_loss_usd:0,total_closed:2,wins:2,losses:0,win_rate:100,profit_factor:null,avg_win_usd:1.50,avg_loss_usd:0,best_usd:2.59,worst_usd:0,avg_pnl_pct:.36,total_fees_usd:1.33,entry_score_breakdown:{good:{count:1,win_rate:100,avg_pnl_pct:.55},marginal:{count:1,win_rate:100,avg_pnl_pct:.18}}},
      positions:[{symbol:"BTCUSDT",side:"LONG",entry_price:103642.2,current_price:104382.65,pnl_pct:.71,pnl_net:{net_usd:1.84},paper_size:{},opened_at:t-2674,manual_sl:102980,manual_tp1:105200,manual_tp:106400,tp1_done:false,exh:{prog:68,target:105900,climax:false},mmd:{alert:{level:"ok"}}},{symbol:"ETHUSDT",side:"SHORT",entry_price:3548.7,current_price:3521.42,pnl_pct:.77,pnl_net:{net_usd:1.12},opened_at:t-5390,manual_sl:3590,manual_tp1:3490,manual_tp:3450,tp1_done:false,exh:{prog:42,target:3470},mmd:{alert:{level:"against",tf:"15m"}}}],
      closed:[{symbol:"SOLUSDT",side:"LONG",entry_price:164.52,exit_price:168.74,pnl_pct:2.56,net_usd:2.59,closed_at:t-7200,reason:"take_profit"},{symbol:"XRPUSDT",side:"SHORT",entry_price:2.191,exit_price:2.1842,pnl_pct:.31,net_usd:.42,closed_at:t-12600,reason:"manual_tp"}]};
  }
  function recRow(sym,side,price,prob,charge){var up=side==="LONG",sl=price*(up?.975:1.025),tp=price*(up?1.045:.955);return{symbol:sym,side:side,price:price,prob_long:prob,charge:charge,charge_in:12,src:sym==="BTCUSDT"?"wl":"scan",bybit:true,sl:sl,sl_pct:up?-2.5:2.5,sl_tf:"1H",tp:tp,tp_pct:up?4.5:-4.5,rr:1.8,m15:{sl:price*(up?.985:1.015),sl_pct:up?-1.5:1.5,sl_tf:"15m",tp:tp,tp_pct:up?4.5:-4.5,rr:3}}}
  function recommended(){var t=now(),long=previewDirection==="LONG";return{ok:true,ts:t,banners:[{key:"scan",ex:"binance",label:"Binance",ok:true,lean:previewDirection,prob_long:long?77:23,coins:18,total:24,need:8,n_long:long?14:4,n_short:long?4:14,fired_long:1,fired_short:5,since:t-3210},{key:"wl",ex:"bybit",label:"Bybit",ok:true,lean:previewDirection,prob_long:long?66:34,coins:5,total:6,need:3,n_long:long?3:2,n_short:long?2:3,fired_long:0,fired_short:1,since:t-3210},{key:"all",lean:previewDirection,since:t-3210}],rows:[recRow("BTCUSDT",previewDirection,104382.65,long?77:23,86),recRow("ETHUSDT",previewDirection,3521.42,long?66:34,74),recRow("SOLUSDT",long?"SHORT":"LONG",168.74,72,68)]}}
  function respond(path,init,q){
    var method=(init&&init.method||"GET").toUpperCase(),body={};if(init&&init.body){try{body=JSON.parse(init.body)}catch(e){}}
    if(path==="/api/me")return{ok:true,email:"demo@vsv.terminal"};
    if(path==="/api/me/watchlist"){
      if(method==="POST"){if(body.add){var s=String(body.add).toUpperCase().replace(/[^A-Z0-9]/g,"");if(s&&!/USDT$/.test(s))s+="USDT";if(state.wl.indexOf(s)<0)state.wl.push(s)}if(body.remove)state.wl=state.wl.filter(function(x){return x!==body.remove});["sel","tf","ob_tf","vob_tf","poc_days","poc_tf","show_imb","show_sbr","show_shl"].forEach(function(k){if(body[k]!==undefined)state[k]=body[k]});if(body.sbr)state.sbr[body.sbr.symbol]=body.sbr.tfs}return state;
    }
    if(path==="/api/info/rows")return{rows:Object.keys(market).map(function(s){return{symbol:s,price:market[s][0],chg24:market[s][1],mm:s==="BTCUSDT"?{dir:"SHORT",str:77}:null}})};
    if(path==="/api/info/universe")return{symbols:Object.keys(market).concat(["ADAUSDT","LINKUSDT","AVAXUSDT","TONUSDT"])};
    if(path==="/api/info/coin")return coin(q);
    if(path==="/api/info/paper")return paper();
    if(path==="/api/info/paper/days"){var t=now();return{ok:true,items:[[t-7200,2.59,2.56],[t-12600,.42,.31]]}}
    if(path==="/api/info/recommended")return recommended();
    if(path==="/api/me/notify")return{ok:true,tg_linked:true};
    if(path.indexOf("/api/me/")===0)return{ok:true};
    return{ok:false,error:"Demo endpoint not found"};
  }
  window.fetch=function(input,init){
    var raw=typeof input==="string"?input:input.url,u;
    try{u=new URL(raw,location.origin)}catch(e){return nativeFetch(input,init)}
    if(u.pathname.indexOf("/api/")!==0)return nativeFetch(input,init);
    var data=respond(u.pathname,init||{},u.searchParams);
    return new Promise(function(resolve){setTimeout(function(){resolve(new Response(JSON.stringify(data),{status:200,headers:{"Content-Type":"application/json"}}))},45)});
  };
})();
