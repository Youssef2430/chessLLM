"use strict";
const $=id=>document.getElementById(id);
const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={'gpt-6-astra':'GPT-6 Astra','gpt-6-sol':'GPT-6 Sol','gpt-6-luna':'GPT-6 Luna','claude-opus-5-5':'Claude Opus 5.5','claude-sonnet-5-5':'Claude Sonnet 5.5','gemini-3.1-pro':'Gemini 3.1 Pro','gemini-3.8-flash':'Gemini 3.8 Flash','gpt-oss-120b':'GPT-OSS-120B','random-baseline':'Random baseline'};
const name=m=>names[m]||m;
const fmt=(v,d=0)=>v===null||v===undefined?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d});
const pct=v=>v==null?'—':fmt(v*100,1)+'%';
const palette=['#315b42','#547858','#80936b','#aa803e','#c4a46a','#6c7e96','#97a8ba'];
let data=null,currentView='overview',selectedModel='all',selectedGame='',frames=[],frameIndex=0,flipped=false,loadSequence=0,follow=true;
const selected=rows=>selectedModel==='all'?rows:rows.filter(r=>(r.bot||r.model)===selectedModel);
const setting=()=>data.manifest.settings||data.manifest.config||{};
function badge(s){return `<span class="badge ${escapeHTML(s)}">${escapeHTML(s||'unreported')}</span>`;}
function metric(label,value,detail){return `<div class="metric"><div class="metric-label">${escapeHTML(label)}</div><div class="metric-value">${escapeHTML(value)}</div><div class="metric-detail">${escapeHTML(detail)}</div></div>`;}
function dl(rows){return rows.map(([k,v])=>`<div><dt>${escapeHTML(k)}</dt><dd>${escapeHTML(v??'Unreported')}</dd></div>`).join('');}
function showError(e){$('error').textContent=e.message||String(e);$('error').hidden=false;}
async function json(url){const r=await fetch(url);const j=await r.json();if(!r.ok)throw new Error(j.error||'Unable to load results');return j;}
async function loadRuns(){
 const runs=await json('/api/runs');runs.sort((a,b)=>String(b.started_at||'').localeCompare(String(a.started_at||'')));
 const old=$('run-select').value;
 $('run-select').replaceChildren(...runs.map(r=>new Option(`${r.id} · ${r.effort||'effort unreported'}`,r.id)));
 if(runs.some(r=>r.id===old))$('run-select').value=old;
 if(!runs.length){$('loading').textContent='No experiments yet. Start a benchmark to collect results.';return false;}
 return true;
}
async function loadRun(reset=false){
 const id=$('run-select').value;if(!id)return;
 const seq=++loadSequence;
 if(reset){$('content').hidden=true;$('loading').hidden=false;}
 try{
  const next=await json('/api/run?id='+encodeURIComponent(id));if(seq!==loadSequence)return;
  data=next;$('error').hidden=true;$('loading').hidden=true;$('content').hidden=false;
  const choices=data.models.map(m=>m.model);
  if(!choices.includes(selectedModel))selectedModel='all';
  $('model-select').replaceChildren(new Option('All models','all'),...choices.map(m=>new Option(name(m),m)));
  $('model-select').value=selectedModel;
  $('health').className='badge '+data.health.status;$('health').textContent=data.health.status==='running'?'Live experiment':data.health.status==='interrupted'?'Interrupted experiment':data.health.status;
  $('updated').textContent='Read '+new Date(data.updated_at).toLocaleTimeString()+' · refreshes every 15s';
  render();
 }catch(e){if(seq===loadSequence){showError(e);$('loading').hidden=true;}}
}
function render(){if(!data)return;renderOverview();renderGames();renderDiagnostics();renderProtocol();if(currentView==='overview')drawOverview();if(currentView==='games')drawEvaluation();}
function renderOverview(){
 const s=setting(), models=selected(data.models),requests=selected(data.requests),gg=selected(data.games);
 const completed=gg.filter(g=>g.result!=='*'),incomplete=gg.length-completed.length;
 const legal=requests.filter(r=>r.legal===true).length;
 const total=models.reduce((n,m)=>n+m.games,0),score=total?models.reduce((n,m)=>n+m.wins+.5*m.draws,0)/total:null;
 const analyses=selected(data.analysis);
 $('metrics').innerHTML=metric('Recorded games',fmt(completed.length),`${incomplete} incomplete / active`)+metric('Model decisions',fmt(new Set(requests.map(r=>r.decision_id||`${r.bot}:${r.game_id}:${r.ply}`)).size),`${fmt(requests.length)} total attempts`)+metric('Observed score',pct(score),`${fmt(total)} completed games · pooled`)+metric('Move analysis',fmt(analyses.length),`${fmt(legal)} accepted moves available`);
 $('experiment-notes').innerHTML=dl([['Protocol',data.manifest.protocol],['Reasoning',s.effort||s.reasoning_effort||'Unreported'],['Opponent','Stockfish '+(s.opponent_elo||s.fixed_opponent_elo||'')],['Schedule',(s.games||s.max_games||'?')+' games / model']]);
 $('run-note').textContent=data.manifest.invalidated_reason?'INVALID EXPERIMENT · '+data.manifest.invalidated_reason:data.manifest.protocol==='3-subscription'?'Legal final answers are recovered. Correction attempts are capped and recorded.':'Strict baseline: extra prose can forfeit a game, even when its final move is legal.';
 const tbody=$('model-table').querySelector('tbody');
 tbody.innerHTML=models.map(m=>`<tr><td><span class="model-name">${escapeHTML(name(m.model))}</span><span class="model-provider">${escapeHTML(m.provider||'No requests yet')}</span></td><td><span class="result-numbers">${m.wins} / ${m.draws} / ${m.losses}</span><div class="wdl" aria-label="Wins ${m.wins}, draws ${m.draws}, losses ${m.losses}"><span class="win" style="flex:${m.wins}"></span><span class="draw" style="flex:${m.draws}"></span><span class="loss" style="flex:${m.losses}"></span></div><span class="td-note">${m.forfeits} response forfeits</span></td><td>${pct(m.score)}</td><td>${m.games}<span class="td-note">${m.incomplete} incomplete</span></td><td>${fmt(m.median_seconds,1)} / ${fmt(m.p95_seconds,1)} s</td><td>${fmt(m.acpl,1)}<span class="td-note">${m.cp_samples} scored moves</span></td><td>${m.recoveries} / ${m.retries}</td><td>${badge(m.status)}</td></tr>`).join('');
 $('token-table').innerHTML=`<table><thead><tr><th>Model</th><th>Total input</th><th>Total output</th><th>Requests with token counts</th><th>Additional charge</th></tr></thead><tbody>${models.map(m=>`<tr><td class="model-name">${escapeHTML(name(m.model))}</td><td>${m.input_tokens==null?'Unreported':fmt(m.input_tokens)}</td><td>${m.output_tokens==null?'Unreported':fmt(m.output_tokens)}</td><td>${pct(m.token_coverage)}</td><td>Unreported</td></tr>`).join('')}</tbody></table>`;
}
function plot(id,traces,layout={}){
 const base={paper_bgcolor:'#fffef9',plot_bgcolor:'#fffef9',font:{family:'Avenir Next, Trebuchet MS, sans-serif',color:'#516253',size:10},margin:{l:115,r:20,t:15,b:42},height:330,showlegend:false,xaxis:{gridcolor:'#eceee4',zerolinecolor:'#cad1c1',rangemode:'tozero'},yaxis:{gridcolor:'#eceee4',automargin:true},...layout};
 Plotly.react($(id),traces,base,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['select2d','lasso2d'],toImageButtonOptions:{format:'svg',filename:id}});
}
function drawOverview(){
 const models=selected(data.models).filter(m=>m.requests), rr=selected(data.requests),qq=selected(data.analysis);
 plot('latency-chart',models.map((m,i)=>({type:'box',orientation:'h',x:rr.filter(r=>r.bot===m.model).map(r=>r.wall_seconds),name:name(m.model),marker:{color:palette[i%palette.length]},boxpoints:'outliers',hovertemplate:`${escapeHTML(name(m.model))}<br>%{x:.2f} seconds<extra></extra>`})),{xaxis:{title:{text:'Seconds',font:{size:10}},gridcolor:'#eceee4',rangemode:'tozero'}});
 const qa=models.filter(m=>m.cp_samples);
 plot('quality-chart',qa.map((m,i)=>({type:'box',orientation:'h',x:qq.filter(q=>q.bot===m.model&&q.centipawn_loss!=null).map(q=>q.centipawn_loss),name:name(m.model),marker:{color:palette[i%palette.length]},boxpoints:'outliers',hovertemplate:`${escapeHTML(name(m.model))}<br>%{x:.0f} centipawns<extra></extra>`})),{xaxis:{title:{text:'Centipawn loss · lower is better',font:{size:10}},gridcolor:'#eceee4',rangemode:'tozero'},annotations:qa.length?[]:[{text:'Analysis has not been recorded yet',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}]});
}
function renderGames(){
 const color=$('color-select').value;
 const games=selected(data.games).filter(g=>color==='all'||(g.color_llm_white?'white':'black')===color);
 const old=selectedGame;
 if(!games.some(g=>`${g.bot}|${g.game_id}`===selectedGame))selectedGame=games.length?`${games[0].bot}|${games[0].game_id}`:'';
 $('game-select').replaceChildren(...games.map(g=>new Option(`${name(g.bot)} · Game ${g.game_id} · ${g.color_llm_white?'White':'Black'} · ${g.result}`,`${g.bot}|${g.game_id}`)));
 $('game-select').value=selectedGame;
 $('game-empty').hidden=Boolean(games.length);$('game-workspace').hidden=!games.length;$('pgn').disabled=!games.length;
 if(!games.length)return;
 const game=games.find(g=>`${g.bot}|${g.game_id}`===selectedGame);
 const pp=data.plies.filter(p=>p.bot===game.bot&&p.game_id===game.game_id).sort((a,b)=>a.ply-b.ply);
 const rr=data.requests.filter(r=>r.bot===game.bot&&r.game_id===game.game_id).sort((a,b)=>a.ply-b.ply||a.attempt-b.attempt);
 const initial=pp[0]?.fen||rr[0]?.fen;
 frames=initial?[{fen:initial,ply:(pp[0]?.ply||rr[0]?.ply)-1},...pp.map(p=>({...p,fen:p.fen_after,before:p.fen}))]:[];
 if(old!==selectedGame||follow)frameIndex=Math.max(0,frames.length-1);
 frameIndex=Math.min(frameIndex,Math.max(0,frames.length-1));
 $('game-title').innerHTML=`<div><strong>${escapeHTML(name(game.bot))}</strong><br><span class="muted">${escapeHTML(game.opening)} · ${game.color_llm_white?'White':'Black'}</span></div><div>${escapeHTML(game.result)}<br><span class="muted">${escapeHTML(game.termination.replaceAll('_',' '))}</span></div>`;
 $('pgn').disabled=!game.path;
 $('ply-range').max=Math.max(0,frames.length-1);
 $('move-list').replaceChildren(...frames.slice(1).map((f,i)=>{const b=document.createElement('button');b.textContent=`${f.ply}. ${f.move_san}`;b.setAttribute('aria-label',`Ply ${f.ply}, ${f.move_san}`);b.onclick=()=>setFrame(i+1);b.dataset.index=i+1;return b;}));
 renderPosition();
}
const pieces={K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
function renderPosition(){
 const f=frames[frameIndex];if(!f){$('board').replaceChildren();return;}
 const [model,gid]=selectedGame.split('|');const game=data.games.find(g=>g.bot===model&&g.game_id===gid);
 const orientation=flipped?!game.color_llm_white:game.color_llm_white;
 const cells={};f.fen.split(' ')[0].split('/').forEach((row,ri)=>{let file=0;for(const p of row){if(/[1-8]/.test(p))file+=Number(p);else{cells['abcdefgh'[file]+(8-ri)]=p;file++;}}});
 const nodes=[];for(let row=0;row<8;row++)for(let col=0;col<8;col++){
  const file=orientation?col:7-col,rank=orientation?8-row:row+1,square='abcdefgh'[file]+rank,p=cells[square];
  const d=document.createElement('div');d.className='square '+((file+rank)%2?'light':'dark');
  if(f.move_uci&&(square===f.move_uci.slice(0,2)||square===f.move_uci.slice(2,4)))d.classList.add('moved');
  d.title=square+(p?' '+p:'');
  if(p){const piece=document.createElement('span');piece.className='piece '+(p===p.toUpperCase()?'white':'black');piece.textContent=pieces[p];d.append(piece);}
  if(col===0||row===7){const c=document.createElement('span');c.className='coordinate';c.textContent=(col===0?rank:'')+(row===7?'abcdefgh'[file]:'');d.append(c);}nodes.push(d);
 }
 $('board').replaceChildren(...nodes);$('board').setAttribute('aria-label',`Chess position at ply ${f.ply}, ${orientation?'White':'Black'} at bottom. FEN ${f.fen}`);
 $('ply-label').textContent=`Ply ${f.ply} / ${frames.at(-1).ply}`;$('ply-range').value=frameIndex;
 $('first').disabled=$('prev').disabled=frameIndex===0;$('last').disabled=$('next').disabled=frameIndex===frames.length-1;
 for(const button of $('move-list').children)button.classList.toggle('selected',Number(button.dataset.index)===frameIndex);
 const candidates=data.requests.filter(r=>r.bot===model&&r.game_id===gid&&r.ply===f.ply);
 const r=candidates.find(r=>r.legal)||candidates.at(-1);
 const q=r?data.analysis.find(q=>q.request_id===r.request_id):null;
 $('move-title').textContent=f.move_san?`${f.ply}. ${f.move_san}`:'Opening position';
 $('decision-stats').innerHTML=[['Player',r?name(model):f.player||'Opening book'],['Request time',r?fmt(r.wall_seconds,2)+' s':'—'],['Resolution',r?r.parse_method.replaceAll('_',' '):'—'],['Attempt',r?r.attempt:'—'],['Centipawn loss',q?fmt(q.centipawn_loss,1):'Not analyzed'],['Best move',q?q.best.pv[0]:'Not analyzed'],['Mate distance',q?.chosen.mate??'—'],['Reported model',r?.reported_model||'Not independently reported']].map(([k,v])=>`<div><small>${escapeHTML(k)}</small>${escapeHTML(v)}</div>`).join('');
 $('response').textContent=r?.response||r?.error||(r?'Response text is unavailable in this snapshot.':frameIndex===0?'Opening book position. No model request.':'Opponent move. No model response.');
 $('prompt').textContent=r?.prompt||(r?'Prompt text is unavailable in this snapshot.':'No model prompt for this position.');$('fen').textContent=f.fen;
 $('position-note').textContent='Highlighted squares show the last played move. Board starts after the prescribed opening.';
}
function setFrame(index){frameIndex=Math.max(0,Math.min(frames.length-1,index));follow=frameIndex===frames.length-1;renderPosition();}
function drawEvaluation(){
 if(!selectedGame)return;
 const [model,gid]=selectedGame.split('|');const q=data.analysis.filter(r=>r.bot===model&&r.game_id===gid).sort((a,b)=>a.ply-b.ply);
 plot('evaluation-chart',[{type:'scatter',mode:'lines+markers',x:q.map(r=>r.ply),y:q.map(r=>r.chosen.cp==null?null:r.chosen.cp/100*(r.side_to_move==='white'?1:-1)),text:q.map(r=>r.move_san),line:{color:'#315b42',width:2},marker:{size:5},hovertemplate:'Ply %{x} · %{text}<br>%{y:.2f} pawns (White)<extra></extra>'}],{margin:{l:55,r:25,t:15,b:45},xaxis:{title:'Game ply',gridcolor:'#eceee4'},yaxis:{title:'Pawns (White)',gridcolor:'#eceee4',zerolinecolor:'#a2af98'},annotations:q.length?[]:[{text:'No analyzed decisions for this game yet',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}]});
 $('evaluation-chart').removeAllListeners?.('plotly_click');$('evaluation-chart').on('plotly_click',e=>{const ply=e.points[0].x;const index=frames.findIndex(f=>f.ply===ply);if(index>=0)setFrame(index);});
}
function renderDiagnostics(){
 const rr=selected(data.requests),which=$('event-select').value;
 const recovered=rr.filter(r=>r.format_recovered),invalid=rr.filter(r=>r.legal===false),interruptions=rr.filter(r=>r.interrupted),errors=rr.filter(r=>r.error&&!r.interrupted),retries=rr.filter(r=>r.attempt>1);
 $('diagnostic-metrics').innerHTML=metric('Recovered answers',fmt(recovered.length),'Legal move extracted locally')+metric('Invalid attempts',fmt(invalid.length),'Formatting or illegal move')+metric('Additional attempts',fmt(retries.length),'All retry calls retained')+metric('Service failures',fmt(errors.length),`${interruptions.length} interrupted requests tracked separately`);
 let events=rr.filter(r=>r.format_recovered||r.legal===false||r.error||r.attempt>1);
 if(which==='recovered')events=recovered;if(which==='invalid')events=invalid;if(which==='error')events=errors;if(which==='interrupted')events=interruptions;if(which==='retry')events=retries;
 events=events.slice().reverse();
 $('diagnostics-list').innerHTML=events.length?events.map(r=>`<article class="event"><div class="event-head"><div><strong>${escapeHTML(name(r.bot))}</strong><small>Game ${escapeHTML(r.game_id)} · ply ${r.ply} · attempt ${r.attempt} · ${fmt(r.wall_seconds,2)} s</small></div>${badge(r.interrupted?'interrupted':r.error?'service error':r.format_recovered?'recovered':r.legal===false?'invalid output':'retry accepted')}</div><p>${escapeHTML(r.interrupted?'Request interrupted when the run stopped; this is not a model failure or chess loss. The saved position can be resumed.':r.error||r.validation_error||(r.format_recovered?'Accepted by '+r.parse_method:'Accepted after an additional attempt'))}</p><details><summary>Inspect response and request</summary><pre>${escapeHTML(r.response||r.error||'No complete response recorded')}</pre><pre>${escapeHTML(r.prompt)}</pre></details></article>`).join(''):'<div class="empty">No matching events. All attempts remain in the raw data.</div>';
}
function renderProtocol(){
 const s=setting(),v3=data.manifest.protocol==='3-subscription',e=data.manifest.engine||{};
 const fields=[['Protocol',data.manifest.protocol],['Reasoning effort',s.effort||s.reasoning_effort],['Prompt',s.prompt_style||'Minimal FEN + legal moves'],['Maximum attempts',s.attempts||1],['Structured output',v3?(s.structured?'Where supported':'Disabled'):'Disabled'],['Games per model',s.games||s.max_games],['Opening seed',s.seed],['Maximum plies',s.max_plies],['Request timeout',(s.timeout||s.llm_timeout)+' s'],['Opponent thinking time',(s.think_time||.3)+' s'],['Engine',e.name||'Stockfish (see run metadata)'],['Opponent UCI_Elo',s.opponent_elo||s.fixed_opponent_elo]];
 $('protocol-content').innerHTML=`<div class="method-grid"><section class="panel"><p class="eyebrow">EXPERIMENT SETTINGS</p><h2>The conditions of play</h2><dl>${dl(fields)}</dl></section><section class="panel"><p class="eyebrow">READING THE EVIDENCE</p><h2>What these numbers mean</h2><ul><li>Games use matching openings with reversed colors. Opponent UCI_Elo is an engine setting, not a measured rating for an LLM.</li><li>${v3?'Explicit legal final answers can be recovered. Correction requests are capped; every attempt remains visible.':'This strict baseline forfeits responses containing extra prose. Treat those losses as output-compliance failures.'}</li><li>Service failures and unfinished games are excluded from the score denominator.</li><li>Move quality uses a fixed Stockfish node budget. Mate scores are kept separate from centipawn loss.</li><li>Latency includes the client harness. Clients have different startup costs and reasoning implementations.</li><li>Tokens are measured only where reported. API-equivalent estimates are not subscription charges.</li><li>Models never receive Stockfish evaluations. Analysis is performed separately.</li></ul></section></div><section class="panel"><p class="eyebrow">FULL PROVENANCE</p><h2>Manifest</h2><details><summary>Inspect experiment metadata</summary><pre>${escapeHTML(JSON.stringify(data.manifest,null,2))}</pre></details></section>`;
}
function downloadCSV(){
 const rows=selected(data.models);if(!rows.length)return;
 const keys=['model','status','games','wins','draws','losses','incomplete','forfeits','score','requests','decisions','recoveries','retries','service_errors','interruptions','median_seconds','p95_seconds','acpl','cp_samples','input_tokens','output_tokens','token_coverage'];
 const cell=v=>{let s=v==null?'':String(v);if(/^[=+@\-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
 const csv=[keys.join(','),...rows.map(r=>keys.map(k=>cell(r[k])).join(','))].join('\r\n');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=data.id+'-model-summary.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
for(const button of document.querySelectorAll('[data-view]'))button.onclick=()=>{currentView=button.dataset.view;for(const b of document.querySelectorAll('[data-view]')){b.classList.toggle('active',b===button);b.setAttribute('aria-current',b===button?'page':'false');}for(const v of document.querySelectorAll('.view'))v.hidden=v.id!==currentView;$('breadcrumb').textContent=button.textContent.slice(2).trim();if(data){if(currentView==='overview')drawOverview();if(currentView==='games')drawEvaluation();}window.scrollTo({top:0,behavior:'instant'});};
$('run-select').onchange=()=>{selectedGame='';follow=true;loadRun(true);};$('model-select').onchange=()=>{selectedModel=$('model-select').value;render();};$('refresh').onclick=async()=>{try{await loadRuns();await loadRun();}catch(e){showError(e);}};
$('game-select').onchange=()=>{selectedGame=$('game-select').value;follow=true;renderGames();drawEvaluation();};$('color-select').onchange=()=>{renderGames();drawEvaluation();};$('event-select').onchange=renderDiagnostics;
$('first').onclick=()=>setFrame(0);$('prev').onclick=()=>setFrame(frameIndex-1);$('next').onclick=()=>setFrame(frameIndex+1);$('last').onclick=()=>setFrame(frames.length-1);$('flip').onclick=()=>{flipped=!flipped;renderPosition();};$('ply-range').oninput=()=>setFrame(Number($('ply-range').value));
$('pgn').onclick=()=>{const [model,game]=selectedGame.split('|');window.location.href='/api/pgn?'+new URLSearchParams({id:data.id,model,game});};$('export').onclick=downloadCSV;
document.addEventListener('keydown',e=>{if(currentView!=='games'||['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();setFrame(frameIndex-1);}if(e.key==='ArrowRight'){e.preventDefault();setFrame(frameIndex+1);}});
(async()=>{try{if(await loadRuns())await loadRun(true);}catch(e){showError(e);$('loading').hidden=true;}})();
setInterval(async()=>{if(document.hidden)return;try{await loadRuns();await loadRun();}catch(e){showError(e);}},15000);
