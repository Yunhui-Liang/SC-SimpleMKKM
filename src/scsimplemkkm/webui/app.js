const $=s=>document.querySelector(s); const viewsEl=$('#views'),tpl=$('#viewTemplate'); let viewCounter=0,pollTimer=null;

const HISTORICAL_12=[
  {enabled:true,name:'linear',kind:'linear',support_family:'cosine'},
  {enabled:true,name:'polyplus2',kind:'polyplus',degree:2,support_family:'cosine'},
  {enabled:true,name:'polyplus4',kind:'polyplus',degree:4,support_family:'cosine'},
  {enabled:true,name:'poly2',kind:'poly',degree:2,support_family:'abs_cosine'},
  {enabled:true,name:'poly4',kind:'poly',degree:4,support_family:'abs_cosine'},
  ...[-3,-2,-1,0,1,2,3].map(e=>({enabled:true,name:`rbf_2^${e}`,kind:'rbf',rbf_scale:2**e,support_family:'cosine'}))
];
const RBF_7=[-3,-2,-1,0,1,2,3].map(e=>({enabled:true,name:`rbf_2^${e}`,kind:'rbf',rbf_scale:2**e,support_family:'cosine'}));
const rbf1Bank=(a=1)=>[{enabled:true,name:Math.abs(Number(a)-1)<1e-12?'rbf_center':`rbf_alpha_${Number(a)}`,kind:'rbf',rbf_scale:Number(a),support_family:'cosine'}];
const selfTunedRbfBank=(k=7)=>[{enabled:true,name:'rbf_self_tuned',kind:'rbf_self_tuned',local_scale_k:Math.max(1,Math.round(Number(k)||7)),support_family:'cosine'}];
const cloneBank=x=>JSON.parse(JSON.stringify(x));
const kernelState={shared:cloneBank(RBF_7),by_view:{}};

function addView(data={}){
  const node=tpl.content.firstElementChild.cloneNode(true);viewCounter++;
  node.querySelector('.v-name').value=data.name||`view${viewCounter}`;
  node.querySelector('.v-path').value=data.path||'';
  node.querySelector('.v-key').value=data.key||'';
  node.querySelector('.v-k').value=data.k||Number($('#globalK').value||10);
  node.querySelector('.remove').onclick=()=>{saveKernelEditor();node.remove();syncK();syncKernelViewChoices()};
  node.querySelector('.v-name').addEventListener('change',()=>{saveKernelEditor();syncKernelViewChoices();renderEffectiveParamsClient()});
  node.querySelector('.v-k').addEventListener('input',renderEffectiveParamsClient);
  viewsEl.appendChild(node);syncK();syncKernelViewChoices();renderEffectiveParamsClient();
}
function syncK(){const linked=$('#linkK').checked,gk=Number($('#globalK').value||10);document.querySelectorAll('.v-k').forEach(x=>{x.disabled=linked;if(linked)x.value=gk});renderEffectiveParamsClient()}
$('#addView').onclick=()=>addView();$('#linkK').onchange=syncK;$('#globalK').oninput=syncK;
function collectViews(){return[...document.querySelectorAll('.view-row')].map(r=>({name:r.querySelector('.v-name').value.trim(),path:r.querySelector('.v-path').value.trim(),key:r.querySelector('.v-key').value.trim()||null,k:Number(r.querySelector('.v-k').value)}))}

function resolvedFamily(row){
  if(row.support_family&&row.support_family!=='auto')return row.support_family;
  return row.kind==='poly'&&Number(row.degree)%2===0?'abs_cosine':'cosine';
}
function kernelParamValue(spec){if(spec.kind==='rbf')return spec.rbf_scale??1;if(spec.kind==='rbf_self_tuned')return spec.local_scale_k??7;if(spec.kind==='poly'||spec.kind==='polyplus')return spec.degree??2;return''}
function addKernelRow(spec={enabled:true,name:'rbf_custom',kind:'rbf',rbf_scale:1,support_family:'auto'}){
  const tr=document.createElement('tr');tr.className='kernel-row';
  tr.innerHTML=`<td><input class="k-enabled" type="checkbox" ${spec.enabled!==false?'checked':''}></td>
    <td><input class="k-name" type="text" value="${String(spec.name||'').replaceAll('&','&amp;').replaceAll('"','&quot;')}"></td>
    <td><select class="k-kind"><option value="linear">linear</option><option value="polyplus">(1+s)^p</option><option value="poly">s^p</option><option value="rbf">RBF</option><option value="rbf_self_tuned">RBF self-tuned</option></select></td>
    <td><input class="k-param" type="number" step="any" value="${kernelParamValue(spec)}"></td>
    <td><select class="k-family"><option value="auto">auto</option><option value="cosine">cosine</option><option value="abs_cosine">abs_cosine</option></select></td>
    <td class="k-resolved"></td><td><button type="button" class="danger k-remove">×</button></td>`;
  tr.querySelector('.k-kind').value=spec.kind||'rbf';tr.querySelector('.k-family').value=spec.support_family||'auto';
  const update=()=>{const kind=tr.querySelector('.k-kind').value,p=tr.querySelector('.k-param');p.disabled=kind==='linear';p.placeholder=kind==='rbf'?'scale':(kind==='rbf_self_tuned'?'K-local':((kind==='poly'||kind==='polyplus')?'degree':'—'));const tmp={kind,degree:Number(p.value),support_family:tr.querySelector('.k-family').value};tr.querySelector('.k-resolved').textContent=resolvedFamily(tmp)};
  tr.querySelector('.k-kind').addEventListener('change',update);tr.querySelector('.k-family').addEventListener('change',update);tr.querySelector('.k-param').addEventListener('input',update);tr.querySelector('.k-remove').onclick=()=>{tr.remove();updateKernelSummary()};
  $('#kernelRows').appendChild(tr);update();
}
function collectKernelRows(){
  return[...document.querySelectorAll('#kernelRows .kernel-row')].map(tr=>{const kind=tr.querySelector('.k-kind').value,param=Number(tr.querySelector('.k-param').value);const fam=tr.querySelector('.k-family').value;const x={enabled:tr.querySelector('.k-enabled').checked,name:tr.querySelector('.k-name').value.trim(),kind,support_family:fam==='auto'?null:fam};if(kind==='rbf')x.rbf_scale=param;if(kind==='rbf_self_tuned')x.local_scale_k=Math.max(1,Math.round(param||7));if(kind==='poly'||kind==='polyplus')x.degree=Math.max(1,Math.round(param));return x})
}
function renderKernelRows(rows){$('#kernelRows').innerHTML='';(rows||[]).forEach(addKernelRow);updateKernelSummary()}
function currentKernelScope(){if($('#kernelMode').value==='shared')return'shared';return $('#kernelViewSelect').value||collectViews()[0]?.name||'view1'}
function saveKernelEditor(){if(!$('#kernelRows'))return;const rows=collectKernelRows();if($('#kernelMode').value==='shared')kernelState.shared=rows;else kernelState.by_view[currentKernelScope()]=rows;updateKernelSummary()}
function loadKernelEditor(){const mode=$('#kernelMode').value;let rows;if(mode==='shared')rows=kernelState.shared;else{const v=currentKernelScope();if(!kernelState.by_view[v])kernelState.by_view[v]=cloneBank(kernelState.shared);rows=kernelState.by_view[v]}renderKernelRows(rows)}
function syncKernelViewChoices(){const sel=$('#kernelViewSelect');if(!sel)return;const names=collectViews().map(v=>v.name||'view');const old=sel.value;sel.innerHTML=names.map(n=>`<option value="${n.replaceAll('&','&amp;').replaceAll('"','&quot;')}">${n}</option>`).join('');if(names.includes(old))sel.value=old;$('#kernelViewWrap').hidden=$('#kernelMode').value==='shared';if(!$('#kernelRows').children.length)loadKernelEditor()}
function updateKernelSummary(){if(!$('#kernelSummary'))return;const rows=collectKernelRows().filter(x=>x.enabled),fams=[...new Set(rows.map(resolvedFamily))];$('#kernelSummary').textContent=`启用 ${rows.length} 个核；需要 ${fams.length} 个 support family：${fams.join(', ')||'—'}`}
function effectiveSpecsForView(viewName){
  saveKernelEditor();
  if($('#kernelMode').value==='shared')return (kernelState.shared||[]).filter(x=>x.enabled!==false);
  return (kernelState.by_view[viewName]||kernelState.shared||[]).filter(x=>x.enabled!==false);
}
function renderEffectiveParamsClient(serverPlan=null){
  const box=$('#effectiveParamBox');if(!box)return;
  if(serverPlan?.effective_parameters){
    const ep=serverPlan.effective_parameters,sk=ep.support_k_by_view||{};
    const support=ep.support_k_active?Object.entries(sk).map(([v,k])=>`${v}=${k}`).join(', '):'Dense：不使用';
    const scales=[];Object.entries(ep.self_tuned_local_scale_by_view||{}).forEach(([v,rows])=>(rows||[]).forEach(x=>scales.push(`${v}:${x.kernel}=${x.local_scale_k}`)));
    const bad=(ep.violations||[]);box.className=`plan ${bad.length?'bad':'ok'}`;box.innerHTML=`<b>最终有效参数（服务器解析）</b>：k_support {${esc(support||'—')}}；K_scale {${esc(scales.join(', ')||'未使用')}}${bad.length?`<br><b>${bad.map(esc).join('<br>')}</b>`:''}`;return;
  }
  const sparse=$('#engine').value==='scalable_sparse',views=collectViews(),scales=[],bad=[];
  for(const v of views){for(const x of effectiveSpecsForView(v.name)){if(x.kind==='rbf_self_tuned'){const K=Number(x.local_scale_k||7);scales.push(`${v.name}:${x.name||'rbf_self_tuned'}=${K}`);if(sparse&&K>Number(v.k))bad.push(`${v.name}: K_scale=${K} > k_support=${v.k}`)}}}
  const support=sparse?views.map(v=>`${v.name}=${v.k}`).join(', '):'Dense：不使用';
  box.className=`plan ${bad.length?'bad':'ok'}`;box.innerHTML=`<b>当前有效参数</b>：k_support {${esc(support||'—')}}；K_scale {${esc(scales.join(', ')||'未使用')}}${bad.length?`<br><b>配置冲突：</b>${bad.map(esc).join('<br>')}`:''}`;
}
function updateEngineUI(){
  const dense=$('#engine').value==='reference_dense';
  $('#engineHint').textContent=dense?'当前：Standard Dense。k_support 不参与计算；若使用 Self-tuned RBF，K_scale 仍参与局部尺度。':'当前：SC-SimpleMKKM。k_support 决定稀疏 support；Self-tuned K_scale 独立决定局部尺度，要求 K_scale ≤ k_support。';
  renderEffectiveParamsClient();
}
function kernelConfig(){saveKernelEditor();const mode=$('#kernelMode').value;if(mode==='shared')return{mode:'shared',shared:kernelState.shared};const by={};for(const v of collectViews()){if(!kernelState.by_view[v.name])kernelState.by_view[v.name]=cloneBank(kernelState.shared);by[v.name]=kernelState.by_view[v.name]}return{mode:'per_view',shared:kernelState.shared,by_view:by}}
function updateSingleRbfUI(){const key=$('#kernelPreset').value;$('#singleRbfScaleWrap').hidden=key!=='rbf1';$('#selfTunedKWrap').hidden=key!=='rbf_self_tuned'}
function applyKernelPreset(){const key=$('#kernelPreset').value;let preset=HISTORICAL_12;if(key==='rbf7')preset=RBF_7;if(key==='rbf1')preset=rbf1Bank(Number($('#singleRbfScale').value||1));if(key==='rbf_self_tuned')preset=selfTunedRbfBank(Number($('#selfTunedK').value||7));renderKernelRows(cloneBank(preset));saveKernelEditor()}
function copyKernelAllViews(){saveKernelEditor();const rows=cloneBank(collectKernelRows());kernelState.shared=cloneBank(rows);for(const v of collectViews())kernelState.by_view[v.name]=cloneBank(rows);updateKernelSummary()}
$('#kernelRows').addEventListener('input',()=>{saveKernelEditor();renderEffectiveParamsClient()});
$('#kernelRows').addEventListener('change',()=>{saveKernelEditor();renderEffectiveParamsClient()});
$('#kernelMode').addEventListener('change',()=>{syncKernelViewChoices();loadKernelEditor()});
$('#kernelViewSelect').addEventListener('change',()=>{loadKernelEditor()});
$('#kernelPreset').addEventListener('change',updateSingleRbfUI);$('#singleRbfScale').addEventListener('change',()=>{if($('#kernelPreset').value==='rbf1')applyKernelPreset()});$('#selfTunedK').addEventListener('input',()=>{if($('#kernelPreset').value==='rbf_self_tuned')applyKernelPreset();else renderEffectiveParamsClient()});
$('#applyKernelPreset').onclick=()=>{applyKernelPreset();renderEffectiveParamsClient()};$('#copyKernelAllViews').onclick=()=>{copyKernelAllViews();renderEffectiveParamsClient()};$('#addKernel').onclick=()=>{addKernelRow();updateKernelSummary();renderEffectiveParamsClient()};
$('#engine').addEventListener('change',updateEngineUI);
updateSingleRbfUI();

function config(){const gt=$('#groundTruth').value.trim();return{engine:$('#engine').value,task_name:$('#taskName').value.trim()||null,runs_root:$('#runsRoot').value.trim()||'runs',clusters:Number($('#clusters').value),auto_clusters_from_ground_truth:$('#autoClusters').checked,k:Number($('#globalK').value),views:collectViews(),kernel_bank:kernelConfig(),ground_truth:gt||null,search:$('#search').value,ann_backend:$('#annBackend').value,support_mode:$('#supportMode').value,support_policy:$('#supportPolicy').value,diffusion_power:Number($('#diffusionPower').value),candidate_factor:Number($('#candidateFactor').value),query_batch_size:Number($('#queryBatch').value),ef_search:Number($('#efSearch').value),hnsw_m:Number($('#hnswM').value),seed:Number($('#seed').value),mmap:true,memory_mode:$('#memoryMode').value,workspace:$('#workspaceDir').value.trim()||'runs/workspace',discretization:{backend:$('#discBackend').value,block_rows:Number($('#discBlock').value),max_iter:Number($('#discIter').value),tol:1e-4,init_sample:200000,init_restarts:Number($('#discRestarts').value)},trajectory:{enabled:$('#trajEnabled').checked,full_threshold:Number($('#trajFullThreshold').value),probe_size:Number($('#trajProbe').value),replicates:Number($('#trajReplicates').value),max_iter:30,tol:1e-4},reference_protocol:{official_eval_enabled:$('#officialEval').checked,official_eval_rounds:20,official_eval_replicates:10,eig_tol:1e-10},spectral_diagnostics:{enabled:$('#spectralDiag').checked,max_n:Number($('#spectralMaxN').value),tol:1e-5,maxiter:400},evaluation:{enabled:$('#evalEnabled').checked,block_rows:Number($('#discBlock').value)},resources:{max_ram_fraction:Number($('#ramFrac').value),reserve_gb:Number($('#reserveGb').value),max_cpu_fraction:Number($('#cpuFrac').value),leave_cores:Number($('#leaveCores').value),max_threads:Number($('#maxThreads').value)||null,monitor_interval:Number($('#monitorInterval').value),allow_oom_risk:$('#allowOom').checked}}}
async function jfetch(url,opts={}){const r=await fetch(url,opts),j=await r.json();if(!r.ok)throw Object.assign(new Error(j.error||r.statusText),{data:j});return j}
function bytes(n){if(n==null)return'—';let x=Number(n);for(const u of['B','KiB','MiB','GiB','TiB']){if(x<1024||u==='TiB')return`${x.toFixed(x<10?2:1)} ${u}`;x/=1024}}
function rate(n){return n==null?'—':`${bytes(n)}/s`}
function fmt(v,d=2){return v==null?'—':Number(v).toFixed(d)}
$('#checkPaths').onclick=async()=>{try{const r=await jfetch('/api/check-paths',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({views:collectViews()})});$('#pathCheck').textContent=r.views.map(v=>`${v.exists?'✓':'✗'} ${v.name}: ${v.path}${v.size?` (${bytes(v.size)})`:''}`).join(' | ')}catch(e){$('#pathCheck').textContent=e.message}};
function renderPlan(p){renderEffectiveParamsClient(p);const box=$('#planBox'),m=p.memory||{},e=m.estimate||{},pol=m.policy||{},kp=p.kernel_plan||{},gt=p.ground_truth||{};const kt=Object.entries(kp).map(([v,x])=>`${v}: ${x.kernels} kernels / ${(x.families||[]).join('+')}`).join('；');if(p.resolved_clusters&&$('#autoClusters').checked)$('#clusters').value=p.resolved_clusters;const gtxt=gt.available?`GT: n=${Number(gt.n).toLocaleString()} / classes=${gt.true_clusters} / c=${p.resolved_clusters??p.clusters}`:'GT: 未提供';const ws=[...(p.warnings||[]),...(m.reasons||[])];box.className=`plan ${p.allowed&&!ws.length?'ok':(p.allowed?'':'bad')}`;box.innerHTML=`<strong>${p.allowed?'✓ 可以运行':'✗ 资源预估超预算'}</strong><br>引擎：${p.engine==='reference_dense'?'Standard Dense SimpleMKKM':'SC-SimpleMKKM'}<br>${gtxt}<br>核配置：${kt||'—'}<br>保守峰值估计：${e.total_human||'—'}；RAM预算：${bytes(pol.memory_budget_bytes)}；推荐线程：${p.recommended_threads??'—'}；机器：${p.system?.physical_cores??'—'} physical / ${p.system?.logical_cores??'—'} logical cores，RAM ${bytes(p.system?.ram_total_bytes)}；模式 ${p.memory_mode_resolved||'standard'}；workspace SSD ${p.workspace_disk?`${bytes(p.workspace_disk.free_bytes)} 可用 / 预计 ${bytes(p.workspace_disk.scratch_estimate_bytes)}`:'—'}${ws.length?`<br><b>提示：</b>${ws.map(esc).join('<br>')}`:''}`}
async function plan(){const p=await jfetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config())});renderPlan(p);return p} $('#planBtn').onclick=()=>plan().catch(e=>{$('#planBox').className='plan bad';$('#planBox').textContent=e.message});
$('#runBtn').onclick=async()=>{$('#resultPanel').hidden=true;try{const p=await plan();if(!p.allowed&&!$('#allowOom').checked)return;const r=await jfetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config())});$('#activeRunId').textContent=`Run ID: ${r.run_id}`;$('#runBtn').disabled=true;$('#cancelBtn').disabled=false;await refreshRuns();poll()}catch(e){alert(e.message)}};
$('#newTaskBtn').onclick=()=>{$('#resultPanel').hidden=true;$('#taskName').value='';$('#activeRunId').textContent='';window.scrollTo({top:0,behavior:'smooth'})};
$('#cancelBtn').onclick=()=>jfetch('/api/cancel',{method:'POST'}).catch(()=>{});
function statusClass(s){return`status ${s||'idle'}`}
function renderTable(rows,cols){if(!rows?.length)return'<p class="muted">无</p>';return`<div class="table-wrap"><table><thead><tr>${cols.map(c=>`<th>${c.label}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${c.fn?c.fn(r[c.key]):(r[c.key]??'')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
function drawLine(canvas,series,opts={}){const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height,p=42;ctx.clearRect(0,0,W,H);if(!series.length)return;const all=series.flatMap(s=>(s.values||[]).filter(Number.isFinite));if(!all.length){ctx.fillStyle='#596273';ctx.font='15px sans-serif';ctx.fillText('暂无轨迹数据',p,p);return;}let ymin=opts.ymin??Math.min(...all),ymax=opts.ymax??Math.max(...all);if(ymax<=ymin){const pad=Math.max(Math.abs(ymax)*1e-6,1e-9);ymin-=pad;ymax+=pad}else if(opts.ymin==null||opts.ymax==null){const pad=(ymax-ymin)*0.06;if(opts.ymin==null)ymin-=pad;if(opts.ymax==null)ymax+=pad}ctx.strokeStyle='#d5dbe3';ctx.strokeRect(p,20,W-p-14,H-62);const palette=['#2563eb','#ef4444','#16a34a','#9333ea','#0891b2','#c2410c','#7c3aed','#4d7c0f'];series.forEach((ss,si)=>{const col=ss.color||palette[si%palette.length];ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=2;ctx.beginPath();const pts=[];(ss.values||[]).forEach((v,i)=>{if(!Number.isFinite(v))return;const x=p+((ss.values.length===1?0.5:i/Math.max(1,ss.values.length-1)))*(W-p-14),y=20+(ymax-v)/(ymax-ymin)*(H-82);pts.push([x,y]);if(pts.length===1)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke();pts.forEach(([x,y])=>{ctx.beginPath();ctx.arc(x,y,3.5,0,Math.PI*2);ctx.fill()});if(ss.label){ctx.font='13px sans-serif';ctx.fillText(ss.label,p+si*110,H-15)}});ctx.fillStyle='#596273';ctx.font='13px sans-serif';ctx.fillText(Number(ymax).toPrecision(5),2,28);ctx.fillText(Number(ymin).toPrecision(5),2,H-42)}
function drawObjective(vals){drawLine($('#objCanvas'),[{values:vals,color:'#2563eb',label:'J (lower)'}])}
function renderCores(vals){const el=$('#coreHeat');el.innerHTML='';(vals||[]).forEach((v,i)=>{const d=document.createElement('div');d.className='core';d.title=`CPU ${i}: ${fmt(v,1)}%`;d.style.setProperty('--load',`${Math.max(0,Math.min(100,v))}%`);d.textContent=i;el.appendChild(d)})}
function renderMetrics(metrics){if(!metrics?.length)return;const last=metrics.at(-1);$('#procCpu').textContent=last.process_cpu_percent==null?'—':`${fmt(last.process_cpu_percent,1)}%`;$('#procCores').textContent=last.process_cpu_cores==null?'—':fmt(last.process_cpu_cores,2);$('#sysCpu').textContent=last.system_cpu_percent==null?'—':`${fmt(last.system_cpu_percent,1)}%`;$('#procRam').textContent=bytes(last.process_rss_bytes);$('#availRam').textContent=bytes(last.ram_available_bytes);$('#diskRate').textContent=`${rate(last.disk_read_bytes_per_sec)} / ${rate(last.disk_write_bytes_per_sec)}`;renderCores(last.per_core_cpu_percent||[]);drawLine($('#cpuCanvas'),[{values:metrics.map(x=>x.process_cpu_percent),color:'#2563eb'},{values:metrics.map(x=>x.system_cpu_percent),color:'#ef4444'}],{ymin:0,ymax:100});const total=last.ram_total_bytes||1;drawLine($('#ramCanvas'),[{values:metrics.map(x=>100*(x.process_rss_bytes||0)/total),color:'#2563eb'},{values:metrics.map(x=>100*(1-(x.ram_available_bytes||0)/total)),color:'#ef4444'}],{ymin:0,ymax:100})}
function flowKey(stage=''){if(stage.startsWith('normalize'))return'normalize';if(stage.startsWith('ann'))return'ann';if(stage.startsWith('support'))return'support';if(stage.startsWith('kernel')||stage.startsWith('operator')||stage.startsWith('dense_kernel'))return'operators';if(stage.startsWith('simplemkkm'))return'simplemkkm';if(stage.startsWith('discretize')||stage.startsWith('postprocess'))return'discretize';if(stage.startsWith('evaluate'))return'evaluate';return''}
function renderFlow(stage){const k=flowKey(stage);document.querySelectorAll('#flow [data-key]').forEach(x=>x.classList.toggle('active',x.dataset.key===k))}
function renderBlocks(blocks){const rows=(blocks||[]).slice(-30).reverse().map(b=>({kind:b.block_kind||'',stage:b.stage||'',view:b.view||'',family:b.family||'',rows:(b.rows_start!=null&&b.rows_end!=null)?`${Number(b.rows_start).toLocaleString()}–${Number(b.rows_end).toLocaleString()}`:'',iter:b.lloyd_iter??b.inner_iter??''}));$('#blockTable').innerHTML=renderTable(rows,[{key:'kind',label:'Block kind'},{key:'stage',label:'Stage'},{key:'view',label:'View'},{key:'family',label:'Family'},{key:'rows',label:'Rows'},{key:'iter',label:'Iter'}])}
function metricCards(obj){if(!obj)return'<p class="muted">未计算。</p>';const rows=[];for(const [group,v] of Object.entries(obj)){for(const [k,x] of Object.entries(v||{}))rows.push({group,metric:k,value:typeof x==='number'?fmt(x,6):x})}return renderTable(rows,[{key:'group',label:'Type'},{key:'metric',label:'Metric'},{key:'value',label:'Value'}])}
function renderResult(r){if(r.effective_parameters)renderEffectiveParamsClient(r.effective_parameters);
  $('#resultPanel').hidden=false;
  const cards=[['Run ID',r.run_id||'—'],['Version',r.software_version||'—'],['Engine',r.engine_label||r.engine||'—'],['SimpleMKKM',r.simplemkkm_status||'—'],['Views',Object.keys(r.shapes||{}).length],['Kernels',r.gamma?.length||0],['Fit seconds',fmt(r.fit_seconds||0,2)],['Peak RAM',bytes(r.peak_rss_bytes)],['Discretization',r.postprocess_backend||'—'],['Labels',r.labels_path||'—']];
  $('#summaryCards').innerHTML=cards.map(([a,b])=>`<div class="card"><span>${a}</span><strong>${b}</strong></div>`).join('');
  const warns=r.warnings||r.diagnostics?.warnings||[]; $('#warningsBox').innerHTML=warns.length?`<div class="plan bad"><strong>诊断警告</strong><br>${warns.map(esc).join('<br>')}</div>`:'<div class="plan ok"><strong>诊断</strong>：未发现已配置的结构/数值警告。</div>';
  const D=r.diagnostics||{}, sd=D.spectral_boundary||{}, gd=D.ground_truth||{}, sol=D.solver||{}; const drows=[{item:'GT classes / requested c',value:gd.available?`${gd.true_clusters} / ${gd.requested_c}`:'—'},{item:'Final inner residual',value:sol.final_inner_residual==null?'—':Number(sol.final_inner_residual).toExponential(3)},{item:'lambda_c',value:sd.lambda_c==null?'—':Number(sd.lambda_c).toPrecision(7)},{item:'lambda_c+1',value:sd.lambda_c_plus_1==null?'—':Number(sd.lambda_c_plus_1).toPrecision(7)},{item:'boundary gap',value:sd.gap==null?'—':Number(sd.gap).toExponential(3)},{item:'relative gap',value:sd.relative_gap==null?'—':Number(sd.relative_gap).toExponential(3)},{item:'spectral diagnostic',value:sd.status||'—'}]; $('#diagnosticSummary').innerHTML=renderTable(drows,[{key:'item',label:'Item'},{key:'value',label:'Value'}]);
  const T=r.iteration_metrics||[]; const objs=T.length?T.map(x=>Number(x.objective)):r.objective||[]; drawObjective(objs);
  const acceptedUpdates=Math.max(0,T.length-1); const j0=objs.length?Number(objs[0]):NaN,jf=objs.length?Number(objs.at(-1)):NaN;
  let gmove=0;if(T.length>1&&Array.isArray(T[0].gamma)&&Array.isArray(T.at(-1).gamma)){for(let i=0;i<Math.min(T[0].gamma.length,T.at(-1).gamma.length);i++)gmove=Math.max(gmove,Math.abs(Number(T.at(-1).gamma[i])-Number(T[0].gamma[i])))}
  const OS=r.optimization_summary||{}; const mono=OS.accepted_objective_monotone===false?'NO':'YES'; const maxUp=Number(OS.max_positive_objective_step||0);
  $('#optimizationNote').innerHTML=acceptedUpdates===0?`<span class="danger-text"><b>没有 accepted outer update。</b> 权重没有真正迭代；请看 strict acceptance / termination / residual。</span>`:`accepted updates=<b>${acceptedUpdates}</b>；strict J: ${Number.isFinite(j0)?j0.toPrecision(8):'—'} → ${Number.isFinite(jf)?jf.toPrecision(8):'—'}；monotone=<b>${mono}</b>；max positive step=${maxUp.toExponential(2)}；max |Δγ|=${gmove.toExponential(3)}；termination=${esc(r.simplemkkm_termination||'—')}`;
  const LS=r.line_search||[];drawLine($('#lineSearchCanvas'),[{values:LS.map(x=>Number(x.objective)),label:'trial J'}]);
  $('#lineSearchTable').innerHTML=renderTable(LS.slice(-40),[{key:'outer',label:'Outer'},{key:'trial',label:'Trial'},{key:'phase',label:'Phase'},{key:'step',label:'Step',fn:x=>Number(x).toExponential(2)},{key:'objective',label:'J',fn:x=>Number(x).toPrecision(9)},{key:'inner_residual',label:'Residual',fn:x=>Number(x).toExponential(2)},{key:'certified',label:'Certified'}]);
  drawLine($('#qualityCanvas'),[
    {values:T.map(x=>Number(x.ACC)),label:'ACC'},
    {values:T.map(x=>Number(x.NMI)),label:'NMI'},
    {values:T.map(x=>Number(x.ARI)),label:'ARI'}
  ],{ymin:0,ymax:1});
  const scopes=[...new Set(T.map(x=>x.metric_scope).filter(Boolean))]; const mn=T.find(x=>x.metric_n)?.metric_n;
  $('#trajectoryNote').textContent=T.length?`accepted states=${T.length}（真正更新 ${acceptedUpdates} 次）；这里的 ACC/NMI/ARI 是 accepted-iterate 诊断轨迹，不等同于最终 evaluation.json；metric scope=${scopes.join(', ')||'OFF'}${mn?`；metric n=${Number(mn).toLocaleString()}`:''}；诊断离散化使用固定随机种子。论文最终表格应使用 Final Evaluation。`:'无 accepted-iteration 轨迹。';
  const m=Math.max(0,...T.map(x=>Array.isArray(x.gamma)?x.gamma.length:0)); const gs=[];for(let j=0;j<m;j++)gs.push({values:T.map(x=>Array.isArray(x.gamma)?Number(x.gamma[j]):NaN),label:`γ${j+1}`});drawLine($('#gammaCanvas'),gs,{ymin:0});
  $('#evaluation').innerHTML=metricCards(r.evaluation);
  const vw=renderTable(r.view_weight_table||[],[{key:'view',label:'View'},{key:'sum_gamma',label:'Σγ',fn:x=>fmt(x,5)},{key:'sum_gamma2',label:'Σγ²',fn:x=>fmt(x,5)}]);
  const kw=renderTable(r.gamma_table||[],[{key:'view',label:'View'},{key:'kernel',label:'Kernel'},{key:'gamma',label:'γ',fn:x=>fmt(x,6)},{key:'gamma2',label:'γ²',fn:x=>fmt(x,6)}]);
  $('#weights').innerHTML=`<h4>按视图汇总</h4>${vw}<h4>逐核</h4>${kw}`;
  const grows=[];Object.entries(r.graphs||{}).forEach(([view,fams])=>Object.entries(fams).forEach(([fam,g])=>grows.push({view,fam,...g})));
  $('#graphs').innerHTML=grows.length?renderTable(grows,[{key:'view',label:'View'},{key:'fam',label:'Family'},{key:'backend',label:'Backend'},{key:'nnz_directed',label:'Edges'},{key:'support_bytes',label:'Support RAM',fn:bytes},{key:'mean_degree',label:'Mean degree',fn:x=>fmt(x,2)},{key:'isolated',label:'Isolated'},{key:'connected_components',label:'Components'}]):'<p class="muted">Standard Dense engine 不使用 sparse support。</p>';
  const pr=renderTable(r.profile||[],[{key:'stage',label:'Stage'},{key:'seconds',label:'Seconds',fn:x=>fmt(x,3)},{key:'cpu_core_equiv',label:'CPU cores(avg)',fn:x=>fmt(x,2)},{key:'read_bytes',label:'Read',fn:bytes},{key:'write_bytes',label:'Write',fn:bytes},{key:'rss_start_bytes',label:'RSS start',fn:bytes},{key:'rss_peak_bytes',label:'RSS peak',fn:bytes},{key:'rss_end_bytes',label:'RSS end',fn:bytes},{key:'ram_available_min_bytes',label:'Min available RAM',fn:bytes}]);
  $('#profile').innerHTML=`${pr}<p class="muted">raw: <code>${r.profile_csv||''}</code> / <code>${r.profile_json||''}</code></p>`;
  $('#handoff').innerHTML=`<p><code>${r.run_report_html||''}</code><br><code>${r.paper_summary_csv||''}</code><br><code>${r.artifact_manifest||''}</code><br><code>${r.handoff_md||''}</code><br><code>${r.handoff_json||''}</code><br><code>${r.events_jsonl||''}</code><br><code>${r.resources_jsonl||''}</code></p>`;
  const root=encodeURIComponent($('#runsRoot').value.trim()||'runs');$('#runReportLink').href=r.run_id?`/run-report?run_id=${encodeURIComponent(r.run_id)}&root=${root}`:'/run-report';
}

async function poll(){clearTimeout(pollTimer);try{const s=await jfetch('/api/status');$('#statusBadge').className=statusClass(s.status);$('#statusBadge').textContent=s.status;$('#elapsed').textContent=s.elapsed_seconds==null?'':`${fmt(s.elapsed_seconds,1)} s`;$('#log').textContent=(s.log||[]).join('\n')||(s.error||'等待输出…');const ev=s.current_event||{},pct=ev.progress==null?0:Math.round(ev.progress*100);$('#stageText').textContent=ev.message||ev.stage||'等待运行…';$('#stagePct').textContent=`${pct}%`;$('#progressFill').style.width=`${pct}%`;renderFlow(ev.stage||'');renderMetrics(s.metrics||[]);renderBlocks(s.blocks||[]);if(s.plan)renderPlan(s.plan);const done=['done','failed','cancelled'].includes(s.status);$('#runBtn').disabled=s.status==='running';$('#cancelBtn').disabled=s.status!=='running';if(s.status==='done'&&s.result){renderResult(s.result);$('#activeRunId').textContent=s.result.run_id?`Run ID: ${s.result.run_id}`:'';refreshRuns()}if(!done&&s.status==='running')pollTimer=setTimeout(poll,1000)}catch(e){$('#log').textContent=e.message;pollTimer=setTimeout(poll,2000)}}
function esc(x){return String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;')}
async function refreshRuns(){try{const root=$('#runsRoot').value.trim()||'runs';const d=await jfetch(`/api/runs?root=${encodeURIComponent(root)}`);const rows=d.runs||[];if(!rows.length){$('#runHistory').innerHTML='<p class="muted">暂无历史任务。</p>';return}$('#runHistory').innerHTML=`<div class="table-wrap"><table><thead><tr><th>Run ID</th><th>任务</th><th>引擎</th><th>状态</th><th>n</th><th>c</th><th>Kernels</th><th>时间(s)</th><th>Peak RAM</th><th>ACC</th><th>NMI</th><th>ARI</th><th>操作</th></tr></thead><tbody>${rows.map(r=>`<tr><td><code>${esc(r.run_id)}</code></td><td>${esc(r.task_name)}</td><td>${esc(r.engine||'—')}</td><td>${esc(r.status)}</td><td>${r.n==null?'—':Number(r.n).toLocaleString()}</td><td>${r.clusters??'—'}</td><td>${r.kernels??'—'}</td><td>${r.total_seconds==null?'—':fmt(r.total_seconds,1)}</td><td>${bytes(r.peak_rss_bytes)}</td><td>${r.ACC==null?'—':fmt(r.ACC,4)}</td><td>${r.NMI==null?'—':fmt(r.NMI,4)}</td><td>${r.ARI==null?'—':fmt(r.ARI,4)}</td><td><button type="button" class="secondary hist-view" data-run="${esc(r.run_id)}">查看</button> <a target="_blank" href="/run-report?run_id=${encodeURIComponent(r.run_id)}&root=${encodeURIComponent(root)}">报告</a></td></tr>`).join('')}</tbody></table></div>`;document.querySelectorAll('.hist-view').forEach(b=>b.addEventListener('click',()=>loadHistoricalRun(b.dataset.run)))}catch(e){$('#runHistory').innerHTML=`<p class="danger-text">${esc(e.message)}</p>`}}
async function loadHistoricalRun(runId){try{const root=$('#runsRoot').value.trim()||'runs';const r=await jfetch(`/api/run-detail?run_id=${encodeURIComponent(runId)}&root=${encodeURIComponent(root)}`);renderResult(r);$('#activeRunId').textContent=`查看历史: ${runId}`;document.querySelector('#resultPanel').scrollIntoView({behavior:'smooth'})}catch(e){alert(e.message)}}
async function generatePaperStats(){try{const root=$('#runsRoot').value.trim()||'runs';$('#paperStats').textContent='正在扫描 run artifacts…';const r=await jfetch('/api/paper-stats',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({root})});const warns=r.warnings||[];const gs=r.groups||[];$('#paperStats').innerHTML=`✓ runs=<b>${r.run_count}</b>，groups=<b>${r.group_count}</b>，dense/sparse paired=<b>${r.pair_count}</b>；输出：<code>${esc(r.out_dir)}</code>${warns.length?`<br><span class="danger-text">${warns.slice(0,6).map(esc).join('<br>')}${warns.length>6?`<br>…还有 ${warns.length-6} 条`:''}</span>`:''}${gs.length?`<br>${gs.map(g=>`${esc(g.task_name)}/${esc(g.engine)}: n=${g.runs}, ACC=${g.ACC_mean==null?'—':fmt(g.ACC_mean,4)}±${g.ACC_std==null?'—':fmt(g.ACC_std,4)}`).join('；')}`:''}`;}catch(e){$('#paperStats').innerHTML=`<span class="danger-text">${esc(e.message)}</span>`}}
$('#paperStatsBtn').onclick=generatePaperStats;$('#refreshRuns').onclick=refreshRuns;$('#runsRoot').addEventListener('change',refreshRuns);

async function syncSharedWorkspace(){
  const w=await jfetch('/api/workspace');
  $('#datasetRoot').value=w.dataset_root;$('#mvCacheRoot').value=w.mv_root;
  $('#paperDataRoot').value=w.data_root;$('#runsRoot').value=w.runs_root;
}
$('#saveSharedWorkspace').onclick=async()=>{
  try{
    await jfetch('/api/workspace',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_root:$('#datasetRoot').value,mv_root:$('#mvCacheRoot').value,data_root:$('#paperDataRoot').value,runs_root:$('#runsRoot').value})});
    await initDatasets();await refreshRuns();$('#datasetStatus').textContent='✓ 共享目录已保存，PaperLab 与单机控制台将读取同一目录。';
  }catch(e){$('#datasetStatus').textContent='共享目录保存失败：'+e.message}
};
(async function init(){addView();renderKernelRows(cloneBank(RBF_7));syncKernelViewChoices();updateEngineUI();try{await syncSharedWorkspace();}catch(e){console.warn('workspace unavailable',e)}try{const d=await jfetch('/api/doctor');$('#envBadge').textContent=`${d.cpu_model||'CPU'} · ${d.physical_cores}C/${d.logical_cores}T · RAM ${bytes(d.ram_total_bytes)} · FAISS ${d.faiss?'✓':'–'}` }catch(e){$('#envBadge').textContent='environment unavailable'}await initDatasets();refreshRuns();poll()})();

let datasetCatalog=[], pendingDatasetAutoApply=false, initialCatalogApplied=false;
function selectedDataset(){return datasetCatalog.find(x=>x.id===$('#datasetSelect').value)||null}
function renderDatasetInfo(){
  const x=selectedDataset();if(!x){$('#datasetInfo').textContent='无数据集信息。';return}
  const eng=x.recommended_engine==='reference_dense'?'Standard Dense SimpleMKKM':'SC-SimpleMKKM';
  $('#datasetInfo').innerHTML=`<b>${esc(x.title)}</b> · n=${Number(x.n||0).toLocaleString()} · dims=${esc(x.d??'—')} · c=${x.c??'—'} · ${esc(x.tier||'')}<br>推荐：<b>${eng}</b>，k_support=${x.recommended_k??10}${x.recommended_kernel_preset?`，kernel=${esc(x.recommended_kernel_preset)}`:''}；状态：${x.ready?'✓ 已就绪':esc((x.status||'not_prepared')+'：'+(x.status_reason||'需在 PaperLab 下载/修复'))}。<br>${esc(x.description||'')}`;
  $('#prepareDataset').textContent=x.ready?'一键载入任务':(x.kind==='paperlab_recipe'?'请先在 PaperLab 准备':'下载/处理并自动载入');
}
function applyDataset(r,info=null){
  if(!r)return;const x=info||selectedDataset()||{};
  viewsEl.innerHTML='';viewCounter=0;
  const kval=Number(r.recommended_k||x.recommended_k||$('#globalK').value||10);$('#globalK').value=kval;
  if(r.view)addView({name:'view1',path:r.view,k:kval});
  else if(Array.isArray(r.views))r.views.forEach((v,i)=>addView({name:v.name||`view${i+1}`,path:v.path||v,k:Number(v.k||kval)}));
  if(r.labels)$('#groundTruth').value=r.labels;
  const cc=Number(r.classes||r.n_clusters||x.c||0);if(cc>=2)$('#clusters').value=cc;
  $('#autoClusters').checked=true;
  const eng=r.recommended_engine||x.recommended_engine||(Number(r.n||r.n_samples||0)<=3000?'reference_dense':'scalable_sparse');$('#engine').value=eng;
  $('#search').value=eng==='reference_dense'?'exact':'ann';
  $('#taskName').value=String(r.dataset||r.name||x.title||x.id||'dataset').replaceAll(' ','-');
  updateEngineUI();syncK();syncKernelViewChoices();
  if(Array.isArray(r.views)&&r.views.length>1){$('#kernelMode').value='shared';$('#kernelPreset').value=r.recommended_kernel_preset||x.recommended_kernel_preset||'rbf_self_tuned';const K=Number(r.single_kernel_rule?.local_scale_k||r.local_scale_k||7);$('#selfTunedK').value=K;updateSingleRbfUI();applyKernelPreset()}
  renderEffectiveParamsClient();
  $('#datasetStatus').textContent=`✓ 已载入任务：${r.dataset||r.name||x.title||''}；views=${Array.isArray(r.views)?r.views.length:1}；c=${$('#clusters').value}；engine=${eng}`;
}
async function initDatasets(){
  try{
    const root=$('#datasetRoot').value.trim()||'datasets';
    const mvroot=$('#mvCacheRoot').value.trim()||'dataset_cache';const paperroot=$('#paperDataRoot').value.trim()||'paper_workspace/data';const d=await jfetch(`/api/datasets?root=${encodeURIComponent(root)}&mv_root=${encodeURIComponent(mvroot)}&paper_root=${encodeURIComponent(paperroot)}`);datasetCatalog=d.catalog||[];const sel=$('#datasetSelect');const old=sel.value;
    const groups={};datasetCatalog.forEach(x=>{const g=x.category||'Other';(groups[g]??=[]).push(x)});
    sel.innerHTML=Object.entries(groups).map(([g,rows])=>`<optgroup label="${esc(g)}">${rows.map(x=>`<option value="${esc(x.id)}">${esc(x.title)} · ${Number(x.n).toLocaleString()}×${x.d} · c=${x.c}${x.ready?' ✓':''}</option>`).join('')}</optgroup>`).join('');
    if(datasetCatalog.some(x=>x.id===old))sel.value=old;renderDatasetInfo();renderDatasetJob(d.job);
    if(!initialCatalogApplied){const x=selectedDataset();if(x&&x.ready&&x.bundled&&x.view&&x.labels){initialCatalogApplied=true;applyDataset({dataset:x.title,view:x.view,labels:x.labels,classes:x.c,n:x.n},x)}}
  }catch(e){$('#datasetStatus').textContent='dataset catalog unavailable: '+e.message}
}
let lastPreparedDataset=null;
function renderDatasetJob(j){
  if(!j)return;let t=`${j.status}: ${j.message||''}`;
  if(j.result){lastPreparedDataset=j.result;if(j.result.batch){t+=` · batch prepared=${j.result.prepared_count||0}, errors=${j.result.error_count||0}`;pendingDatasetAutoApply=false}else{t+=` · ready: ${j.result.view||j.result.name||j.result.dataset||''}${Array.isArray(j.result.views)?` (${j.result.views.length} views)`:''}`;if(pendingDatasetAutoApply&&j.status==='done'){pendingDatasetAutoApply=false;applyDataset(j.result)}}}
  if(j.error)t+=' · '+j.error.split('\n').slice(-2).join(' ');$('#datasetStatus').textContent=t;
}
$('#datasetSelect').addEventListener('change',()=>{
  renderDatasetInfo();
  const x=selectedDataset();if(!x)return;
  // Always prefill metadata immediately so c/k/engine/task name cannot be forgotten.
  if(Number(x.c)>=2)$('#clusters').value=Number(x.c);
  if(Number(x.recommended_k)>0){$('#globalK').value=Number(x.recommended_k);syncK()}
  if(x.recommended_engine){$('#engine').value=x.recommended_engine;$('#search').value=x.recommended_engine==='reference_dense'?'exact':'ann';updateEngineUI()}
  $('#taskName').value=String(x.title||x.id||'dataset').replaceAll(' ','-');
  // Bundled ready datasets require no second confirmation: selecting one loads paths/GT too.
  if(x.ready&&x.bundled&&x.view&&x.labels)applyDataset({dataset:x.title,view:x.view,labels:x.labels,classes:x.c,n:x.n},x);
  else if(x.ready&&Array.isArray(x.views)&&x.views.length>1)applyDataset({dataset:x.title,views:x.views,labels:x.labels,classes:x.c,n:x.n,recommended_engine:x.recommended_engine,recommended_k:x.recommended_k,recommended_kernel_preset:x.recommended_kernel_preset},x);
});
$('#datasetRoot').addEventListener('change',initDatasets);$('#mvCacheRoot').addEventListener('change',initDatasets);$('#paperDataRoot').addEventListener('change',initDatasets);
$('#prepareDataset').onclick=async()=>{
  try{const x=selectedDataset();if(!x)return;if(x.kind==='paperlab_recipe'&&!x.ready)throw new Error(`PaperLab 数据集 ${x.id} 尚未准备好：${x.status} ${x.status_reason||''}。请在 PaperLab 准备后点击刷新。`);if(x.ready&&x.view&&x.labels){applyDataset({dataset:x.title,view:x.view,labels:x.labels,classes:x.c,n:x.n},x);return}if(x.ready&&Array.isArray(x.views)&&x.views.length>1){applyDataset({dataset:x.title,views:x.views,labels:x.labels,classes:x.c,n:x.n,recommended_engine:x.recommended_engine,recommended_k:x.recommended_k,recommended_kernel_preset:x.recommended_kernel_preset},x);return}
    pendingDatasetAutoApply=true;await jfetch('/api/dataset-prepare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'real',dataset:x.id,out_root:$('#datasetRoot').value.trim()||'datasets'})});pollDataset();
  }catch(e){pendingDatasetAutoApply=false;$('#datasetStatus').textContent=e.message}
};
$('#prepareScale').onclick=async()=>{try{const dims=$('#scaleDims').value.split(',').map(x=>Number(x.trim())).filter(x=>Number.isFinite(x)&&x>=2);if(!dims.length)throw new Error('模拟 MVC 至少需要一个合法维数，例如 32,24,16');pendingDatasetAutoApply=true;await jfetch('/api/dataset-prepare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'scale',n:Number($('#scaleN').value),dims,clusters:Number($('#clusters').value),noise:Number($('#scaleNoise').value),out_root:$('#mvCacheRoot').value.trim()||'dataset_cache'})});pollDataset()}catch(e){pendingDatasetAutoApply=false;$('#datasetStatus').textContent=e.message}};
async function pollDataset(){try{const j=await jfetch('/api/dataset-status');renderDatasetJob(j);if(j.status==='running')setTimeout(pollDataset,600);else if(j.status==='done')initDatasets()}catch(e){$('#datasetStatus').textContent=e.message}}
async function inspectMvMat(){try{const path=$('#mvMatPath').value.trim();if(!path)throw new Error('请先填写 .mat 路径');const views_key=$('#mvViewsKey').value.trim()||'auto',labels_key=$('#mvLabelsKey').value.trim()||'auto';const r=await jfetch('/api/mv-mat-inspect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path,views_key,labels_key})});$('#mvMatInfo').innerHTML=`✓ <b>${esc(r.dataset)}</b>：${esc(r.source_format||'MAT')}，n=${Number(r.n).toLocaleString()}，V=${r.view_count}，dims=${(r.view_dims||[]).join('/')}，c=${r.classes??'—'}；视图=${esc(r.views_key)}，标签=${esc(r.labels_key||'无')}；keys=${(r.keys||[]).join(', ')}${r.aux_key?`；${esc(r.aux_key)} shape=${(r.aux_shape||[]).join('×')}（仅记录，不作为特征）`:''}`;return r}catch(e){$('#mvMatInfo').textContent=e.message;throw e}}
$('#inspectMvMat').onclick=()=>inspectMvMat().catch(()=>{});
$('#prepareMvMat').onclick=async()=>{try{const info=await inspectMvMat();pendingDatasetAutoApply=true;await jfetch('/api/dataset-prepare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'mv_mat',path:$('#mvMatPath').value.trim(),out_root:$('#mvCacheRoot').value.trim()||'dataset_cache',name:$('#mvDatasetName').value.trim()||info.dataset,views_key:info.views_key,labels_key:info.labels_key,k:Number($('#mvImportK').value||10),search:'exact'})});pollDataset()}catch(e){pendingDatasetAutoApply=false;$('#mvMatInfo').textContent=e.message}};
$('#prepareMvMatDir').onclick=async()=>{try{const path=$('#mvMatDir').value.trim();if(!path)throw new Error('请填写 MVC .mat 目录');pendingDatasetAutoApply=false;await jfetch('/api/dataset-prepare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'mv_mat_dir',path,out_root:$('#mvCacheRoot').value.trim()||'dataset_cache',pattern:$('#mvMatPattern').value.trim()||'*.mat',views_key:$('#mvViewsKey').value.trim()||'auto',labels_key:$('#mvLabelsKey').value.trim()||'auto',k:Number($('#mvImportK').value||10),search:'exact'})});pollDataset()}catch(e){$('#mvMatInfo').textContent=e.message}};
