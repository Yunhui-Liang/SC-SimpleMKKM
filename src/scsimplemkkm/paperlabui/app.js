'use strict';
const $ = selector => document.querySelector(selector);
let recipes = [];
let planPath = null;
let planSignature = null;
let refreshRequestId = 0;
let lastPrepareCompletion = null;
const selectedDatasetIds = new Set();

async function api(url, opt) {
  const response = await fetch(url, opt);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}
function nfmt(n) { return n == null ? '—' : Number(n).toLocaleString(); }
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[ch]));
}
function visibleRecipes() {
  const filter = $('#groupFilter').value;
  return recipes.filter(r => filter === 'all' || r.group === filter);
}
function selected() {
  // Selection is independent of the current filter and DOM re-rendering.
  return recipes.filter(r => selectedDatasetIds.has(String(r.id))).map(r => String(r.id));
}
function updateCount() {
  const total = selected().length;
  const visible = visibleRecipes().filter(r => selectedDatasetIds.has(String(r.id))).length;
  $('#selCount').textContent = `已选 ${total} 个数据集（当前列表 ${visible} 个）`;
}
function invalidatePlan() {
  planPath = null;
  planSignature = null;
  $('#planInfo').textContent = '数据集或参数已修改。请重新预览计划，或直接点击“开始无人值守”以生成新计划。';
}
function render() {
  $('#datasets').innerHTML = visibleRecipes().map(r => {
    const id = String(r.id);
    return `<tr><td><input class="dscheck" type="checkbox" value="${escapeHtml(id)}" ${selectedDatasetIds.has(id) ? 'checked' : ''} aria-label="选择 ${escapeHtml(r.title)}"></td>` +
      `<td><b>${escapeHtml(r.title)}</b><br><small>${escapeHtml(id)}</small></td>` +
      `<td>${escapeHtml(r.group)}</td><td>${nfmt(r.n)}</td><td>${escapeHtml(r.views ?? '—')}</td><td>${escapeHtml(r.classes ?? '—')}</td>` +
      `<td class="${r.ready ? 'ready' : 'missing'}" title="${escapeHtml(r.status_reason || '')}"><b>${escapeHtml(r.status || 'unknown')}</b>${r.ready ? ' ✓' : ''}${r.relocated ? ' · copied/relinked' : ''}<br><small>${escapeHtml((r.status_reason || '').slice(0,150))}</small></td></tr>`;
  }).join('');
  updateCount();
}
async function refresh() {
  const requestId = ++refreshRequestId;
  const dataRoot = $('#dataRoot').value;
  const response = await api('/api/recipes?root=' + encodeURIComponent(dataRoot));
  // An old in-flight refresh must not replace a newer catalog.
  if (requestId !== refreshRequestId || $('#dataRoot').value !== dataRoot) return;
  recipes = response.recipes;
  const valid = new Set(recipes.map(r => String(r.id)));
  for (const id of selectedDatasetIds) if (!valid.has(id)) selectedDatasetIds.delete(id);
  render();
}
function req() {
  return {
    datasets: selected(), data_root: $('#dataRoot').value, runs_root: $('#runsRoot').value,
    experiment_mode: $('#mode').value, repeat_policy: $('#repeatPolicy').value,
    repeats: Number($('#repeats').value), seed_start: Number($('#seedStart').value),
    dense_compare: $('#denseCompare').checked, dense_max_n: Number($('#denseMax').value)
  };
}
function signature(q) { return JSON.stringify(q); }
function ensureSelected(requireReady = false) {
  if (!selected().length) {
    $('#planInfo').textContent = '请先勾选至少一个数据集，然后生成或运行实验计划。';
    return false;
  }
  const invalid = recipes.filter(r => selectedDatasetIds.has(String(r.id)) && !r.ready);
  if (requireReady && invalid.length) {
    $('#planInfo').textContent = invalid.map(r => `${r.id}: ${r.status} — ${r.status_reason || '数据未就绪'}`).join('\n') + '\n请先完成准备或修复路径，系统不会在运行中悄悄重新下载。';
    return false;
  }
  return true;
}
function sendJson(url, payload) {
  return api(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
}

$('#datasets').addEventListener('change', event => {
  const input = event.target;
  if (!input.matches('.dscheck')) return;
  if (input.checked) selectedDatasetIds.add(input.value);
  else selectedDatasetIds.delete(input.value);
  updateCount();
  invalidatePlan();
});
$('#refresh').onclick = () => refresh().catch(e => { $('#planInfo').textContent = e.message; });
$('#groupFilter').onchange = render;
$('#clearSel').onclick = () => { selectedDatasetIds.clear(); render(); invalidatePlan(); };
$('#selectStd').onclick = () => {
  for (const r of recipes) if (r.role === 'standard_mvc') selectedDatasetIds.add(String(r.id));
  render(); invalidatePlan();
};
$('#selectLarge').onclick = () => {
  for (const r of recipes) if (r.n != null && Number(r.n) >= 10000) selectedDatasetIds.add(String(r.id));
  render(); invalidatePlan();
};
for (const field of ['#dataRoot','#runsRoot','#mode','#repeatPolicy','#repeats','#seedStart','#denseCompare','#denseMax']) {
  $(field).addEventListener('input', invalidatePlan);
  $(field).addEventListener('change', invalidatePlan);
}
$('#dataRoot').addEventListener('change', () => refresh().catch(e => { $('#planInfo').textContent = e.message; }));
$('#auditData').onclick = async()=>{
  try{const r=await sendJson('/api/audit-data',{data_root:$('#dataRoot').value});$('#auditInfo').textContent=`数据状态: ready ${r.ready}/${r.rows}，CSV：${r.path}`;}
  catch(e){$('#auditInfo').textContent='状态导出失败：'+e.message;}
};
$('#prepare').onclick = async () => {
  if (!ensureSelected()) return;
  try { await sendJson('/api/prepare', { datasets: selected(), data_root: $('#dataRoot').value, overwrite:$('#redoInvalid').checked }); }
  catch (e) { $('#planInfo').textContent = '数据准备失败：' + e.message; }
};
$('#preview').onclick = async () => {
  if (!ensureSelected(true)) return;
  const q = req(), sig = signature(q);
  try {
    const response = await sendJson('/api/plan', q);
    if (signature(req()) !== sig) return; // User changed config while request was in flight.
    planPath = response.plan_path;
    planSignature = sig;
    $('#planInfo').textContent = `Plan: ${planPath}\n` + JSON.stringify(response.status.counts, null, 2);
  } catch (e) { $('#planInfo').textContent = e.message; }
};
$('#run').onclick = async () => {
  if (!ensureSelected(true)) return;
  const q = req();
  if (planPath && planSignature === signature(q)) q.plan_path = planPath;
  try {
    const response = await sendJson('/api/run', q);
    planPath = response.plan_path;
    planSignature = signature(req());
    $('#planInfo').textContent = '正在执行计划：' + planPath;
  } catch (e) { $('#planInfo').textContent = '启动失败：' + e.message; }
};
$('#stop').onclick = () => sendJson('/api/stop', {}).catch(e => { $('#planInfo').textContent = e.message; });
$('#outputs').onclick = async () => {
  try {
    const d = await sendJson('/api/outputs', { runs_root: $('#runsRoot').value });
    $('#outputInfo').textContent = JSON.stringify(d, null, 2);
  } catch (e) { $('#outputInfo').textContent = e.message; }
};
async function poll() {
  try {
    const state = await api('/api/state');
    const counts = state.plan_status?.counts || {};
    $('#statusCards').innerHTML = `<div class="card">state: <b>${escapeHtml(state.mode)}</b></div>` +
      Object.entries(counts).map(([k,v]) => `<div class="card">${escapeHtml(k)}: <b>${escapeHtml(v)}</b></div>`).join('');
    $('#log').textContent = (state.log || []).join('\n') + (state.error ? '\n' + state.error : '');
    // Refresh once after each *preparation* completes, not every second forever.
    if ((state.mode === 'done' || state.mode === 'done_with_errors') &&
        state.result && Array.isArray(state.result.items) && state.finished != null &&
        lastPrepareCompletion !== state.finished) {
      lastPrepareCompletion = state.finished;
      await refresh();
    }
  } catch (e) { /* transient server state; keep selections intact */ }
  setTimeout(poll, 1000);
}
async function loadWorkspace() {
  const w = await api('/api/workspace');
  if(w.data_root)$('#dataRoot').value = w.data_root;
  if(w.runs_root)$('#runsRoot').value = w.runs_root;
  $('#workspaceInfo').textContent = `共享配置：${w.settings_path}`;
}
$('#saveWorkspace').onclick = async () => {
  try {
    const w = await sendJson('/api/workspace', {data_root:$('#dataRoot').value, runs_root:$('#runsRoot').value});
    $('#workspaceInfo').textContent = `已同步：${w.settings_path}`;
    invalidatePlan(); await refresh(); await loadSharedRuns();
  } catch (e) { $('#workspaceInfo').textContent = '目录保存失败：'+e.message; }
};
async function loadSharedRuns(){
  try {
    const root=$('#runsRoot').value;
    const d=await api('/api/runs?root='+encodeURIComponent(root));
    const rows=d.runs||[];
    $('#runHistoryCount').textContent=`${rows.length} 条（${root}）`;
    $('#runHistory').innerHTML=rows.map(r=>`<tr><td>${escapeHtml(r.run_id)}</td><td>${escapeHtml(r.task_name)}</td><td>${escapeHtml(r.engine)}</td><td>${escapeHtml(r.status)}</td><td>${nfmt(r.n)}</td><td>${r.ACC==null?'—':Number(r.ACC).toFixed(4)}</td><td>${r.NMI==null?'—':Number(r.NMI).toFixed(4)}</td><td>${r.ARI==null?'—':Number(r.ARI).toFixed(4)}</td><td><button class="historyView" data-id="${escapeHtml(r.run_id)}">查看</button></td></tr>`).join('');
    for(const b of document.querySelectorAll('.historyView'))b.onclick=async()=>{
      try{ const rec=await api('/api/run-detail?run_id='+encodeURIComponent(b.dataset.id)+'&root='+encodeURIComponent(root));
        $('#runDetail').textContent=JSON.stringify({run_id:rec.run_id,evaluation:rec.evaluation,total_seconds:rec.total_seconds,peak_rss_bytes:rec.peak_rss_bytes,artifact_manifest:rec.artifact_manifest,run_report_html:rec.run_report_html},null,2);
      }catch(e){$('#runDetail').textContent=e.message}
    };
  }catch(e){$('#runHistoryCount').textContent=e.message;}
}
$('#refreshRunHistory').onclick=loadSharedRuns;
$('#runsRoot').addEventListener('change', loadSharedRuns);
(async () => {
  try { await loadWorkspace(); } catch (e) { $('#workspaceInfo').textContent = '共享路径读取失败：'+e.message; }
  try {
    const env = await api('/api/env');
    $('#env').textContent = `${env.physical_cores || '?'}C/${env.logical_cores || '?'}T · RAM ${(Number(env.ram_total_bytes || 0) / 2**30).toFixed(1)} GiB`;
  } catch (e) { $('#env').textContent = '环境信息暂不可用'; }
  try { await refresh(); } catch (e) { $('#planInfo').textContent = e.message; }
  loadSharedRuns(); poll();
})();
