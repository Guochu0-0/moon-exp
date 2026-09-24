/* 实验工作台前端：思路树 + 节点详情。数据来自 /api/data，逐对数据来自 /api/pair，影像来自 /img。
   口径（「实验工作台原型」的结论）：节点同列 Val / Test；保留/放弃只依据 Val；逐对可视化在 Test 上。 */

let D = null;                 // /api/data
const byId = {};
const S = { sel: null, method: null, ref: 'parent', cat: null, pair: null, layer: 'checker', msg: '', thr: 5 };
const PAIR_SPLIT = 'test';
const IDENT = '未配准';

/* ---------------- 数据访问 ---------------- */
const MAIN = () => D.protocol.main;
const keyOf = (rid, m) => `${rid}/${m}`;
function evalOf(key, split) {
  if (key === IDENT) return D.identity[split];
  const [rid, m] = key.split('/');
  return byId[rid]?.metrics?.[m]?.[split];
}
const sumOf = (key, split) => evalOf(key, split)?.summary;
const mainOf = (key, split) => sumOf(key, split)?.[MAIN()];
function repMethod(r) {   // 多方法节点（如 baseline）以 Val 主指标最好的方法为代表
  if (r.methods.length <= 1) return r.methods[0];
  return r.methods.slice().sort((a, b) => (mainOf(keyOf(r.id, b), 'val') ?? -1) - (mainOf(keyOf(r.id, a), 'val') ?? -1))[0];
}
const repKey = r => r.methods.length ? keyOf(r.id, repMethod(r)) : null;
function parentRef(r) {
  if (r.meta.init) return r.meta.init.includes('/') ? r.meta.init : repKey(byId[r.meta.init]);
  const p = byId[r.meta.parent];
  return p ? repKey(p) : IDENT;
}
const label = key => key === IDENT ? IDENT : key.endsWith('/main') ? key.slice(0, -5) : key;
function allKeys() { return D.runs.flatMap(r => r.methods.map(m => keyOf(r.id, m))); }

/* ---------------- 格式化 ---------------- */
const pct = x => x == null ? '—' : (x * 100).toFixed(1);
const px = x => x == null ? '∞' : x.toFixed(1);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function delta(a, b, lowerBetter = false, scale = 100, unit = ' pt') {
  if (a == null || b == null) return '<span class="muted">—</span>';
  const d = (a - b) * scale;
  if (Math.abs(d) < 0.05) return '<span class="muted">±0</span>';
  const good = lowerBetter ? d < 0 : d > 0;
  return `<span class="${good ? 'up' : 'down'}">${d > 0 ? '+' : ''}${d.toFixed(1)}${unit}</span>`;
}
function pill(st) { const t = { kept: '保留', dropped: '放弃', running: '运行中', baseline: 'baseline' }[st] || st; return `<span class="pill ${st}">${t}</span>`; }
const PAL = ['#0969da', '#cf222e', '#1a7f37', '#8250df', '#bf8700', '#1b7c83', '#e16f24', '#6e7781', '#bf3989', '#57606a'];

/* ---------------- 图表 ---------------- */
function cdfChart(items, w = 660, h = 210) {   // items: [{name, errors, color, dash, width}]
  const L = 44, R = 10, T = 8, B = 30, W = w - L - R, H = h - T - B, xMax = D.protocol.t_coarse;
  const X = x => L + x / xMax * W, Y = y => T + H - y * H;
  let g = '';
  for (let i = 0; i <= 4; i++) g += `<line x1="${L}" x2="${L + W}" y1="${Y(i / 4)}" y2="${Y(i / 4)}" stroke="#eaeef2"/><text x="${L - 6}" y="${Y(i / 4) + 4}" font-size="10" fill="#656d76" text-anchor="end">${i * 25}%</text>`;
  for (let i = 0; i <= 5; i++) { const x = xMax * i / 5; g += `<text x="${X(x)}" y="${T + H + 14}" font-size="10" fill="#656d76" text-anchor="middle">${x}</text>`; }
  for (const t of D.protocol.sr_thresholds.filter(t => t < xMax)) g += `<line x1="${X(t)}" x2="${X(t)}" y1="${T}" y2="${T + H}" stroke="#eaeef2" stroke-dasharray="3 3"/>`;
  g += `<text x="${L + W / 2}" y="${h - 2}" font-size="10.5" fill="#656d76" text-anchor="middle">pair 误差阈值 (px)</text>`;
  items.forEach((it, i) => {
    const e = it.errors.map(v => v == null ? Infinity : v).sort((a, b) => a - b), n = e.length;
    if (!n) return;
    let d = `M${X(0)} ${Y(0)}`;
    e.forEach((v, k) => { if (v <= xMax) d += `L${X(v).toFixed(1)} ${Y(k / n).toFixed(1)}L${X(v).toFixed(1)} ${Y((k + 1) / n).toFixed(1)}`; });
    const last = e.filter(v => v <= xMax).length / n;
    d += `L${X(xMax)} ${Y(last)}`;
    g += `<path d="${d}" fill="none" stroke="${it.color || PAL[i % PAL.length]}" stroke-width="${it.width || 1.8}" ${it.dash ? `stroke-dasharray="${it.dash}"` : ''}><title>${esc(it.name)}</title></path>`;
  });
  return `<svg width="${w}" height="${h}">${g}</svg>` + legend(items);
}
function legend(items) { return `<div class="legend">${items.map((it, i) => `<span><i style="background:${it.color || PAL[i % PAL.length]}"></i>${esc(it.name)}</span>`).join('')}</div>`; }
function outcomeBars(rows, w = 660) {   // rows: [{name, errors}]；按协议分：≤阈值 / 成功但超阈值 / 错配 / 失败
  const tc = D.protocol.t_coarse, thr = S.thr, rh = 20, L = 150, W = w - L - 10, h = rows.length * rh + 6;
  const segs = [[`≤ ${thr} px`, '#2da44e', e => e != null && e <= thr], [`成功但 > ${thr} px`, '#a2d9ae', e => e != null && e > thr && e <= tc],
                [`错配 (> ${tc} px)`, '#bf8700', e => e != null && e > tc], ['失败', '#cf222e', e => e == null]];
  let g = '';
  rows.forEach((r, i) => {
    const y = i * rh + 3, n = r.errors.length || 1; let x = L;
    g += `<text x="${L - 6}" y="${y + 12}" font-size="11" text-anchor="end">${esc(r.name)}</text>`;
    for (const [nm, col, f] of segs) { const c = r.errors.filter(f).length, ww = c / n * W;
      g += `<rect x="${x}" y="${y}" width="${ww}" height="${rh - 6}" fill="${col}"><title>${nm}：${c} 对（${pct(c / n)}%）</title></rect>`; x += ww; }
  });
  return `<svg width="${w}" height="${h}">${g}</svg><div class="legend">${segs.map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join('')}</div>`;
}

/* ---------------- 思路树 ---------------- */
function layout() {
  const kids = {}; D.runs.forEach(r => { const p = r.meta.parent || ''; (kids[p] = kids[p] || []).push(r); });
  Object.values(kids).forEach(a => a.sort((x, y) => String(x.meta.date).localeCompare(String(y.meta.date)) || x.id.localeCompare(y.id)));
  const pos = {}; let row = 0;
  const walk = (r, depth) => { pos[r.id] = [depth, row]; (kids[r.id] || []).forEach((c, i) => { if (i) row++; walk(c, depth + 1); }); };
  (kids[''] || []).forEach((r, i) => { if (i) row++; walk(r, 0); });
  D.runs.filter(r => !(r.id in pos)).forEach(r => { row++; walk(r, 0); });  // 父实验不存在时兜底
  return { pos, rows: row + 1, depth: Math.max(0, ...Object.values(pos).map(p => p[0])) + 1 };
}
function tree() {
  const { pos, rows, depth } = layout(), bw = 190, bh = 86, gx = 232, gy = 108, ox = 4, oy = 4;
  const xy = id => [ox + pos[id][0] * gx, oy + pos[id][1] * gy];
  const vals = D.runs.map(r => mainOf(repKey(r), 'val')).filter(v => v != null), lo = Math.min(...vals, 0), hi = Math.max(...vals, 0.01);
  const heat = v => v == null ? '#f6f8fa' : `hsl(${Math.round((v - lo) / (hi - lo || 1) * 140)},65%,88%)`;
  let g = '';
  for (const r of D.runs) if (r.meta.parent && byId[r.meta.parent]) {
    const [x1, y1] = xy(r.meta.parent), [x2, y2] = xy(r.id), dead = r.meta.status === 'dropped';
    const a = mainOf(repKey(r), 'val'), b = mainOf(parentRef(r), 'val'), d = a != null && b != null ? (a - b) * 100 : null;
    g += `<path d="M${x1 + bw} ${y1 + bh / 2} C${x1 + bw + 30} ${y1 + bh / 2} ${x2 - 30} ${y2 + bh / 2} ${x2} ${y2 + bh / 2}" fill="none" stroke="${dead ? '#d0d7de' : '#8c959f'}" stroke-width="2" ${dead ? 'stroke-dasharray="5 4"' : ''}/>`;
    if (d != null) g += `<text x="${(x1 + bw + x2) / 2}" y="${(y1 + y2) / 2 + bh / 2 - 6}" font-size="11" text-anchor="middle" fill="${d >= 0 ? '#1a7f37' : '#cf222e'}" font-weight="600">${d >= 0 ? '+' : ''}${d.toFixed(1)}</text>`;
  }
  for (const r of D.runs) {
    const [x, y] = xy(r.id), dead = r.meta.status === 'dropped', k = repKey(r), v = mainOf(k, 'val'), t = mainOf(k, 'test');
    const title = r.meta.title || '', multi = r.methods.length > 1 ? ` · ${r.methods.length} 个方法` : '';
    g += `<g class="node ${S.sel === r.id ? 'sel' : ''}" onclick="selectNode('${r.id}')" opacity="${dead ? .55 : 1}">
      <rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="8" fill="${heat(v)}" stroke="${dead ? '#cf222e' : '#8c959f'}"/>
      <text x="${x + 10}" y="${y + 18}" font-size="12" font-weight="700">${esc(r.id)} <tspan font-weight="400" fill="#656d76">${esc(String(r.meta.date || '').slice(5))} · ${esc(String(r.meta.commit || '').slice(0, 7))}</tspan></text>
      <text x="${x + bw - 8}" y="${y + 18}" font-size="10.5" text-anchor="end">${{ kept: '✓', dropped: '✗', running: '◔', baseline: '◆' }[r.meta.status] || ''}</text>
      <text x="${x + 10}" y="${y + 37}" font-size="11.5"><title>${esc(title)}</title>${esc(title.length > 15 ? title.slice(0, 15) + '…' : title)}</text>
      <text x="${x + 10}" y="${y + 66}" font-size="18" font-weight="700">${pct(v)}</text><text x="${x + 10}" y="${y + 79}" font-size="9.5" fill="#656d76">Val ${MAIN().toUpperCase()}${multi}</text>
      <text x="${x + 110}" y="${y + 66}" font-size="15">${pct(t)}</text><text x="${x + 110}" y="${y + 79}" font-size="9.5" fill="#656d76">Test</text></g>`;
  }
  return `<div class="muted" style="font-size:12px;margin-bottom:6px">节点 = 实验；颜色、边上数字 = Val ${MAIN().toUpperCase()}（边上是相对父节点 / 起点方法的变化，pt）；虚线淡色 = 放弃。保留/放弃只依据 Val。</div>
    <svg width="${ox + depth * gx}" height="${oy + rows * gy}">${g}</svg>`;
}

/* ---------------- 节点详情 ---------------- */
function metricRows() {
  const p = D.protocol;
  return [...p.auc_thresholds.map(t => [`AUC@${t}`, `auc@${t}`, false, 'pt']), ...p.sr_thresholds.map(t => [`SR@${t}`, `sr@${t}`, false, 'pt']),
    ['失败率', 'fail_rate', true, 'pt'], [`错配率 (> ${p.t_coarse} px)`, 'mis_rate', true, 'pt'], ['成功子集误差均值', 'succ_mean', true, 'px'], ['成功子集误差中位数', 'succ_median', true, 'px']];
}
function metricTable(key, ref) {
  const sv = sumOf(key, 'val') || {}, st = sumOf(key, 'test') || {}, rv = sumOf(ref, 'val') || {}, rt = sumOf(ref, 'test') || {};
  const cell = (s, k, u) => { const v = s[k]; if (v == null) return '—'; const ci = s[k + '_ci'];
    return (u === 'px' ? px(v) : pct(v)) + (k === MAIN() && ci ? ` <span class="muted" style="font-size:11px">[${pct(ci[0])}, ${pct(ci[1])}]</span>` : ''); };
  const dl = (a, b, k, low, u) => u === 'px' ? delta(a[k], b[k], true, 1, ' px') : delta(a[k], b[k], low);
  const cov = s => evalOf(key, s)?.coverage;
  return `<table style="margin:4px 0 6px"><tr><th>指标</th><th>Val</th><th>Test</th><th>Δ Val vs ${esc(label(ref))}</th><th>Δ Test</th></tr>
    ${metricRows().map(([n, k, low, u]) => `<tr ${k === MAIN() ? 'style="font-weight:700"' : ''}><td>${n}</td><td>${cell(sv, k, u)}</td><td>${cell(st, k, u)}</td><td>${dl(sv, rv, k, low, u)}</td><td>${dl(st, rt, k, low, u)}</td></tr>`).join('')}
    <tr class="muted"><td>有标注 pair / 缺预测</td><td>${cov('val') ? `${cov('val').n_labelled} / ${cov('val').missing}` : '—'}</td><td>${cov('test') ? `${cov('test').n_labelled} / ${cov('test').missing}` : '—'}</td><td></td><td></td></tr></table>
    <div class="muted" style="font-size:11.5px">AUC、SR、率均为 %；主指标 ${MAIN().toUpperCase()} 附 bootstrap 95% CI。缺预测的有标注 pair 按失败计。</div>`;
}
function methodTable(r) {
  const rows = r.methods.map(m => ({ m, k: keyOf(r.id, m) })).sort((a, b) => (mainOf(b.k, 'val') ?? -1) - (mainOf(a.k, 'val') ?? -1));
  const cols = [[`Val ${MAIN().toUpperCase()}`, 'val', MAIN(), pct], [`Test ${MAIN().toUpperCase()}`, 'test', MAIN(), pct], ['Test SR@5', 'test', 'sr@5', pct],
    ['Test 失败率', 'test', 'fail_rate', pct], ['Test 错配率', 'test', 'mis_rate', pct], ['Test 成功中位误差', 'test', 'succ_median', px]];
  const idv = [IDENT, ...cols.map(([, s, k, f]) => f(sumOf(IDENT, s)?.[k]))];
  return `<table style="margin:6px 0"><tr><th>方法（点选）</th>${cols.map(c => `<th>${c[0]}</th>`).join('')}</tr>
    ${rows.map(({ m, k }) => `<tr class="click ${S.method === m ? 'on' : ''}" onclick="S.method='${m}';S.cat=null;S.pair=null;render()"><td>${esc(r.meta.methods?.[m]?.name || m)}</td>${cols.map(([, s, kk, f]) => `<td>${f(sumOf(k, s)?.[kk])}</td>`).join('')}</tr>`).join('')}
    <tr class="muted"><td>${idv[0]}</td>${idv.slice(1).map(v => `<td>${v}</td>`).join('')}</tr></table>`;
}
function panel() {
  const r = byId[S.sel]; if (!r) return '<p class="muted">点左侧节点查看详情。</p>';
  const m = r.meta, multi = r.methods.length > 1, key = S.method ? keyOf(r.id, S.method) : null;
  const ref = S.ref === 'parent' ? parentRef(r) : S.ref, kids = D.runs.filter(x => x.meta.parent === r.id);
  const par = byId[m.parent];
  let h = `<div class="muted">${esc(m.date)} · commit <code>${esc(m.commit)}</code>${r.git?.subject ? ` <span class="muted">${esc(r.git.subject)}</span>` : ''}${par ? ` · 父节点 <a onclick="selectNode('${par.id}')">${esc(par.id)}</a>` : ''}${m.init ? ` · 起点 <code>${esc(m.init)}</code>` : ''}</div>
    <h2>${esc(r.id)} ${esc(m.title)} ${pill(m.status)}</h2>`;
  if (!r.methods.length) return h + '<p class="muted">（还没有 preds/）</p>' + textBlocks(r, kids);
  if (multi) h += `<h4>方法（Val ${MAIN().toUpperCase()} 排序）</h4>${methodTable(r)}`;
  h += `<h4>指标 · ${esc(label(key))}</h4>${metricTable(key, ref)}`;
  if (m.hypothesis?.trim()) h += `<h4>为什么做</h4><p class="prose">${esc(m.hypothesis.trim())}</p>`;
  if (m.change?.trim() || r.git?.diffstat) h += `<h4>相对父节点改了什么</h4>${m.change?.trim() ? `<p class="prose">${esc(m.change.trim())}</p>` : ''}${r.git?.diffstat ? `<div class="diff">${esc(r.git.diffstat)}</div>` : ''}`;

  h += `<div class="zone">固定区 <span class="muted" style="font-weight:400;font-size:12px">每个实验都有，格式统一</span></div>`;
  const cmp = multi ? r.methods.map((mm, i) => ({ name: label(keyOf(r.id, mm)), errors: evalOf(keyOf(r.id, mm), PAIR_SPLIT)?.errors || [], color: PAL[i % PAL.length], width: mm === S.method ? 2.6 : 1.5 }))
    .concat([{ name: IDENT, errors: evalOf(IDENT, PAIR_SPLIT).errors, color: '#8c959f', dash: '4 3' }])
    : [{ name: label(key), errors: evalOf(key, PAIR_SPLIT)?.errors || [], color: '#0969da', width: 2.4 }, { name: label(ref), errors: evalOf(ref, PAIR_SPLIT)?.errors || [], color: '#8c959f', dash: '4 3' }];
  h += `<h4>误差累积分布（Test）</h4>${cdfChart(cmp)}`;
  h += `<h4>结果分解（Test）</h4>${outcomeBars(cmp.map(c => ({ name: c.name, errors: c.errors })))}`;
  h += pairSection(r, key, ref);

  h += `<div class="zone">灵活区 <span class="muted" style="font-weight:400;font-size:12px">实验自己产出的额外可视化（runs/${esc(r.id)}/extra/）</span></div>`;
  h += r.extras.length ? r.extras.map(f => `<div class="card extra" style="margin-bottom:10px"><h3 style="font-size:13.5px">${esc(f.replace(/^extra\//, ''))}</h3><div data-extra="/runs/${esc(r.id)}/${esc(f)}"></div></div>`).join('') : '<p class="muted">（没有 extra/ 产物）</p>';
  return h + textBlocks(r, kids);
}
function textBlocks(r, kids) {
  const m = r.meta;
  return `<h4>结论 <span class="muted" style="font-weight:400">（依据 Val）</span></h4><p class="prose">${esc(m.verdict?.trim() || '（未写）')}</p>
    <h4>由此派生</h4><p>${kids.length ? kids.map(c => `<a onclick="selectNode('${c.id}')">${esc(c.id)} ${esc(c.meta.title)}</a>`).join('<br>') : `<span class="muted">（暂无）</span>${m.next?.trim() ? ` 计划：${esc(m.next.trim())}` : ''}`}</p>`;
}

/* ---------------- 逐对可视化 ---------------- */
const CAT = { fixed: ['修好了', '#1a7f37'], broke: ['变坏了', '#cf222e'], still: ['仍不达标', '#bf8700'], ok: ['一直达标', '#8c959f'] };
const pairNo = (split, pair) => D.pairs[split].indexOf(pair);
const pairName = (split, pair) => `${split === 'test' ? 'Test' : 'Val'} #${String(pairNo(split, pair)).padStart(4, '0')} · ${pair}`;
function pairsFor(key, ref) {
  const ea = evalOf(ref, PAIR_SPLIT)?.errors || [], eb = evalOf(key, PAIR_SPLIT)?.errors || [], t = S.thr;
  return D.labelled[PAIR_SPLIT].map((p, i) => {
    const a = ea[i], b = eb[i], okA = a != null && a <= t, okB = b != null && b <= t;
    return { p, a, b, cat: !okA && okB ? 'fixed' : okA && !okB ? 'broke' : okA ? 'ok' : 'still' };
  });
}
function pairSearch(v) {
  v = String(v).trim(); const all = D.pairs[PAIR_SPLIT]; let p = null;
  if (/^#?\d+$/.test(v)) p = all[parseInt(v.replace('#', ''), 10)];
  else { const m = v.match(/(?:ROI_)?(\d+)\s*\/\s*(?:patch_)?(\d+)/i); if (m) p = `ROI_${m[1].padStart(3, '0')}/patch_${+m[2]}`; if (!all.includes(p)) p = null; }
  if (!p) { S.msg = `没有找到 Test「${v}」（编号 0–${all.length - 1}，或写 ROI_037/1）`; render(); return; }
  S.pair = p; S.cat = null; S.msg = ''; render();
}
function pairSection(r, key, ref) {
  const pairs = pairsFor(key, ref), cnt = {}; for (const k in CAT) cnt[k] = pairs.filter(q => q.cat === k).length;
  const cat = S.cat || (cnt.fixed ? 'fixed' : 'still');
  const sel = S.pair;
  let list = [];
  if (sel) list = [pairs.find(q => q.p === sel) || { p: sel, a: null, b: null, cat: null, unlabelled: !D.labelled[PAIR_SPLIT].includes(sel) }];
  else list = pairs.filter(q => q.cat === cat).sort((x, y) => gain(y) - gain(x)).slice(0, 3);
  const opts = [['parent', `父节点 / 起点（${label(parentRef(r))}）`], [IDENT, '未配准（数据集粗对齐）'], ...allKeys().filter(k => k !== key).map(k => [k, label(k)])];
  return `<h4>逐对可视化（Test）</h4>
    <div class="ctl">
      <label>和谁比 <select onchange="S.ref=this.value;S.cat=null;render()">${opts.map(([k, t]) => `<option value="${esc(k)}" ${S.ref === k ? 'selected' : ''}>${esc(t)}</option>`).join('')}</select></label>
      <label>达标线 <select onchange="S.thr=+this.value;S.cat=null;render()">${D.protocol.sr_thresholds.map(t => `<option ${S.thr === t ? 'selected' : ''}>${t}</option>`).join('')}</select> px</label>
      <label>按编号查 <input id="pq" placeholder="如 10、#0010、ROI_037/1" onkeydown="if(event.key==='Enter')pairSearch(this.value)"> <button onclick="pairSearch(document.getElementById('pq').value)">查</button></label>
    </div>
    ${S.msg ? `<div class="down" style="font-size:12px">${esc(S.msg)}</div>` : ''}
    <div class="muted" style="font-size:12px">每个点是一个有标注的 Test pair：横轴 = ${esc(label(ref))} 的误差，纵轴 = ${esc(label(key))} 的误差；贴顶/贴右 = 失败。点一个点看影像。</div>
    ${scatter(pairs, label(key), label(ref))}
    <div class="chips">${Object.entries(CAT).map(([k, [t, c]]) => { const on = k === cat && !sel;
      return `<span class="chip" style="${on ? `background:${c};border-color:${c};color:#fff` : `color:${c}`}" onclick="S.cat='${k}';S.pair=null;render()">${t} ${cnt[k]}</span>`; }).join('')}
      <span style="margin-left:auto" class="seg">${[['checker', '棋盘格'], ['match', '点对连线']].map(([k, t]) => `<button class="${S.layer === k ? 'on' : ''}" onclick="S.layer='${k}';render()">${t}</button>`).join('')}</span></div>
    ${list.map(q => pairCard(q, key, ref)).join('') || '<p class="muted">（这一类没有样本）</p>'}
    <div class="muted" style="font-size:11.5px">${S.layer === 'checker'
      ? '棋盘格：光学与「按估计仿射变回光学坐标系的 SAR」（偏蓝格）交替，看边缘在格子交界处是否连续。黄点 = 光学检查点，箭头指向对应 SAR 检查点落在这张图上的位置（×3 放大）。'
      : '点对连线：左光学、右 SAR（原始）。绿 = 与该方法自己的估计仿射相差 ≤ 3 px 的点对，红 = 其余。只有记录了 matches 的方法才有。'}</div>`;
}
const gain = q => (q.a == null ? 1e3 : q.a) - (q.b == null ? 1e3 : q.b);
function scatter(pairs, kn, rn) {
  const w = 660, h = 270, L = 46, B = 34, T = 8, R = 12, W = w - L - R, H = h - T - B, lo = Math.log10(0.5), hi = Math.log10(300);
  const lg = v => v == null ? hi : Math.log10(Math.max(0.5, Math.min(300, v)));
  const X = v => L + (lg(v) - lo) / (hi - lo) * W, Y = v => T + H - (lg(v) - lo) / (hi - lo) * H, t = S.thr;
  let g = `<rect x="${X(t)}" y="${Y(t)}" width="${L + W - X(t)}" height="${T + H - Y(t)}" fill="#e6f6ea"/><rect x="${L}" y="${T}" width="${X(t) - L}" height="${Y(t) - T}" fill="#fff0f0"/>
    <line x1="${X(.5)}" y1="${Y(.5)}" x2="${X(300)}" y2="${Y(300)}" stroke="#afb8c1" stroke-dasharray="3 3"/>`;
  for (const v of [.5, 1, 2, 5, 10, 20, 50, 100]) g += `<text x="${X(v)}" y="${T + H + 13}" font-size="10" fill="#656d76" text-anchor="middle">${v}</text><text x="${L - 5}" y="${Y(v) + 3}" font-size="10" fill="#656d76" text-anchor="end">${v}</text>`;
  g += `<text x="${L + W / 2}" y="${h - 3}" font-size="10.5" fill="#656d76" text-anchor="middle">${esc(rn)} 误差 (px, log)</text><text x="11" y="${T + H / 2}" font-size="10.5" fill="#656d76" text-anchor="middle" transform="rotate(-90 11 ${T + H / 2})">${esc(kn)} 误差</text>`;
  for (const q of pairs) { const sel = S.pair === q.p;
    g += `<circle cx="${X(q.a)}" cy="${Y(q.b)}" r="${sel ? 6 : 3}" fill="${CAT[q.cat][1]}" fill-opacity="${sel ? 1 : .6}" ${sel ? 'stroke="#0969da" stroke-width="2.5"' : ''} onclick="S.pair='${q.p}';S.msg='';render()"><title>${esc(pairName(PAIR_SPLIT, q.p))}\n${esc(rn)} ${px(q.a)} → ${esc(kn)} ${px(q.b)} px</title></circle>`; }
  return `<svg class="scat" width="${w}" height="${h}">${g}</svg>`;
}
function pairCard(q, key, ref) {
  const fmt = v => q.unlabelled ? '无标注' : `${px(v)} px ${v != null && v <= S.thr ? '<span class="up">✓</span>' : '<span class="down">✗</span>'}`;
  const id = `pc-${q.p.replace(/\W/g, '_')}`;
  const figs = S.layer === 'checker'
    ? [['opt', null, '光学'], ['sar', null, 'SAR（原始）'], ['ov', ref, `${label(ref)}：${fmt(q.a)}`], ['ov', key, `${label(key)}：${fmt(q.b)}`]]
    : [['mt', ref, `${label(ref)}：${fmt(q.a)}`], ['mt', key, `${label(key)}：${fmt(q.b)}`]];
  return `<div class="vcard ${q.p === S.pair ? 'hl' : ''}" id="${id}"><div style="font-size:12px;margin-bottom:5px"><b>${esc(pairName(PAIR_SPLIT, q.p))}</b> ${q.cat ? `<span class="pill" style="color:${CAT[q.cat][1]}">${CAT[q.cat][0]}</span>` : ''}</div>
    <div class="imgs">${figs.map(([k, who, cap]) => `<figure><canvas data-pair="${esc(q.p)}" data-kind="${k}" data-who="${esc(who || '')}" width="${k === 'mt' ? 2 * VS + 6 : VS}" height="${VS}"></canvas>${cap}</figure>`).join('')}</div></div>`;
}

/* ---------------- 影像绘制（canvas 上按仿射变换） ---------------- */
const VS = 150, IMG = {}, PAIRDATA = {};
function loadImg(split, pair, kind) {
  const k = `${split}|${pair}|${kind}`;
  return IMG[k] = IMG[k] || new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = bad; im.src = `/img?split=${split}&pair=${encodeURIComponent(pair)}&kind=${kind}`; });
}
function loadPair(split, pair) {
  const k = `${split}|${pair}`;
  return PAIRDATA[k] = PAIRDATA[k] || fetch(`/api/pair?split=${split}&pair=${encodeURIComponent(pair)}`).then(r => r.json());
}
function inv(A) {   // 2×3 仿射求逆
  const [[a, b, c], [d, e, f]] = A, det = a * e - b * d;
  return [[e / det, -b / det, (b * f - c * e) / det], [-d / det, a / det, (c * d - a * f) / det]];
}
const ap = (A, x, y) => [A[0][0] * x + A[0][1] * y + A[0][2], A[1][0] * x + A[1][1] * y + A[1][2]];
function predFor(pd, who) {
  if (who === IDENT) return { A: [[1, 0, 0], [0, 1, 0]] };
  return pd.preds[who] || null;
}
async function drawCanvases() {
  for (const cv of document.querySelectorAll('canvas[data-pair]')) {
    const pair = cv.dataset.pair, kind = cv.dataset.kind, who = cv.dataset.who, ctx = cv.getContext('2d');
    try {
      const [opt, sar, pd] = await Promise.all([loadImg(PAIR_SPLIT, pair, 'opt'), loadImg(PAIR_SPLIT, pair, 'sar'), loadPair(PAIR_SPLIT, pair)]);
      const s = VS / opt.width;
      if (kind === 'opt' || kind === 'sar') { ctx.setTransform(s, 0, 0, s, 0, 0); ctx.drawImage(kind === 'opt' ? opt : sar, 0, 0); continue; }
      const pr = predFor(pd, who);
      if (kind === 'ov') {
        ctx.setTransform(s, 0, 0, s, 0, 0); ctx.drawImage(opt, 0, 0);
        if (!pr || !pr.A) { overlayText(ctx, pr ? `失败：${pr.fail || ''}` : '无预测'); continue; }
        const Ai = inv(pr.A), cell = VS / 8;
        ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.beginPath();
        for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) if ((i + j) % 2) ctx.rect(i * cell, j * cell, cell, cell);
        ctx.clip(); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, VS, VS);
        ctx.setTransform(s * Ai[0][0], s * Ai[1][0], s * Ai[0][1], s * Ai[1][1], s * Ai[0][2], s * Ai[1][2]); ctx.drawImage(sar, 0, 0);
        ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = 'rgba(60,120,255,.18)'; ctx.fillRect(0, 0, VS, VS); ctx.restore();
        if (pd.opt_pts) {
          ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.strokeStyle = ctx.fillStyle = '#ffd33d'; ctx.lineWidth = 1.5;
          pd.opt_pts.forEach(([x, y], i) => { const [u, v] = ap(Ai, ...pd.sar_pts[i]), k = 3;
            ctx.beginPath(); ctx.arc(x * s, y * s, 2.2, 0, 7); ctx.fill();
            ctx.beginPath(); ctx.moveTo(x * s, y * s); ctx.lineTo((x + k * (u - x)) * s, (y + k * (v - y)) * s); ctx.stroke(); });
        }
      } else if (kind === 'mt') {
        const off = VS + 6; ctx.setTransform(s, 0, 0, s, 0, 0); ctx.drawImage(opt, 0, 0); ctx.setTransform(s, 0, 0, s, off, 0); ctx.drawImage(sar, 0, 0); ctx.setTransform(1, 0, 0, 1, 0, 0);
        const M = pr?.matches;
        if (!M || !M.length) { overlayText(ctx, pr ? '没有记录点对' : '无预测'); continue; }
        const step = Math.max(1, Math.floor(M.length / 150)); let ok = 0; ctx.lineWidth = 1;
        M.forEach((m, i) => { const good = pr.A && Math.hypot(...ap(pr.A, m[0], m[1]).map((v, j) => v - m[2 + j])) <= 3; if (good) ok++;
          if (i % step) return; ctx.strokeStyle = good ? 'rgba(46,160,67,.85)' : 'rgba(207,34,46,.7)';
          ctx.beginPath(); ctx.moveTo(m[0] * s, m[1] * s); ctx.lineTo(off + m[2] * s, m[3] * s); ctx.stroke(); });
        overlayText(ctx, `一致 ${ok}/${M.length}`);
      }
    } catch (e) { ctx.setTransform(1, 0, 0, 1, 0, 0); overlayText(ctx, '读图失败'); console.error(e); }
  }
}
function overlayText(ctx, t) { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.font = '11px sans-serif'; const w = ctx.measureText(t).width + 8; ctx.fillStyle = 'rgba(0,0,0,.65)'; ctx.fillRect(0, 0, w, 16); ctx.fillStyle = '#fff'; ctx.fillText(t, 4, 12); }

/* ---------------- 灵活区 ---------------- */
async function drawExtras() {
  for (const el of document.querySelectorAll('[data-extra]')) {
    const url = el.dataset.extra, ext = url.split('.').pop().toLowerCase();
    if (['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp'].includes(ext)) { el.innerHTML = `<img src="${url}">`; continue; }
    if (ext === 'html') { el.innerHTML = `<iframe src="${url}"></iframe>`; continue; }
    const txt = await (await fetch(url)).text();
    if (ext === 'json') { try { const j = JSON.parse(txt); if (j.type === 'line' || j.type === 'bar') { el.innerHTML = extraChart(j); continue; } } catch (e) { /* 按文本显示 */ } }
    el.innerHTML = `<pre>${esc(txt.slice(0, 20000))}</pre>`;
  }
}
function extraChart(j) {   // {"type":"line","x":[..],"series":{"名称":[..]},"xlabel":"step"} 或 {"type":"bar","labels":[..],"values":[..]}
  const w = 660, h = 200, L = 50, R = 10, T = 8, B = 30, W = w - L - R, H = h - T - B;
  if (j.type === 'bar') {
    const mx = Math.max(...j.values, 1e-12), rh = 22;
    return `<svg width="${w}" height="${j.values.length * rh + 6}">${j.labels.map((k, i) => `<text x="120" y="${i * rh + 15}" font-size="11" text-anchor="end">${esc(k)}</text><rect x="128" y="${i * rh + 4}" width="${j.values[i] / mx * 420}" height="${rh - 8}" fill="#8250df" opacity=".75"/><text x="${134 + j.values[i] / mx * 420}" y="${i * rh + 15}" font-size="10.5" fill="#656d76">${j.values[i]}</text>`).join('')}</svg>`;
  }
  const ser = Object.entries(j.series), all = ser.flatMap(([, v]) => v).filter(Number.isFinite);
  const x0 = Math.min(...j.x), x1 = Math.max(...j.x), y0 = Math.min(...all), y1 = Math.max(...all);
  const X = x => L + (x - x0) / (x1 - x0 || 1) * W, Y = y => T + H - (y - y0) / (y1 - y0 || 1) * H;
  let g = '';
  for (let i = 0; i <= 4; i++) { const y = y0 + (y1 - y0) * i / 4; g += `<line x1="${L}" x2="${L + W}" y1="${Y(y)}" y2="${Y(y)}" stroke="#eaeef2"/><text x="${L - 5}" y="${Y(y) + 4}" font-size="10" fill="#656d76" text-anchor="end">${+y.toPrecision(3)}</text>`; }
  for (let i = 0; i <= 5; i++) { const x = x0 + (x1 - x0) * i / 5; g += `<text x="${X(x)}" y="${T + H + 14}" font-size="10" fill="#656d76" text-anchor="middle">${+x.toPrecision(3)}</text>`; }
  g += `<text x="${L + W / 2}" y="${h - 2}" font-size="10.5" fill="#656d76" text-anchor="middle">${esc(j.xlabel || '')}</text>`;
  ser.forEach(([n, v], i) => { g += `<path d="${v.map((y, k) => (k ? 'L' : 'M') + X(j.x[k]).toFixed(1) + ' ' + Y(y).toFixed(1)).join('')}" fill="none" stroke="${PAL[i % PAL.length]}" stroke-width="1.8"/>`; });
  return `<svg width="${w}" height="${h}">${g}</svg>` + legend(ser.map(([n]) => ({ name: n })));
}

/* ---------------- 主循环 ---------------- */
function selectNode(id) {
  const r = byId[id]; S.sel = id; S.method = r ? repMethod(r) : null; S.ref = 'parent'; S.cat = null; S.pair = null; S.msg = '';
  syncHash(); render();
  document.querySelector('.panel')?.scrollTo(0, 0);
}
function syncHash() {   // #E1&pair=ROI_037/patch_1&layer=match，可直接分享
  const q = new URLSearchParams(); if (S.pair) q.set('pair', S.pair); if (S.layer !== 'checker') q.set('layer', S.layer);
  history.replaceState(null, '', '#' + encodeURIComponent(S.sel || '') + (q.toString() ? '&' + q : ''));
}
function render() {
  if (D && S.sel) syncHash();
  const p = D.protocol;
  document.getElementById('app').innerHTML = `<div class="topbar"><b>🌙 moon-exp 实验工作台</b><span class="muted">${D.runs.length} 个实验 · Val ${D.labelled.val.length}/${D.pairs.val.length} · Test ${D.labelled.test.length}/${D.pairs.test.length} 有标注</span>
      ${p.provisional ? `<span class="warnbox">档位暂定（AUC@${p.auc_thresholds.join('/')}，T_粗=${p.t_coarse}），待「锁定评价阈值档位」</span>` : ''}
      <button style="margin-left:auto" onclick="reload()">重新加载 runs/</button></div>
    <div class="main"><div class="canvas">${D.runs.length ? tree() : '<p class="muted">runs/ 下还没有实验。用 <code>python -m workbench new &lt;id&gt;</code> 新建。</p>'}</div><div class="panel">${panel()}</div></div>`;
  drawCanvases(); drawExtras();
  if (S.pair) document.getElementById(`pc-${S.pair.replace(/\W/g, '_')}`)?.scrollIntoView({ block: 'nearest' });
}
async function load() {
  const r = await fetch('/api/data'); if (!r.ok) { document.getElementById('app').innerHTML = `<pre style="padding:20px">${esc(await r.text())}</pre>`; return; }
  D = await r.json(); for (const k in byId) delete byId[k]; D.runs.forEach(x => byId[x.id] = x);
  const [wantRaw, qs] = location.hash.slice(1).split(/&(.*)/s), want = decodeURIComponent(wantRaw || ''), hq = new URLSearchParams(qs || '');
  if (!byId[S.sel]) S.sel = null;
  if (byId[want] && !S.sel) { selectNode(want); if (hq.get('layer')) S.layer = hq.get('layer'); if (hq.get('pair')) S.pair = hq.get('pair'); render(); } else if (!S.sel && D.runs.length) selectNode(D.runs[0].id); else render();
}
async function reload() { await fetch('/api/reload'); for (const k in PAIRDATA) delete PAIRDATA[k]; await load(); }
load();
