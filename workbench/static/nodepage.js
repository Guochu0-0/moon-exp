// 节点页（#/exp/<id>）：一份严谨的实验记录。纸面、衬线；只呈现事实，唯一可编辑的是 Notes。
// 版式参照 prototype/node-page 分支的定稿：左栏章节导航，右栏 实验信息（含比较设置）→ 结果 → 可视化结果 → 训练图表 → Notes。
// 可视化结果见 visual.js。训练图表读 runs/<id>/tb/ 的 TensorBoard scalars，每个 tag 一张小图。没有内容的章节不显示。
import { api } from './api.js';
import { mountVisual } from './visual.js';
import { mountNotes } from './notes.js';

const BINS = [5, 20];          // 误差分档：≤5、5–20、>20 px（错配）、失败
const CDF_MAX = 30;            // 累积分布横轴上限（px）
const RUN_COLORS = ['var(--me)', '#C26A00', '#2E8B57', '#8E44AD', '#B03A48', '#5D6D7E'];   // 训练图表：同一 tag 下的各 run

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const num = x => typeof x === 'number' && isFinite(x);
const pct = x => num(x) ? (x * 100).toFixed(1) : '—';
const pxf = x => num(x) ? x.toFixed(2) : '—';
const signed = (d, f) => { if (!num(d)) return '—'; const t = f(Math.abs(d)); return Number(t) === 0 ? t : (d > 0 ? '+' : '−') + t; };
const ci = (c, f = pct) => c ? `<span class="ci">[${f(c[0])}, ${f(c[1])}]</span>` : '';
const up = k => k.replace(/^auc/, 'AUC').replace(/^sr/, 'SR');
const link = id => `<a href="#/exp/${encodeURIComponent(id)}">${esc(id)}</a>`;

// 中位数：summary 里失败计 ∞，过半失败时为 ∞（JSON 里是 null）
const median = s => s && s.n > 0 && s.median == null ? '∞' : pxf(s?.median);

export function mountNodePage(root, { onBack }) {
  root.className = 'np';
  let D = null, CMP = null, TB = null, NOTES = null, seq = 0, FIGS = 0, VFIGS = 0;
  const vis = mountVisual(() => !root.hidden);
  const S = { method: null, ref: undefined, split: 'val' };
  const $ = s => root.querySelector(s);
  window.addEventListener('beforeunload', ev => { if (NOTES?.dirty()) ev.preventDefault(); });
  // 离开节点页前：Notes 有未保存的改动就问一次，问过了（或没改动）就不再管
  function leave() {
    if (NOTES?.dirty() && !confirm('Notes 有未保存的改动，仍然离开？')) return false;
    NOTES = null; return true;
  }

  // ---------------- 选择 ----------------
  const multi = () => D.methods.length > 1;
  const cur = () => D.methods.find(m => m.id === S.method) || D.methods[0];
  const refKey = () => S.ref === undefined ? cur().ref : S.ref;
  const options = () => D.reference.groups.flatMap(g => g.options);
  const refSplits = () => refKey() == null ? ['val', 'test'] : options().find(o => o.key === refKey())?.splits || [];
  const splitOk = s => s in cur().results && refSplits().includes(s);
  const SPLIT = () => S.split === 'val' ? 'Val' : 'Test';

  async function load(id) {
    const my = ++seq;
    root.innerHTML = `<div class="layout"><nav class="nav">${backLink()}</nav><div class="content"><p class="muted">读取中…</p></div></div>`;
    bindBack(id);
    try {
      const d = await api.detail(id);
      if (my !== seq) return;
      D = d; Object.assign(S, { method: d.methods[0]?.id ?? null, ref: undefined, split: 'val' });
      CMP = null; TB = null; VFIGS = 0;
      page();
      if (D.tb.length) loadScalars(d.id);
      if (D.lit) await compare();
    } catch (err) {
      if (my !== seq) return;
      $('.content').innerHTML = `<p>${err.status === 404 ? `没有实验 ${esc(id)}。` : `读取失败：${esc(err.message)}`}</p>`;
    }
  }

  async function compare() {
    if (!splitOk(S.split)) S.split = 'val';
    const my = ++seq, id = D.id;
    try {
      CMP = await api.compare(`${id}/${cur().id}`, refKey(), S.split);
    } catch (err) {
      if (my !== seq) return;
      CMP = { error: err.message };
    }
    if (my !== seq || D?.id !== id) return;
    info(); results(); visual(); train();
  }

  // ---------------- 页面骨架 ----------------
  const backLink = () => '<a class="back" href="#/">← 画布</a>';
  function bindBack(id) {
    $('.back').addEventListener('click', ev => {
      ev.preventDefault();
      if (leave()) onBack(id);
    });
  }
  function page() {
    const secs = [['info', '实验信息'], ['results', '结果'], ...(D.lit ? [['visual', '可视化结果']] : []),
      ...(D.tb.length ? [['train', '训练图表']] : []), ['notes', 'Notes']];
    root.innerHTML = `<div class="layout">
      <nav class="nav" aria-label="章节">${backLink()}<ol>${secs.map(([k, t]) => `<li><a href="#sec-${k}" data-sec="${k}">${t}</a></li>`).join('')}</ol></nav>
      <div class="content">${secs.map(([k, t]) => `<section class="sec" id="sec-${k}">${k === 'info' ? '' : `<h2>${t}</h2>`}<div class="body"></div></section>`).join('')}</div></div>`;
    bindBack(D.id);
    root.querySelectorAll('.nav a[data-sec]').forEach(a => a.addEventListener('click', ev => {
      ev.preventDefault(); $(`#sec-${a.dataset.sec}`).scrollIntoView({ behavior: 'smooth', block: 'start' });
    }));
    info(); results(); visual(); train(); notes(); markNav();
  }
  function markNav() {
    const secs = [...root.querySelectorAll('.content .sec')];
    if (!secs.length) return;
    const top = root.getBoundingClientRect().top + root.clientHeight / 4;
    let on = secs[0].id;
    for (const el of secs) if (el.getBoundingClientRect().top <= top) on = el.id;
    if (root.scrollTop + root.clientHeight >= root.scrollHeight - 2) on = secs[secs.length - 1].id;
    root.querySelectorAll('.nav a[data-sec]').forEach(a => a.setAttribute('aria-current', String(`sec-${a.dataset.sec}` === on)));
  }
  root.addEventListener('scroll', markNav, { passive: true });

  // ---------------- 实验信息与比较设置 ----------------
  function commitText(c) {
    const short = s => esc(String(s).slice(0, 10));
    if (c.state === 'unknown') return '未知';
    if (c.state === 'single') return `<span class="mono">${short(c.commit)}</span>${c.dirty ? '（工作区有未提交改动）' : ''}`;
    const rows = Object.entries(c.methods).flatMap(([m, sp]) => Object.entries(sp).map(([s, v]) =>
      `<li><span class="mono">${esc(m)}</span> ${s}：${v?.commit ? `<span class="mono">${short(v.commit)}</span>${v.dirty ? '（dirty）' : ''}` : '未知'}</li>`));
    return `<details class="commits"><summary>多个</summary><ul>${rows.join('')}</ul></details>`;
  }
  function info() {
    const p = D.parent, rows = [
      ['父实验', p ? link(p) : '无'],
      ['起点方法', D.init ? `<span class="mono">${esc(D.init)}</span>` : '无'],
      ['子实验', D.children.length ? D.children.map(link).join('，') : '无'],
      ['状态', (D.lit ? '已点亮' : '未点亮') + (D.baseline ? '，基线' : '')],
      ['日期', esc(D.date || '—')],
      ['commit', commitText(D.commit)]];
    if (multi()) rows.splice(3, 0, ['方法数', `<span class="num">${D.methods.length}</span>`]);
    $('#sec-info .body').innerHTML = `<h1><span class="id">${esc(D.id)}</span>${esc(D.title)}</h1>
      <dl class="meta">${rows.map(([t, d]) => `<div><dt>${t}</dt><dd>${d}</dd></div>`).join('')}</dl>
      ${D.lit ? settings() : ''}`;
    if (!D.lit) return;
    $('#selM')?.addEventListener('change', ev => pickMethod(ev.target.value));
    $('#selR').addEventListener('change', ev => { const v = ev.target.value; S.ref = v === '' ? null : v; compare(); });
    root.querySelectorAll('.radio button').forEach(b => b.addEventListener('click', () => { S.split = b.dataset.split; compare(); }));
  }
  function settings() {
    const m = cur(), def = m.ref, now = refKey(), me = `${D.id}/${m.id}`;
    const opt = o => `<option value="${esc(o.key ?? '')}" ${o.key === now ? 'selected' : ''}>${esc(o.label)}${o.key === def ? '（默认）' : ''}</option>`;
    const groups = D.reference.groups.map(g => {
      const os = g.options.filter(o => o.key !== me);
      return os.length ? `<optgroup label="${esc(g.label)}">${os.map(opt).join('')}</optgroup>` : '';
    }).join('');
    const methodCtl = multi()
      ? `<label for="selM"><span class="key"></span>方法</label><select id="selM" class="me">${D.methods.map(x =>
          `<option value="${esc(x.id)}" ${x.id === m.id ? 'selected' : ''}>${esc(x.name)}</option>`).join('')}</select>`
      : `<span class="lbl"><span class="key"></span>方法</span>${esc(m.name === 'main' ? D.id : m.name)}`;
    const radio = ['val', 'test'].map(s => `<button data-split="${s}" aria-pressed="${S.split === s}" ${splitOk(s) ? '' : 'disabled title="没有 Test 结果"'}>${s === 'val' ? 'Val' : 'Test'}</button>`).join('');
    return `<div class="cmp">
      <span>${methodCtl}</span>
      <span><label for="selR"><span class="key ref"></span>参考方法</label><select id="selR">${groups}</select></span>
      <span><span class="lbl">数据集</span><span class="radio" role="group" aria-label="数据集">${radio}</span></span>
      ${S.split === 'test' ? '<span class="testflag" role="status">当前显示 Test 结果</span>' : ''}</div>`;
  }
  function pickMethod(id) { S.method = id; S.ref = undefined; compare(); }

  // ---------------- 结果 ----------------
  function results() {
    const body = $('#sec-results .body');
    if (!D.lit) { body.innerHTML = '<p>尚无结果。</p>'; return; }
    if (!CMP) { body.innerHTML = '<p class="muted">计算中…</p>'; return; }
    if (CMP.error) { body.innerHTML = `<p>读取比较结果失败：${esc(CMP.error)}</p>`; return; }
    const n = { tab: 0, fig: 0 };
    let h = S.split === 'test' ? '<p class="testflag block">以下为 Test 集上的结果。</p>' : '';
    if (multi()) h += methodsTable(n);
    if (!multi() || CMP.ref) h += pairTable(n);
    h += `<div class="figs">${cdfFigure(n)}${binFigure(n)}</div>`;
    body.innerHTML = h;
    FIGS = n.fig;
    body.querySelectorAll('tr.pick').forEach(tr => tr.addEventListener('click', () => pickMethod(tr.dataset.m)));
  }
  // 全表与分档图的行序：按当前数据集上的主指标降序，没有该数据集结果的排最后
  const ranked = () => D.methods.slice().sort((a, b) =>
    (b.results[S.split]?.summary[D.protocol.main] ?? -1) - (a.results[S.split]?.summary[D.protocol.main] ?? -1));
  function methodsTable(n) {
    const P = D.protocol, MAIN = P.main, s = S.split, withBase = !!D.parent;
    const cols = [...P.table_auc.map(t => `auc@${t}`), ...P.table_sr.map(t => `sr@${t}`)];
    const rows = ranked();
    const delta = D.delta_key;
    return `<p class="caption"><b>表 ${++n.tab}</b>${esc(D.id)} 各方法在 ${SPLIT()} 上的指标，按 ${up(MAIN)} 降序。AUC、SR、失败率单位为 %；${up(MAIN)} 后方括号内为 bootstrap 95% 置信区间；误差中位数统计全部有标注 pair，失败计为无穷大。${withBase ? `末列为相对父实验 ${esc(D.parent)} 中同名基础方法的 ${up(delta)} 差值（百分点），父实验没有同名基础方法时为「—」。` : ''}点击一行设为当前方法。</p>
      <div class="scroll"><table class="tab num"><thead><tr><th>方法</th>${cols.map(k => `<th>${up(k)}</th>`).join('')}<th>失败率</th><th>误差中位数（px）</th>${withBase ? `<th>Δ${up(delta)}</th>` : ''}</tr></thead>
      <tbody>${rows.map(m => { const v = m.results[s]?.summary || {};
        return `<tr class="pick ${m.id === cur().id ? 'on' : ''}" data-m="${esc(m.id)}" title="${esc(m.id)}"><td>${esc(m.name)}</td>
          ${cols.map(k => `<td>${pct(v[k])}${k === MAIN ? ci(v[k + '_ci']) : ''}</td>`).join('')}<td>${pct(v.fail_rate)}</td><td>${median(v)}</td>
          ${withBase ? `<td>${signed(m.delta[s] * 100, a => a.toFixed(1))}</td>` : ''}</tr>`; }).join('')}</tbody></table></div>`;
  }
  function pairTable(n) {
    const P = D.protocol, a = CMP.method, b = CMP.ref, d = CMP.diff || {};
    const rows = [...P.auc_thresholds.map(t => [`AUC@${t}`, `auc@${t}`, 'pct']), ...P.sr_thresholds.map(t => [`SR@${t}`, `sr@${t}`, 'pct']),
      ['失败率', 'fail_rate', 'pct'], ['误差中位数（px）', 'median', 'px']];
    const val = (sm, k, u) => u === 'px' ? median(sm) : pct(sm[k]);
    const dv = (k, u) => u === 'px' ? signed(d[k], x => x.toFixed(2)) : signed(d[k] * 100, x => x.toFixed(1));
    const dci = k => d[k + '_ci'] ? `<span class="ci">[${d[k + '_ci'].map(x => signed(x * 100, y => y.toFixed(1))).join(', ')}]</span>` : '';
    return `<p class="caption"><b>表 ${++n.tab}</b>${esc(a.label)}${b ? ` 与参考方法 ${esc(b.label)}` : ''} 在 ${SPLIT()} 上的全部指标（n = <span class="num">${a.summary.n}</span> 个有标注 pair）。AUC、SR、失败率单位为 %；AUC 与 SR 后方括号内为 bootstrap 95% 置信区间${b ? '；差值 = 方法 − 参考方法，单位为百分点或 px，其置信区间由配对 bootstrap 得到' : ''}。</p>
      <table class="tab num"><thead><tr><th>指标</th><th>${esc(a.label)}</th>${b ? `<th>${esc(b.label)}</th><th>差值</th>` : ''}</tr></thead>
      <tbody>${rows.map(([nm, k, u]) => `<tr><td>${nm}</td><td class="me">${val(a.summary, k, u)}${ci(a.summary[k + '_ci'])}</td>
        ${b ? `<td class="ref">${val(b.summary, k, u)}${ci(b.summary[k + '_ci'])}</td><td>${dv(k, u)}${dci(k)}</td>` : ''}</tr>`).join('')}</tbody></table>`;
  }

  // ---------------- 可视化结果 ----------------
  function visual() {
    const body = $('#sec-visual .body');
    if (!body) return;
    if (!CMP) body.innerHTML = '<p class="muted">计算中…</p>';
    else if (CMP.error) body.innerHTML = '<p class="muted">比较结果读取失败，见上节。</p>';
    else VFIGS = vis.render(body, CMP, FIGS);
  }

  // ---------------- 训练图表 ----------------
  async function loadScalars(id) {
    let tb;
    try { tb = await api.scalars(id); } catch (err) { tb = { error: err.message }; }
    if (D?.id !== id) return;
    TB = tb; train();
  }
  function train() {
    const body = $('#sec-train .body');
    if (!body) return;
    const head = `<div class="tb-head"><span class="muted">日志：<span class="mono">runs/${esc(D.id)}/tb/</span></span>
      <button class="tbopen">在 TensorBoard 中打开</button><span class="tbstatus muted" role="status"></span></div>`;
    let h = '';
    if (!TB) h = '<p class="muted">读取中…</p>';
    else if (TB.error) h = `<p>读取训练日志失败：${esc(TB.error)}</p>`;
    else {
      let fig = VFIGS || FIGS;
      const many = TB.methods.length > 1;
      h = TB.methods.map(m => {
        const name = D.methods.find(x => x.id === m.method)?.name || m.method;
        const tags = [...new Set(m.runs.flatMap(r => Object.keys(r.tags)))].sort();
        const color = i => RUN_COLORS[i % RUN_COLORS.length];
        const legend = m.runs.length > 1 ? `<div class="legend">${m.runs.map((r, i) =>
          `<span><span class="key" style="border-top-color:${color(i)}"></span><span class="mono">${esc(r.run)}</span></span>`).join('')}</div>` : '';
        const thinned = m.runs.some(r => Object.values(r.tags).some(t => t.n > t.step.length));
        const charts = tags.map(tag => chart(tag, m.runs.map((r, i) => ({ s: r.tags[tag], color: color(i), run: r.run })).filter(x => x.s))).join('');
        const one = m.runs.length === 1 ? `TensorBoard run 为 <span class="mono">${esc(m.runs[0].run)}</span>。` : '不同 TensorBoard run（含 events 文件的各个目录，如 version_N、add_scalars 的子目录）分开画，不拼接。';
        return `${many ? `<h3>${esc(name)}</h3>` : ''}${legend}<div class="tbgrid">${charts}</div>
          <p class="caption figcap"><b>图 ${++fig}</b>${many ? `${esc(name)} ` : ''}训练日志 <span class="mono">tb/${esc(m.method)}/</span> 中的 scalar，每个 tag 一张：横轴为 step（不同 tag 的 step 含义可能不同，例如以 epoch 计），纵轴为取值。${one}同一目录里续训重叠的 step 已按 tag 截断，保留后写的点。${thinned ? `点数超过 ${TB.max_points} 的曲线按序号分桶，只画每桶的最小、最大值和首尾点。` : ''}</p>`;
      }).join('');
    }
    body.innerHTML = head + h;
    const btn = body.querySelector('.tbopen'), st = body.querySelector('.tbstatus'), id = D.id;
    btn.addEventListener('click', async () => {
      const w = window.open('', '_blank');   // 先开窗口：等子进程就绪后再开会被拦截
      btn.disabled = true; st.textContent = '正在启动 TensorBoard…';
      try {
        const { url } = await api.tensorboard(id);
        if (w) w.location = url;
        st.innerHTML = `已打开 <a href="${esc(url)}" target="_blank" rel="noopener">${esc(url)}</a>`;
      } catch (err) {
        w?.close(); st.textContent = `启动失败：${err.message}`;
      }
      btn.disabled = false;
    });
  }
  // 一张小图：真实坐标轴（step、取值），各 run 一条线；只有一个点时画成点
  function chart(tag, series) {
    const W = 300, H = 196, L = 50, B = 30, T = 8, R = 10, pw = W - L - R, ph = H - B - T;
    const xs = series.flatMap(x => x.s.step), ys = series.flatMap(x => x.s.value).filter(num);
    let [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = ys.length ? [Math.min(...ys), Math.max(...ys)] : [0, 1];
    if (x1 === x0) { x0 -= 1; x1 += 1; }
    if (y1 === y0) { const d = Math.abs(y0) * .1 || 1; y0 -= d; y1 += d; }
    const xt = ticks(x0, x1, 4), yt = ticks(y0, y1, 4);
    [x0, x1] = [Math.min(x0, xt.lo), Math.max(x1, xt.hi)]; [y0, y1] = [Math.min(y0, yt.lo), Math.max(y1, yt.hi)];
    const X = v => L + (v - x0) / (x1 - x0) * pw, Y = v => T + ph - (v - y0) / (y1 - y0) * ph;
    let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(tag)}"><g class="ax">`;
    for (const t of xt.v) g += `<line x1="${X(t)}" x2="${X(t)}" y1="${T + ph}" y2="${T + ph + 4}"/><text x="${X(t)}" y="${T + ph + 16}" text-anchor="middle">${tick(t, xt.step)}</text>`;
    for (const t of yt.v) g += `<line class="grid" x1="${L}" x2="${L + pw}" y1="${Y(t)}" y2="${Y(t)}"/><line x1="${L - 4}" x2="${L}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${L - 7}" y="${Y(t) + 4}" text-anchor="end">${tick(t, yt.step)}</text>`;
    g += `<line x1="${L}" x2="${L}" y1="${T}" y2="${T + ph}"/><line x1="${L}" x2="${L + pw}" y1="${T + ph}" y2="${T + ph}"/><text x="${L + pw}" y="${H - 2}" text-anchor="end">step</text></g>`;
    for (const { s, color, run } of series) {
      let d = '', pen = false;
      s.step.forEach((x, i) => { const v = s.value[i]; if (!num(v)) { pen = false; return; } d += `${pen ? 'L' : 'M'}${X(x).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; });
      const n = s.value.filter(num).length;
      g += n === 1 ? s.step.map((x, i) => num(s.value[i]) ? `<circle cx="${X(x)}" cy="${Y(s.value[i])}" r="2.5" fill="${color}"/>` : '').join('')
        : `<path class="tbln" d="${d}" stroke="${color}"><title>${esc(run)}：${s.n} 个点</title></path>`;
    }
    return `<figure class="tbchart"><figcaption class="mono">${esc(tag)}</figcaption>${g}</svg></figure>`;
  }

  // ---------------- 图 ----------------
  function series() {   // [{label, errors, role}]：当前方法、参考方法；多方法且无参考时其余方法作背景
    const me = { id: cur().id, label: CMP.ref || !multi() ? CMP.method.label : cur().name, errors: CMP.method.errors, role: 'me' };
    if (CMP.ref) return [me, { label: CMP.ref.label, errors: CMP.ref.errors, role: 'ref' }];
    if (!multi()) return [me];
    const s = S.split;
    return [me, ...D.methods.filter(m => m.id !== cur().id && m.results[s]).map(m => ({ id: m.id, label: m.name, errors: m.results[s].errors, role: 'other' }))];
  }
  function cdfFigure(n) {
    const W = 640, H = 330, L = 50, B = 42, T = 12, pw = W - L - 14, ph = H - B - T;
    const xs = v => L + v / CDF_MAX * pw, ys = f => T + ph - f * ph;
    let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="误差累积分布"><g class="ax">`;
    for (let t = 0; t <= CDF_MAX; t += 5) g += `<line x1="${xs(t)}" x2="${xs(t)}" y1="${T + ph}" y2="${T + ph + 4}"/><text x="${xs(t)}" y="${T + ph + 18}" text-anchor="middle">${t}</text>`;
    for (const f of [0, .25, .5, .75, 1]) g += `<line class="grid" x1="${L}" x2="${L + pw}" y1="${ys(f)}" y2="${ys(f)}"/><line x1="${L - 4}" x2="${L}" y1="${ys(f)}" y2="${ys(f)}"/><text x="${L - 8}" y="${ys(f) + 4}" text-anchor="end">${f * 100}</text>`;
    g += `<line x1="${L}" x2="${L}" y1="${T}" y2="${T + ph}"/><line x1="${L}" x2="${L + pw}" y1="${T + ph}" y2="${T + ph}"/>
      <text x="${L + pw / 2}" y="${H - 4}" text-anchor="middle">误差阈值（px）</text><text transform="translate(13 ${T + ph / 2}) rotate(-90)" text-anchor="middle">累积占比（%）</text></g>`;
    const path = e => {
      const v = e.filter(num).sort((a, b) => a - b), N = e.length;
      let d = '', i = 0;
      for (let x = 0; x <= CDF_MAX + 1e-9; x += .1) { while (i < v.length && v[i] <= x) i++; d += `${d ? 'L' : 'M'}${xs(x).toFixed(1)},${ys(N ? i / N : 0).toFixed(1)}`; }
      return d;
    };
    const ss = series();
    for (const r of ['other', 'ref', 'me']) for (const x of ss.filter(x => x.role === r)) g += `<path class="ln ${r}" d="${path(x.errors)}"/>`;
    g += '</svg>';
    const others = ss.filter(x => x.role === 'other').length;
    const legend = ss.filter(x => x.role !== 'other').map(x => `<span><span class="key ${x.role === 'ref' ? 'ref' : ''}"></span>${esc(x.label)}</span>`).join('')
      + (others ? `<span><span class="key other"></span>${esc(D.id)} 的其他 ${others} 个方法</span>` : '');
    return `<figure>${g}<div class="legend">${legend}</div><p class="caption figcap"><b>图 ${++n.fig}</b>${SPLIT()} 上配准误差的累积分布：纵轴为误差不超过横轴阈值的有标注 pair 占比。失败的 pair 计为误差无穷大，不进入曲线。</p></figure>`;
  }
  function binFigure(n) {
    const C = [[`≤${BINS[0]} px`, 'b0'], [`${BINS[0]}–${BINS[1]} px`, 'b1'], [`>${BINS[1]} px（错配）`, 'b2'], ['失败', 'b3']];
    const bin = e => !num(e) ? 3 : e <= BINS[0] ? 0 : e <= BINS[1] ? 1 : 2;
    const ss = series(), order = ranked().map(m => m.id);
    if (!CMP.ref && multi()) ss.sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));   // 与全表同序
    const rows = ss.map(x => {
      const cnt = [0, 0, 0, 0]; x.errors.forEach(e => cnt[bin(e)]++);
      return `<div class="oc-row ${x.role === 'me' ? 'on' : ''}"><span>${esc(x.label)}</span><span class="oc-bar">${cnt.map((c, j) =>
        `<i class="${C[j][1]}" style="flex:${c}" title="${C[j][0]}：${c} / ${x.errors.length}"></i>`).join('')}</span></div>`;
    }).join('');
    return `<figure>${rows}<div class="legend">${C.map(([t, c]) => `<span><span class="swatch ${c}"></span>${t}</span>`).join('')}</div>
      <p class="caption figcap"><b>图 ${++n.fig}</b>${SPLIT()} 上有标注 pair 按配准误差分档的占比。</p></figure>`;
  }

  // ---------------- Notes ----------------
  function notes() {
    const id = D.id;
    NOTES = mountNotes($('#sec-notes .body'), {
      text: D.notes, base: `/runs/${encodeURIComponent(id)}/`, path: `runs/${id}/notes.md`,
      save: async text => { await api.saveNotes(id, text); if (D?.id === id) D.notes = text; },
    });
  }

  return {
    show(id) { root.hidden = false; root.scrollTop = 0; load(id); },
    shown: () => !root.hidden,
    leave,
    hide() { root.hidden = true; root.innerHTML = ''; D = null; NOTES = null; seq++; },
  };
}

// 坐标轴刻度：步长取 1、2、5 × 10^k，大约 n 个
function ticks(lo, hi, n) {
  const raw = (hi - lo) / n, p = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 5, 10].map(k => k * p).find(k => k >= raw);
  const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step, v = [];
  for (let t = a; t <= b + step * 1e-9; t += step) v.push(+t.toPrecision(12));
  return { v, step, lo: a, hi: b };
}
function tick(t, step) {
  if (Math.abs(t) >= 1e5 || (t !== 0 && Math.abs(step) < 1e-4)) return t.toExponential(1);
  return t.toFixed(Math.max(0, -Math.floor(Math.log10(step))));
}
