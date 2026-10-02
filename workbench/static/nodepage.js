// 节点页（#/exp/<id>）：一份严谨的实验记录。纸面、衬线；只呈现事实，唯一可编辑的是 Notes。
// 版式参照 prototype/node-page 分支的定稿：左栏章节导航，右栏 实验信息（含比较设置）→ 结果 → 可视化结果 → Notes。
// 可视化结果见 visual.js；训练图表由后续票填充。没有内容的章节不显示。
import { api } from './api.js';
import { mountVisual } from './visual.js';

const BINS = [5, 20];          // 误差分档：≤5、5–20、>20 px（错配）、失败
const CDF_MAX = 30;            // 累积分布横轴上限（px）

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
  let D = null, CMP = null, seq = 0, FIGS = 0;
  const vis = mountVisual(() => !root.hidden);
  const S = { method: null, ref: undefined, split: 'val', editing: false };
  const $ = s => root.querySelector(s);

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
      D = d; Object.assign(S, { method: d.methods[0]?.id ?? null, ref: undefined, split: 'val', editing: false });
      CMP = null;
      page();
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
    info(); results(); visual();
  }

  // ---------------- 页面骨架 ----------------
  const backLink = () => '<a class="back" href="#/">← 画布</a>';
  function bindBack(id) {
    $('.back').addEventListener('click', ev => { ev.preventDefault(); onBack(id); });
  }
  function page() {
    const secs = [['info', '实验信息'], ['results', '结果'], ...(D.lit ? [['visual', '可视化结果']] : []), ['notes', 'Notes']];
    root.innerHTML = `<div class="layout">
      <nav class="nav" aria-label="章节">${backLink()}<ol>${secs.map(([k, t]) => `<li><a href="#sec-${k}" data-sec="${k}">${t}</a></li>`).join('')}</ol></nav>
      <div class="content">${secs.map(([k, t]) => `<section class="sec" id="sec-${k}">${k === 'info' ? '' : `<h2>${t}</h2>`}<div class="body"></div></section>`).join('')}</div></div>`;
    bindBack(D.id);
    root.querySelectorAll('.nav a[data-sec]').forEach(a => a.addEventListener('click', ev => {
      ev.preventDefault(); $(`#sec-${a.dataset.sec}`).scrollIntoView({ behavior: 'smooth', block: 'start' });
    }));
    info(); results(); visual(); notes(); markNav();
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
    else vis.render(body, CMP, FIGS);
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
    const body = $('#sec-notes .body'), path = `runs/${esc(D.id)}/notes.md`;
    if (S.editing) {
      body.innerHTML = `<textarea class="notes-edit" spellcheck="false" aria-label="Notes（Markdown）"></textarea>
        <div class="notes-foot"><button class="save">保存</button><button class="cancel">取消</button><span>Markdown；Ctrl+S 保存，Esc 取消。保存到 ${path}</span></div>`;
      const ta = body.querySelector('textarea');
      ta.value = D.notes;
      ta.style.height = `${Math.max(220, ta.scrollHeight + 8)}px`;
      ta.focus();
      ta.addEventListener('input', () => { ta.style.height = 'auto'; ta.style.height = `${Math.max(220, ta.scrollHeight + 8)}px`; });
      ta.addEventListener('keydown', ev => {
        if ((ev.ctrlKey || ev.metaKey) && ev.key === 's') { ev.preventDefault(); save(ta.value); }
        else if (ev.key === 'Escape') cancel();
      });
      const cancel = () => {   // 有未保存的改动时先确认，免得误按 Esc 丢掉
        if (ta.value !== D.notes && !confirm('放弃未保存的 Notes 改动？')) return;
        S.editing = false; notes();
      };
      body.querySelector('.save').addEventListener('click', () => save(ta.value));
      body.querySelector('.cancel').addEventListener('click', cancel);
      return;
    }
    body.innerHTML = `<div class="notes md" title="双击编辑">${D.notes.trim() ? markdown(D.notes, `/runs/${encodeURIComponent(D.id)}/`) : '<p class="empty">（空）</p>'}</div>
      <div class="notes-foot"><button class="edit">编辑</button><span class="status">${path}</span></div>`;
    const edit = () => { S.editing = true; notes(); };
    body.querySelector('.edit').addEventListener('click', edit);
    body.querySelector('.notes').addEventListener('dblclick', ev => { if (!ev.target.closest('a')) edit(); });
  }
  async function save(text) {
    const id = D.id;
    try {
      await api.saveNotes(id, text);
    } catch (err) {
      const st = $('#sec-notes .notes-foot span'); if (st) st.textContent = `保存失败：${err.message}`;
      return;
    }
    if (D?.id !== id) return;
    D.notes = text.replace(/\r\n?/g, '\n'); S.editing = false; notes();
    $('#sec-notes .status').textContent = `已保存到 runs/${id}/notes.md`;
  }

  return {
    show(id) { root.hidden = false; root.scrollTop = 0; load(id); },
    hide() { root.hidden = true; root.innerHTML = ''; D = null; seq++; },
  };
}

// ---------------- Markdown ----------------
// 只覆盖记录里会用到的子集：标题、段落、列表、引用、代码块、分隔线，以及行内代码、粗体、斜体、链接、图片。
// 先整体转义再生成标签，不放行任何原始 HTML；相对路径按实验目录解析（附件如 extra/offset/offset_val.png）。
export function markdown(src, base) {
  const lines = src.replace(/\r\n?/g, '\n').split('\n'), out = [];
  let i = 0;
  const para = [];
  const flush = () => { if (para.length) { out.push(`<p>${inline(join(para), base)}</p>`); para.length = 0; } };
  while (i < lines.length) {
    const l = lines[i];
    let m;
    if ((m = l.match(/^\s*(```|~~~)/))) {
      flush();
      const fence = m[1], code = [];
      for (i++; i < lines.length && !lines[i].trimStart().startsWith(fence); i++) code.push(lines[i]);
      out.push(`<pre><code>${esc(code.join('\n'))}</code></pre>`); i++; continue;
    }
    if (!l.trim()) { flush(); i++; continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/))) { flush(); const k = Math.min(m[1].length + 2, 6); out.push(`<h${k}>${inline(m[2], base)}</h${k}>`); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(l)) { flush(); out.push('<hr>'); i++; continue; }
    if (/^\s*>/.test(l)) {
      flush(); const q = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]); i++) q.push(lines[i].replace(/^\s*>\s?/, ''));
      out.push(`<blockquote>${markdown(q.join('\n'), base)}</blockquote>`); continue;
    }
    if ((m = l.match(/^\s*([-*+]|\d+[.)])\s+/))) {
      flush(); const ordered = /\d/.test(m[1]), items = [];
      for (; i < lines.length; i++) {
        const li = lines[i].match(/^\s*([-*+]|\d+[.)])\s+(.*)$/);
        if (li && /\d/.test(li[1]) === ordered) items.push(li[2]);
        else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) items[items.length - 1] += '\n' + lines[i].trim();
        else break;
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map(t => `<li>${inline(t, base)}</li>`).join('')}</${tag}>`); continue;
    }
    para.push(l); i++;
  }
  flush();
  return out.join('\n');
}

// 段内换行：两侧都是中文（含全角标点）时直接相连，否则按 Markdown 惯例成为空白
const CJK = '⺀-鿿　-〿＀-￯';
const join = lines => lines.join('\n').replace(new RegExp(`([${CJK}])\\n(?=[${CJK}])`, 'g'), '$1');

function url(raw, base) {
  const u = raw.trim().replace(/^<|>$/g, '');
  if (/^(https?:|mailto:)/i.test(u) || u.startsWith('#')) return u;
  if (/^[a-z][a-z0-9+.-]*:/i.test(u) || u.startsWith('/') || u.startsWith('\\')) return null;   // 其他协议、绝对路径不放行
  return base + u.replace(/^\.\//, '');
}

function inline(text, base) {
  const keep = [];
  const hold = h => `\u0000${keep.push(h) - 1}\u0000`;
  let s = text.replace(/`([^`]+)`/g, (_, c) => hold(`<code>${esc(c)}</code>`));
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (all, alt, src) => {
    const u = url(src, base); return u ? hold(`<img src="${esc(u)}" alt="${esc(alt)}" loading="lazy">`) : all;
  });
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (all, t, href) => {
    const u = url(href, base);
    return u ? hold(`<a href="${esc(u)}"${/^https?:/i.test(u) ? ' target="_blank" rel="noopener"' : ''}>${inlineText(t)}</a>`) : all;
  });
  s = inlineText(s);
  return s.replace(/\u0000(\d+)\u0000/g, (_, k) => keep[+k]);
}

function inlineText(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^\w])__(?!\s)(.+?)__(?!\w)/g, '$1<strong>$2</strong>')   // 方法名 a__b 里的 __ 不算
    .replace(/(^|[^*\w])\*(?!\s)(.+?)\*(?!\w)/g, '$1<em>$2</em>')
    .replace(/  \n/g, '<br>');
}
