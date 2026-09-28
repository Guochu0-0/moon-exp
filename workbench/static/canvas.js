// 画布：浏览实验，新建、派生、改父、改编号、删除。视觉与交互参照 prototype/canvas 定稿 commit 86242d5。
// 手写画布：HTML div 卡片 + 一层 SVG 连线，world 层整体 transform 做平移缩放。
// 语义改动都走 API 写回 exp.toml；卡片坐标写回 runs/canvas.json；视口存 localStorage。
import { api } from './api.js';

const VIEW_KEY = 'workbench.canvas.view';
const POLL_MS = 3000;
const SHOWN = 4;             // 默认显示前几个方法；与 workbench/canvas.py 的 SHOWN_METHODS 一致
const DRAG_PX = 4;           // 移动超过这么多才算拖动，否则是单击

const fam = m => !m ? 'none' : /roma/i.test(m) ? 'roma' : /loftr/i.test(m) ? 'loftr' : 'other';
const pct = x => x == null ? '—' : (x * 100).toFixed(1);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const sel = (attr, v) => `[${attr}="${CSS.escape(v)}"]`;
const curve = (a, b) => { const dx = Math.max(60, Math.abs(b[0] - a[0]) * .45); return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`; };
const ICON_PLUS = '<svg class="i" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>';
const ICON_FIT = '<svg class="i" viewBox="0 0 24 24"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>';

export function mountCanvas(root, { onOpen }) {
  root.className = 'cv';
  root.innerHTML = `
    <div class="viewport" tabindex="0"><div class="world"><svg class="edges"></svg><div class="layer"></div></div></div>
    <div class="title"><b>实验画布</b><span class="count"></span></div>
    <div class="act">
      <button class="pri" data-cmd="exp">${ICON_PLUS}新建实验 <kbd>E</kbd></button>
      <button data-cmd="zout" aria-label="缩小">−</button><span class="pc"></span><button data-cmd="zin" aria-label="放大">+</button>
      <button data-cmd="fit" title="适应窗口（.）" aria-label="适应窗口">${ICON_FIT}</button>
    </div>
    <div class="menu" hidden></div>
    <div class="toast" role="status"></div>`;
  const $ = s => root.querySelector(s);
  const vp = $('.viewport'), world = $('.world'), layer = $('.layer'), edges = $('.edges'), menu = $('.menu');

  // ---------------- 状态 ----------------
  let EXPS = [], byId = new Map(), pos = new Map(), lastText = '', active = false;
  const S = { tx: 0, ty: 0, k: 1, sel: null, mouse: [innerWidth / 2, innerHeight / 2], link: null, fresh: null,
              expanded: new Set(), editing: false };
  let drag = null;
  const saved = JSON.parse(localStorage.getItem(VIEW_KEY) || 'null');
  if (saved) Object.assign(S, { tx: saved.tx, ty: saved.ty, k: saved.k });

  const kids = id => EXPS.filter(e => e.parent === id);
  const initOf = (parent, m) => m && m !== '_' ? `${parent}/${m}` : null;   // 端口 '_' 是通用端口，不带 init
  const initMethod = e => e.parent && e.init && e.init.startsWith(e.parent + '/') ? e.init.slice(e.parent.length + 1) : null;

  async function refresh(force = false) {
    let d;
    try { d = await api.data(); } catch (err) { toast(`读取数据失败：${err.message}`, true); return; }
    const text = JSON.stringify(d);
    if (text === lastText && !force) return;
    lastText = text;
    EXPS = d.experiments;
    byId = new Map(EXPS.map(e => [e.id, e]));
    pos = new Map(EXPS.map(e => [e.id, [e.x, e.y]]));
    if (S.sel && !byId.has(S.sel)) S.sel = null;
    render();
  }
  const busy = () => drag || S.editing || !menu.hidden;
  setInterval(() => { if (active && !document.hidden && !busy()) refresh(); }, POLL_MS);
  window.addEventListener('focus', () => { if (active && !busy()) refresh(); });

  // ---------------- 渲染 ----------------
  function nodeHTML(e) {
    let body;
    if (e.lit) {
      const ms = e.methods;
      let show = S.expanded.has(e.id) ? ms : ms.slice(0, SHOWN);
      const used = new Set(kids(e.id).map(initMethod).filter(Boolean));   // 被子实验用作 init 的方法总是显示，好接线
      show = show.concat(ms.filter(m => used.has(m.id) && !show.includes(m)));
      body = show.map(m => `<div class="row" data-m="${esc(m.id)}">
          <span class="mn" title="${esc(m.name === m.id ? m.id : `${m.id}（${m.name}）`)}">${ms.length > 1 ? esc(m.id) : '本实验'}</span>
          <div class="bar"><i style="width:${Math.max(0, Math.min(1, m.auc || 0)) * 100}%"></i><span>${pct(m.auc)}</span></div>
          <span class="port out fam-${fam(m.id)}" data-m="${esc(m.id)}" title="从 ${esc(m.id)} 拖出：拖到实验上改它的父实验，拖到空白处派生新实验"></span></div>`).join('')
        + (ms.length > show.length ? `<div class="more">▸ 另外 ${ms.length - show.length} 个方法</div>`
          : S.expanded.has(e.id) && ms.length > SHOWN ? '<div class="more">▾ 收起</div>' : '<div class="rows-end"></div>');
    } else body = '<div class="empty">尚无结果</div>';
    return `<div class="node${e.lit ? '' : ' off'}${S.sel === e.id ? ' sel' : ''}${S.fresh === e.id ? ' new' : ''}" data-id="${esc(e.id)}">
      <div class="hdr"><span class="nid">${esc(e.id)}</span><span class="ttl" title="${esc(e.title)}">${esc(e.title)}</span>${e.baseline ? '<span class="chip">基线</span>' : ''}<span class="st" title="${e.lit ? '已点亮' : '未点亮'}"></span></div>
      <div class="io"><span class="port in"></span><span>父实验</span><span>${e.lit ? 'AUC@10（Val）' : ''}</span><span>子实验</span><span class="port out fam-none" data-m="_" title="拖出：拖到实验上改它的父实验，拖到空白处派生新实验"></span></div>
      ${body}</div>`;
  }
  function render() {
    layer.innerHTML = EXPS.map(nodeHTML).join('');
    for (const e of EXPS) place(e.id);
    S.fresh = null;
    $('.count').textContent = `${EXPS.length} 个实验，${EXPS.filter(e => e.lit).length} 个已点亮`;
    applyView(); drawEdges();
  }
  const nodeEl = id => layer.querySelector(`.node${sel('data-id', id)}`);
  function place(id) { const el = nodeEl(id), [x, y] = pos.get(id); el.style.left = x + 'px'; el.style.top = y + 'px'; }
  function centerOf(el) {
    const r = el.getBoundingClientRect(), w = world.getBoundingClientRect();
    return [(r.left + r.width / 2 - w.left) / S.k, (r.top + r.height / 2 - w.top) / S.k];
  }
  function outPort(id, m) { const el = nodeEl(id)?.querySelector(`.port.out${sel('data-m', m || '_')}`); return el && centerOf(el); }
  function drawEdges() {
    let svg = EXPS.filter(e => e.parent && byId.has(e.parent)).map(e => {
      const m = initMethod(e), a = outPort(e.parent, m) || outPort(e.parent, '_'), b = centerOf(nodeEl(e.id).querySelector('.port.in'));
      return `<path class="edge fam-${fam(m)}${e.lit ? '' : ' off'}" d="${curve(a, b)}"/>`;
    }).join('');
    if (S.link) svg += `<path class="edge tmp fam-${fam(S.link.m === '_' ? '' : S.link.m)}" d="${curve(S.link.a, S.link.b)}"/>`;
    edges.innerHTML = svg;
  }
  function applyView() {
    world.style.transform = `translate(${S.tx}px,${S.ty}px) scale(${S.k})`;
    const g = 20 * S.k;
    vp.style.backgroundSize = `${g}px ${g}px, ${g}px ${g}px, ${g * 5}px ${g * 5}px, ${g * 5}px ${g * 5}px`;
    vp.style.backgroundPosition = `${S.tx}px ${S.ty}px`;
    $('.pc').textContent = Math.round(S.k * 100) + '%';
    localStorage.setItem(VIEW_KEY, JSON.stringify({ tx: S.tx, ty: S.ty, k: S.k }));
  }
  function fit() {
    const els = [...layer.children]; if (!els.length) return;
    const w = world.getBoundingClientRect(), rs = els.map(el => el.getBoundingClientRect());
    const x0 = Math.min(...rs.map(r => r.left)), y0 = Math.min(...rs.map(r => r.top)), x1 = Math.max(...rs.map(r => r.right)), y1 = Math.max(...rs.map(r => r.bottom));
    const b = { x0: (x0 - w.left) / S.k, y0: (y0 - w.top) / S.k, x1: (x1 - w.left) / S.k, y1: (y1 - w.top) / S.k };
    const r = vp.getBoundingClientRect(), pad = 80;
    S.k = Math.min(1.1, Math.max(.3, Math.min((r.width - 2 * pad) / (b.x1 - b.x0), (r.height - 2 * pad) / (b.y1 - b.y0))));
    S.tx = (r.width - (b.x1 - b.x0) * S.k) / 2 - b.x0 * S.k; S.ty = (r.height - (b.y1 - b.y0) * S.k) / 2 - b.y0 * S.k;
    applyView();
  }
  function zoomAt(k, mx, my) { k = Math.min(2, Math.max(.3, k)); S.tx = mx - (mx - S.tx) * k / S.k; S.ty = my - (my - S.ty) * k / S.k; S.k = k; applyView(); }
  function zoomBy(f) { const r = vp.getBoundingClientRect(); zoomAt(S.k * f, r.width / 2, r.height / 2); }
  function centerOn(id) {
    const el = nodeEl(id); if (!el) return;
    const [x, y] = pos.get(id), r = vp.getBoundingClientRect();
    S.tx = r.width / 2 - (x + el.offsetWidth / 2) * S.k; S.ty = r.height / 2 - (y + el.offsetHeight / 2) * S.k; applyView();
  }
  const toWorld = (cx, cy) => { const r = vp.getBoundingClientRect(); return [(cx - r.left - S.tx) / S.k, (cy - r.top - S.ty) / S.k]; };

  // ---------------- 操作：都经 API 写回，再重拉 ----------------
  async function attempt(p, ok) {
    try { const r = await p; if (ok) toast(typeof ok === 'function' ? ok(r) : ok); return r; }
    catch (err) { toast(err.message, true); await refresh(true); return null; }
  }
  async function created(r, parent, init) {
    if (!r) return;
    S.fresh = S.sel = r.id; await refresh(true);
    toast(`已新建 ${r.id}${parent ? `，父实验 ${parent}${init ? `，init = ${init}` : ''}` : ''}`);
    editTitle(r.id);
  }
  async function newExpAt(x, y, parent = null, init = null) {
    await created(await attempt(api.create({ x: Math.round(x), y: Math.round(y), parent, init })), parent, init);
  }
  async function derive(parent, m) {   // 由服务端摆到父实验右侧、往下避开已有实验
    const init = initOf(parent, m);
    await created(await attempt(api.create({ parent, init })), parent, init);
  }
  async function reparent(child, parent, m) {
    const init = initOf(parent, m);
    if (await attempt(api.setParent(child, parent, init), `${child} 的父实验改为 ${parent}${init ? `（init = ${init}）` : ''}`)) await refresh(true);
  }
  async function disconnect(id) { if (await attempt(api.setParent(id, null), `${id} 已断开父实验`)) await refresh(true); }
  async function del(id) {
    const e = byId.get(id); if (!e) return;
    if (e.lit) { toast(`${id} 已点亮，不能删除`); return; }
    if (await attempt(api.remove(id), `已删除 ${id}（runs/${id}/）${kids(id).length ? '；它的子实验不再有父实验' : ''}`)) {
      if (S.sel === id) S.sel = null; await refresh(true);
    }
  }
  function editTitle(id) {
    const el = nodeEl(id)?.querySelector('.ttl'); if (!el) return;
    editText(el, async t => { if (t !== byId.get(id).title && await attempt(api.setTitle(id, t))) await refresh(true); else render(); });
  }
  function editId(id) {
    const el = nodeEl(id)?.querySelector('.nid'); if (!el) return;
    editText(el, async t => {
      if (!t || t === id) { render(); return; }
      if (await attempt(api.rename(id, t), `${id} 已改为 ${t}`)) { if (S.sel === id) S.sel = t; await refresh(true); }
    });
  }
  function open(id) { S.sel = id; onOpen(id); }
  async function savePosition(id) {   // 其余还没存过坐标的实验由服务端按当前位置一并钉住
    const [x, y] = pos.get(id);
    await attempt(api.saveCanvas({ experiments: { [id]: { x: Math.round(x), y: Math.round(y) } } }));
  }
  function editText(el, done) {
    const before = el.textContent;
    S.editing = true; el.contentEditable = 'true'; el.focus({ preventScroll: true }); document.getSelection().selectAllChildren(el);
    let cancelled = false;
    const fin = () => {
      el.removeEventListener('blur', fin); el.contentEditable = 'false'; S.editing = false;
      const t = el.innerText.replace(/\s+/g, ' ').trim();
      if (cancelled) { el.textContent = before; render(); } else done(t);
    };
    el.addEventListener('blur', fin);
    el.addEventListener('keydown', ev => {
      if (ev.key === 'Enter') { ev.preventDefault(); el.blur(); }
      if (ev.key === 'Escape') { cancelled = true; el.blur(); }
      ev.stopPropagation();
    });
  }

  // ---------------- 右键菜单 ----------------
  function showMenu(cx, cy, head, items) {
    menu.innerHTML = (head ? `<div class="h">${head}</div>` : '') + items.map((it, i) => it === '-' ? '<hr>'
      : `<button data-i="${i}" ${it.off ? `disabled title="${esc(it.why || '')}"` : ''}><span>${esc(it.t)}</span>${it.k ? `<kbd>${it.k}</kbd>` : ''}</button>`).join('');
    menu.hidden = false;
    const r = menu.getBoundingClientRect();
    menu.style.left = Math.min(cx, innerWidth - r.width - 8) + 'px'; menu.style.top = Math.min(cy, innerHeight - r.height - 8) + 'px';
    menu.onclick = ev => { const b = ev.target.closest('button'); if (!b || b.disabled) return; hideMenu(); items[+b.dataset.i].f(); };
  }
  const hideMenu = () => { menu.hidden = true; };
  vp.addEventListener('contextmenu', ev => {
    ev.preventDefault();
    const [wx, wy] = toWorld(ev.clientX, ev.clientY);
    const n = ev.target.closest('.node'), row = ev.target.closest('.row');
    if (n) {
      const e = byId.get(n.dataset.id), m = row && row.dataset.m;
      if (S.sel !== e.id) { S.sel = e.id; render(); }
      const items = [{ t: '打开节点页', k: '双击', f: () => open(e.id) }, '-'];
      if (m) items.push({ t: `从 ${m} 派生新实验`, f: () => derive(e.id, m) });
      items.push({ t: `从 ${e.id} 派生新实验`, f: () => derive(e.id, null) });
      if (e.parent) items.push({ t: '断开父实验', f: () => disconnect(e.id) });
      items.push('-', { t: '改标题', f: () => editTitle(e.id) },
        { t: '改编号', off: e.lit, why: '已点亮的实验不能改编号', f: () => editId(e.id) },
        '-', { t: '删除实验', k: 'Del', off: e.lit, why: '已点亮的实验不能删除', f: () => del(e.id) });
      showMenu(ev.clientX, ev.clientY, `${esc(e.id)}　${esc(e.title)}`, items);
    } else {
      showMenu(ev.clientX, ev.clientY, null, [
        { t: '新建实验', k: 'E', f: () => newExpAt(wx - 150, wy - 40) }, '-',
        { t: '适应窗口', k: '.', f: fit }]);
    }
  });

  // ---------------- 鼠标：平移、拖动实验、拖线、缩放 ----------------
  vp.addEventListener('pointerdown', ev => {
    if (ev.button !== 0) return;
    hideMenu();
    if (ev.target.closest('[contenteditable=true]')) return;
    vp.focus({ preventScroll: true });
    const [wx, wy] = toWorld(ev.clientX, ev.clientY);
    const port = ev.target.closest('.port.out'), n = ev.target.closest('.node');
    if (port) {
      const from = n.dataset.id, m = port.dataset.m;
      S.link = { from, m, a: outPort(from, m), b: [wx, wy] }; drag = { kind: 'link' }; return;
    }
    if (n) {
      const id = n.dataset.id;
      if (S.sel !== id) { S.sel = id; layer.querySelectorAll('.node.sel').forEach(x => x.classList.remove('sel')); n.classList.add('sel'); }
      const [x, y] = pos.get(id);
      drag = { kind: 'node', id, moved: 0, sx: ev.clientX, sy: ev.clientY, ox: wx - x, oy: wy - y, target: ev.target };
      return;
    }
    drag = { kind: 'pan', sx: ev.clientX - S.tx, sy: ev.clientY - S.ty, px: ev.clientX, py: ev.clientY, moved: 0 };
    vp.classList.add('panning');
  });
  window.addEventListener('pointermove', ev => {
    S.mouse = [ev.clientX, ev.clientY];
    if (!drag) return;
    const [wx, wy] = toWorld(ev.clientX, ev.clientY);
    if (drag.kind === 'pan') {
      drag.moved = Math.max(drag.moved, Math.hypot(ev.clientX - drag.px, ev.clientY - drag.py));
      S.tx = ev.clientX - drag.sx; S.ty = ev.clientY - drag.sy; applyView();
    } else if (drag.kind === 'node') {
      drag.moved = Math.max(drag.moved, Math.hypot(ev.clientX - drag.sx, ev.clientY - drag.sy));
      if (drag.moved < DRAG_PX) return;
      pos.set(drag.id, [Math.round(wx - drag.ox), Math.round(wy - drag.oy)]); place(drag.id); drawEdges();
    } else if (drag.kind === 'link') {
      S.link.b = [wx, wy]; drawEdges();
      layer.querySelectorAll('.drop').forEach(x => x.classList.remove('drop'));
      const t = document.elementFromPoint(ev.clientX, ev.clientY)?.closest('.node');
      if (t && t.dataset.id !== S.link.from) t.classList.add('drop');
    }
  });
  window.addEventListener('pointerup', ev => {
    const d = drag; drag = null; vp.classList.remove('panning');
    if (!d) return;
    if (d.kind === 'link') {
      const t = document.elementFromPoint(ev.clientX, ev.clientY)?.closest('.node'), L = S.link; S.link = null;
      layer.querySelectorAll('.drop').forEach(x => x.classList.remove('drop'));
      if (t && t.dataset.id !== L.from) reparent(t.dataset.id, L.from, L.m);
      else if (!t && root.contains(document.elementFromPoint(ev.clientX, ev.clientY))) {
        const [wx, wy] = toWorld(ev.clientX, ev.clientY); newExpAt(wx + 6, wy - 50, L.from, initOf(L.from, L.m));
      } else drawEdges();
    } else if (d.kind === 'node') {
      if (d.moved >= DRAG_PX) savePosition(d.id);
      else if (d.target.closest('.more')) { S.expanded.has(d.id) ? S.expanded.delete(d.id) : S.expanded.add(d.id); render(); }
    } else if (d.kind === 'pan' && d.moved < DRAG_PX && S.sel) { S.sel = null; render(); }
  });
  vp.addEventListener('dblclick', ev => {
    if (ev.target.closest('[contenteditable=true], .port')) return;
    const n = ev.target.closest('.node'); if (n) { open(n.dataset.id); return; }
    const [wx, wy] = toWorld(ev.clientX, ev.clientY); newExpAt(wx - 150, wy - 40);
  });
  vp.addEventListener('scroll', () => { vp.scrollLeft = vp.scrollTop = 0; });   // 平移只走 transform；聚焦等引起的滚动一律撤销
  vp.addEventListener('wheel', ev => { ev.preventDefault(); const r = vp.getBoundingClientRect(); zoomAt(S.k * Math.exp(-ev.deltaY * .0015), ev.clientX - r.left, ev.clientY - r.top); }, { passive: false });

  // ---------------- 键盘 ----------------
  window.addEventListener('keydown', ev => {
    if (!active || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (ev.target.closest && ev.target.closest('[contenteditable=true], input, textarea, select')) return;
    const [wx, wy] = toWorld(...S.mouse);
    switch (ev.key) {
      case 'e': case 'E': newExpAt(wx - 150, wy - 40); break;
      case '.': fit(); break;
      case '+': case '=': zoomBy(1.2); break;
      case '-': case '_': zoomBy(1 / 1.2); break;
      case 'Escape':
        hideMenu();
        if (S.link) { S.link = null; drag = null; layer.querySelectorAll('.drop').forEach(x => x.classList.remove('drop')); drawEdges(); }
        else if (S.sel) { S.sel = null; render(); }
        break;
      case 'Delete': {
        const id = S.sel || document.elementFromPoint(...S.mouse)?.closest('.node')?.dataset.id;
        if (id) del(id);
        break;
      }
      case 'Enter': if (S.sel) open(S.sel); break;
      default: return;
    }
    ev.preventDefault();
  });
  document.addEventListener('pointerdown', ev => { if (!ev.target.closest('.menu')) hideMenu(); });
  $('.act').addEventListener('click', ev => {
    const b = ev.target.closest('[data-cmd]'); if (!b) return;
    const r = vp.getBoundingClientRect(), [cx, cy] = toWorld(r.left + r.width / 2, r.top + r.height / 2);
    ({ exp: () => newExpAt(cx - 150, cy - 60), fit, zin: () => zoomBy(1.2), zout: () => zoomBy(1 / 1.2) })[b.dataset.cmd]();
  });
  let tt;
  function toast(s, err = false) {
    const t = $('.toast'); t.textContent = s; t.classList.toggle('err', err); t.classList.add('on');
    clearTimeout(tt); tt = setTimeout(() => t.classList.remove('on'), 3800);
  }
  window.addEventListener('resize', () => { if (active) applyView(); });

  let first = true;
  return {
    async show(focus) {
      root.hidden = false; active = true;
      await refresh(true);
      if (focus && byId.has(focus)) { S.sel = focus; render(); centerOn(focus); }
      else if (first && !saved) fit();
      first = false;
      if (document.fonts) document.fonts.ready.then(() => { if (active) drawEdges(); });   // 字体换上后端口位置会变
      vp.focus({ preventScroll: true });
    },
    hide() { root.hidden = true; active = false; hideMenu(); },
  };
}
