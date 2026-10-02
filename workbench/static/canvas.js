// 画布：浏览与组织实验。视觉与交互参照 prototype/canvas 定稿 commit 86242d5。
// 手写画布：HTML div 卡片 + 一层 SVG 连线，world 层整体 transform 做平移缩放。
// 语义改动都走 API 写回 exp.toml；卡片坐标、便签、分组框写回 runs/canvas.json；视口存 localStorage。
import { api } from './api.js';

const VIEW_KEY = 'workbench.canvas.view';
const POLL_MS = 3000;
const SHOWN = 4;             // 默认显示前几个方法；与 workbench/canvas.py 的 SHOWN_METHODS 一致
const DRAG_PX = 4;           // 移动超过这么多才算拖动，否则是单击
const COLORS = ['c1', 'c2', 'c3', 'c4'];   // 分组框颜色；与 workbench/canvas.py 的 COLORS 一致
const MINI_FILL = { c1: 'rgba(62,122,168,.3)', c2: 'rgba(160,110,56,.3)', c3: 'rgba(110,140,90,.3)', c4: 'rgba(140,90,150,.3)' };
const MINI_W = 232, MINI_H = 130;

const fam = m => !m ? 'none' : /roma/i.test(m) ? 'roma' : /loftr/i.test(m) ? 'loftr' : 'other';
const pct = x => x == null ? '—' : (x * 100).toFixed(1);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const sel = (attr, v) => `[${attr}="${CSS.escape(v)}"]`;
const curve = (a, b) => { const dx = Math.max(60, Math.abs(b[0] - a[0]) * .45); return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`; };
const ICON_PLUS = '<svg class="i" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>';
const ICON_FIT = '<svg class="i" viewBox="0 0 24 24"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>';
const KEYS = [
  ['右键', '上下文菜单：空白处新建，实验 / 方法行 / 分组 / 便签上各有操作'],
  ['双击空白', '在此处新建实验'],
  ['<kbd>E</kbd>', '在鼠标处新建实验'],
  ['<kbd>N</kbd>', '在鼠标处新建便签'],
  ['<kbd>G</kbd>', '在鼠标处新建分组框'],
  ['<kbd>.</kbd>', '适应窗口'],
  ['<kbd>+</kbd> <kbd>−</kbd>', '放大 / 缩小（滚轮也可以）'],
  ['<kbd>Ctrl</kbd> <kbd>A</kbd>', '全选实验和便签'],
  ['<kbd>Ctrl</kbd> <kbd>G</kbd>', '把选中的对象打成一组'],
  ['<kbd>Del</kbd>', '删除选中的对象（已点亮实验跳过）；没有选中时删除鼠标下的对象'],
  ['<kbd>Enter</kbd>', '打开选中的实验的节点页'],
  ['<kbd>Esc</kbd>', '关闭菜单 / 取消拖线 / 取消选择'],
  ['拖空白', '平移画布'],
  ['从端口拖线', '拖到实验上：改它的父实验；拖到空白：派生新实验'],
  ['分组右下角', '拖动调整大小；拖标题栏移动整组'],
  ['双击标题 / 便签', '重命名分组、编辑便签'],
  ['单击', '选中实验或便签；单击空白取消选择'],
  ['Ctrl / Shift 单击', '加选 / 减选'],
  ['Ctrl 拖空白', '框选'],
  ['拖选中的对象', '一起移动'],
  ['双击实验', '进入节点页'],
];

// 选择集里的键：实验 'e:<id>'，便签 's:<id>'（两者的 id 可能重名）
const ek = id => 'e:' + id, sk = id => 's:' + id;

export function mountCanvas(root, { onOpen }) {
  root.className = 'cv';
  root.innerHTML = `
    <div class="viewport" tabindex="0"><div class="world"><svg class="edges"></svg><div class="layer"></div></div></div>
    <div class="title"><b>实验画布</b><span class="count"></span></div>
    <div class="act">
      <button class="pri" data-cmd="exp">${ICON_PLUS}新建实验 <kbd>E</kbd></button>
      <button data-cmd="sticky">便签 <kbd>N</kbd></button><button data-cmd="group">分组框 <kbd>G</kbd></button>
      <button data-cmd="zout" aria-label="缩小">−</button><span class="pc"></span><button data-cmd="zin" aria-label="放大">+</button>
      <button data-cmd="fit" title="适应窗口（.）" aria-label="适应窗口">${ICON_FIT}</button>
    </div>
    <div class="mini"><svg width="${MINI_W}" height="${MINI_H}" aria-label="小地图"></svg></div>
    <button class="keysbtn">快捷键 <kbd>?</kbd></button>
    <div class="keys"><h3>快捷键与鼠标</h3><dl>${KEYS.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl></div>
    <div class="menu" hidden></div>
    <div class="toast" role="status"></div>`;
  const $ = s => root.querySelector(s);
  const vp = $('.viewport'), world = $('.world'), layer = $('.layer'), edges = $('.edges'), menu = $('.menu'),
        mini = $('.mini svg'), keys = $('.keys');

  // ---------------- 状态 ----------------
  let EXPS = [], byId = new Map(), pos = new Map(), groups = [], stickies = [], lastText = '', active = false;
  let rev = 0;                 // 本地布局改动的版本号：重拉期间有过改动，就丢掉这次重拉的结果
  const S = { tx: 0, ty: 0, k: 1, sel: new Set(), mouse: [innerWidth / 2, innerHeight / 2], link: null, fresh: null,
              expanded: new Set(), editing: false };
  let drag = null;
  const saved = JSON.parse(localStorage.getItem(VIEW_KEY) || 'null');
  if (saved) Object.assign(S, { tx: saved.tx, ty: saved.ty, k: saved.k });

  const kids = id => EXPS.filter(e => e.parent === id);
  const initOf = (parent, m) => m && m !== '_' ? `${parent}/${m}` : null;   // 端口 '_' 是通用端口，不带 init
  const initMethod = e => e.parent && e.init && e.init.startsWith(e.parent + '/') ? e.init.slice(e.parent.length + 1) : null;
  const groupOf = id => groups.find(g => g.id === id), stickyOf = id => stickies.find(s => s.id === id);
  const keyExists = k => k.startsWith('e:') ? byId.has(k.slice(2)) : !!stickyOf(k.slice(2));
  const selExps = () => [...S.sel].filter(k => k.startsWith('e:')).map(k => k.slice(2));

  async function refresh(force = false) {
    const r0 = rev;
    let d;
    try { d = await api.data(); } catch (err) { toast(`读取数据失败：${err.message}`, true); return; }
    if (rev !== r0 && !force) return;
    const text = JSON.stringify(d);
    if (text === lastText && !force) return;
    lastText = text;
    EXPS = d.experiments;
    byId = new Map(EXPS.map(e => [e.id, e]));
    pos = new Map(EXPS.map(e => [e.id, [e.x, e.y]]));
    groups = d.canvas.groups;
    stickies = d.canvas.stickies;
    S.sel = new Set([...S.sel].filter(keyExists));
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
    return `<div class="node${e.lit ? '' : ' off'}${S.sel.has(ek(e.id)) ? ' sel' : ''}${S.fresh === e.id ? ' new' : ''}" data-id="${esc(e.id)}">
      <div class="hdr"><span class="nid">${esc(e.id)}</span><span class="ttl" title="${esc(e.title)}">${esc(e.title)}</span>${e.baseline ? '<span class="chip">基线</span>' : ''}<span class="st" title="${e.lit ? '已点亮' : '未点亮'}"></span></div>
      <div class="io"><span class="port in"></span><span>父实验</span><span>${e.lit ? 'AUC@10（Val）' : ''}</span><span>子实验</span><span class="port out fam-none" data-m="_" title="拖出：拖到实验上改它的父实验，拖到空白处派生新实验"></span></div>
      ${body}</div>`;
  }
  const groupHTML = g => `<div class="grp ${esc(g.color)}" data-g="${esc(g.id)}" style="left:${g.x}px;top:${g.y}px;width:${g.w}px;height:${g.h}px">
      <div class="gt">${esc(g.title)}</div><span class="rz" title="拖动调整大小"></span></div>`;
  const stickyHTML = s => `<div class="sticky${S.sel.has(sk(s.id)) ? ' sel' : ''}" data-s="${esc(s.id)}" style="left:${s.x}px;top:${s.y}px;width:${s.w}px;min-height:${s.h}px">
      <div class="hdr">便签</div><div class="body">${esc(s.text)}</div></div>`;
  function render() {
    layer.innerHTML = groups.map(groupHTML).join('') + stickies.map(stickyHTML).join('') + EXPS.map(nodeHTML).join('');
    for (const e of EXPS) place(e.id);
    S.fresh = null;
    $('.count').textContent = `${EXPS.length} 个实验，${EXPS.filter(e => e.lit).length} 个已点亮`;
    applyView(); drawEdges();
  }
  const nodeEl = id => layer.querySelector(`.node${sel('data-id', id)}`);
  const groupEl = id => layer.querySelector(`.grp${sel('data-g', id)}`);
  const stickyEl = id => layer.querySelector(`.sticky${sel('data-s', id)}`);
  function place(id) { const el = nodeEl(id), [x, y] = pos.get(id); el.style.left = x + 'px'; el.style.top = y + 'px'; }
  function placeBox(el, o) { el.style.left = o.x + 'px'; el.style.top = o.y + 'px'; }
  function paintSel() {
    layer.querySelectorAll('.node').forEach(el => el.classList.toggle('sel', S.sel.has(ek(el.dataset.id))));
    layer.querySelectorAll('.sticky').forEach(el => el.classList.toggle('sel', S.sel.has(sk(el.dataset.s))));
  }
  function setSel(keys) { S.sel = new Set(keys); paintSel(); }
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
    drawMini();
  }
  function applyView() {
    world.style.transform = `translate(${S.tx}px,${S.ty}px) scale(${S.k})`;
    const g = 20 * S.k;
    vp.style.backgroundSize = `${g}px ${g}px, ${g}px ${g}px, ${g * 5}px ${g * 5}px, ${g * 5}px ${g * 5}px`;
    vp.style.backgroundPosition = `${S.tx}px ${S.ty}px`;
    $('.pc').textContent = Math.round(S.k * 100) + '%';
    localStorage.setItem(VIEW_KEY, JSON.stringify({ tx: S.tx, ty: S.ty, k: S.k }));
    drawMini();
  }
  // 世界坐标里的外接框：卡片按 left / top 摆放，offset* 不受 world 的缩放影响
  const rectOf = el => ({ el, x: el.offsetLeft, y: el.offsetTop, w: el.offsetWidth, h: el.offsetHeight });
  const rects = () => [...layer.children].map(rectOf);
  function hull(rs) {
    if (!rs.length) return null;
    const x0 = Math.min(...rs.map(r => r.x)), y0 = Math.min(...rs.map(r => r.y));
    return { x0, y0, x1: Math.max(...rs.map(r => r.x + r.w)), y1: Math.max(...rs.map(r => r.y + r.h)) };
  }
  const inside = (r, g) => r.x >= g.x && r.y >= g.y && r.x + r.w <= g.x + g.w && r.y + r.h <= g.y + g.h;
  function members(g) {   // 几何包含：整个落在框里的实验和便签
    const rs = rects().filter(r => !r.el.classList.contains('grp') && inside(r, g));
    return { exps: rs.filter(r => r.el.dataset.id).map(r => r.el.dataset.id), stickies: rs.filter(r => r.el.dataset.s).map(r => r.el.dataset.s) };
  }
  function fit() {
    const b = hull(rects()); if (!b) return;
    const r = vp.getBoundingClientRect(), pad = 80;
    S.k = Math.min(1.1, Math.max(.3, Math.min((r.width - 2 * pad) / (b.x1 - b.x0), (r.height - 2 * pad) / (b.y1 - b.y0))));
    S.tx = (r.width - (b.x1 - b.x0) * S.k) / 2 - b.x0 * S.k; S.ty = (r.height - (b.y1 - b.y0) * S.k) / 2 - b.y0 * S.k;
    applyView();
  }
  function zoomAt(k, mx, my) { k = Math.min(2, Math.max(.3, k)); S.tx = mx - (mx - S.tx) * k / S.k; S.ty = my - (my - S.ty) * k / S.k; S.k = k; applyView(); }
  function zoomBy(f) { const r = vp.getBoundingClientRect(); zoomAt(S.k * f, r.width / 2, r.height / 2); }
  function lookAt(wx, wy) { const r = vp.getBoundingClientRect(); S.tx = r.width / 2 - wx * S.k; S.ty = r.height / 2 - wy * S.k; applyView(); }
  function centerOn(id) {
    const el = nodeEl(id); if (!el) return;
    const [x, y] = pos.get(id);
    lookAt(x + el.offsetWidth / 2, y + el.offsetHeight / 2);
  }
  const toWorld = (cx, cy) => { const r = vp.getBoundingClientRect(); return [(cx - r.left - S.tx) / S.k, (cy - r.top - S.ty) / S.k]; };

  // 小地图：全部对象的外接框留 50 px 边，白框是当前视口；点击把视口中心移到那里
  let miniMap = null;
  function drawMini() {
    const rs = rects(), b = hull(rs); if (!b) { mini.innerHTML = ''; miniMap = null; return; }
    const s = Math.min(MINI_W / (b.x1 - b.x0 + 100), MINI_H / (b.y1 - b.y0 + 100));
    const X = x => (x - b.x0 + 50) * s, Y = y => (y - b.y0 + 50) * s, r = vp.getBoundingClientRect();
    miniMap = { s, x0: b.x0 - 50, y0: b.y0 - 50 };
    mini.innerHTML = rs.map(o => {
      const e = o.el.dataset.id && byId.get(o.el.dataset.id);
      const fill = o.el.dataset.g ? MINI_FILL[groupOf(o.el.dataset.g)?.color] || MINI_FILL.c1
        : o.el.dataset.s ? '#574F37' : e && e.lit ? '#4A6E96' : '#3A3C42';
      return `<rect x="${X(o.x)}" y="${Y(o.y)}" width="${o.w * s}" height="${o.h * s}" rx="1.5" fill="${fill}"/>`;
    }).join('') + `<rect class="vp" x="${X(-S.tx / S.k)}" y="${Y(-S.ty / S.k)}" width="${r.width / S.k * s}" height="${r.height / S.k * s}" fill="rgba(255,255,255,.05)" stroke="#E4E5E8"/>`;
  }
  mini.addEventListener('click', ev => {
    if (!miniMap) return;
    const p = mini.getBoundingClientRect();
    lookAt((ev.clientX - p.left) / miniMap.s + miniMap.x0, (ev.clientY - p.top) / miniMap.s + miniMap.y0);
  });

  // ---------------- 操作：语义改动经 API 写回再重拉；布局改动本地先改，再整体存 canvas.json ----------------
  async function attempt(p, ok) {
    try { const r = await p; if (ok) toast(typeof ok === 'function' ? ok(r) : ok); return r; }
    catch (err) { toast(err.message, true); await refresh(true); return null; }
  }
  async function saveLayout(expIds = []) {   // 便签、分组框整体替换；实验只带动过的（其余没存过坐标的由服务端一并钉住）
    rev++;
    const experiments = Object.fromEntries(expIds.filter(id => pos.has(id)).map(id => {
      const [x, y] = pos.get(id); return [id, { x: Math.round(x), y: Math.round(y) }];
    }));
    return attempt(api.saveCanvas({ experiments, groups, stickies }));
  }
  async function created(r, parent, init) {
    if (!r) return;
    S.fresh = r.id; S.sel = new Set([ek(r.id)]); await refresh(true);
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
      S.sel.delete(ek(id)); await refresh(true);
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
      if (await attempt(api.rename(id, t), `${id} 已改为 ${t}`)) {
        if (S.sel.delete(ek(id))) S.sel.add(ek(t));
        await refresh(true);
      }
    });
  }
  function open(id) { S.sel = new Set([ek(id)]); onOpen(id); }

  const uid = (prefix, taken) => { let id; do id = prefix + Date.now().toString(36) + Math.floor(Math.random() * 36 ** 2).toString(36); while (taken(id)); return id; };
  function newSticky(x, y) {
    const s = { id: uid('s', stickyOf), x: Math.round(x), y: Math.round(y), w: 230, h: 80, text: '' };
    stickies.push(s); S.sel = new Set([sk(s.id)]); render(); saveLayout();
    editSticky(s.id);
  }
  function editSticky(id) {
    const el = stickyEl(id)?.querySelector('.body'); if (!el) return;
    editText(el, t => { const s = stickyOf(id); if (s && t !== s.text) { s.text = t; saveLayout(); } render(); }, { multiline: true });
  }
  function delSticky(id) { stickies = stickies.filter(s => s.id !== id); S.sel.delete(sk(id)); render(); saveLayout(); }
  function newGroup(x, y, box = { w: 460, h: 300 }) {
    const g = { id: uid('g', groupOf), x: Math.round(x), y: Math.round(y), w: Math.round(box.w), h: Math.round(box.h),
                title: '新分组', color: COLORS[groups.length % COLORS.length] };
    groups.push(g); render(); saveLayout();
    return g;
  }
  function renameGroup(id) {
    const el = groupEl(id)?.querySelector('.gt'); if (!el) return;
    editText(el, t => { const g = groupOf(id); if (g && t && t !== g.title) { g.title = t; saveLayout(); } render(); });
  }
  function recolor(g) { g.color = COLORS[(COLORS.indexOf(g.color) + 1) % COLORS.length]; render(); saveLayout(); }
  function delGroup(id) { groups = groups.filter(g => g.id !== id); render(); saveLayout(); toast('已删除分组框，框内的实验和便签不动'); }
  const PAD = { x: 30, top: 60, bottom: 30 };   // 打组 / 贴合时内容到框边的留白；顶上让出标题带
  const around = b => ({ x: b.x0 - PAD.x, y: b.y0 - PAD.top, w: b.x1 - b.x0 + 2 * PAD.x, h: b.y1 - b.y0 + PAD.top + PAD.bottom });
  function groupSel() {   // 按选中对象的外接框建分组，并进入重命名
    const rs = rects().filter(r => S.sel.has(r.el.dataset.id ? ek(r.el.dataset.id) : r.el.dataset.s ? sk(r.el.dataset.s) : ''));
    if (!rs.length) { toast('先选中实验或便签'); return; }
    const b = around(hull(rs)), g = newGroup(b.x, b.y, b);
    renameGroup(g.id);
  }
  function fitGroup(g) {   // 按当前几何包含的内容收紧框
    const m = members(g), rs = [...m.exps.map(nodeEl), ...m.stickies.map(stickyEl)].map(rectOf);
    if (!rs.length) { toast('框里没有内容'); return; }
    Object.assign(g, Object.fromEntries(Object.entries(around(hull(rs))).map(([k, v]) => [k, Math.round(v)])));
    render(); saveLayout();
  }
  async function deleteSel() {
    let n = 0, skipped = 0, failed = false;
    const ids = selExps(), notes = [...S.sel].filter(k => k.startsWith('s:')).map(k => k.slice(2));
    if (notes.length) { stickies = stickies.filter(s => !notes.includes(s.id)); n += notes.length; }
    S.sel = new Set(); render();
    if (notes.length) await saveLayout();
    for (const id of ids) {
      if (byId.get(id)?.lit) { skipped++; continue; }
      try { await api.remove(id); n++; } catch (err) { toast(err.message, true); failed = true; break; }
    }
    if (ids.length) await refresh(true);
    if (!failed) toast(`已删除 ${n} 个对象${skipped ? `；${skipped} 个已点亮的实验不能删除，已跳过` : ''}`);
  }
  function editText(el, done, { multiline = false } = {}) {
    const before = el.textContent;
    S.editing = true; el.contentEditable = 'true'; el.focus({ preventScroll: true }); document.getSelection().selectAllChildren(el);
    let cancelled = false;
    const fin = () => {
      el.removeEventListener('blur', fin); el.contentEditable = 'false'; S.editing = false;
      const raw = el.innerText.replace(/ /g, ' ');
      const t = multiline ? raw.replace(/[ \t]+$/gm, '').trim() : raw.replace(/\s+/g, ' ').trim();
      if (cancelled) { el.textContent = before; render(); } else done(t);
    };
    el.addEventListener('blur', fin);
    el.addEventListener('keydown', ev => {
      if (ev.key === 'Enter' && !(multiline && ev.shiftKey)) { ev.preventDefault(); el.blur(); }   // 便签里 Shift+Enter 换行
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
  const groupItem = () => ({ t: S.sel.size ? `把选中的 ${S.sel.size} 个打成一组` : '把选中的对象打成一组', k: 'Ctrl G',
                             off: !S.sel.size, why: '先选中实验或便签', f: groupSel });
  vp.addEventListener('contextmenu', ev => {
    ev.preventDefault();
    const [wx, wy] = toWorld(ev.clientX, ev.clientY);
    const n = ev.target.closest('.node'), row = ev.target.closest('.row'), st = ev.target.closest('.sticky'), gr = ev.target.closest('.grp');
    if (n) {
      const e = byId.get(n.dataset.id), m = row && row.dataset.m;
      if (!S.sel.has(ek(e.id))) setSel([ek(e.id)]);
      const items = [{ t: '打开节点页', k: '双击', f: () => open(e.id) }, '-'];
      if (S.sel.size > 1) items.push(groupItem());
      if (m) items.push({ t: `从 ${m} 派生新实验`, f: () => derive(e.id, m) });
      items.push({ t: `从 ${e.id} 派生新实验`, f: () => derive(e.id, null) });
      if (e.parent) items.push({ t: '断开父实验', f: () => disconnect(e.id) });
      items.push('-', { t: '改标题', f: () => editTitle(e.id) },
        { t: '改编号', off: e.lit, why: '已点亮的实验不能改编号', f: () => editId(e.id) },
        '-', { t: '删除实验', k: 'Del', off: e.lit, why: '已点亮的实验不能删除', f: () => del(e.id) });
      showMenu(ev.clientX, ev.clientY, `${esc(e.id)}　${esc(e.title)}`, items);
    } else if (st) {
      const id = st.dataset.s;
      if (!S.sel.has(sk(id))) setSel([sk(id)]);
      const items = [{ t: '编辑', k: '双击', f: () => editSticky(id) }];
      if (S.sel.size > 1) items.push(groupItem());
      items.push('-', { t: '删除便签', k: 'Del', f: () => delSticky(id) });
      showMenu(ev.clientX, ev.clientY, '便签', items);
    } else if (gr) {
      const g = groupOf(gr.dataset.g);
      showMenu(ev.clientX, ev.clientY, `分组：${esc(g.title)}`, [
        { t: '在此新建实验', k: 'E', f: () => newExpAt(wx - 150, wy - 40) },
        { t: '重命名', k: '双击标题', f: () => renameGroup(g.id) },
        { t: '换颜色', f: () => recolor(g) },
        { t: '贴合框内内容', f: () => fitGroup(g) },
        '-', { t: '删除分组框（内容保留）', k: 'Del', f: () => delGroup(g.id) }]);
    } else {
      showMenu(ev.clientX, ev.clientY, null, [
        { t: '新建实验', k: 'E', f: () => newExpAt(wx - 150, wy - 40) },
        { t: '新建便签', k: 'N', f: () => newSticky(wx, wy) },
        { t: '新建分组框', k: 'G', f: () => newGroup(wx, wy) },
        groupItem(), '-',
        { t: '适应窗口', k: '.', f: fit }]);
    }
  });

  // ---------------- 鼠标：平移、选择、拖动、拖线、调整分组大小、缩放 ----------------
  vp.addEventListener('pointerdown', ev => {
    if (ev.button !== 0) return;
    hideMenu();
    if (ev.target.closest('[contenteditable=true]')) return;
    vp.focus({ preventScroll: true });
    const [wx, wy] = toWorld(ev.clientX, ev.clientY);
    const port = ev.target.closest('.port.out'), n = ev.target.closest('.node'), st = ev.target.closest('.sticky'),
          gr = ev.target.closest('.grp'), multi = ev.ctrlKey || ev.shiftKey || ev.metaKey;
    if (port) {
      const from = n.dataset.id, m = port.dataset.m;
      S.link = { from, m, a: outPort(from, m), b: [wx, wy] }; drag = { kind: 'link' }; return;
    }
    if (n || st) {
      const key = n ? ek(n.dataset.id) : sk(st.dataset.s);
      if (multi) { S.sel.has(key) ? S.sel.delete(key) : S.sel.add(key); paintSel(); return; }   // 加选 / 减选
      if (!S.sel.has(key)) setSel([key]);
      const items = [...S.sel].map(k => k.startsWith('e:') ? { id: k.slice(2), p0: [...pos.get(k.slice(2))] }
        : (s => s && { s, p0: [s.x, s.y] })(stickyOf(k.slice(2)))).filter(Boolean);
      drag = { kind: 'move', key, moved: 0, sx: ev.clientX, sy: ev.clientY, w0: [wx, wy], items, target: ev.target };
      return;
    }
    if (gr && ev.target.closest('.rz')) {
      const g = groupOf(gr.dataset.g);
      drag = { kind: 'resize', g, el: gr, sx: wx, sy: wy, w: g.w, h: g.h, moved: 0 }; return;
    }
    if (gr && ev.target.closest('.gt')) {   // 拖标题栏：几何包含的实验和便签一起移动
      const g = groupOf(gr.dataset.g), m = members(g);
      drag = { kind: 'group', g, el: gr, w0: [wx, wy], g0: [g.x, g.y], moved: 0, sx: ev.clientX, sy: ev.clientY,
               exps: m.exps.map(id => ({ id, p0: [...pos.get(id)] })),
               stickies: m.stickies.map(id => (s => ({ s, p0: [s.x, s.y] }))(stickyOf(id))) };
      return;
    }
    if (ev.ctrlKey || ev.metaKey) {   // 框选：在已选的基础上加
      const box = document.createElement('div'); box.className = 'boxsel'; root.appendChild(box);
      drag = { kind: 'box', sx: ev.clientX, sy: ev.clientY, box, base: new Set(S.sel) };
      return;
    }
    drag = { kind: 'pan', sx: ev.clientX - S.tx, sy: ev.clientY - S.ty, px: ev.clientX, py: ev.clientY, moved: 0 };
    vp.classList.add('panning');
  });
  const movedBy = (d, ev) => (d.moved = Math.max(d.moved, Math.hypot(ev.clientX - d.sx, ev.clientY - d.sy)));
  window.addEventListener('pointermove', ev => {
    S.mouse = [ev.clientX, ev.clientY];
    if (!drag) return;
    const [wx, wy] = toWorld(ev.clientX, ev.clientY), d = drag;
    if (d.kind === 'pan') {
      d.moved = Math.max(d.moved, Math.hypot(ev.clientX - d.px, ev.clientY - d.py));
      S.tx = ev.clientX - d.sx; S.ty = ev.clientY - d.sy; applyView();
    } else if (d.kind === 'move') {
      if (movedBy(d, ev) < DRAG_PX) return;
      const dx = wx - d.w0[0], dy = wy - d.w0[1];
      for (const it of d.items) {
        const x = Math.round(it.p0[0] + dx), y = Math.round(it.p0[1] + dy);
        if (it.id) { pos.set(it.id, [x, y]); place(it.id); } else { Object.assign(it.s, { x, y }); placeBox(stickyEl(it.s.id), it.s); }
      }
      drawEdges();
    } else if (d.kind === 'group') {
      if (movedBy(d, ev) < DRAG_PX) return;
      const dx = Math.round(wx - d.w0[0]), dy = Math.round(wy - d.w0[1]);
      Object.assign(d.g, { x: d.g0[0] + dx, y: d.g0[1] + dy }); placeBox(d.el, d.g);
      for (const it of d.exps) { pos.set(it.id, [it.p0[0] + dx, it.p0[1] + dy]); place(it.id); }
      for (const it of d.stickies) { Object.assign(it.s, { x: it.p0[0] + dx, y: it.p0[1] + dy }); placeBox(stickyEl(it.s.id), it.s); }
      drawEdges();
    } else if (d.kind === 'resize') {
      d.moved = 1;
      d.g.w = Math.max(220, Math.round(d.w + wx - d.sx)); d.g.h = Math.max(120, Math.round(d.h + wy - d.sy));
      d.el.style.width = d.g.w + 'px'; d.el.style.height = d.g.h + 'px'; drawMini();
    } else if (d.kind === 'box') {
      const x0 = Math.min(d.sx, ev.clientX), y0 = Math.min(d.sy, ev.clientY), x1 = Math.max(d.sx, ev.clientX), y1 = Math.max(d.sy, ev.clientY);
      const rr = root.getBoundingClientRect();
      Object.assign(d.box.style, { left: x0 - rr.left + 'px', top: y0 - rr.top + 'px', width: x1 - x0 + 'px', height: y1 - y0 + 'px' });
      const hit = [...layer.querySelectorAll('.node, .sticky')].filter(el => {
        const r = el.getBoundingClientRect(); return r.left < x1 && r.right > x0 && r.top < y1 && r.bottom > y0;
      }).map(el => el.dataset.id ? ek(el.dataset.id) : sk(el.dataset.s));
      setSel([...d.base, ...hit]);
    } else if (d.kind === 'link') {
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
    } else if (d.kind === 'move') {
      if (d.moved >= DRAG_PX) saveLayout(d.items.filter(it => it.id).map(it => it.id));
      else {
        if (S.sel.size > 1) setSel([d.key]);   // 在多选里单击（没拖动）：只留这一个
        const id = d.key.startsWith('e:') && d.key.slice(2);
        if (id && d.target.closest('.more')) { S.expanded.has(id) ? S.expanded.delete(id) : S.expanded.add(id); render(); }
      }
    } else if (d.kind === 'group') {
      if (d.moved >= DRAG_PX) saveLayout(d.exps.map(it => it.id));
    } else if (d.kind === 'resize') {
      if (d.moved) saveLayout();
    } else if (d.kind === 'box') d.box.remove();
    else if (d.kind === 'pan' && d.moved < DRAG_PX && S.sel.size) setSel([]);
  });
  vp.addEventListener('dblclick', ev => {
    if (ev.target.closest('[contenteditable=true], .port, .rz')) return;
    const gt = ev.target.closest('.gt'); if (gt) { renameGroup(gt.parentElement.dataset.g); return; }
    const st = ev.target.closest('.sticky'); if (st) { editSticky(st.dataset.s); return; }
    const n = ev.target.closest('.node'); if (n) { open(n.dataset.id); return; }
    const [wx, wy] = toWorld(ev.clientX, ev.clientY); newExpAt(wx - 150, wy - 40);
  });
  vp.addEventListener('scroll', () => { vp.scrollLeft = vp.scrollTop = 0; });   // 平移只走 transform；聚焦等引起的滚动一律撤销
  vp.addEventListener('wheel', ev => { ev.preventDefault(); const r = vp.getBoundingClientRect(); zoomAt(S.k * Math.exp(-ev.deltaY * .0015), ev.clientX - r.left, ev.clientY - r.top); }, { passive: false });

  // ---------------- 键盘 ----------------
  const toggleKeys = on => keys.classList.toggle('on', on);
  window.addEventListener('keydown', ev => {
    if (!active || ev.altKey) return;
    if (ev.target.closest && ev.target.closest('[contenteditable=true], input, textarea, select')) return;
    if (ev.ctrlKey || ev.metaKey) {
      const k = ev.key.toLowerCase();
      if (k === 'a') setSel([...EXPS.map(e => ek(e.id)), ...stickies.map(s => sk(s.id))]);
      else if (k === 'g') groupSel();
      else return;
      ev.preventDefault(); return;
    }
    const [wx, wy] = toWorld(...S.mouse);
    switch (ev.key) {
      case 'e': case 'E': newExpAt(wx - 150, wy - 40); break;
      case 'n': case 'N': newSticky(wx, wy); break;
      case 'g': case 'G': newGroup(wx, wy); break;
      case '.': fit(); break;
      case '+': case '=': zoomBy(1.2); break;
      case '-': case '_': zoomBy(1 / 1.2); break;
      case '?': toggleKeys(); break;
      case 'Escape':   // 逐级退出：先关菜单 / 面板，再取消拖线，最后取消选择
        if (!menu.hidden || keys.classList.contains('on')) { hideMenu(); toggleKeys(false); }
        else if (S.link) { S.link = null; drag = null; layer.querySelectorAll('.drop').forEach(x => x.classList.remove('drop')); drawEdges(); }
        else if (S.sel.size) setSel([]);
        break;
      case 'Delete': {
        if (S.sel.size) { deleteSel(); break; }
        const under = document.elementFromPoint(...S.mouse);
        const n = under?.closest('.node'), st = under?.closest('.sticky'), gr = under?.closest('.grp');
        if (n) del(n.dataset.id); else if (st) delSticky(st.dataset.s); else if (gr) delGroup(gr.dataset.g);
        break;
      }
      case 'Enter': { const ids = selExps(); if (S.sel.size === 1 && ids.length === 1) open(ids[0]); break; }
      default: return;
    }
    ev.preventDefault();
  });
  document.addEventListener('pointerdown', ev => {
    if (!ev.target.closest('.menu')) hideMenu();
    if (!ev.target.closest('.keys, .keysbtn')) toggleKeys(false);
  });
  $('.keysbtn').addEventListener('click', () => toggleKeys());
  $('.act').addEventListener('click', ev => {
    const b = ev.target.closest('[data-cmd]'); if (!b) return;
    const r = vp.getBoundingClientRect(), [cx, cy] = toWorld(r.left + r.width / 2, r.top + r.height / 2);
    ({ exp: () => newExpAt(cx - 150, cy - 60), sticky: () => newSticky(cx - 115, cy - 40), group: () => newGroup(cx - 230, cy - 150),
       fit, zin: () => zoomBy(1.2), zout: () => zoomBy(1 / 1.2) })[b.dataset.cmd]();
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
      if (focus && byId.has(focus)) { S.sel = new Set([ek(focus)]); render(); centerOn(focus); }
      else if (first && !saved) fit();
      first = false;
      if (document.fonts) document.fonts.ready.then(() => { if (active) drawEdges(); });   // 字体换上后端口位置会变
      vp.focus({ preventScroll: true });
    },
    hide() { root.hidden = true; active = false; hideMenu(); toggleKeys(false); },
  };
}
