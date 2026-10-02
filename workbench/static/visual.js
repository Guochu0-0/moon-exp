// 节点页「可视化结果」：逐对查看一个方法相对参考方法的表现。视觉参照 prototype/node-page 分支第五版。
// 散点总览（逐级放大后点选）→ 筛选 → 排序 → 缩略图网格（每次 12 张）→ 详情（方法与参考方法并排，← → 在当前列表里移动）。
// 影像视图都是 SVG：底图取 /img，卷帘里的 SAR 按估计仿射的逆变换摆到光学坐标，残差箭头和点对画成矢量。
// 详情里的各面板共用一个 viewBox，所以缩放和平移同步。
// 方法写了中间结果时，详情多出对应的视图：标量图画成热力图叠加，点集画成点加取值，位移场画成 warp，图片原样显示。
import { api } from './api.js';

const GOOD = 5;            // 改善 / 退化以 5 px 为界：误差不超过 5 px 算配准
const CAP = 50;            // 按误差差值排序时，失败与超过 50 px 的误差都按 50 px 计
const PAGE = 12;           // 网格每次显示的张数
const EDGE = 30;           // 散点：失败或超过 30 px 的点画进 30.6–33.8 的边缘带
const FULL = { x0: 0, x1: 34, y0: 0, y1: 34 };
const SC = { W: 340, H: 340, L: 46, B: 40, T: 8 };
SC.pw = SC.W - SC.L - 10; SC.ph = SC.H - SC.B - SC.T;
const CATS = [['all', '全部'], ['better', '改善'], ['worse', '退化'], ['neither', '均未配准'], ['both', '均配准']];
const SORTS_REF = [['gain', '误差下降量大在前'], ['loss', '误差上升量大在前'], ['me', '方法误差大在前'], ['no', '编号']];
const SORTS_NOREF = [['me', '方法误差大在前'], ['no', '编号']];
const VIEWS = [['swipe', '卷帘'], ['resid', '残差'], ['raw', '原图']];
const POINT_VIEWS = [['lines', '点对连线'], ['dots', '残差点图']];
const INLIER = '#1B9E77', OUTLIER = '#D95F02', NEUTRAL = '#8A8F99';
const RESID_STOPS = [[0, [26, 152, 80]], [3, [254, 224, 139]], [10, [215, 48, 39]]];   // 残差点图：0 → 3 → ≥10 px
const MAX_ZOOM = 40;
// 筛选切换时自动换成合适的排序；其余筛选回到默认的误差下降量
const FILTER_SORT = { worse: 'loss', neither: 'me', both: 'no' };
const isPointView = v => POINT_VIEWS.some(([x]) => x === v);
const INTER = 'inter:';    // 中间结果视图的 id 前缀，后接名称
const interName = v => v.startsWith(INTER) ? v.slice(INTER.length) : null;
// 与服务端热力图同一色图（viridis 的 9 个采样点）
const VIRIDIS = [[68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141], [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37]];
const FRAME = { opt: '光学', sar: 'SAR' };
const KIND = { scalar: '标量图', points: '点集', flow: '位移场', image: '图片' };
const POINT_LABELS = 60;   // 点集不超过这么多点时，在点旁标出取值

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fin = v => v == null ? Infinity : v;
const errTxt = e => e == null ? '失败' : e.toFixed(1);
const order = (x, y) => x < y ? -1 : x > y ? 1 : 0;
const pad = n => String(n).padStart(4, '0');
const img = (split, pair, kind) => `/img?${new URLSearchParams({ split, pair, kind })}`;
const f2 = v => +v.toFixed(2);

// 仿射 A（光学 → SAR）的逆，写成 SVG matrix(a b c d e f)：把 SAR 坐标摆回光学坐标
function inverse(A) {
  const [[a, b, c], [d, e, f]] = A, det = a * e - b * d;
  if (!det) return null;
  const p = e / det, q = -b / det, r = -d / det, s = a / det;
  return { m: [p, r, q, s, -(p * c + q * f), -(r * c + s * f)], apply: (x, y) => [p * (x - c) + q * (y - f), r * (x - c) + s * (y - f)] };
}

function viridis(t) {
  if (t == null || !isFinite(t)) return NEUTRAL;
  const x = Math.min(Math.max(t, 0), 1) * (VIRIDIS.length - 1), i = Math.min(Math.floor(x), VIRIDIS.length - 2), f = x - i;
  return `rgb(${VIRIDIS[i].map((c, j) => Math.round(c + (VIRIDIS[i + 1][j] - c) * f))})`;
}
const RAMP = `linear-gradient(to right, ${VIRIDIS.map((c, i) => `rgb(${c}) ${i / (VIRIDIS.length - 1) * 100}%`).join(', ')})`;
const fmt = v => v == null ? '—' : Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(2) : +v.toPrecision(4);

function residColor(v) {
  if (v == null) return NEUTRAL;
  const st = RESID_STOPS;
  if (v >= st[st.length - 1][0]) return `rgb(${st[st.length - 1][1]})`;
  for (let i = 1; i < st.length; i++) if (v <= st[i][0]) {
    const [x0, c0] = st[i - 1], [x1, c1] = st[i], t = (v - x0) / (x1 - x0);
    return `rgb(${c0.map((c, j) => Math.round(c + (c1[j] - c) * t))})`;
  }
}

// 残差箭头：从光学检查点指向其 SAR 真值经估计仿射变回光学坐标后的位置，原尺寸。k = 每个屏幕 px 对应的影像 px。
function arrows(cp, A, k) {
  const inv = A && inverse(A);
  if (!cp || !inv) return '';
  let g = '';
  cp.opt.forEach(([ox, oy], i) => {
    const [ex, ey] = inv.apply(...cp.sar[i]), L = Math.hypot(ex - ox, ey - oy);
    g += `<line class="arr" x1="${f2(ox)}" y1="${f2(oy)}" x2="${f2(ex)}" y2="${f2(ey)}"/>`;
    if (L > 4 * k) {
      const ux = (ex - ox) / L, uy = (ey - oy) / L, h = 7 * k, w = 4 * k;
      g += `<polygon class="arr" points="${f2(ex)},${f2(ey)} ${f2(ex - h * ux + w * uy)},${f2(ey - h * uy - w * ux)} ${f2(ex - h * ux - w * uy)},${f2(ey - h * uy + w * ux)}"/>`;
    }
    g += `<circle class="cp" cx="${f2(ox)}" cy="${f2(oy)}" r="${f2(3.5 * k)}"/>`;
  });
  return g;
}

// conf 过滤：保留 conf 不低于第 q 百分位的点；没有 conf 时全保留。thr 为实际阈值（不过滤时为 null）
function confCut(mt, q) {
  const all = [...Array(mt.n).keys()];
  if (!mt.conf || !q) return { ks: all, thr: null };
  const v = mt.conf.filter(x => x != null).sort((a, b) => a - b);
  if (!v.length) return { ks: all, thr: null };
  const thr = v[Math.min(v.length - 1, Math.floor(q / 100 * v.length))];
  return { ks: all.filter(i => mt.conf[i] != null && mt.conf[i] >= thr), thr };
}
// conf 全部相同（如 RoMa 系截到前 2000 个点后 certainty 都是 1）时，按分位数过滤不起作用
const flatConf = mt => { const v = mt.conf.filter(x => x != null); return v.every(x => x === v[0]) ? v[0] : null; };

export function mountVisual(isShown) {
  let host = null, C = null, ctxKey = '', FIG = 0, V = null, seq = 0;
  const cache = new Map();
  const fresh = (prev, ref) => ({ filter: 'all', sort: ref ? 'gain' : 'me', n: PAGE, sel: null, zoom: null, zstack: [],
    view: prev?.view || 'swipe', swipe: prev?.swipe ?? 50, vb: null, vbl: null, conf: 0, find: '', alpha: prev?.alpha ?? 60, iv: null });
  const $ = s => host.querySelector(s);
  const SPLIT = () => C.split === 'val' ? 'Val' : 'Test';

  function pairData(pair, points) {
    const k = `${ctxKey}|${pair}|${points}`;
    if (!cache.has(k)) {
      const p = api.pair(C.method.key, C.ref?.key, C.split, pair, points);
      p.catch(() => cache.delete(k));
      cache.set(k, p);
    }
    return cache.get(k);
  }

  // ---------------- 列表：分类、筛选、排序 ----------------
  function info() {
    const eM = C.method.errors, eR = C.ref?.errors || null, n = C.pairs.length, idx = [...Array(n).keys()];
    const cap = v => Math.min(fin(v), CAP);
    const cat = eR && (i => {
      const a = fin(eM[i]) <= GOOD, b = fin(eR[i]) <= GOOD;
      return a && !b ? 'better' : !a && b ? 'worse' : a ? 'both' : 'neither';
    });
    const count = cat && Object.fromEntries(CATS.map(([c]) => [c, c === 'all' ? n : idx.filter(i => cat(i) === c).length]));
    const key = { gain: i => cap(eM[i]) - cap(eR[i]), loss: i => cap(eR[i]) - cap(eM[i]), me: i => -fin(eM[i]), no: i => C.no[i] }[V.sort];
    const list = idx.filter(i => !cat || V.filter === 'all' || cat(i) === V.filter)
      .sort((a, b) => order(key(a), key(b)) || C.no[a] - C.no[b]);
    return { eM, eR, cat, count, list };
  }

  // ---------------- 渲染 ----------------
  function render(el, cmp, fig) {
    host = el; C = cmp; FIG = fig;
    const k = `${cmp.method.key}|${cmp.ref?.key ?? ''}|${cmp.split}`;
    if (k !== ctxKey) { ctxKey = k; V = fresh(V, cmp.ref); cache.clear(); }
    if (!C.pairs.length) { host.innerHTML = `<p>${SPLIT()} 上没有有标注 pair。</p>`; return FIG; }
    draw();
    return V.detailFig;   // 本节用到的最后一个图号，供后面的章节接着编
  }

  function draw() {
    const I = info();
    if (!C.pairs.includes(V.sel)) V.sel = I.list.length ? C.pairs[I.list[0]] : null;
    const n = { fig: FIG };
    const flag = C.split === 'test' ? '<p class="testflag block">以下为 Test 集上的逐对结果。</p>' : '';
    const cap = `<p class="caption">缩略图为 ${esc(C.method.label)} 的残差图（光学原图上的检查点与原尺寸残差箭头）；数字为方法误差${I.eR ? ` / 参考方法 ${esc(C.ref.label)} 的误差` : ''}（px）。${I.eR ? `改善、退化以 ${GOOD} px 为界。` : ''}点击缩略图在下方查看详情。</p>`;
    const side = `<div>${controls(I)}<div class="grid"></div><div class="more"></div></div>`;
    host.innerHTML = flag + cap + (I.eR
      ? `<div class="vr-top"><figure class="sc-fig"><div class="sc"></div><p class="caption figcap"><b>图 ${++n.fig}</b>每个点为一个有标注 pair，横轴为参考方法误差，纵轴为方法误差；虚线为 y = x，细线为 ${GOOD} px。灰色边缘带内的点误差大于 ${EDGE} px 或估计失败，带内位置只为错开显示，不代表数值。深色点属于当前筛选。</p></figure>${side}</div>`
      : `<div class="vr-top noref">${side}</div>`) + '<div class="detail"></div>';
    V.detailFig = n.fig + 1;
    bindControls();
    drawList(I);
    drawDetail();
  }

  function controls(I) {
    const chips = I.eR ? `<span><span class="lbl">筛选</span><span class="chips" role="group" aria-label="筛选">${CATS.map(([id, t]) =>
      `<button data-f="${id}" aria-pressed="${V.filter === id}">${t} <span class="num">${I.count[id]}</span></button>`).join('')}</span></span>` : '';
    const sorts = I.eR ? SORTS_REF : SORTS_NOREF;
    return `<div class="vr-ctl">${chips}
      <span><label for="vsort">排序</label><select id="vsort">${sorts.map(([v, t]) => `<option value="${v}" ${V.sort === v ? 'selected' : ''}>${t}</option>`).join('')}</select></span>
      <span class="find"><label for="vfind">查找编号</label><input id="vfind" class="num" size="7" placeholder="如 12" value="${esc(V.find)}"><button id="vgo">查找</button><span class="msg" role="status"></span></span></div>`;
  }

  function bindControls() {
    host.querySelectorAll('.chips button').forEach(b => b.addEventListener('click', () => {
      V.filter = b.dataset.f; V.sort = FILTER_SORT[V.filter] || 'gain';
      V.n = PAGE; V.sel = null; draw();
    }));
    $('#vsort').addEventListener('change', ev => { V.sort = ev.target.value; V.n = PAGE; V.sel = null; draw(); });
    const go = () => find($('#vfind').value);
    $('#vgo').addEventListener('click', go);
    $('#vfind').addEventListener('keydown', ev => { if (ev.key === 'Enter') go(); });
  }

  function find(text) {
    V.find = text.trim();
    const t = V.find.replace(/^#/, ''), msg = $('.vr-ctl .msg');
    const i = /^\d+$/.test(t) ? C.no.indexOf(+t) : C.pairs.indexOf(t);
    if (i < 0) { msg.textContent = `${SPLIT()} 中没有编号为 ${t} 的有标注 pair`; return; }
    msg.textContent = info().list.includes(i) ? '' : '不在当前筛选中';
    select(C.pairs[i], { scroll: true });
  }

  // 选中一个 pair：网格只改高亮，不重建；← → 翻到网格之外时网格多显示一页
  function select(pair, { scroll = false, grow = false } = {}) {
    V.sel = pair;
    const I = info(), pos = I.list.indexOf(C.pairs.indexOf(pair));
    if (grow && pos >= V.n) { V.n = Math.ceil((pos + 1) / PAGE) * PAGE; drawList(I); }
    else {
      host.querySelectorAll('.grid .th').forEach(b => b.setAttribute('aria-current', String(b.dataset.pair === pair)));
      if (I.eR) redrawScatter(I);
    }
    drawDetail();
    if (scroll) $('.detail').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  function step(d) {
    const I = info();
    if (!I.list.length) return;
    let pos = I.list.indexOf(C.pairs.indexOf(V.sel));
    pos = pos < 0 ? 0 : Math.max(0, Math.min(I.list.length - 1, pos + d));
    select(C.pairs[I.list[pos]], { grow: true });
  }

  // ---------------- 散点与网格 ----------------
  function drawList(I) {
    if (I.eR) redrawScatter(I);
    const shown = I.list.slice(0, V.n);
    $('.grid').innerHTML = shown.map(i => `<button class="th" data-pair="${esc(C.pairs[i])}" aria-current="${C.pairs[i] === V.sel}">
      <div class="im"></div><div class="cap num"><span>#${pad(C.no[i])}</span><span>${errTxt(I.eM[i])}${I.eR ? ` / ${errTxt(I.eR[i])}` : ''}</span></div></button>`).join('')
      || '<p class="muted">当前筛选下没有 pair。</p>';
    $('.more').innerHTML = `<span class="muted num">${shown.length ? `显示第 1–${shown.length} 个，` : ''}共 ${I.list.length} 个</span>${I.list.length > V.n ? `<button>再显示 ${PAGE} 个</button>` : ''}`;
    $('.more button')?.addEventListener('click', () => { V.n += PAGE; drawList(info()); });
    host.querySelectorAll('.grid .th').forEach(b => {
      b.addEventListener('click', () => select(b.dataset.pair, { scroll: true }));
      thumb(b);
    });
  }

  async function thumb(b) {
    const pair = b.dataset.pair, my = ctxKey;
    let d;
    try { d = await pairData(pair, false); } catch (err) { b.querySelector('.im').textContent = '读取失败'; return; }
    if (my !== ctxKey || !b.isConnected) return;
    const [w, h] = d.size, m = d.method, k = w / 110;
    b.querySelector('.im').innerHTML = `<svg viewBox="0 0 ${w} ${h}" aria-hidden="true"><image href="${img(C.split, pair, 'opt')}" width="${w}" height="${h}"/>
      ${m.A ? `<g class="ov">${arrows(d.checkpoints, m.A, k)}</g>` : ''}</svg>${m.A ? '' : `<span class="fail">失败</span>`}`;
  }

  // 散点：全图点击 = 以该处为中心放大 5 倍；放大后点点 = 选中，点空白 = 再放大 3 倍。
  function sPos(i, I) {
    const f = (v, g) => v == null || v > EDGE ? EDGE + .6 + ((C.no[i] * g) % 1) * 3.2 : v;
    return [f(I.eR[i], 0.618034), f(I.eM[i], 0.381966)];
  }
  function niceStep(span) { for (const s of [0.1, 0.2, 0.5, 1, 2, 5, 10]) if (span / s <= 7) return s; return 10; }
  function scatter(I) {
    const z = V.zoom || FULL, { W, H, L, T, pw, ph } = SC;
    const xs = v => L + (v - z.x0) / (z.x1 - z.x0) * pw, ys = v => T + ph - (v - z.y0) / (z.y1 - z.y0) * ph;
    let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="逐 pair 误差散点"><defs><clipPath id="vr-plot"><rect x="${L}" y="${T}" width="${pw}" height="${ph}"/></clipPath></defs><g class="ax">`;
    const st = niceStep(z.x1 - z.x0), sty = niceStep(z.y1 - z.y0);
    for (let t = Math.ceil(z.x0 / st) * st; t <= Math.min(z.x1, EDGE) + 1e-9; t += st) g += `<line x1="${xs(t)}" x2="${xs(t)}" y1="${T + ph}" y2="${T + ph + 4}"/><text x="${xs(t)}" y="${T + ph + 17}" text-anchor="middle">${+t.toFixed(1)}</text>`;
    for (let t = Math.ceil(z.y0 / sty) * sty; t <= Math.min(z.y1, EDGE) + 1e-9; t += sty) g += `<line x1="${L - 4}" x2="${L}" y1="${ys(t)}" y2="${ys(t)}"/><text x="${L - 7}" y="${ys(t) + 4}" text-anchor="end">${+t.toFixed(1)}</text>`;
    if (z.x1 > EDGE + 1) g += `<text x="${xs(32.2)}" y="${T + ph + 17}" text-anchor="middle">&gt;${EDGE}</text>`;
    if (z.y1 > EDGE + 1) g += `<text x="${L - 7}" y="${ys(32.2) + 4}" text-anchor="end">&gt;${EDGE}</text>`;
    g += `<text x="${L + pw / 2}" y="${H - 3}" text-anchor="middle">参考方法误差（px）</text><text transform="translate(12 ${T + ph / 2}) rotate(-90)" text-anchor="middle">方法误差（px）</text></g>`;
    g += `<g clip-path="url(#vr-plot)"><rect class="band" x="${xs(EDGE + .3)}" y="${T}" width="${pw}" height="${ph}"/><rect class="band" x="${L}" y="${ys(34)}" width="${pw}" height="${ys(EDGE + .3) - ys(34)}"/>`;
    g += `<line class="diag" x1="${xs(0)}" y1="${ys(0)}" x2="${xs(EDGE)}" y2="${ys(EDGE)}"/>`;
    g += `<line class="five" x1="${xs(GOOD)}" x2="${xs(GOOD)}" y1="${T}" y2="${T + ph}"/><line class="five" x1="${L}" x2="${L + pw}" y1="${ys(GOOD)}" y2="${ys(GOOD)}"/>`;
    const inList = new Set(I.list), r0 = V.zoom ? 3 : 2.1;
    C.pairs.forEach((p, i) => {
      const [x, y] = sPos(i, I);
      if (x < z.x0 - 1 || x > z.x1 + 1 || y < z.y0 - 1 || y > z.y1 + 1) return;
      const on = inList.has(i);
      g += `<circle class="pt ${on ? 'on' : ''}" data-i="${i}" cx="${xs(x).toFixed(1)}" cy="${ys(y).toFixed(1)}" r="${on ? r0 : r0 * .75}"><title>#${C.no[i]}  方法 ${errTxt(I.eM[i])} / 参考 ${errTxt(I.eR[i])} px</title></circle>`;
    });
    const s = C.pairs.indexOf(V.sel);
    if (s >= 0) { const [x, y] = sPos(s, I); g += `<circle class="selring" cx="${xs(x)}" cy="${ys(y)}" r="${r0 + 4}"/>`; }
    g += `</g><rect class="frame" x="${L}" y="${T}" width="${pw}" height="${ph}"/></svg>`;
    const f1 = v => v > EDGE ? `>${EDGE}` : v.toFixed(1);
    const bar = V.zoom ? `<span class="muted num">放大区域：参考 ${f1(z.x0)}–${f1(z.x1)} px，方法 ${f1(z.y0)}–${f1(z.y1)} px</span><button data-z="out">缩小一级</button><button data-z="full">返回全图</button>`
      : '<span class="muted">点击图中任意位置，放大该区域后再点选</span>';
    return `<div class="${V.zoom ? 'zoomed' : ''}">${g}<div class="zbar">${bar}</div></div>`;
  }
  function redrawScatter(I) { $('.sc').innerHTML = scatter(I); bindScatter(); }
  function bindScatter() {
    const svg = $('.sc svg'), redraw = () => redrawScatter(info());
    svg.addEventListener('click', ev => {
      const pt = ev.target.closest('circle.pt');
      if (pt && V.zoom) { select(C.pairs[+pt.dataset.i], { scroll: true }); return; }
      const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
      const v = p.matrixTransform(svg.getScreenCTM().inverse()), z = V.zoom || FULL, { L, T, pw, ph } = SC;
      if (v.x < L || v.x > L + pw || v.y < T || v.y > T + ph) return;
      const cx = z.x0 + (v.x - L) / pw * (z.x1 - z.x0), cy = z.y0 + (T + ph - v.y) / ph * (z.y1 - z.y0);
      const f = V.zoom ? 3 : 5, w = Math.max((z.x1 - z.x0) / f, 0.4), h = Math.max((z.y1 - z.y0) / f, 0.4);
      const clamp = (c, s) => Math.min(Math.max(c - s / 2, 0), FULL.x1 - s);
      if (V.zoom) V.zstack.push(V.zoom);
      V.zoom = { x0: clamp(cx, w), x1: clamp(cx, w) + w, y0: clamp(cy, h), y1: clamp(cy, h) + h };
      redraw();
    });
    $('.sc [data-z="out"]')?.addEventListener('click', () => { V.zoom = V.zstack.pop() || null; redraw(); });
    $('.sc [data-z="full"]')?.addEventListener('click', () => { V.zoom = null; V.zstack = []; redraw(); });
  }

  // ---------------- 详情 ----------------
  async function drawDetail() {
    const el = $('.detail'), pair = V.sel, my = ++seq;
    if (!pair) { el.innerHTML = ''; return; }
    const I = info(), i = C.pairs.indexOf(pair), pos = I.list.indexOf(i);
    el.innerHTML = `<div class="dhead"><span class="pn num">${SPLIT()} #${pad(C.no[i])}</span><span class="muted">${esc(pair)}</span><span class="sp"></span>
      <span class="muted num">${pos >= 0 ? `当前列表第 ${pos + 1} / ${I.list.length} 个` : '不在当前筛选中'}</span>
      <button data-step="-1" title="←">← 上一个</button><button data-step="1" title="→">下一个 →</button></div><div class="dbody"><p class="muted">读取中…</p></div>`;
    el.querySelectorAll('[data-step]').forEach(b => b.addEventListener('click', () => step(+b.dataset.step)));
    let d;
    try { d = await pairData(pair, true); } catch (err) {
      if (my === seq) el.querySelector('.dbody').innerHTML = `<p>读取失败：${esc(err.message)}</p>`;
      return;
    }
    if (my !== seq || !el.isConnected) return;
    V.d = d;
    body();
  }

  function body() {
    const d = V.d, el = $('.detail .dbody'), sides = [d.method, d.ref].filter(Boolean);
    const hasPts = sides.some(s => s.matches?.state === 'ok');
    if (!hasPts && isPointView(V.view)) V.view = 'swipe';
    const inters = [...new Map(sides.flatMap(s => s.inter || []).map(x => [x.name, x])).values()];
    if (interName(V.view) != null && !inters.some(x => x.name === interName(V.view))) V.view = 'swipe';
    const views = [...VIEWS, ...(hasPts ? POINT_VIEWS : [])];
    if (interName(V.view) != null) { interBody(el, sides, views, inters); return; }
    const point = isPointView(V.view);
    const missing = sides.filter(s => s.matches?.state === 'not_synced');
    const cmds = [...new Set(missing.map(s => s.matches.sync))];
    const cmd = cmds.length > 1 ? `python -m workbench sync ${[...new Set(missing.map(s => s.exp))].join(' ')}` : cmds[0];
    const hint = cmd ? `<p class="synchint">本地没有 ${missing.map(s => esc(s.exp)).filter((x, j, a) => a.indexOf(x) === j).join('、')} 的点对，拉回后可看点对视图：<code>${esc(cmd)}</code><button class="copy" data-cmd="${esc(cmd)}">复制</button></p>` : '';
    const withConf = point ? sides.filter(s => s.matches?.state === 'ok' && s.matches.conf) : [];
    const flat = withConf.map(s => flatConf(s.matches));
    const flatNote = withConf.length && flat.every(v => v != null)
      ? `<span class="muted">conf 全部为 ${[...new Set(flat)].join(' / ')}，按分位数过滤不起作用</span>` : '';
    const caveats = point ? sides.filter(s => s.caveat).map(s => `<p class="caveat">${esc(s.label)}：${esc(s.caveat)}</p>`).join('') : '';
    const panes = V.view === 'raw'
      ? pane(d, null, '光学', '', 'opt') + pane(d, null, 'SAR', '', 'sar')
      : sides.map(s => pane(d, s, esc(s.label), s.A ? `误差 ${errTxt(s.error)} px` : `估计失败（${esc(s.fail)}）`, V.view)).join('');
    const cap = {
      swipe: '左侧为光学，右侧为按估计仿射变换到光学坐标系的 SAR；拖动下方滑杆改变分界位置。估计失败的一侧只显示光学原图。',
      resid: '光学原图上标出检查点（黄圈）与残差：红色箭头从检查点指向其 SAR 真值经估计仿射变回光学坐标后的位置，按原尺寸绘制。',
      raw: '光学与 SAR 原图，未做变换。',
      lines: `左为光学、右为 SAR，连线为方法给出的点对（RANSAC 之前的全部点）。内点、外点由估计仿射重新判定：残差不超过 ${d.inlier_px} px 为内点（绿），其余为外点（橙）；估计失败时不区分（灰）。`,
      dots: '光学图上的匹配点，颜色为该点经估计仿射映射后到对应 SAR 点的距离（残差），图例见下；估计失败时为灰色。',
    }[V.view];
    el.innerHTML = `${viewBar(views, inters)}
      ${hint}${caveats}
      <div class="panes ${V.view === 'lines' ? 'wide' : ''}">${panes}</div>
      ${V.view === 'swipe' ? `<input class="swipe" type="range" min="0" max="100" value="${V.swipe}" aria-label="卷帘分界位置">` : ''}
      ${withConf.length ? `<div class="conf"><label for="vconf">conf 过滤</label><input id="vconf" type="range" min="0" max="95" step="5" value="${V.conf}"><span class="num">${confText()}</span>${flatNote}</div>` : ''}
      ${V.view === 'dots' ? `<div class="legend"><span>残差</span><span class="num">0</span><span class="ramp" title="0 px 绿，3 px 黄，10 px 及以上红"></span><span class="num">≥10 px</span><span>（3 px 处为黄色）</span></div>` : ''}
      ${V.view === 'lines' ? `<div class="legend"><span><span class="swatch" style="background:${INLIER}"></span>内点</span><span><span class="swatch" style="background:${OUTLIER}"></span>外点</span></div>` : ''}
      <p class="caption figcap"><b>图 ${V.detailFig}</b>${cap}滚轮缩放、拖动平移、双击复原，各面板同步。</p>`;
    bindBody(el);
    el.querySelector('#vconf')?.addEventListener('input', ev => {
      V.conf = +ev.target.value; el.querySelector('.conf .num').textContent = confText(); zoomApply();
    });
    bindZoom();
  }
  const confText = () => V.conf ? `只保留 conf 不低于第 ${V.conf} 百分位的点` : '显示全部点';

  function viewBar(views, inters) {
    const btn = ([v, t, title]) => `<button data-v="${esc(v)}" aria-pressed="${V.view === v}"${title ? ` title="${esc(title)}"` : ''}>${esc(t)}</button>`;
    return `<div class="views" role="group" aria-label="视图">${views.map(btn).join('')}${inters.length
      ? `<span class="vsep">中间结果</span>${inters.map(x => btn([INTER + x.name, x.name, x.desc])).join('')}` : ''}</div>`;
  }
  function bindBody(el) {
    el.querySelectorAll('.views button').forEach(b => b.addEventListener('click', () => { V.view = b.dataset.v; body(); }));
    el.querySelectorAll('.copy').forEach(b => b.addEventListener('click', ev => {
      navigator.clipboard?.writeText(ev.target.dataset.cmd).then(() => { ev.target.textContent = '已复制'; }, () => {});
    }));
    el.querySelector('.swipe')?.addEventListener('input', ev => { V.swipe = +ev.target.value; zoomApply(); });
  }

  // ---------------- 中间结果 ----------------
  // 只给声明了这个中间结果的一侧出面板；数据按 pair 与名称取一次，翻页或切换视图后重取。
  const side = s => s === V.d.method ? 'method' : 'ref';
  function interBody(el, sides, views, inters) {
    const d = V.d, name = interName(V.view), meta = inters.find(x => x.name === name);
    const own = sides.filter(s => s.inter?.some(x => x.name === name));
    const key = `${ctxKey}|${d.pair}|${name}`;
    if (V.iv?.key !== key) {
      const my = V.iv = { key, data: {} };
      Promise.all(own.map(async s => {
        const st = s.inter.find(x => x.name === name);
        my.data[side(s)] = st.state === 'ok' ? await api.inter(s.key, d.split, d.pair, name) : st;
      })).then(() => { if (V.iv === my && V.d === d && interName(V.view) === name) body(); },
        err => { my.error = err.message; if (V.iv === my && V.d === d) body(); });
    }
    const iv = V.iv, head = viewBar(views, inters);
    if (iv.error || !own.every(s => iv.data[side(s)])) {
      el.innerHTML = `${head}${iv.error ? `<p>读取中间结果失败：${esc(iv.error)}</p>` : '<p class="muted">读取中…</p>'}`;
      bindBody(el);
      return;
    }
    const all = own.map(s => [s, iv.data[side(s)]]);
    const cmds = [...new Set(all.filter(([, x]) => x.state === 'not_synced').map(([, x]) => x.sync))];
    const hint = cmds.map(c => `<p class="synchint">本地没有中间结果 ${esc(name)} 的数据，拉回后可看：<code>${esc(c)}</code><button class="copy" data-cmd="${esc(c)}">复制</button></p>`).join('');
    const ok = all.filter(([, x]) => x.state === 'ok');
    const unit = meta.unit ? `（${esc(meta.unit)}）` : '';
    const legend = ok.filter(([, x]) => x.kind !== 'image').map(([s, x]) => `<div class="legend"><span>${ok.length > 1 ? `${esc(s.label)}：` : ''}${x.kind === 'flow' ? '位移大小' : '取值'}${unit}</span>
      <span class="num">${fmt(x.vmin)}</span><span class="ramp" style="background:${RAMP}"></span><span class="num">${fmt(x.vmax)}</span></div>`).join('');
    const res = [...new Set(ok.filter(([, x]) => x.kind === 'scalar').map(([, x]) => x.shape.join('×')))];
    const F = FRAME[meta.frame], O = FRAME[meta.frame === 'opt' ? 'sar' : 'opt'];
    const cap = `中间结果 ${esc(name)}（${KIND[meta.kind]}，${F}坐标系）${meta.desc ? `：${esc(meta.desc)}` : ''}。` + (!ok.length ? '' : {
      scalar: `热力图叠加在${F}原图上，颜色按该 pair 自身的最小、最大值映射（见色标）${res.length ? `；原分辨率 ${res.join(' / ')}，拉伸到 patch 大小` : ''}。滑杆调节热力图的不透明度。`,
      points: `点画在${F}原图上，颜色为取值（见色标）；不超过 ${POINT_LABELS} 个点时在点旁标出取值，悬停可看单点取值。`,
      flow: `左侧为${F}原图，右侧为${O}按位移场摆到${F}坐标后的影像，拖动下方滑杆改变分界位置；位移落到${O}之外的像素留空。`,
      image: `方法输出的图片，原样拉伸到${F} patch 大小。`,
    }[meta.kind]);
    const kinds = ok.length ? meta.kind : null;
    el.innerHTML = `${head}${hint}<div class="panes">${all.map(([s, x]) => interPane(d, s, meta, x)).join('')}</div>
      ${kinds === 'flow' ? `<input class="swipe" type="range" min="0" max="100" value="${V.swipe}" aria-label="卷帘分界位置">` : ''}
      ${kinds === 'scalar' ? `<div class="conf"><label for="valpha">热力图不透明度</label><input id="valpha" type="range" min="0" max="100" step="5" value="${V.alpha}"><span class="num">${V.alpha}%</span></div>` : ''}
      ${legend}
      <p class="caption figcap"><b>图 ${V.detailFig}</b>${cap}滚轮缩放、拖动平移、双击复原，各面板同步。</p>`;
    bindBody(el);
    el.querySelector('#valpha')?.addEventListener('input', ev => {
      V.alpha = +ev.target.value; el.querySelector('.conf .num').textContent = `${V.alpha}%`;
      el.querySelectorAll('image.heat').forEach(im => im.setAttribute('opacity', V.alpha / 100));
    });
    bindZoom();
  }

  function interPane(d, s, meta, x) {
    const sd = side(s), [w, h] = meta.frame === 'sar' ? d.sar_size : d.size, full = [0, 0, w, h];
    const layer = cls => `<image class="${cls}" href="${esc(x.png)}" width="${w}" height="${h}" preserveAspectRatio="none"${cls === 'heat' ? ` opacity="${V.alpha / 100}"` : ''}/>`;
    let base = `<image href="${img(d.split, d.pair, meta.frame)}" width="${w}" height="${h}"/>`, msg = '';
    if (x.state === 'absent') msg = '此 pair 没有该中间结果';
    else if (x.state === 'not_synced') msg = '本地没有该中间结果的数据';
    else if (meta.kind === 'scalar') base += layer('heat');
    else if (meta.kind === 'flow') base += `<g class="clip">${layer('warp')}</g><line class="cut"/>`;
    else if (meta.kind === 'image') base = layer('raw');
    const right = x.state !== 'ok' ? '' : meta.kind === 'points' ? `${x.points.length} 个点` : meta.kind === 'image' ? '' : x.shape.slice(0, 2).join('×');
    return `<div class="pane"><div class="ttl"><span>${esc(s.label)}</span><span class="num stat">${right}</span></div>
      <div class="box" style="aspect-ratio:${w} / ${h}"><svg class="zoom" data-side="${sd}" data-full="${full.join(' ')}" viewBox="${full.join(' ')}">
      <defs><clipPath id="vc-${sd}" clipPathUnits="userSpaceOnUse"><rect class="cliprect" x="0" y="${-1e4}" width="${2e4}" height="${2e4}"/></clipPath></defs>
      ${base}<g class="ov"></g></svg>${msg ? `<div class="msg">${msg}</div>` : ''}</div></div>`;
  }

  // 点集：颜色为取值，按该 pair 的最小、最大值映射；点少时旁边标出取值
  function interPoints(x, k) {
    if (x?.kind !== 'points' || x.state !== 'ok') return '';
    const span = x.vmax - x.vmin, t = v => v == null ? null : span > 0 ? (v - x.vmin) / span : 0.5;
    const label = x.points.length <= POINT_LABELS;
    return x.points.map(([px, py, v]) => `<circle class="ipt" cx="${px}" cy="${py}" r="${f2(3 * k)}" fill="${viridis(t(v))}"><title>(${px}, ${py})  ${fmt(v)}</title></circle>`
      + (label ? `<text class="ival" x="${f2(px + 4.5 * k)}" y="${f2(py - 4.5 * k)}" font-size="${f2(11 * k)}">${fmt(v)}</text>` : '')).join('');
  }

  // 一个面板：底图 + 叠加层（叠加层随缩放重画，线宽、圆点在屏幕上大小不变）
  function pane(d, s, title, right, view) {
    const [w, h] = view === 'sar' ? d.sar_size : d.size, [sw, sh] = d.sar_size;
    let full = [0, 0, w, h], base = '', msg = '';
    if (view === 'opt' || view === 'sar') base = `<image href="${img(d.split, d.pair, view)}" width="${w}" height="${h}"/>`;
    else if (view === 'lines') {
      const gap = Math.round(w * .04);
      full = [0, 0, w + gap + sw, Math.max(h, sh)];
      base = `<image href="${img(d.split, d.pair, 'opt')}" width="${w}" height="${h}"/><image href="${img(d.split, d.pair, 'sar')}" x="${w + gap}" width="${sw}" height="${sh}"/>`;
    } else {
      base = `<image href="${img(d.split, d.pair, 'opt')}" width="${w}" height="${h}"/>`;
      const inv = s.A && inverse(s.A);
      if (view === 'swipe' && inv) base += `<g class="clip"><image href="${img(d.split, d.pair, 'sar')}" width="${sw}" height="${sh}" transform="matrix(${inv.m.map(f2).join(' ')})"/></g><line class="cut"/>`;
    }
    if (s && !s.A && (view === 'swipe' || view === 'resid')) msg = '估计失败，只显示光学原图';
    if (s && isPointView(view)) {
      const mt = s.matches;
      if (!mt) msg = '未配准没有点对';
      else if (mt.state === 'not_synced') msg = '本地没有点对';
      else if (mt.state === 'absent') msg = '没有点对（该 pair 出错）';
      else if (!mt.n) msg = '0 个点';
    }
    const side = s ? (s === d.method ? 'method' : 'ref') : view;
    return `<div class="pane"><div class="ttl"><span>${title}</span><span class="num stat">${right}</span></div>
      <div class="box ${view === 'lines' ? 'ln' : ''}" style="aspect-ratio:${full[2]} / ${full[3]}"><svg class="zoom" data-side="${side}" data-full="${full.join(' ')}" viewBox="${full.join(' ')}">
      <defs><clipPath id="vc-${side}" clipPathUnits="userSpaceOnUse"><rect class="cliprect" x="0" y="${-1e4}" width="${2e4}" height="${2e4}"/></clipPath></defs>
      ${base}<g class="ov"></g></svg>${msg ? `<div class="msg">${msg}</div>` : ''}</div></div>`;
  }

  // ---------------- 同步缩放与平移 ----------------
  const vbKey = () => V.view === 'lines' ? 'vbl' : 'vb';
  function zoomApply() {
    const d = V.d;
    host.querySelectorAll('.detail svg.zoom').forEach(svg => {
      const full = svg.dataset.full.split(' ').map(Number), vb = V[vbKey()] || full;
      svg.setAttribute('viewBox', vb.join(' '));
      const k = vb[2] / (svg.getBoundingClientRect().width || 1), side = svg.dataset.side, s = d[side];
      const cut = full[0] + full[2] * V.swipe / 100;
      svg.querySelector('.cliprect')?.setAttribute('x', cut);
      const g = svg.querySelector('g.clip');
      if (g) { g.setAttribute('clip-path', `url(#vc-${side})`); svg.querySelector('.cut').setAttribute('x1', cut); svg.querySelector('.cut').setAttribute('x2', cut);
        svg.querySelector('.cut').setAttribute('y1', -1e4); svg.querySelector('.cut').setAttribute('y2', 1e4); }
      svg.querySelector('.ov').innerHTML = s ? overlay(d, s, k, full) : '';
      const stat = svg.closest('.pane').querySelector('.stat');
      if (s && isPointView(V.view) && s.matches?.state === 'ok') stat.textContent = ptsText(s, d);
    });
  }
  function ptsText(s, d) {
    const mt = s.matches, { ks, thr } = confCut(mt, V.conf);
    const t = `${ks.length} / ${mt.n} 个点${thr != null ? `（conf ≥ ${+thr.toFixed(4)}）` : ''}`;
    return mt.resid ? `${t}，内点 ${ks.filter(i => mt.resid[i] <= d.inlier_px).length}` : t;
  }
  function overlay(d, s, k, full) {
    if (interName(V.view) != null) return interPoints(V.iv?.data[side(s)], k);
    if (V.view === 'resid') return arrows(d.checkpoints, s.A, k);
    const mt = s.matches;
    if (!mt || mt.state !== 'ok' || !mt.n) return '';
    const { ks } = confCut(mt, V.conf);
    if (V.view === 'dots') return ks.map(i => `<circle cx="${mt.opt[i][0]}" cy="${mt.opt[i][1]}" r="${f2(2.4 * k)}" fill="${residColor(mt.resid?.[i])}"/>`).join('');
    if (V.view === 'lines') {
      const off = full[2] - d.sar_size[0];
      const inl = i => !!mt.resid && mt.resid[i] <= d.inlier_px;
      const col = i => !mt.resid ? NEUTRAL : inl(i) ? INLIER : OUTLIER;
      return ks.slice().sort((a, b) => inl(a) - inl(b)).map(i =>   // 内点后画，压在外点上面
        `<line class="mline" x1="${mt.opt[i][0]}" y1="${mt.opt[i][1]}" x2="${f2(mt.sar[i][0] + off)}" y2="${mt.sar[i][1]}" stroke="${col(i)}"/>`).join('');
    }
    return '';
  }
  function bindZoom() {
    const svgs = [...host.querySelectorAll('.detail svg.zoom')];
    zoomApply();
    svgs.forEach(svg => {
      const full = () => svg.dataset.full.split(' ').map(Number);
      const user = ev => { const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      svg.addEventListener('wheel', ev => {
        ev.preventDefault();
        const F = full(), vb = V[vbKey()] || F, p = user(ev);
        const s = Math.min(Math.max(vb[2] * Math.exp(ev.deltaY * 0.0015), F[2] / MAX_ZOOM), F[2]) / vb[2];
        V[vbKey()] = s * vb[2] >= F[2] ? null : [p.x - (p.x - vb[0]) * s, p.y - (p.y - vb[1]) * s, vb[2] * s, vb[3] * s];
        zoomApply();
      }, { passive: false });
      svg.addEventListener('pointerdown', ev => {
        if (ev.button !== 0 || !V[vbKey()]) return;
        const start = [ev.clientX, ev.clientY], vb0 = V[vbKey()], k = vb0[2] / svg.getBoundingClientRect().width;
        svg.setPointerCapture(ev.pointerId); svg.classList.add('grab');
        const move = e => { V[vbKey()] = [vb0[0] - (e.clientX - start[0]) * k, vb0[1] - (e.clientY - start[1]) * k, vb0[2], vb0[3]]; zoomApply(); };
        const up = () => { svg.removeEventListener('pointermove', move); svg.classList.remove('grab'); };
        svg.addEventListener('pointermove', move);
        svg.addEventListener('pointerup', up, { once: true });
        svg.addEventListener('pointercancel', up, { once: true });
      });
      svg.addEventListener('dblclick', () => { V[vbKey()] = null; zoomApply(); });
    });
  }

  // ← → 在当前列表里移动（焦点在输入控件里时不响应）
  document.addEventListener('keydown', ev => {
    if (!host?.isConnected || !isShown() || !V?.sel || ev.ctrlKey || ev.altKey || ev.metaKey) return;
    if (ev.target.closest?.('input, select, textarea, [contenteditable]')) return;
    if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') { ev.preventDefault(); step(ev.key === 'ArrowLeft' ? -1 : 1); }
  });
  addEventListener('resize', () => { if (host?.isConnected && V?.d) zoomApply(); });

  return { render };
}
