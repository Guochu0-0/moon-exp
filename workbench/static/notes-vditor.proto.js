// 【原型，用完即删】Notes 编辑器换成 Vditor 4.0.0 IR 模式的试用版（「选定编辑器方案与快捷键表」那张票）。
// 节点页地址带 ?editor=vditor 时代替 notes.js 的 mountNotes，接口相同；不带时一切照旧。
// 要回答的问题：Vditor IR 在真实 Notes 上用起来是否像 Typora，写回的 Markdown 是否干净，Typora 键位能否压住它自带的键。

const VD = '/static/vendor/vditor-4.0.0';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let loading;
function loadVditor() {
  return loading ??= new Promise((ok, fail) => {
    const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = `${VD}/dist/index.css`; document.head.append(css);
    const own = document.createElement('link'); own.rel = 'stylesheet'; own.href = '/static/notes-vditor.proto.css'; document.head.append(own);
    const js = document.createElement('script'); js.src = `${VD}/dist/index.min.js`;
    js.onload = () => ok(window.Vditor); js.onerror = () => fail(new Error('Vditor 加载失败'));
    document.head.append(js);
  });
}

// 原型里的键位表（Windows），页面上「快捷键表」按钮显示的就是它
export const KEYMAP_DOC = [
  ['Ctrl+B', '粗体'], ['Ctrl+I', '斜体'], ['Ctrl+Shift+`', '行内代码'], ['Alt+Shift+5', '删除线'], ['Ctrl+K', '链接'],
  ['Ctrl+1…6', '标题 1–6（显示时降两级）'], ['Ctrl+0', '变回段落'], ['Ctrl+= / Ctrl+-', '标题升 / 降一级（表格里：下方加行 / 删行）'],
  ['Ctrl+Shift+K', '代码块'], ['Ctrl+Shift+M', '公式块'], ['Ctrl+Shift+Q', '引用'],
  ['Ctrl+Shift+[ / Ctrl+Shift+]', '有序 / 无序列表'], ['Ctrl+Alt+X', '任务列表（Typora 无此键，暂借）'],
  ['Tab / Shift+Tab，Ctrl+[ / Ctrl+]', '列表缩进 / 反缩进'], ['Ctrl+Alt+T', '表格（Typora 的 Ctrl+T 浏览器拦不住，暂借）'],
  ['Ctrl+Enter', '表格里：下方加行（Vditor 自带）'], ['Ctrl+Shift+Backspace', '表格里：删除当前行'],
  ['Ctrl+L / Ctrl+D', '选中当前段 / 当前词'], ['Ctrl+/', '源码模式'], ['Ctrl+Z / Ctrl+Y', '撤销 / 重做（Vditor 自带）'],
  ['Ctrl+S', '保存'], ['Ctrl+单击', '打开链接'],
];

export function mountNotes(body, { text, base, path, save }) {
  const orig = text.replace(/\r\n?/g, '\n');
  let saved = orig, baseline = null, source = false, busy = false, vd = null;
  body.innerHTML = `<div class="proto-bar">原型 · Vditor 4.0.0 IR 模式 · <a href="${esc(location.pathname + location.hash)}">回到现有编辑器</a>
      <button class="keys">快捷键表</button><button class="diff">看写回差异</button></div>
    <div class="proto-keys" hidden><table>${KEYMAP_DOC.map(([k, t]) => `<tr><td><kbd>${esc(k)}</kbd></td><td>${esc(t)}</td></tr>`).join('')}</table></div>
    <pre class="proto-diff" hidden></pre>
    <div class="notes-vd"></div>
    <textarea class="notes-src" spellcheck="false" aria-label="Notes（Markdown 源码）" hidden></textarea>
    <div class="notes-foot"><button class="save">保存</button><button class="revert">撤销改动</button><button class="mode" aria-pressed="false">源码</button><span class="status"></span></div>`;
  const host = body.querySelector('.notes-vd'), ta = body.querySelector('.notes-src'), $ = s => body.querySelector(`.notes-foot .${s}`);
  const diffBox = body.querySelector('.proto-diff');

  // Lute 的 SetUnorderedListMarker 管不到 IR：无序列表符号记在 DOM 的 data-marker 上，取值前统一改成 -
  const dash = () => host.querySelectorAll('.vditor-ir [data-marker="*"], .vditor-ir [data-marker="+"]').forEach(e => e.setAttribute('data-marker', '-'));
  const current = () => source ? ta.value.replace(/\r\n?/g, '\n') : vd ? (dash(), vd.getValue()) : saved;
  // 「改过」按 Vditor 规整后的文本比：载入后 getValue() 一次作为基准，免得一载入就因为规整显示未保存
  const dirty = () => baseline != null && current() !== baseline;
  function refresh(msg) {
    const d = dirty();
    $('save').disabled = !d || busy; $('revert').disabled = !d;
    $('status').textContent = msg ?? (d ? `未保存 · Ctrl+S 保存到 ${path}` : path);
  }

  async function doSave() {
    if (busy || !dirty()) return;
    const t = current();
    busy = true; refresh('保存中…');
    try { await save(t); } catch (err) { busy = false; refresh(`保存失败：${err.message}`); return; }
    busy = false; saved = t; baseline = t;
    refresh(`已保存到 ${path}`);
  }
  function revert() {
    if (!confirm('放弃未保存的 Notes 改动？')) return;
    if (source) { ta.value = saved; baseline = saved; } else { vd.setValue(saved, true); baseline = current(); }
    refresh();
  }
  // 切换源码时页面不滚动：记下滚动位置，聚焦时不让浏览器把光标滚进视野，切换后再恢复
  const scroller = () => { for (let e = body.parentElement; e; e = e.parentElement) if (e.scrollHeight > e.clientHeight && /auto|scroll/.test(getComputedStyle(e).overflowY)) return e; return document.scrollingElement; };
  function toggle() {
    const sc = scroller(), top = sc.scrollTop;
    if (source) { vd.setValue(ta.value.replace(/\r\n?/g, '\n')); }
    else ta.value = current();
    source = !source;
    host.hidden = source; ta.hidden = !source;
    $('mode').setAttribute('aria-pressed', source);
    if (source) {
      ta.style.height = 'auto'; ta.style.height = `${Math.max(260, ta.scrollHeight + 8)}px`;
      ta.setSelectionRange(0, 0); ta.focus({ preventScroll: true });
    } else irEl()?.focus({ preventScroll: true });
    sc.scrollTop = top;
    requestAnimationFrame(() => { sc.scrollTop = top; });
    refresh();
  }

  // 写回差异：文件原文 vs 现在保存会写进去的内容（逐行 LCS，只显示改动附近 2 行）
  function showDiff() {
    if (!diffBox.hidden) { diffBox.hidden = true; return; }
    const a = orig.split('\n'), b = current().split('\n'), n = a.length, m = b.length;
    const L = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    const out = []; let i = 0, j = 0, changed = 0;
    while (i < n || j < m) {
      if (i < n && j < m && a[i] === b[j]) { out.push(['=', a[i]]); i++; j++; }
      else if (j < m && (i === n || L[i][j + 1] >= L[i + 1][j])) { out.push(['+', b[j++]]); changed++; }
      else { out.push(['-', a[i++]]); changed++; }
    }
    const keep = out.map((x, k) => out.slice(Math.max(0, k - 2), k + 3).some(y => y[0] !== '='));
    let html = `<b>文件原文 → 现在保存会写入的内容：${changed} 行不同</b>\n`, gap = false;
    out.forEach(([t, l], k) => {
      if (!keep[k]) { if (!gap) html += '<span class="gap">…</span>\n'; gap = true; return; }
      gap = false;
      html += `<span class="${{ '+': 'add', '-': 'del', '=': '' }[t]}">${t === '=' ? ' ' : t} ${esc(l)}</span>\n`;
    });
    diffBox.innerHTML = changed ? html : '<b>和文件原文完全一致</b>';
    diffBox.hidden = false;
  }

  // ---- Typora 键位：捕获阶段先于 Vditor 处理；命令借隐藏工具栏的按钮执行 ----
  // 按键用 event.code 判断，避免 Shift 改变 event.key（Ctrl+Shift+` 的 key 是 ~）
  // 注意：IR 编辑区的根节点本身是 <pre class="vditor-reset">，找代码块时要排除它
  const irEl = () => host.querySelector('.vditor-ir .vditor-reset');
  const tool = name => host.querySelector(`.vditor-toolbar button[data-type="${name}"]`);
  // 按钮的可用状态要等 keyup 才刷新，键位触发时可能还是旧的「不可用」，先去掉再点
  const click = el => {
    if (el) { el.classList.remove('vditor-menu--disabled'); el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })); }
    return true;
  };
  const at = sel => {
    const s = getSelection(), n = s.rangeCount && s.anchorNode, e = n && (n.nodeType === 1 ? n : n.parentElement);
    const hit = e?.closest(sel); return hit && irEl()?.contains(hit) ? hit : null;
  };
  const headingLevel = () => { const h = at('h1,h2,h3,h4,h5,h6'); return h ? +h.nodeName[1] : 0; };
  function heading(k) {
    if (k <= 0) {   // 变回段落：Vditor 里对「当前级别」的标题按钮再按一次就取消
      if (!headingLevel()) return true;
      const btn = tool('headings'); btn.classList.add('vditor-menu--current'); return click(btn);
    }
    return click(host.querySelector(`.vditor-toolbar [data-tag="h${Math.min(6, k)}"]`));
  }
  // 表格里借 Vditor 自带的 Ctrl+= 加行、Ctrl+- 删行：发一个合成按键，放它过去
  let passing = false;
  function vditorKey(key, code) {
    passing = true;
    try { irEl().dispatchEvent(new KeyboardEvent('keydown', { key, code, ctrlKey: true, bubbles: true, cancelable: true })); } finally { passing = false; }
    return true;
  }
  function select(unit) {
    const s = getSelection(); if (!s.rangeCount) return true;
    if (unit === 'word') { s.modify('move', 'backward', 'word'); s.modify('extend', 'forward', 'word'); return true; }
    const b = at('p,li,h1,h2,h3,h4,h5,h6,td,th,pre:not(.vditor-reset)');
    if (b) { const r = document.createRange(); r.selectNodeContents(b); s.removeAllRanges(); s.addRange(r); }
    return true;
  }
  const lv = () => headingLevel();
  function keydown(ev) {
    if (passing || ev.isComposing) return;
    const c = ev.ctrlKey || ev.metaKey, sh = ev.shiftKey, alt = ev.altKey, k = ev.code;
    const is = (cc, ss, aa, code) => c === cc && sh === ss && alt === aa && k === code;
    let act = null;
    if (is(true, false, false, 'KeyS')) act = () => doSave();
    else if (is(true, false, false, 'Slash')) act = () => toggle();
    else if (source || !vd) return;
    else if (is(true, false, false, 'KeyB')) act = () => click(tool('bold'));
    else if (is(true, false, false, 'KeyI')) act = () => click(tool('italic'));
    else if (is(true, true, false, 'Backquote')) act = () => click(tool('inline-code'));
    else if (is(false, true, true, 'Digit5')) act = () => click(tool('strike'));
    else if (is(true, false, false, 'KeyK')) act = () => click(tool('link'));
    else if (c && !sh && !alt && /^Digit[1-6]$/.test(k)) act = () => heading(+k.at(-1));
    else if (is(true, false, false, 'Digit0')) act = () => heading(0);
    else if (is(true, false, false, 'Equal')) act = () => at('table') ? vditorKey('=', 'Equal') : lv() > 1 && heading(lv() - 1);
    else if (is(true, false, false, 'Minus')) act = () => at('table') ? vditorKey('-', 'Minus') : heading(lv() ? (lv() === 6 ? 0 : lv() + 1) : 0);
    else if (is(true, true, false, 'KeyK')) act = () => click(tool('code'));
    else if (is(true, true, false, 'KeyM')) act = () => vd.insertValue('\n$$\n\n$$\n');
    else if (is(true, true, false, 'KeyQ')) act = () => click(tool('quote'));
    else if (is(true, true, false, 'BracketLeft')) act = () => click(tool('ordered-list'));
    else if (is(true, true, false, 'BracketRight')) act = () => click(tool('list'));
    else if (is(true, false, true, 'KeyX')) act = () => click(tool('check'));
    // Vditor 只在光标位于列表项开头时把 Tab 当缩进；Typora 在列表项里任何位置都缩进
    else if (!c && !alt && k === 'Tab' && at('li') && !at('table, pre:not(.vditor-reset)')) act = () => click(tool(sh ? 'outdent' : 'indent'));
    else if (is(true, false, false, 'BracketLeft')) act = () => click(tool('indent'));
    else if (is(true, false, false, 'BracketRight')) act = () => click(tool('outdent'));
    else if (is(true, false, true, 'KeyT')) act = () => click(tool('table'));
    else if (is(true, true, false, 'Backspace')) act = () => at('table') && vditorKey('-', 'Minus');
    else if (is(true, false, false, 'KeyL')) act = () => select('block');
    else if (is(true, false, false, 'KeyD')) act = () => select('word');
    // Vditor 自带、和 Typora 不同的组合（Ctrl+U 代码块、Ctrl+G 行内代码、Ctrl+J 任务、Ctrl+O 有序、Ctrl+; 引用、Ctrl+H 分隔线……）一律吞掉
    else if (c && !alt && ['KeyU', 'KeyG', 'KeyJ', 'KeyO', 'Semicolon', 'KeyH'].includes(k)) act = () => true;
    if (!act) return;
    ev.preventDefault(); ev.stopPropagation();
    act();
    setTimeout(() => refresh());
  }
  body.addEventListener('keydown', keydown, true);
  ta.addEventListener('input', () => refresh());

  // Ctrl+单击打开链接（IR 里链接是一串 span，地址在 marker--link 里）
  host.addEventListener('click', ev => {
    if (!(ev.ctrlKey || ev.metaKey)) return;
    const a = ev.target.closest('[data-type="a"]'); if (!a) return;
    const href = a.querySelector('.vditor-ir__marker--link')?.textContent?.trim(); if (!href) return;
    ev.preventDefault();
    window.open(/^(https?:|mailto:|#)/i.test(href) ? href : base + href.replace(/^\.\//, ''), '_blank', 'noopener');
  }, true);

  $('save').addEventListener('click', doSave);
  $('revert').addEventListener('click', revert);
  $('mode').addEventListener('click', toggle);
  body.querySelector('.proto-bar .keys').addEventListener('click', () => { const t = body.querySelector('.proto-keys'); t.hidden = !t.hidden; });
  body.querySelector('.proto-bar .diff').addEventListener('click', showDiff);
  refresh('载入 Vditor…');

  const off = name => ({ name, hotkey: '' });   // 工具栏只借它的命令；自带热键清空，由上面的 Typora 键位接管
  loadVditor().then(Vditor => {
    vd = new Vditor(host, {
      mode: 'ir', value: orig, cdn: VD, lang: 'zh_CN', icon: 'ant', minHeight: 220,
      cache: { enable: false }, counter: { enable: false }, toolbarConfig: { hide: true, pin: false },
      toolbar: ['headings', 'bold', 'italic', 'strike', 'link', 'list', 'ordered-list', 'check', 'outdent', 'indent',
        'quote', 'line', 'code', 'inline-code', 'table', 'undo', 'redo'].map(off),
      hint: { emoji: {} },
      preview: {
        hljs: { enable: false }, math: { engine: 'KaTeX', inlineDigit: false },
        markdown: { linkBase: base, gfmAutoLink: false, autoSpace: false, fixTermTypo: false, toc: false, footnotes: false,
          mark: false, sanitize: true, paragraphBeginningSpace: false, codeBlockPreview: true, mathBlockPreview: true },
      },
      input: () => refresh(),
      after: () => {
        vd.vditor.lute.SetUnorderedListMarker('-');
        baseline = current();
        window.__protoVd = vd;   // 原型：方便在控制台里看
        refresh();
      },
    });
  }, err => refresh(err.message));

  return { dirty };
}
