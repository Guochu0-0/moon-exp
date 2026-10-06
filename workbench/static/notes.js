// Notes 编辑器：Vditor 4.0.0 的 IR 模式（光标进入时露出 Markdown 源码，和 Typora 一样），键位按 Typora。
// Vditor 放在 static/vendor/vditor-4.0.0/，第一次打开 Notes 时才加载，不访问外网。
// 写回时 Vditor 会规整写法（表格补齐空格、表格前加空行、无序列表统一 `-`），GitHub 上渲染不变。

const VD = '/static/vendor/vditor-4.0.0';

let loading;
function loadVditor() {
  return loading ??= new Promise((ok, fail) => {
    // notes.css 放在 Vditor 自己的样式后面，才能盖掉它
    for (const href of [`${VD}/dist/index.css`, '/static/notes.css']) {
      const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = href; document.head.append(css);
    }
    const js = document.createElement('script'); js.src = `${VD}/dist/index.min.js`;
    js.onload = () => ok(window.Vditor); js.onerror = () => { loading = null; fail(new Error('Vditor 加载失败')); };
    document.head.append(js);
  });
}

// body 里放编辑区和底栏。save(text) 写盘，失败时抛错。保存要手动（按钮或 Ctrl+S），不自动保存。
export function mountNotes(body, { text, base, path, save }) {
  let saved = text.replace(/\r\n?/g, '\n'), baseline = null, source = false, busy = false, vd = null;
  body.innerHTML = `<div class="notes-vd"></div>
    <textarea class="notes-src" spellcheck="false" aria-label="Notes（Markdown 源码）" hidden></textarea>
    <div class="notes-foot"><button class="save">保存</button><button class="revert">撤销改动</button><button class="mode" aria-pressed="false">源码</button><span class="status"></span></div>`;
  const host = body.querySelector('.notes-vd'), ta = body.querySelector('.notes-src'), $ = s => body.querySelector(`.notes-foot .${s}`);

  // Lute 的 SetUnorderedListMarker 管不到 IR：无序列表符号记在 DOM 的 data-marker 上，取值前统一改成 -
  const dash = () => host.querySelectorAll('.vditor-ir [data-marker="*"], .vditor-ir [data-marker="+"]').forEach(e => e.setAttribute('data-marker', '-'));
  const current = () => source ? ta.value.replace(/\r\n?/g, '\n') : vd ? (dash(), vd.getValue()) : saved;
  // 「改过」按 Vditor 规整后的文本比：载入后取一次值作为基准，免得一打开就因为规整显示未保存
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
    if (source) { ta.value = saved; baseline = saved; grow(); } else { vd.setValue(saved, true); baseline = current(); }
    refresh();
  }
  // 切换源码时页面不滚动：记下滚动位置，聚焦时不让浏览器把光标滚进视野，切换后再恢复
  const scroller = () => {
    for (let e = body.parentElement; e; e = e.parentElement) if (e.scrollHeight > e.clientHeight && /auto|scroll/.test(getComputedStyle(e).overflowY)) return e;
    return document.scrollingElement;
  };
  const grow = () => { ta.style.height = 'auto'; ta.style.height = `${Math.max(260, ta.scrollHeight + 8)}px`; };
  function toggle() {
    if (!vd) return;
    const sc = scroller(), top = sc.scrollTop;
    if (source) {   // 源码里没改过（或刚保存、撤销过）就以规整后的文本为基准，免得切回来就显示未保存
      const same = !dirty();
      vd.setValue(ta.value.replace(/\r\n?/g, '\n'));
      if (same) baseline = current();
    } else ta.value = current();
    source = !source;
    host.hidden = source; ta.hidden = !source;
    $('mode').setAttribute('aria-pressed', source);
    if (source) { grow(); ta.setSelectionRange(0, 0); ta.focus({ preventScroll: true }); }
    else irEl()?.focus({ preventScroll: true });
    sc.scrollTop = top;
    requestAnimationFrame(() => { sc.scrollTop = top; });
    refresh();
  }

  // ---- Typora 键位：捕获阶段先于 Vditor 处理；命令借隐藏工具栏的按钮执行 ----
  // 按键用 event.code 判断，避免 Shift 改变 event.key（Ctrl+Shift+` 的 key 是 ~）
  // IR 编辑区的根节点本身是 <pre class="vditor-reset">，找代码块时要排除它
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
  const level = () => { const h = at('h1,h2,h3,h4,h5,h6'); return h ? +h.nodeName[1] : 0; };
  function heading(k) {
    if (k <= 0) {   // 变回段落：Vditor 里对「当前级别」的标题按钮再按一次就取消
      if (!level()) return true;
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
  function keydown(ev) {
    if (passing || ev.isComposing || ev.keyCode === 229) return;   // 输入法组字时的按键全交给输入法
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
    else if (is(true, false, false, 'Equal')) act = () => at('table') ? vditorKey('=', 'Equal') : level() > 1 && heading(level() - 1);
    else if (is(true, false, false, 'Minus')) act = () => at('table') ? vditorKey('-', 'Minus') : heading(level() ? (level() === 6 ? 0 : level() + 1) : 0);
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
  ta.addEventListener('input', () => { grow(); refresh(); });

  // 编辑区里单击是放光标，Ctrl+单击才打开链接（IR 里链接是一串 span，地址在 marker--link 里）
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
  refresh('载入编辑器…');

  const off = name => ({ name, hotkey: '' });   // 工具栏隐藏，只借它的命令；自带热键清空，由上面的 Typora 键位接管
  loadVditor().then(Vditor => {
    if (!host.isConnected) return;   // 加载期间已经离开节点页
    vd = new Vditor(host, {
      mode: 'ir', value: saved, cdn: VD, lang: 'zh_CN', icon: 'ant', minHeight: 220, placeholder: '（空）',
      cache: { enable: false }, counter: { enable: false }, toolbarConfig: { hide: true, pin: false },
      toolbar: ['headings', 'bold', 'italic', 'strike', 'link', 'list', 'ordered-list', 'check', 'outdent', 'indent',
        'quote', 'line', 'code', 'inline-code', 'table', 'undo', 'redo'].map(off),
      hint: { emoji: {} },
      // 单击只放光标：Vditor 默认单击链接就按原始地址打开、单击图片弹出大图，都关掉；Ctrl+单击见上面
      link: { isOpen: false }, image: { isPreview: false },
      preview: {
        hljs: { enable: false }, math: { engine: 'KaTeX', inlineDigit: false },
        // 关 GFM 自动链接：不关的话 AUC@5 会被写坏成 AUC@5mailto:AUC@5
        markdown: { linkBase: base, gfmAutoLink: false, autoSpace: false, fixTermTypo: false, toc: false, footnotes: false,
          mark: false, sanitize: true, paragraphBeginningSpace: false, codeBlockPreview: true, mathBlockPreview: true },
      },
      input: () => refresh(),
      after: () => {
        vd.vditor.lute.SetUnorderedListMarker('-');
        baseline = current();
        refresh();
      },
    });
  }, err => refresh(err.message));

  return { dirty };
}
