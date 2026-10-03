// Notes：Markdown 渲染、反向序列化（DOM → Markdown），以及节点页上所见即所得的编辑器。
// 编辑器就是渲染结果本身（contenteditable）：输入 Markdown 语法（`# `、`- `、`**粗**`、`| a | b |` 回车……）当场变成格式，
// 保存时把 DOM 写回 Markdown。复杂的改动（表格增删列、图片、代码块语言）用「源码」切到纯文本编辑。

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---------------- Markdown → HTML ----------------
// 只覆盖记录里会用到的子集：标题、段落、列表、引用、代码块、分隔线、表格，以及行内代码、粗体、斜体、链接、图片。
// 先整体转义再生成标签，不放行任何原始 HTML；相对路径按实验目录解析（附件如 extra/offset/offset_val.png）。
// 写回 Markdown 要用的信息放在 data-* 上：原始相对路径、代码块语言、表格对齐。
export function markdown(src, base) {
  const lines = src.replace(/\r\n?/g, '\n').split('\n'), out = [];
  let i = 0;
  const para = [];
  const flush = () => { if (para.length) { out.push(`<p>${inline(join(para), base)}</p>`); para.length = 0; } };
  while (i < lines.length) {
    const l = lines[i];
    let m;
    if ((m = l.match(/^\s*(```|~~~)\s*([^\s`]*)/))) {
      flush();
      const fence = m[1], code = [];
      for (i++; i < lines.length && !lines[i].trimStart().startsWith(fence); i++) code.push(lines[i]);
      out.push(`<pre${m[2] ? ` data-lang="${esc(m[2])}"` : ''}><code>${esc(code.join('\n'))}</code></pre>`); i++; continue;
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
    if (l.includes('|') && DELIM.test(lines[i + 1] ?? '')) {
      const head = cells(l), aligns = cells(lines[i + 1]).map(c => ({ '::': 'center', ':-': 'left', '-:': 'right' }[c[0] + c.at(-1)] ?? ''));
      if (head.length === aligns.length) {
        flush();
        const rows = [];
        for (i += 2; i < lines.length && lines[i].trim() && lines[i].includes('|'); i++) rows.push(cells(lines[i]));
        const row = (cs, tag) => `<tr>${aligns.map((a, k) => `<${tag}${a ? ` data-align="${a}"` : ''}>${inline(cs[k] ?? '', base)}</${tag}>`).join('')}</tr>`;
        out.push(`<table><thead>${row(head, 'th')}</thead><tbody>${rows.map(r => row(r, 'td')).join('')}</tbody></table>`);
        continue;
      }
    }
    para.push(l); i++;
  }
  flush();
  return out.join('\n');
}

// GFM 表格：分隔行；单元格按没转义的 | 切开，首尾的 | 可省
const DELIM = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
function cells(l) {
  let s = l.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
  return s.split(/(?<!\\)\|/).map(c => c.trim().replace(/\\\|/g, '|'));
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
    const u = url(src, base); return u ? hold(`<img src="${esc(u)}" data-src="${esc(src)}" alt="${esc(alt)}" loading="lazy">`) : all;
  });
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (all, t, href) => {
    const u = url(href, base);
    return u ? hold(`<a href="${esc(u)}" data-href="${esc(href)}"${/^https?:/i.test(u) ? ' target="_blank" rel="noopener"' : ''}>${inlineText(t)}</a>`) : all;
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

// ---------------- DOM → Markdown ----------------
// 把 markdown() 的输出（以及编辑器里浏览器顺手生成的 b/i/div/嵌套列表）写回 Markdown。
// 不加转义：渲染器本来就不认转义，正文里的 `**` 之类写回去会再次成为格式，这正是边打边渲染要的效果。
// caret 给出时在光标处插入 MARK，编辑器据此在重新渲染后找回光标。
const MARK = '\uE000', ZW = '\u200b';
const BLOCKS = new Set(['P', 'DIV', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'UL', 'OL', 'LI', 'BLOCKQUOTE', 'PRE', 'HR', 'TABLE',
  'THEAD', 'TBODY', 'TR', 'TH', 'TD', 'SECTION', 'FIGURE']);
const isBlock = n => n.nodeType === 1 && BLOCKS.has(n.nodeName);
const isList = n => n.nodeType === 1 && (n.nodeName === 'UL' || n.nodeName === 'OL');
let CARET = null;

export function toMarkdown(root, caret = null) {
  CARET = caret;
  try {
    const s = blocks(kids(root)).join('\n\n');
    return s ? s + '\n' : '';
  } finally { CARET = null; }
}

function kids(n) {
  const k = [...n.childNodes];
  if (CARET && CARET.node === n) k.splice(CARET.offset, 0, { nodeType: 3, nodeValue: MARK, childNodes: [] });
  return k;
}

function blocks(nodes) {
  const out = [], run = [];
  const flush = () => { const t = para(run); if (t) out.push(t); run.length = 0; };
  for (const n of nodes) {
    if (!isBlock(n)) { run.push(n); continue; }
    flush();
    const b = block(n);
    if (b) out.push(b);
  }
  flush();
  return out;
}

const para = nodes => inl(nodes).replace(/^[ \t\n]+|[ \t\n]+$/g, '');

function block(n) {
  const N = n.nodeName;
  if (/^H[1-6]$/.test(N)) { const t = para(kids(n)).replace(/ *\n/g, ' '); return t && `${'#'.repeat(Math.max(1, +N[1] - 2))} ${t}`; }
  if (N === 'HR') return '---';
  if (N === 'PRE') {
    const code = raw(n).replace(/\n$/, ''), fence = /^\s*```/m.test(code) ? '~~~' : '```';
    return `${fence}${n.getAttribute('data-lang') || ''}\n${code}\n${fence}`;
  }
  if (isList(n)) return items(n).map((t, k) => `${N === 'OL' ? `${k + 1}.` : '-'} ${t.replace(/\n/g, '\n  ')}`).join('\n');
  if (N === 'BLOCKQUOTE') return blocks(kids(n)).join('\n\n').split('\n').map(l => l ? `> ${l}` : '>').join('\n').replace(/^>$/, '');
  if (N === 'TABLE') return table(n);
  return blocks(kids(n)).join('\n\n');   // P、DIV，以及落单的 LI、TD 等
}

// 列表不支持嵌套：浏览器（Tab、粘贴）造出的子列表摊平成同级的项
function items(list) {
  const out = [];
  for (const li of kids(list)) {
    if (isList(li)) { out.push(...items(li)); continue; }
    if (li.nodeType !== 1) { const t = para([li]); if (t) out.push(t); continue; }
    const k = kids(li), t = blocks(k.filter(c => !isList(c))).join('\n');
    if (t) out.push(t);
    for (const s of k.filter(isList)) out.push(...items(s));
  }
  return out;
}

function table(t) {
  const rows = [];
  const walk = e => { for (const c of kids(e)) if (c.nodeType === 1) c.nodeName === 'TR' ? rows.push(c) : walk(c); };
  walk(t);
  if (!rows.length) return '';
  const cellsOf = tr => kids(tr).filter(c => c.nodeType === 1 && (c.nodeName === 'TD' || c.nodeName === 'TH'));
  const n = Math.max(...rows.map(r => cellsOf(r).length));
  const text = c => c ? para(kids(c)).replace(/ *\n/g, ' ').replace(/\|/g, '\\|') : '';
  const line = r => { const cs = cellsOf(r); return `|${Array.from({ length: n }, (_, k) => ` ${text(cs[k])} `.replace(/^  $/, ' ')).join('|')}|`; };
  const head = cellsOf(rows[0]);
  const delim = Array.from({ length: n }, (_, k) => ({ left: ':--', right: '--:', center: ':-:' }[head[k]?.getAttribute('data-align')] ?? '---'));
  return [line(rows[0]), `|${delim.join('|')}|`, ...rows.slice(1).map(line)].join('\n');   // 和手写的表格同一种写法
}

// 代码块里只取文字（浏览器可能在里面插 <br>）
const raw = n => kids(n).map(c => c.nodeType === 3 ? one(c) : c.nodeName === 'BR' ? '\n' : c.nodeType === 1 ? raw(c) : '').join('');

function inl(nodes) {
  let s = '';
  for (const n of nodes) s += one(n);
  return s;
}

function one(n) {
  if (n.nodeType === 3) {
    let t = n.nodeValue;
    if (CARET && CARET.node === n) t = t.slice(0, CARET.offset) + MARK + t.slice(CARET.offset);
    return t.replace(/\u00a0/g, ' ').replaceAll(ZW, '');
  }
  if (n.nodeType !== 1) return '';
  const wrap = m => {
    const [, a, body, b] = inl(kids(n)).match(/^(\s*)([^]*?)(\s*)$/);
    return body.replaceAll(MARK, '') ? `${a}${m}${body}${m}${b}` : a + body + b;
  };
  switch (n.nodeName) {
    case 'BR': return '  \n';
    case 'STRONG': case 'B': return wrap('**');
    case 'EM': case 'I': return wrap('*');
    case 'CODE': { const t = inl(kids(n)); return t.replaceAll(MARK, '') ? `\`${t}\`` : t; }
    case 'A': return `[${inl(kids(n))}](${n.getAttribute('data-href') ?? n.getAttribute('href') ?? ''})`;
    case 'IMG': return `![${n.getAttribute('alt') ?? ''}](${n.getAttribute('data-src') ?? n.getAttribute('src') ?? ''})`;
    default: return isBlock(n) ? `\n${blocks(kids(n)).join('\n')}\n` : inl(kids(n));
  }
}

// ---------------- 编辑器 ----------------
// body 里放编辑区和底栏。save(text) 写盘，失败时抛错。保存要手动（按钮或 Ctrl+S），不自动保存。
// 撤销自己做：边打边渲染会替换 DOM，浏览器自带的撤销栈接不上。快照是带光标标记的 Markdown。
export function mountNotes(body, { text, base, path, save }) {
  let saved = text.replace(/\r\n?/g, '\n'), source = false, composing = false, busy = false, lastSnap = 0;
  const undo = [], redo = [];
  body.innerHTML = `<div class="notes md" contenteditable="true" spellcheck="false" role="textbox" aria-multiline="true" aria-label="Notes"></div>
    <textarea class="notes-src" spellcheck="false" aria-label="Notes（Markdown 源码）" hidden></textarea>
    <div class="notes-foot"><button class="save">保存</button><button class="revert">撤销改动</button><button class="mode" aria-pressed="false">源码</button><span class="status"></span></div>`;
  const ed = body.querySelector('.notes'), ta = body.querySelector('.notes-src'), $ = s => body.querySelector(`.notes-foot .${s}`);
  const render = md => markdown(md, base);
  const current = () => source ? ta.value.replace(/\r\n?/g, '\n') : toMarkdown(ed);
  let savedHtml = render(saved);   // 每次输入都要比，存一份
  const dirty = () => source ? current() !== saved : render(current()) !== savedHtml;

  function fill(md) {
    ed.innerHTML = render(md) || '<p><br></p>';
    for (const c of ed.querySelectorAll('pre > code')) if (c.textContent.endsWith('\n')) c.append('\n');   // 见 codeEnter
    place(ed);
  }
  function refresh(msg) {
    const d = dirty();
    $('save').disabled = !d || busy; $('revert').disabled = !d;
    ed.classList.toggle('blank', !toMarkdown(ed));
    $('status').textContent = msg ?? (d ? `未保存 · Ctrl+S 保存到 ${path}` : path);
  }

  // ---- 光标 ----
  const caret = () => {
    const s = getSelection();
    return s.rangeCount && ed.contains(s.anchorNode) ? { node: s.anchorNode, offset: s.anchorOffset } : null;
  };
  function setCaret(node, offset) {
    const r = document.createRange(); r.setStart(node, offset); r.collapse(true);
    const s = getSelection(); s.removeAllRanges(); s.addRange(r);
  }
  // 找到 MARK，删掉并把光标放在那里；返回是否找到
  function place(root) {
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let t; (t = w.nextNode());) {
      const k = t.nodeValue.indexOf(MARK);
      if (k < 0) continue;
      t.nodeValue = t.nodeValue.replace(MARK, '');
      if (t.nodeValue) { setCaret(t, k); return true; }
      const p = t.parentNode;
      if (p.childNodes.length > 1) { t.nodeValue = ZW; setCaret(t, 1); }   // 紧跟在粗体等后面：留个零宽字符，免得光标被吸进去
      else { const br = document.createElement('br'); p.replaceChild(br, t); setCaret(p, 0); }
      return true;
    }
    return false;
  }
  const empty = () => { const p = document.createElement('p'); p.innerHTML = '<br>'; return p; };
  const blockOf = n => { while (n && n.parentNode !== ed) n = n.parentNode; return n; };
  const inside = (n, ...names) => { for (; n && n !== ed; n = n.parentNode) if (names.includes(n.nodeName)) return n; return null; };
  const offsetIn = (el, node, offset) => { const r = document.createRange(); r.setStart(el, 0); r.setEnd(node, offset); return r.toString().length; };

  // ---- 撤销 ----
  const snap = () => toMarkdown(ed, caret() ?? { node: ed, offset: ed.childNodes.length });
  function remember(force) {
    const now = Date.now();
    if (force || now - lastSnap > 800) { undo.push(snap()); if (undo.length > 200) undo.shift(); redo.length = 0; }
    lastSnap = now;
  }
  function history(from, to) {
    if (!from.length) return;
    to.push(snap()); fill(from.pop()); lastSnap = 0; refresh();
  }

  // ---- 边打边渲染：光标所在的块按 Markdown 重新渲染，结构变了才替换 ----
  const sig = nodes => [...nodes].map(n => n.nodeType === 3 ? n.nodeValue.replace(/[\uE000\u200b]/g, '').replace(/[\s\u00a0]+/g, ' ').trim()
    : n.nodeType !== 1 || (n.nodeName === 'BR' && n === n.parentNode?.lastChild) ? '' : `<${n.nodeName}>${sig(n.childNodes)}</>`).join('');
  function live() {
    const c = caret(); if (!c || !getSelection().isCollapsed) return;
    let blk = blockOf(c.node);
    if (blk && !isBlock(blk)) { blk = loose(blk); setCaret(c.node, c.offset); }
    if (!blk || blk.nodeName === 'PRE') return;
    let md;
    CARET = c;
    try { md = blocks([blk]).join('\n\n'); } finally { CARET = null; }
    if (/^\s*(```|~~~)/.test(md)) return;   // 代码块等回车再变，免得还在打语言名就成了代码块
    const tpl = document.createElement('template');
    tpl.innerHTML = render(md);
    if (sig(tpl.content.childNodes) === sig([blk])) return;
    remember(true);
    replace(blk, tpl.content);
  }
  // 直接落在编辑区里的文字（空白 Notes 里打的第一个字常这样）连同相邻的行内节点收进一个段落
  function loose(n) {
    const run = [n];
    for (let s = n.previousSibling; s && !isBlock(s); s = s.previousSibling) run.unshift(s);
    for (let s = n.nextSibling; s && !isBlock(s); s = s.nextSibling) run.push(s);
    const p = document.createElement('p');
    run[0].before(p); p.append(...run);
    return p;
  }
  function replace(blk, frag) {
    for (const t of [...frag.childNodes]) if (t.nodeType === 3 && !t.nodeValue.trim()) t.remove();
    if (!frag.childNodes.length) frag.append(empty());
    const last = frag.lastChild;
    blk.replaceWith(frag);
    if (!place(ed)) setCaret(last, last.childNodes.length);
  }

  // ---- 回车 ----
  function enter(ev) {
    const c = caret(); if (!c) return;
    const pre = inside(c.node, 'PRE'), cell = inside(c.node, 'TD', 'TH');
    if (pre) { ev.preventDefault(); remember(true); codeEnter(pre); return; }
    if (cell) { ev.preventDefault(); moveCell(cell, 'down'); return; }
    const blk = blockOf(c.node);
    if (!blk || blk.nodeName !== 'P') return;
    const t = blk.textContent.replaceAll(ZW, '').trim();
    let md = null;
    if (/^(```|~~~)[^\s`]*$/.test(t)) md = `${t}\n${MARK}\n${t.slice(0, 3)}`;                     // 代码块
    else if (/^([-*_])(\s*\1){2,}$/.test(t)) md = `---\n\n${MARK}`;                               // 分隔线
    else if (/^\|.*\|.*\|$/.test(t)) {                                                          // 表头 → 表格，光标到第一格
      const n = cells(t).length;
      md = `${t}\n|${' --- |'.repeat(n)}\n| ${MARK}${' | '.repeat(n - 1)} |`;
    }
    if (md == null) return;
    ev.preventDefault(); remember(true);
    const tpl = document.createElement('template'); tpl.innerHTML = render(md);
    replace(blk, tpl.content);
    refresh();
  }
  // 代码块按纯文本改：在光标处插入 ins，光标后移 move 个字
  function codeInsert(pre, r, ins, move = ins.length) {
    const off = offsetIn(pre, r.startContainer, r.startOffset), t = pre.textContent, code = pre.querySelector('code') || pre;
    code.textContent = t.slice(0, off) + ins + t.slice(off);
    setCaret(code.firstChild, off + move);
  }
  // 代码块里回车是换行；在末尾的空行上再按一次回车就离开代码块
  function codeEnter(pre) {
    const r = getSelection().getRangeAt(0);
    r.deleteContents();
    const off = offsetIn(pre, r.startContainer, r.startOffset), t = pre.textContent;
    if (t.slice(off).replace(/\n$/, '') === '' && (t[off - 1] === '\n' || !t.replaceAll(ZW, ''))) {
      (pre.querySelector('code') || pre).textContent = t.slice(0, Math.max(0, off - 1)).replaceAll(ZW, '');
      const p = empty(); pre.after(p); setCaret(p, 0);
    } else codeInsert(pre, r, off === t.length ? '\n\n' : '\n', 1);   // 末尾多留一个换行，新行才看得见
    refresh();
  }
  // 表格里：回车到下一行同一列，Tab / Shift+Tab 到下一格 / 上一格；走出最后一行时加一行
  function moveCell(cell, dir) {
    const tr = cell.parentNode, table = tr.closest('table'), rows = [...table.querySelectorAll('tr')];
    const col = [...tr.children].indexOf(cell), row = rows.indexOf(tr);
    let to = dir === 'down' ? rows[row + 1]?.children[col]
      : dir === 'next' ? cell.nextElementSibling || rows[row + 1]?.children[0]
        : cell.previousElementSibling || rows[row - 1]?.lastElementChild;
    if (!to && dir !== 'prev') {
      remember(true);
      const nr = (table.tBodies[0] || table.appendChild(document.createElement('tbody'))).insertRow();
      for (const h of rows[0].children) {
        const td = document.createElement('td'); if (h.dataset.align) td.dataset.align = h.dataset.align;
        td.innerHTML = '<br>'; nr.append(td);
      }
      to = nr.children[dir === 'down' ? col : 0];
      refresh();
    }
    if (!to) return;
    const r = document.createRange(); r.selectNodeContents(to); r.collapse(false);
    getSelection().removeAllRanges(); getSelection().addRange(r);
  }

  // ---- 粘贴：只取纯文本，按 Markdown 并入 ----
  function paste(ev) {
    ev.preventDefault();
    const text = ev.clipboardData.getData('text/plain').replace(/\r\n?/g, '\n');
    if (!text) return;
    remember(true);
    const r = getSelection().getRangeAt(0); r.deleteContents();
    const pre = inside(r.startContainer, 'PRE');
    if (pre) codeInsert(pre, r, text);
    else fill(toMarkdown(ed, { node: r.startContainer, offset: r.startOffset }).replace(MARK, text + MARK));
    refresh();
  }

  async function doSave() {
    if (busy || !dirty()) return;
    const t = current();
    busy = true; refresh('保存中…');
    try {
      await save(t);
    } catch (err) {
      busy = false; refresh(`保存失败：${err.message}`); return;
    }
    busy = false; saved = t; savedHtml = render(t);
    refresh(`已保存到 ${path}`);
  }
  function revert() {
    if (!confirm('放弃未保存的 Notes 改动？')) return;
    if (source) { ta.value = saved; grow(); } else { remember(true); fill(saved); }
    refresh();
  }
  function toggle() {
    if (source) { fill(ta.value.replace(/\r\n?/g, '\n')); undo.length = redo.length = 0; }
    else ta.value = dirty() ? toMarkdown(ed) : saved;   // 没改过就给原文，免得写回时的规整化改动了原来的排版
    source = !source;
    ed.hidden = source; ta.hidden = !source;
    $('mode').setAttribute('aria-pressed', source);
    if (source) { grow(); ta.focus(); } else ed.focus();
    refresh();
  }
  const grow = () => { ta.style.height = 'auto'; ta.style.height = `${Math.max(220, ta.scrollHeight + 8)}px`; };

  const saveKey = ev => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 's') { ev.preventDefault(); doSave(); return true; }
    return false;
  };
  ed.addEventListener('keydown', ev => {
    if (saveKey(ev) || ev.isComposing || composing) return;
    const mod = ev.ctrlKey || ev.metaKey, k = ev.key.toLowerCase();
    if (mod && (k === 'z' || k === 'y')) { ev.preventDefault(); k === 'y' || ev.shiftKey ? history(redo, undo) : history(undo, redo); return; }
    if (ev.key === 'Enter' && !ev.shiftKey && !mod) { enter(ev); return; }
    if (ev.key === 'Tab') {
      const c = caret(), cell = c && inside(c.node, 'TD', 'TH');
      if (cell) { ev.preventDefault(); moveCell(cell, ev.shiftKey ? 'prev' : 'next'); }
      return;
    }
    if (ev.key === 'Backspace' && !mod && getSelection().isCollapsed) {   // 标题开头退格：变回段落
      const c = caret(), blk = c && blockOf(c.node);
      if (!blk || !/^H[1-6]$/.test(blk.nodeName) || offsetIn(blk, c.node, c.offset)) return;
      ev.preventDefault(); remember(true);
      const p = blk.firstChild ? document.createElement('p') : empty(); p.append(...blk.childNodes);
      blk.replaceWith(p); setCaret(p, 0); refresh();
    }
  });
  ed.addEventListener('beforeinput', ev => {
    if (ev.inputType === 'historyUndo' || ev.inputType === 'historyRedo') {
      ev.preventDefault(); ev.inputType === 'historyUndo' ? history(undo, redo) : history(redo, undo); return;
    }
    if (!composing) remember(false);
  });
  ed.addEventListener('input', ev => { if (!ev.isComposing && !composing) live(); refresh(); });
  ed.addEventListener('compositionstart', () => { composing = true; remember(false); });
  ed.addEventListener('compositionend', () => { composing = false; live(); refresh(); });
  ed.addEventListener('paste', paste);
  ed.addEventListener('drop', ev => ev.preventDefault());   // 不接受拖进来的 HTML
  ed.addEventListener('click', ev => {                     // 编辑区里单击是放光标，Ctrl+单击才打开链接
    const a = ev.target.closest('a');
    if (a && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); window.open(a.href, '_blank', 'noopener'); }
  });
  ta.addEventListener('keydown', saveKey);
  ta.addEventListener('input', () => { grow(); refresh(); });
  $('save').addEventListener('click', doSave);
  $('revert').addEventListener('click', revert);
  $('mode').addEventListener('click', toggle);

  document.execCommand('defaultParagraphSeparator', false, 'p');
  fill(saved);
  refresh();
  return { dirty };
}
