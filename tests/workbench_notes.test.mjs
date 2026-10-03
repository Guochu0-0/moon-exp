// notes.js 的 Markdown 渲染与反向序列化（所见即所得编辑靠它把 DOM 写回 notes.md）。
// 由 test_workbench_notes.py 用 `node --test` 跑；DOM 用下面的小解析器模拟，只够解析 markdown() 自己的输出。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { markdown, toMarkdown } from '../workbench/static/notes.js';

const VOID = new Set(['BR', 'HR', 'IMG']);
const ENT = { amp: '&', lt: '<', gt: '>', quot: '"', nbsp: ' ' };
const unent = s => s.replace(/&(\w+);/g, (_, e) => ENT[e]);

function el(name, attrs = {}) {
  return { nodeType: 1, nodeName: name, attrs, childNodes: [], getAttribute: k => attrs[k] ?? null,
    get textContent() { return this.childNodes.map(c => c.textContent).join(''); } };
}
function text(s) { return { nodeType: 3, nodeValue: s, get textContent() { return this.nodeValue; } }; }

function parse(html) {
  const root = el('DIV'), stack = [root];
  for (const [, close, name, attrs, txt] of html.matchAll(/<(\/?)([a-zA-Z0-9]+)([^>]*)>|([^<]+)/g)) {
    const top = stack[stack.length - 1];
    if (txt !== undefined) { top.childNodes.push(text(unent(txt))); continue; }
    const N = name.toUpperCase();
    if (close) { assert.equal(stack.pop().nodeName, N, `标签没有配对：${html}`); continue; }
    const a = {};
    for (const [, k, v] of attrs.matchAll(/([\w-]+)="([^"]*)"/g)) a[k] = unent(v);
    const e = el(N, a);
    top.childNodes.push(e);
    if (!VOID.has(N)) stack.push(e);
  }
  assert.equal(stack.length, 1, `标签没有闭合：${html}`);
  return root;
}

const BASE = '/runs/E1/';
const render = src => markdown(src, BASE);
const back = src => toMarkdown(parse(render(src)));
// 往返：渲染 → 写回 → 再渲染，结果不变
const roundTrip = src => assert.equal(render(back(src)), render(src), `往返后变了：\n${back(src)}`);

test('GFM 表格：表头、对齐、单元格内行内格式、转义的竖线、缺格补空', () => {
  const html = render('| 方法 | Val | 备注 |\n|:---|---:|:-:|\n| **M1** | 0.306 | a \\| b |\n| M2 | 0.300 |\n');
  assert.equal(html, '<table><thead><tr><th data-align="left">方法</th><th data-align="right">Val</th><th data-align="center">备注</th></tr></thead>'
    + '<tbody><tr><td data-align="left"><strong>M1</strong></td><td data-align="right">0.306</td><td data-align="center">a | b</td></tr>'
    + '<tr><td data-align="left">M2</td><td data-align="right">0.300</td><td data-align="center"></td></tr></tbody></table>');
});

test('表格：不写首尾竖线也算；遇到空行结束；分隔行列数不符时不是表格', () => {
  assert.match(render('a | b\n--|--\n1 | 2\n\n后文'), /^<table>.*<td>2<\/td><\/tr><\/tbody><\/table>\n<p>后文<\/p>$/s);
  assert.equal(render('| a | b |\n|---|\n'), '<p>| a | b |\n|---|</p>');
});

test('表格可以紧跟在段落后面', () => {
  assert.match(render('标签说明：\n| a | b |\n|---|---|\n| 1 | 2 |'), /^<p>标签说明：<\/p>\n<table>/);
});

test('实验 M 的表格能渲染出来', () => {
  const src = '标签：`p2`。\n\n| 方法 | 路线 | 改动 | 峰值 step | Val AUC@5 | Test AUC@5 |\n|---|---|---|---|---|---|\n'
    + '| M1 | 伪标签 | p2，geo+photo 扰动 | 5000 | 0.306 | 0.283 |\n| M2 | 伪标签 | p2，扰动，lr 3e-6 | 8000 | 0.300 | |\n';
  const html = render(src);
  assert.equal((html.match(/<tr>/g) || []).length, 3);
  assert.match(html, /<td>0\.300<\/td><td><\/td><\/tr>/);
});

test('代码块保留语言，链接和图片保留原始相对路径', () => {
  assert.equal(render('```python\nx = 1\n```'), '<pre data-lang="python"><code>x = 1</code></pre>');
  assert.equal(render('![图](extra/a.png) [表](extra/b.csv)'),
    '<p><img src="/runs/E1/extra/a.png" data-src="extra/a.png" alt="图" loading="lazy"> <a href="/runs/E1/extra/b.csv" data-href="extra/b.csv">表</a></p>');
});

test('写回 Markdown：各种块', () => {
  assert.equal(back('# 标题\n\n段落 **粗** *斜* `码`\n\n- a\n- b\n\n1. x\n2. y\n\n> 引\n\n---\n\n```py\nq\n```'),
    '# 标题\n\n段落 **粗** *斜* `码`\n\n- a\n- b\n\n1. x\n2. y\n\n> 引\n\n---\n\n```py\nq\n```\n');
  assert.equal(back('| a | b |\n|:--|--:|\n| 1 | x \\| y |'), '| a | b |\n|:--|--:|\n| 1 | x \\| y |\n');
  assert.equal(back('| a | b |\n|---|---|\n| 1 | |'), '| a | b |\n|---|---|\n| 1 | |\n');
  assert.equal(back('![图](extra/a.png)'), '![图](extra/a.png)\n');
  assert.equal(back(''), '');
});

test('往返不改变渲染结果', () => {
  for (const src of [
    '## 记录来源\n\n本实验在 #76 之前跑完：\n\n- 代码：用 `git archive` 副本跑，M1–M6 为 8457109。\n- 任务清单：`a.txt`\n  第二行续写',
    '中文段落\n接着写\n\nEnglish\nsoft break\n\n行尾两个空格  \n换行',
    '> 引用里有 **粗体**\n>\n> 第二段\n\n```\n<b>不是 HTML</b>\n```',
    '| 方法 | 改动 |\n|---|---|\n| M1 | `--cert-out`，a__b |\n| M2 | [链接](https://x.org) |',
    '方法名 roma__v 和 a*b 里的符号不是格式；<b>原样</b> & 符号',
  ]) roundTrip(src);
});

test('写回时会处理编辑器产生的结构：b/i、div、首尾空白在粗体外、不换行空格', () => {
  const dom = parse('<div>前<b> 粗 </b>后&nbsp;<i>斜</i></div><div><br></div><h4>小标题</h4><ul><li>一<ul><li>二</li></ul></li></ul>');
  assert.equal(toMarkdown(dom), '前 **粗** 后 *斜*\n\n## 小标题\n\n- 一\n- 二\n');
});
