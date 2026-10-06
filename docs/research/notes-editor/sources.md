# 来源（2026-10-06 读取）

## Typora 官方支持站

- [S1] Shortcut Keys — https://support.typora.io/Shortcut-Keys/
- [S2] Markdown Reference — https://support.typora.io/Markdown-Reference/
- [S3] Table Editing — https://support.typora.io/Table-Editing/
- [S4] Math and Academic Functions — https://support.typora.io/Math/
- [S5] Whitespace and Line Breaks — https://support.typora.io/Line-Break/
- [S6] What's New 0.10 — https://support.typora.io/What's-New-0.10/
- [S7] Code Fences — https://support.typora.io/Code-Fences/
- [S8] What's New 1.5 — https://support.typora.io/What's-New-1.5/

## Vditor

- [S10] 仓库首页 README — https://github.com/Vanessa219/vditor
- [S11] README（中文，含完整 options 表、CDN 切换说明），npm 包 vditor@4.0.0 内的 `README.md` — https://cdn.jsdelivr.net/npm/vditor@4.0.0/README.md
- [S12] npm registry `vditor/latest`（版本、许可证、依赖）— https://registry.npmjs.org/vditor/latest ；v4.0.0 release notes — https://github.com/Vanessa219/vditor/releases/tag/v4.0.0 ；GitHub API `repos/Vanessa219/vditor`
- [S13] jsDelivr 文件清单（`dist/index.min.js` 297357 B、`dist/js/lute/lute.min.js` 3743299 B、`dist/js/katex/katex.min.js` 277038 B 等）— https://data.jsdelivr.com/v1/packages/npm/vditor@4.0.0?structure=flat ；默认工具栏与快捷键 `src/ts/util/Options.ts`（v4.0.0）
- [S14] 源码（v4.0.0）：`src/ts/ir/expandMarker.ts`（光标进入节点展开标记符）、`src/ts/ir/index.ts`（compositionstart/end）、`src/ts/util/hotKey.ts`（⌘ 映射到 Ctrl）、`src/ts/toolbar/Headings.ts`（标题 ⌥⌘1…6）
- [S15] 源码（v4.0.0）：`src/ts/util/fixBrowserBehavior.ts`（fixList、fixBlockquote、fixTask、fixTable、fixCodeBlock、fixCJKPosition 等）、`src/ts/ir/processKeydown.ts`
- [S16] Lute：`lute.go`（全部 `Set*` 开关，无关闭 HTML 解析的开关）、`render/renderer.go`（`UnorderedListMarker` 默认 `*`，`SoftBreak2HardBreak` 默认 true）、`parse/parse.go`（`GFMStrikethrough1` 单个 `~`）、`LICENSE`（木兰宽松许可证第 2 版）— https://github.com/88250/lute

## Milkdown

- [S17] npm registry `@milkdown/crepe/latest`（7.22.2，MIT，只有 ESM/CJS 导出）；GitHub API `repos/Milkdown/milkdown`
- [S18] 文档 Keyboard Shortcuts — https://github.com/Milkdown/website/blob/main/docs/guide/keyboard-shortcuts.md （即 https://milkdown.dev/docs/guide/keyboard-shortcuts ）
- [S19] 文档 Getting Started、FAQ（remark-stringify 选项、`getMarkdown()`）— https://github.com/Milkdown/website/tree/main/docs/guide
- [S20] 文档 Using Crepe（功能列表）— https://github.com/Milkdown/website/blob/main/docs/guide/using-crepe.md
- [S21] 源码 `packages/crepe/src/feature/latex/`（`index.ts`、`remark.ts`、`block-latex.ts`、`input-rule.ts`）— https://github.com/Milkdown/milkdown
- 实测：`https://esm.sh/@milkdown/crepe@7.22.2/es2022/crepe.bundle.mjs` 下载 2830307 B，gzip -9 后 936742 B，唯一外部引用 `/node/process.mjs`

## Tiptap

- [S22] npm registry `@tiptap/core`、`@tiptap/markdown`、`@tiptap/extension-mathematics`、`@tiptap/extension-list`、`@tiptap/starter-kit`（均 3.31.4，MIT）；GitHub API `repos/ueberdosis/tiptap`
- [S23] Keyboard shortcuts — https://tiptap.dev/docs/editor/core-concepts/keyboard-shortcuts
- [S24] Markdown — https://tiptap.dev/docs/editor/markdown ；源码 `packages/markdown/src/MarkdownManager.ts`（html token 走 `parseHTMLToken`，失败退回文字）
- [S25] CDN 安装 — https://tiptap.dev/docs/editor/getting-started/install/cdn
- [S26] 源码 `packages/extension-mathematics/src/extensions/InlineMath.ts`、`BlockMath.ts`（`renderMarkdown`、`markdownTokenizer`）；`packages/extension-list/src/*`；文档 https://tiptap.dev/docs/editor/extensions/nodes/mathematics
- 实测：`https://esm.sh/@tiptap/core@3.31.4?bundle` 仍外链 `@tiptap/pm` 的 8 个子模块；`@tiptap/starter-kit@3.31.4?bundle` 是独立的一个 bundle

## 其他

- [S27] KaTeX 0.19.0（MIT）jsDelivr 文件清单：`katex.min.js` 272868 B、`katex.mjs` 602889 B、`katex.min.css` 24793 B、20 个 woff2 字体共 259792 B — https://data.jsdelivr.com/v1/packages/npm/katex@0.19.0?structure=flat
- [S28] GitHub Docs: Writing mathematical expressions — https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions
- Issue 搜索（GitHub API，2026-10-06）：Vanessa219/vditor #1940、#1922；Milkdown/milkdown #1542；ueberdosis/tiptap #7135、#7271
