# Notes 编辑器调研：Typora 的行为与快捷键，候选编辑器库比较

> 调研日期：2026-10-06 · 对应 [#108](https://github.com/Guochu0-0/moon-exp/issues/108)，属于地图 [#1](https://github.com/Guochu0-0/moon-exp/issues/1)
> 标注规则：**「文档」** 指直接读自官方文档、官方仓库源码、npm registry / jsDelivr / GitHub API；**「推断」** 是我的判断；**「待实测」** 是官方没写、需要在 Typora 或原型里按一下才能确定的。来源编号见 [sources.md](sources.md)。

## 结论（推荐）

**推荐：先用 Vditor 的 IR（即时渲染）模式做一个一页的原型，过了下面「原型必须验证的 8 点」就换掉手写编辑器；有任何一点过不了，退回继续手写，并引入 KaTeX 预编译文件来渲染公式。Milkdown 和 Tiptap 不推荐。**

理由（推荐部分是我的判断，依据都标在后面各节）：

1. 这次的核心要求是「光标进入行内元素时露出 Markdown 源码」。四个选项里只有 Vditor IR 和手写能做到：Vditor IR 就是照 Typora 的方式设计的，源码里有专门的「光标进入节点就展开标记符」逻辑（`expandMarker.ts`，[S14]）。Milkdown 和 Tiptap 都基于 ProseMirror，是「输入规则一触发就变成格式、标记符消失」的所见即所得，官方文档里没有露出源码的功能（[S20][S24]，没找到），要做得自己写节点视图，工作量不比手写少。
2. Vditor 有官方打包好的 `dist/`（`index.min.js` 是可以直接用 `<script>` 加载的 UMD），按需加载的子文件（Lute 解析器、KaTeX、语言包、图标）都从 `options.cdn` 指定的目录读，官方 README 写明可以把 `dist` 目录拷到自己的位置、把 `cdn` 指过去（[S11]）。这正好满足「放进 `workbench/static/vendor/`、不走 CDN、不加构建」。Tiptap 没有单文件包；Milkdown 只能借 esm.sh 生成一个 2.8 MB 的单文件（下文实测），都不如 Vditor 正。
3. Vditor 的代价：体积大（Lute 解析器 3.7 MB，gzip 后约 580 KB）；默认快捷键和 Typora 不同（标题是 Ctrl+Alt+1…6，表格加行是 Ctrl+=，和 Typora 的「标题升级」冲突）；不能彻底关掉原始 HTML（只有 XSS 过滤）；保存出来的 Markdown 经过 Lute 重新格式化（无序列表默认写成 `*`）。这些都要原型确认能不能压住，所以推荐是有条件的。
4. 继续手写的好处是现有约定全部天然保留、零依赖、体积小；坏处是嵌套列表、任务列表、源码露出、输入法、撤销这几块每块都是坑，Vditor 的 `fixBrowserBehavior.ts` 一个文件就有 1600 多行专门修浏览器行为（[S15]），可以看出量级。作为退路合适，作为首选风险更大。（推断）

### 原型必须验证的 8 点（任何一点不过就退回手写）

1. 把现有 20 份 `runs/*/notes.md` 读进 Vditor 再原样 `getValue()` 写出，`git diff` 里只出现可接受的规整（列表符号、表格空格、空行），GitHub 渲染不变。
2. 无序列表符号能改成 `-`（Lute 有 `SetUnorderedListMarker`，[S16]，但 Vditor 的选项表里没暴露，需要拿到内部的 lute 实例去设）。
3. 快捷键能改成 Typora 的 Windows 键位：工具栏按钮的 `hotkey` 可以配置（[S11]），但表格、列表里的一些键是写死在 `fixBrowserBehavior.ts` 里的（Ctrl+= 加行、Ctrl+- 删行，[S15]），要看能不能用 `keydown` 回调先拦下来。
4. 相对路径：`preview.markdown.linkBase`（[S11]）设成实验目录后，图片和链接都按实验目录解析；Ctrl+单击打开链接要自己加一个 click 监听。
5. `#` 标题显示时降两级：Vditor 里只能靠 CSS 把 h1 画成 h3 的样子（不改 DOM 层级）。看看是否可接受。（推断）
6. 中文输入法：在列表项、表格格子、标题、行内代码、公式里连续输入中文，不吞字、不重复、光标不跳。
7. `cache.enable` 设为 `false`（默认是 `true`，会把内容存进 localStorage，[S11]），确认保存仍然只走我们自己的手动保存。
8. 原始 HTML：确认粘贴或输入 `<b>x</b>` 时的表现可接受（Vditor 会把它当 HTML 渲染，只做 XSS 过滤，[S11][S16]）。现有 20 份 Notes 里没有任何 HTML 标签、`$`、嵌套列表或 `*` 列表符号（2026-10-06 用 grep 查过），所以只影响以后新写的内容。

---

## 一、Typora 的行为与快捷键清单

### 1.1 快捷键（Windows）

来源：Typora 官方《Shortcut Keys》（[S1]）。只列和 Notes 编辑相关的；macOS 键位放在最后一列备查。「浏览器里能否用」一列是我的判断：网页拿不到某些被浏览器保留的组合键。

| 功能 | Windows | macOS | 浏览器里能否用（推断） |
|---|---|---|---|
| 新段落 | Enter | Enter | 能 |
| 段内换行 | Shift+Enter | Shift+Enter | 能 |
| 保存 | Ctrl+S | ⌘S | 能（现有已实现） |
| 粗体 | Ctrl+B | ⌘B | 能 |
| 斜体 | Ctrl+I | ⌘I | 能 |
| 下划线 | Ctrl+U | ⌘U | 不需要（票里不要下划线） |
| 行内代码 | Ctrl+Shift+` | ⌘⇧` | 能 |
| 删除线 | Alt+Shift+5 | ⌃⇧` | 能 |
| 超链接 | Ctrl+K | ⌘K | 能（Chrome 的 Ctrl+K 是地址栏搜索，网页可以 preventDefault，待实测） |
| 图片 | Ctrl+Shift+I | ⌘⌃I | **很可能不能**：Ctrl+Shift+I 是 Chrome 开发者工具，待实测 |
| 清除格式 | Ctrl+\ | ⌘\ | 能 |
| 标题 1–6 | Ctrl+1…6 | ⌘1…6 | 能（Chrome 的 Ctrl+1…8 是切标签页，网页可以 preventDefault，待实测） |
| 段落（取消标题） | Ctrl+0 | ⌘0 | 能（会盖掉浏览器的「重置缩放」） |
| 标题升一级 / 降一级 | Ctrl+= / Ctrl+- | ⌘= / ⌘- | 能（会盖掉浏览器缩放） |
| 表格 | Ctrl+T | ⌘⌥T | **不能**：Ctrl+T 新建标签页，网页拦不住 |
| 代码块 | Ctrl+Shift+K | ⌘⌥C | 能 |
| 公式块 | Ctrl+Shift+M | ⌘⌥B | 能 |
| 引用 | Ctrl+Shift+Q | ⌘⌥Q | 能 |
| 有序列表 | Ctrl+Shift+[ | ⌘⌥O | 能 |
| 无序列表 | Ctrl+Shift+] | ⌘⌥U | 能 |
| 增加缩进 | Ctrl+[ 或 Tab | ⌘[ 或 Tab | 能 |
| 减少缩进 | Ctrl+] 或 Shift+Tab | ⌘] 或 Shift+Tab | 能 |
| 选中当前行/句 | Ctrl+L | ⌘L | 能（盖掉地址栏聚焦） |
| 选中当前样式范围 | Ctrl+E | ⌘E | 能 |
| 选中当前词 | Ctrl+D | ⌘D | 能（盖掉加书签） |
| 删除当前词 | Ctrl+Shift+D | ⌘⇧D | 能 |
| 删除表格行 | Ctrl+Shift+Backspace | ⌘⇧Backspace | 能 |
| 复制为 Markdown | Ctrl+Shift+C | ⌘⇧C | 待实测（Chrome 里是检查元素） |
| 粘贴为纯文本 | Ctrl+Shift+V | ⌘⇧V | 能（现有粘贴本来就只取纯文本） |
| 查找 / 替换 | Ctrl+F / Ctrl+H | ⌘F / ⌘H | 用浏览器自带查找即可（推断） |
| 源码模式 | Ctrl+/ | ⌘/ | 能（对应现有的「源码」按钮） |
| 撤销 / 重做 | Ctrl+Z / Ctrl+Y | ⌘Z / ⌘⇧Z | 能（现有已实现；Typora 官方表里没列，是系统通用键） |

表格编辑另有（[S3]）：Ctrl+Enter 在当前行下面插入空行；新版本里在最后一格按 Tab 也会加行（[S3]，0.10 起「Add new table row on Tab key」[S6]）；Ctrl+Shift+Backspace 删除当前行。增删列、移动行列靠右键菜单和拖动，没有快捷键（[S3]）。

公式块（[S4]）：输入状态下按上/下方向键或 Ctrl+Enter 结束编辑。行内公式 `$…$` 在 Typora 里默认关闭，要到偏好设置的 Markdown 页打开（[S2][S4]）；渲染用 MathJax（[S4]）。

链接：Typora 文档写的是「按住 Ctrl 单击」跳转（[S2]），和现有「Ctrl+单击打开链接」一致。

### 1.2 块级元素的进出

Typora 官方文档对「怎么创建」写得清楚，对「回车、退格、Tab 在块里具体做什么」大多没写。下表分开标注。

| 场景 | 行为 | 依据 |
|---|---|---|
| 段落里 Enter | 新段落；源码里是两个换行 | 文档 [S5] |
| 段落里 Shift+Enter | 段内换行（单个换行） | 文档 [S2][S5] |
| 行首输入 `#`…`######` 加空格 + 文字，回车 | 变标题 | 文档 [S2] |
| 行首 `>` | 引用 | 文档 [S2] |
| 行首 `* `、`- `、`+ ` / `1. ` | 无序 / 有序列表 | 文档 [S2] |
| 列表项 `[ ]` / `[x]` | 任务列表；单击复选框切换完成状态 | 文档 [S2] |
| ```` ``` ```` 加可选语言名，回车 | 代码块 | 文档 [S2][S7] |
| `$$` 回车 | 公式块 | 文档 [S2][S4] |
| `\| a \| b \|` 回车 | 表格 | 文档 [S2] |
| 空行上 `***` 或 `---` 回车 | 分隔线 | 文档 [S2] |
| 列表里 Tab / Shift+Tab | 缩进 / 反缩进（变成嵌套 / 回到上一级） | 文档（快捷键表把 Tab 列为 Indent）[S1] |
| 代码块里 Shift+Tab | 1.5 起默认对选中行做缩进调整（可在偏好里关） | 文档 [S7][S8] |
| 代码块里 Ctrl+A | 只选中当前代码块 | 文档 [S7] |
| 表格里 Tab | 下一格；最后一格加新行 | 文档 [S3][S6] |
| 表格里 Ctrl+Enter | 下面插一行 | 文档 [S3] |
| 公式块里 ↑ / ↓ / Ctrl+Enter | 结束编辑、离开公式块 | 文档 [S4] |
| 列表里在**有内容**的项上 Enter | 新建同级列表项 | 待实测（官方没写，所有同类编辑器都如此） |
| 列表里在**空**项上 Enter | 退出列表（嵌套时退一级） | 待实测 |
| 列表项行首 Backspace | 去掉列表符号变成普通段落 / 并入上一项 | 待实测 |
| 引用里空行 Enter | 跳出引用 | 待实测 |
| 引用行首 Backspace | 去掉引用 | 待实测 |
| 代码块里 Enter | 换行；怎样离开代码块（↓ 到块末尾之外 / 末尾连按回车？） | 待实测（官方没写 [S7]） |
| 标题行首 Backspace | 变回段落 | 待实测（现有手写编辑器已这样做） |

建议在拍板「Typora 一致」的具体规格前，用户在 Typora 里把「待实测」的 9 行各按一次，把结果补进这张表。这是 10 分钟的事，比猜靠谱。

### 1.3 露出 Markdown 源码的规则

- 文档原话（[S2]）：「Moving the cursor to the middle of a span element will expand that element into the Markdown source.」——光标移进一个行内元素（粗体、斜体、行内代码、链接、删除线、行内公式等）中间，它就展开成 Markdown 源码；离开后重新渲染。
- 行内公式：输入 `$` 再按 Esc 可以边打边预览（[S2]）。
- 块级元素（标题的 `#`、列表符号、引用的 `>`）光标进入时是否露出：官方没写，待实测。（我的印象是 Typora 对标题会在左边显示级别提示而不是露出 `#`，对列表和引用不露出符号；不作为依据。）
- 软换行：Typora 默认「编辑时保留换行、导出时忽略」（[S5]）。也就是说编辑区里看到的换行和源码一致。这和现有「中文段内换行直接相连」的**显示**约定不同；如果要跟 Typora 对齐，编辑区里应该按原样显示换行，只在只读渲染（如果还有的话）里相连。（推断，需要用户定）

---

## 二、候选编辑器库比较

### 2.1 总表

| | Vditor（IR 模式） | Milkdown（Crepe） | Tiptap | 继续手写 |
|---|---|---|---|---|
| 版本 / 发布 | 4.0.0（2026-08-30）[S12] | 7.22.2 [S17] | 3.31.4 [S22] | 现有 `notes.js` 468 行 |
| 许可证 | MIT；附带的 Lute 是木兰宽松许可证 v2（宽松型）[S12][S16] | MIT [S17] | MIT（核心和本次要用的扩展都是）[S22] | — |
| 维护（GitHub，2026-10-06） | 11.4k★，最后 push 2026-10-02，开着的 issue 96；基本是作者一人维护（b3log） | 12.0k★，最后 push 2026-10-06，开着 40 | 38.6k★，最后 push 2026-10-02，开着 837；有公司 | 自己 |
| 和第一节的贴合 | **高**：IR 模式就是 Typora 式；光标进入行内节点展开标记符 [S14]；列表 Tab 缩进、空项回车退出、引用逐层跳出、任务列表退格、表格 Tab/回车等都有专门处理 [S15]。快捷键不同，需要改 | **低**：所见即所得，标记符打完就消失，不露出源码；快捷键也是另一套（标题 Ctrl+Alt+1…6）[S18] | **低**：同 Milkdown；快捷键另一套（标题 Ctrl+Alt+1…6，删除线 Ctrl+Shift+S）[S23] | 能做到完全一致，但全部要自己写 |
| 无构建、单文件放进 vendor | **能**，官方支持：`dist/index.min.js`（UMD，297 KB）+ `index.css`，按需从 `options.cdn` 目录加载 Lute、KaTeX 等；拷 `dist` 并改 `cdn` 即可 [S11][S13] | **勉强**：官方只给 npm（ESM 里有裸模块名）[S19]；esm.sh 能生成一个自包含的 `crepe.bundle.mjs`（2.83 MB，gzip 约 915 KB，只额外引用一个很小的 `/node/process.mjs`）——实测，但这是第三方构建服务产物 | **不能**：官方无构建方案是 jsDelivr `+esm` 逐包加载 [S25]；esm.sh 的 `?bundle` 是每个包各自打包，starter-kit 和 core 会各带一份 ProseMirror（实测），要单文件只能自己用 esbuild 打一次 | 天然满足 |
| 体积（实测，min） | 主体 297 KB（gzip 71 KB）+ Lute 3.74 MB（gzip 580 KB）+ KaTeX 277 KB + 字体 | 约 2.83 MB 一个文件（含 KaTeX、CodeMirror） | 视组合，未测 | 手写 + KaTeX 273 KB（`katex.min.js`）+ 25 KB CSS + 约 254 KB woff2 字体 [S27] |
| 嵌套列表 | 有 [S10][S15] | 有（ListItem）[S20] | 有（extension-list）[S26] | 要写 |
| 任务列表 | 有 [S10][S15] | 有 [S20] | 有 [S23][S26] | 要写 |
| 删除线 | 有（`~~`，也认单个 `~`）[S16] | 有（GFM preset）[S18] | 有 [S23] | 要写（简单） |
| 行内 / 独立公式（GitHub 认 `$…$` 和单独成行的 `$$…$$`，[S28]） | 有，KaTeX 或 MathJax；`$` 后紧跟数字默认不算公式 [S11] | 有，KaTeX；块公式内部借用 `LaTeX` 语言的代码块表示，存回时写 `$$`——副作用是原文里语言为 `latex` 的代码块也会被存成 `$$` [S21] | 有（extension-mathematics，KaTeX），存回 `$…$` 和 `$$\n…\n$$` [S26] | 要写；渲染用 KaTeX |
| 关掉原始 HTML | **不能彻底关**：只有 `sanitize`（XSS 过滤，默认开）；Lute 没有关闭 HTML 解析的开关 [S11][S16] | 待实测（commonmark preset 有 html 节点，推断是原样保留） | 不是透传：HTML 用各扩展的 `parseHTML` 转成节点，转不了的退回成纯文字 [S24] | 现在就是全转义，天然满足 |
| 存回 Markdown 是否干净 | Lute 重新格式化：无序列表默认 `*`（可改，[S16]）；4.0.0 刚修了「保存时写入不换行空格」[S12]。待原型 diff | remark-stringify 重新格式化；`bullet` 等可配 [S19] | 官方说是早期版本，可能有边角情况；表格每格只允许一个子节点 [S24] | 完全可控（现在就会规整成固定写法） |
| 相对路径按实验目录解析 | `linkBase` 选项 [S11] | 要自己写（改 image/link 的 DOM 渲染，推断） | 要自己写（推断） | 已有 |
| `#` 标题显示降两级 | 只能 CSS（推断） | 只能 CSS 或改 schema（推断） | 同左 | 已有 |
| 中文段内换行直接相连 | IR 显示源码换行（Typora 也是保留换行显示 [S5]）；默认 `SoftBreak2HardBreak` 为真 [S16] | 不处理（推断） | 不处理（推断） | 已有 |
| Ctrl+单击开链接 | 要自己加监听（推断） | 有 LinkTooltip，非 Ctrl+单击（推断） | Link 扩展有点击打开的选项（推断） | 已有 |
| 中文输入法 | 有专门的 composition 处理和若干中文输入修复（`fixCJKPosition` 等）[S14][S15]；作者是中文开发者，主要用户也是中文 | ProseMirror 原生处理 IME，有零星中文光标问题报告（如 #1542，已关）| 有若干 IME 相关 open issue（如 #7135 行内可编辑节点、#7271 WebKit 标题重复字），我们只用 Chrome/Edge | 现有代码已处理 composition，但嵌套结构越多越难 |
| 整体风险 | 体积、快捷键冲突、HTML、单人维护 | 不满足核心需求 | 不满足核心需求，且无单文件 | 工作量大 |

### 2.2 各项要点

**Vditor IR**（[S10]–[S16]）

- 三种模式：所见即所得、IR（即时渲染）、分屏；IR「类似 Typora」。默认模式就是 `ir`。
- 加载：README 给的是 unpkg 的 `<script>`；`options.cdn` 默认是 `https://unpkg.com/vditor@版本号`，自建时把 `dist` 拷到正确位置并在 `options` 和渲染方法里传 `cdn`（[S11]）。`dist/js/` 下还有 mermaid、echarts、graphviz 等子目录，我们只需拷 `lute`、`katex`、`i18n`、`icons` 和 css（推断，原型时确认最小集合）。
- 快捷键（源码默认值，⌘ 在 Windows 上就是 Ctrl，[S13][S14]）：粗体 Ctrl+B，斜体 Ctrl+I，删除线 Ctrl+D，链接 Ctrl+K，无序列表 Ctrl+L，有序列表 Ctrl+O，任务列表 Ctrl+J，缩进 Ctrl+Shift+O，反缩进 Ctrl+Shift+I，引用 Ctrl+;，分隔线 Ctrl+Shift+H，代码块 Ctrl+U，行内代码 Ctrl+G，表格 Ctrl+M，标题 Ctrl+Alt+1…6，撤销/重做 Ctrl+Z/Ctrl+Y；表格内 Ctrl+= 下方加行、Ctrl+- 删行、Ctrl+Shift+= 加列等；任务项 Ctrl+Shift+J 切换完成。和 Typora 对不上的大约一半。工具栏按钮的 `hotkey` 可以配置，工具栏本身可以隐藏（`toolbarConfig.hide`）。
- `options.tab` 可以设 Tab 键插入的字符串；有 `keydown`、`input`、`esc`、`ctrlEnter` 回调（[S11]）。
- 近期 IR 相关的 issue：#1940「IR 模式输入 `- [ ]` 后立即回车不生成任务列表项」2026-09 已关；#1922「点击回车会产生两个换行符」开着。说明 IR 模式还在修边角。

**Milkdown / Crepe**（[S17]–[S21]）

- 「插件驱动的所见即所得 Markdown 编辑器框架」，ProseMirror + remark。Crepe 是开箱即用的完整编辑器，功能项：CodeMirror、ListItem、LinkTooltip、ImageBlock、BlockEdit、Table、Toolbar、Cursor、Placeholder、Latex、TopBar。Latex 依赖 CodeMirror 功能，不开 CodeMirror 会报错（[S21]）。
- 安装只写了 npm / yarn / pnpm（[S19]）。
- 快捷键（[S18]）：标题 Ctrl+Alt+1…6，引用 Ctrl+Shift+B，无序 Ctrl+Alt+8，有序 Ctrl+Alt+7，代码块 Ctrl+Alt+C，删除线 Ctrl+Alt+X，行内代码 Ctrl+E；表格 Tab / Shift+Tab 换格。可以通过 `ctx.set(xxxKeymap.key, …)` 改。

**Tiptap**（[S22]–[S26]）

- 无构建用法：`<script type="module">` 从 jsDelivr 的 `+esm` 地址 import（[S25]）。
- Markdown 支持是 `@tiptap/markdown`，基于 marked，官方自称「early release」（[S24]）。列表、任务列表、公式扩展都实现了 `renderMarkdown`（[S26]）。
- 快捷键见 [S23]，与 Typora 差异和 Milkdown 类似。

**继续手写**

- 现有 `notes.js` 的结构是「Markdown → HTML 渲染 + DOM → Markdown 反序列化 + 光标所在块按 Markdown 重新渲染」。要加的东西：嵌套列表（渲染器和 `items()` 现在会把子列表摊平）、任务列表、删除线、公式（KaTeX 渲染，`$`/`$$` 解析）、光标进入行内元素时把那个元素换成「标记符 + 文字」的源码形态、Typora 键位表、列表/引用/代码块/公式块的进出键。
- 公式可以直接用 KaTeX 官方的预编译文件 `katex.min.js`（UMD）或 `katex.mjs`（ES module，603 KB 未压缩），MIT（[S27]）。
- 工作量估计：源码露出 + 嵌套列表两块最难，整体可能把文件推到 1500 行以上。（推断）

---

## 三、给下一张票（#109）的输入

- 若采用 Vditor：先做原型（第一节「原型必须验证的 8 点」），vendor 目录建议是 `workbench/static/vendor/vditor-4.0.0/`（带版本号，升级时整体替换），只拷用到的子目录。
- 若退回手写：KaTeX 放 `workbench/static/vendor/katex-0.19.0/`；键位按 1.1 表，去掉浏览器拦不住的（Ctrl+T 表格、Ctrl+Shift+I 图片），给它们另找键或只留输入语法。
- 无论哪条路，1.2 表里「待实测」的 9 行先请用户在 Typora 里按一遍补全。
