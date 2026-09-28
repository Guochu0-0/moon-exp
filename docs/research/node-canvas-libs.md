# 节点画布库技术调研（工作台 v2）

> 调研日期：2026-09-27 · 对应 [#17](https://github.com/Guochu0-0/moon-exp/issues/17)，属于地图 [#15](https://github.com/Guochu0-0/moon-exp/issues/15)
> 标注规则：**「读到的」** 指直接读自官方文档、GitHub API、npm registry 或 CDN 实测；**「推断」** 是我的判断，文中都会注明。

## 结论（先说）

**推荐手写：用 HTML `div` 做节点，放在一个平移/缩放的容器里，底下垫一层 SVG 画边（总共约 300–500 行 vanilla JS），不引入画布库，也不加构建步骤。**

理由：

1. 我们要的交互很少：自由拖节点、平移/缩放、从节点拖出连线（只有「派生自」一种含义，单父）、保存坐标。节点数量在几十、最多一两百的量级。这些功能手写很快，库的主要价值在更复杂的地方（大图虚拟化、minimap、多选框选、自动布局、端口类型系统），我们用不上。（推断）
2. v1 就是 vanilla 模板字符串 + 内联 SVG（`workbench/static/app.js` 已经手写了 SVG 思路树，见 `tree()`），手写方案风格一致，零依赖，也不依赖 CDN。
3. 节点里要放富 HTML（标题、状态 pill、小指标）。用 DOM 节点可以直接复用 v1 的 `.pill` 等 CSS，这比 Canvas 类库（litegraph、Cytoscape）好得多。
4. 数据的「真相」在我们自己的 repo 文件里（画布布局文件 + `runs/`）。用库的话还得把库的内部模型（比如 Drawflow 的 export JSON）和我们的文件格式来回同步；手写方案里，坐标和父指针就是我们自己的数据结构。

**如果以后需要更多，第二选择是 React Flow（`@xyflow/react`）。** 它可以零构建使用（esm.sh + importmap + `htm`，下文有实测），但这条路比较别扭。一旦决定上 React Flow，**我认为加一个 Vite 构建步骤是值得的**。触发条件（推断）：画布需要 minimap、框选/多选、撤销重做、自动布局（dagre/elk）、上千节点，或者节点页编辑器也要做成复杂的 SPA。

## 对比总表

数据来源：GitHub REST API（`gh api repos/...`，2026-09-27 读取）、npm registry `/<pkg>/latest`、bundlephobia `api/size`。「零构建」一列是本次实测 CDN 链接和官方文档的结果。

| 库 | 许可证 | 体积（min+gzip） | 零构建可用？ | 节点渲染 | 拖拽连线 | 位置持久化 | 维护（最后 push / 最新 release） | ★ |
|---|---|---|---|---|---|---|---|---|
| **手写 DOM+SVG** | 自有 | 0 | 是 | 任意 HTML | 自己写（约 80 行，推断） | 自己的 JSON | — | — |
| React Flow `@xyflow/react` 12.12.0 | MIT | 60 KB（bp），**另需 React 19 + ReactDOM 约 62 KB** | 可行但不官方：React 19 已无 UMD，只能走 esm.sh | 任意 React 组件（`nodeTypes`） | 内建（`onConnect`、`isValidConnection`） | `onNodesChange`/`onNodeDragStop` 拿坐标 | 2026-09-24 / 12.12.0（2026-09-24），约每月发版 | 38.5k |
| Svelte Flow `@xyflow/svelte` 1.7.0 | MIT | bp 无法计算 | **否**：需要 Svelte 编译器（推断） | Svelte 组件 | 内建 | 同上 | 同 xyflow 仓库 | — |
| Rete.js v2 `rete` 2.0.6 | MIT | 核心 3 KB，加 area/connection/render-utils 约 25 KB，**另需 React/Vue/Angular/Svelte/Lit 其中之一** | 核心有 UMD；但渲染插件必须配框架，`rete-lit-plugin` 在 jsDelivr 上没有 min.js（404） | 取决于框架插件 | 内建（connection plugin） | 自己序列化 | 2026-09-27 / v2.0.6（2025-06-30） | 12.3k |
| litegraph.js 0.7.18 | MIT | 121 KB | 是（script 标签） | **Canvas2D**，不支持 HTML 节点 | 内建（slot 连线） | `graph.serialize()` | 原仓库 2024-08 后无 push；ComfyUI 的 fork `@comfyorg/litegraph` 最后发版 2025-08 | 8.2k |
| tldraw 5.4.2 | **专有 license**（生产需要 license key） | 524 KB（bp） | 否（React + 大量依赖） | React shape | 箭头绑定 | 自带 store | 非常活跃（2026-09-27） | 50.6k |
| JointJS `@joint/core` 4.3.3 | MPL-2.0 | 144 KB（实测 gzip `joint.min.js`） | 是（官方有 buildless 指南，全局变量 `joint`） | SVG markup；HTML 需要额外做 | 内建（magnet 拖线） | `graph.toJSON()` | 2026-09-25 / 4.3.x（2026-09） | 5.4k |
| Drawflow 0.0.60 | MIT | 8.7 KB | 是（CDN script） | HTML 字符串 | 内建（output→input） | `export()` 含 `pos_x/pos_y` | **停滞**：最后提交 2024-09-03，273 个 open issue | 6.1k |
| Cytoscape.js 3.34.3 | MIT | 137 KB | 是（UMD + ESM） | **Canvas**；HTML 需要插件 `cytoscape-node-html-label` 叠加 | 需要插件 `cytoscape-edgehandles` | `cy.json()` | 2026-09-25 / 3.34.3（2026-09-07） | 11.2k |
| AntV X6 3.1.8 | MIT | bp 数据异常（1.3 KB，不可信） | 有 UMD `dist/x6.min.js` | SVG；HTML 节点要插件 | 内建 | `toJSON()` | 2026-08-11 / v3.1.7（2026-03） | 6.7k |
| maxGraph 0.24.0 | Apache-2.0 | 170 KB | 仅 ESM/CJS，没有 unpkg 字段 | SVG（mxGraph 风格） | 内建 | codec | 活跃（2026-09-25），仍是 0.x | 1.1k |

## 逐项说明与出处

### React Flow / xyflow

- **许可证与定位**：README 写明 "React Flow and Svelte Flow are MIT licensed"；另有付费的 React Flow Pro（Pro 示例和支持），核心功能不设付费门槛。（读到的：[xyflow/xyflow README](https://github.com/xyflow/xyflow)）
- **monorepo 结构**：`@xyflow/react`、`@xyflow/svelte`、`@xyflow/system`（"shared helper library"）。`@xyflow/system` 依赖 d3-drag/d3-zoom/d3-selection，官方没有文档化的 vanilla 版本。（读到的：README 和 npm registry 的 dependencies 字段）
- **官方安装文档只写了包管理器**（npm/pnpm/yarn/bun）和 Vite 模板，没有提 CDN，并强调 "You must import the css stylesheet"。（读到的：[Installation](https://reactflow.dev/learn/getting-started/installation-and-requirements)）
- **零构建路径实测**：
  - React 19 不再发布 UMD：官方原话是 "Starting with React 19, React will no longer produce UMD builds … we recommend using an ESM-based CDN such as esm.sh"。（读到的：[React 19 Upgrade Guide](https://react.dev/blog/2024/04/25/react-19-upgrade-guide)）实测 `react@19.2.0/umd/...` 在 jsDelivr 返回 404，`react-dom@18.3.1/umd/` 仍然存在。
  - `@xyflow/react@12.12.0/dist/umd/index.js` 存在（jsDelivr 200，188 KB，gzip 后 59 KB），依赖全局 `React`、`ReactDOM`、`jsxRuntime`。
  - `https://esm.sh/@xyflow/react@12.12.0?deps=react@19.2.0,react-dom@19.2.0` 返回 200，会引入 system、classcat、zustand 等模块。
  - 所以零构建写法是：`<script type="importmap">` + esm.sh + [`htm`](https://esm.sh/htm@3)（代替 JSX）。可以跑通（推断，本次没有写 demo 页验证），但：不能写 JSX；调试和报错不友好；运行时依赖外部 CDN，除非把文件 vendor 进 `static/`；esm.sh 的依赖图要用 `?deps=` 固定版本。
- **体积**：bundlephobia 给出 `@xyflow/react` 12.12.0 为 min 188 KB / gzip 60 KB。React DOM client 在 esm.sh 的 bundle 实测 gzip 约 58 KB，react 约 4 KB。合计约 **120 KB gzip**。
- **我们需要的功能都有**：`nodeTypes` 自定义节点组件；`onConnect` 加 `isValidConnection` 可以实现「单父」（文档没有内建的每个 handle 只允许一条入边，需要在回调里检查已有边）；`onNodesChange` / `onNodeDragStop` 拿到坐标后写回文件。（读到的：[ReactFlow API reference](https://reactflow.dev/api-reference/react-flow)）
- **维护**：非常活跃，2026-09-24 同时发布了 react 12.12.0、svelte 1.7.0、system 0.0.83；上一轮是 2026-09-01。（读到的：GitHub releases API）

### Svelte Flow

- 与 React Flow 同仓库、同为 MIT，peer 依赖 `svelte ^5.25.0`。Svelte 组件必须经过编译器，所以**无法零构建**（推断；npm 包 `module` 指向 `./dist/lib/index.js`，内部是 `.svelte` 组件）。

### Rete.js v2

- 官方 Getting started 提供 CDN 方式：`<script src="https://cdn.jsdelivr.net/npm/rete/rete.min.js">`，通过全局 `Rete`、`ReteAreaPlugin` 等使用。（读到的：[retejs.org/docs/getting-started](https://retejs.org/docs/getting-started)）
- 但**渲染必须通过框架插件**：React.js、Vue.js、Angular、Svelte、Lit，文档里没有纯 DOM renderer。（读到的：同上）
- `rete-lit-plugin@2.0.3` 的 package.json 里没有 main/unpkg 字段，`rete-lit-plugin.min.js` 在 jsDelivr 返回 404，所以要零构建只能走 esm.sh。（实测）
- 定位是 dataflow / visual programming（socket 类型、引擎执行），对「想法树」来说模型太重。（推断）

### litegraph.js

- README 原文："Renders on Canvas2D"，节点定制靠 widgets 和 custom rendering 回调，**不支持在节点里放任意 HTML**。（读到的：[jagenjo/litegraph.js](https://github.com/jagenjo/litegraph.js)）
- 原仓库最后 push 是 2024-08-01，最后 release 在 2024-03。ComfyUI 维护的 fork `@comfyorg/litegraph` 最新版本 0.17.2 发布于 2025-08。（读到的：GitHub API）
- 结论：做我们的富 HTML 节点不合适。

### tldraw

- `LICENSE.md` 原文："Not to use the Software in Production Environments"（除非有 trial 或 commercial license），并且软件内含 "technical measures to verify License Key validity … and ensure proper watermark display"。（读到的：[tldraw LICENSE.md](https://raw.githubusercontent.com/tldraw/tldraw/main/LICENSE.md)）
- pricing 页面有可申请的免费 Hobby license、Commercial license 和 100 天 trial。（读到的：[tldraw.dev/pricing](https://tldraw.dev/pricing)）
- bundlephobia 为 gzip 524 KB，依赖 React、radix-ui、tiptap。它是白板，不是节点图。对我们来说许可证麻烦、体积过大、需要构建，**排除**。

### JointJS（`@joint/core`）

- 官方文档说明 "JointJS can also be used without any build process"：`<script src="https://cdn.jsdelivr.net/npm/@joint/core/dist/joint.js">`，全局变量为 `joint`、`g`、`V`。（读到的：[Installation](https://docs.jointjs.com/learn/quickstart/installation/)、[JavaScript integration](https://docs.jointjs.com/learn/integration/javascript/)）
- 许可证 MPL-2.0（文件级 copyleft：只有修改了库文件本身才需要开源那些文件，直接使用不受影响；推断，依据 MPL-2.0 的一般条款）。更高级的功能在商业版 JointJS+ 中。
- 实测 `joint.min.js` 为 474 KB，gzip 后 144 KB。它是 SVG 优先，富 HTML 节点需要额外做（foreignObject 或 HTML 覆盖层），本次读到的集成文档没有涉及这部分。
- 活跃度好（2026-09 有发版）。对我们的需求来说偏重，属于「能用但不划算」。

### Drawflow

- README：零依赖 vanilla，支持 CDN；`editor.addNode(name, inputs, outputs, posx, posy, class, data, html)` 可以直接塞 HTML；export 的 JSON 含 `pos_x` / `pos_y`；有 `connectionCreated` 等事件。（读到的：[jerosoler/Drawflow](https://github.com/jerosoler/Drawflow)）
- gzip 只有 8.7 KB，**从功能看最贴近零构建的需求**。
- 但维护基本停了：最后一次提交是 2024-09-03（"Update verion 0.0.60"），273 个 open issue，版本仍是 0.0.x。数据模型是 input/output 端口加它自己的 JSON，要和我们的文件格式同步。（读到的：GitHub API；后半句是推断）
- 结论：可以作为参考实现，读它的源码学拖线逻辑，但不作为依赖。

### Cytoscape.js

- MIT、活跃（3.34.3，2026-09-07），有 UMD 和 ESM，零构建没有问题。gzip 137 KB。
- Canvas 渲染：富 HTML 节点要靠 `cytoscape-node-html-label` 插件在上面叠 DOM，拖线要靠 `cytoscape-edgehandles` 插件（最后 push 2026-03）。它的强项是图分析和大图布局，用来做交互式编辑器不顺手。（读到的：npm registry；判断部分是推断）

### 其他（简列）

- **AntV X6**：MIT，有 `dist/x6.min.js` UMD，功能全（HTML 节点要插件）。文档以中文为主，版本 3.x。bundlephobia 给出的数据异常，本次没有实测体积。
- **maxGraph**（mxGraph 的继任者）：Apache-2.0，仅 ESM/CJS，仍是 0.x，gzip 170 KB。

## 手写方案的要点（给实现票参考，推断）

- **结构**：`<div class="viewport">` 里放一个 `<div class="world" style="transform: translate(x,y) scale(k)">`；`world` 内有一个 `<svg class="edges">`（`overflow: visible`）和若干绝对定位的 `.node` div。
- **交互**：用 Pointer Events 统一处理鼠标和触控，`setPointerCapture` 处理拖拽。在空白处拖动是平移，滚轮以光标为中心缩放。把 screen→world 坐标换算写成一个函数。
- **连线**：节点边缘有一个「拖出」把手，pointerdown 后画一条临时的三次贝塞尔曲线，在目标节点上 pointerup 就完成。连完只改子节点的 `parent`，旧的父边自然被替换，从而强制单父。还要检查成环：沿祖先链往上走，确认目标不是自己的后代。
- **持久化**：`dragend` / 连线完成后 debounce，POST 到 Python server，写布局文件（格式归 #15 的「写回的文件格式」一项）。
- **规模**：节点少于 200 时，全量重渲染 DOM 加上用 SVG `path` 画边，性能不会有问题。
- 预计 300–500 行，不含样式。v1 的 `tree()` 布局代码可以用来给首次出现、还没有坐标的节点算初始位置。

## 什么时候该加构建步骤（坦白说）

- 只做上面这个画布的话，**不值得**。构建会带来 node_modules、锁文件、构建产物是否入 git、服务器上要不要装 node 等一串成本，收益却很小。
- **值得的情况**：v2 的「整页节点详情」如果要做成富交互 SPA（拖动排序视图块、内联编辑、路由、状态管理），或画布需要 minimap、框选、撤销重做、自动布局，那就上 **Vite + React + React Flow**。可以让 Python server 继续只提供 `dist/` 静态文件，保留「本地起服务器」的用法；构建产物要不要提交进 git，由实现票决定。
- 两者之间的折中：用 esm.sh + importmap + htm 零构建使用 React Flow，把依赖文件 vendor 到 `static/vendor/` 以免依赖外网。能跑，但这是在不写 JSX 的前提下使用一个以 JSX 为主的库，长期维护体验一般。（推断）
