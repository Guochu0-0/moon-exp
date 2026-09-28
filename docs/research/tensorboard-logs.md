# TensorBoard 日志：读取、合并、同步与「在 TensorBoard 中打开」

> 调研日期：2026-09-27。对应工作台 v2 节点详情页「训练曲线」区（方案 B：Python 端解析 scalars → JSON → 浏览器画图）以及「在 TensorBoard 中打开」按钮。
> 实测环境：Windows 11 本机。读取端 venv：Python 3.11.5 + `tensorboard 2.21.0` + `tbparse 0.0.9`（protobuf 7.36.2，upb 后端），**未装 TensorFlow、未装 torch**。写入端：Python 3.9 + `torch 2.8.0+cpu` 的 `torch.utils.tensorboard.SummaryWriter`、`lightning 2.6.0` 的 `TensorBoardLogger`。没有登录服务器，体积全部由本地生成的日志测得。

## 结论

1. **读取器：自己写约 20 行的流式解析器**（按 TFRecord 帧逐条读，用 `tensorboard.compat.proto.event_pb2.Event` 反序列化，只取 `simple_value`）。它唯一的依赖是 `tensorboard` pip 包（实际只用到 protobuf 定义），不需要 TF，也不需要起服务器。实测 50 万个点约 1 s；events 文件里混有图片时仍是约 1 s，而官方读取器要 33–37 s。它不做蓄水池采样；写到一半被截断的尾部会被安全跳过。多次运行怎么合并也由我们自己决定（见下文）。`EventAccumulator` 和 `tbparse` 都能用，但慢一个数量级，前者默认还会悄悄降采样。
2. **同步范围：同步 `version_N/` 或 writer 的目录，但排除 `checkpoints/`。** 只含 scalar 的日志约 **50 B/点**（10 个 tag × 5 万步 = 24 MiB，gzip 后 6.5 MiB）。体积主要来自图片：本次合成的每张图约 1.5 MiB，而且基本压缩不动。我们自己的 finetune 循环建议**用两个 writer，把 scalars 和图片/直方图写进不同子目录**，这样可以只拉 scalars。上游 Lightning 代码的图片和 scalar 混在同一个文件里，默认整拉；如果太大，再到服务器端抽出 scalars.json（本例约 2.3 MiB gz）。
3. **打开 TensorBoard：由工作台用 `subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", <dir>, "--host", "127.0.0.1", "--port", <空闲端口>])` 启动。** 进程归工作台管：每个 logdir 复用一个实例，工作台退出时统一 `terminate()`。不建议用进程内的 `tensorboard.program.TensorBoard().launch()`。

## 读取方式对比

在同一台机器上测量，每种方法各跑一个独立子进程，内存取峰值工作集（空进程加上 import protobuf 约 20 MB）。数据集 A：10 个 scalar tag × 50 000 步，单个 events 文件 24.0 MiB。数据集 B：A 再加上每 1000 步 2 张 1280×480 PNG 和 3 个直方图（共 100 张图、150 个直方图），单个文件 175.4 MiB。

| 方法 | 依赖 | A 耗时 / 峰值内存 | B 耗时 / 峰值内存 | 返回点数 | 备注 |
|---|---|---|---|---|---|
| `EventAccumulator(d)` 默认 | tensorboard | 10.7 s / 90 MB | 37.5 s / 107 MB | **100 000**（每个 tag 只剩 1 万） | 蓄水池采样，**丢点** |
| `EventAccumulator(d, size_guidance={"scalars": 0, ...})` | tensorboard | 9.9 s / 178 MB | 36.1 s / 181 MB | 500 000 | 不递归子目录 |
| `tbparse.SummaryReader(d, event_types={"scalars"})` | tensorboard + pandas | 11.8 s / 218 MB | 36.9 s / 222 MB | 500 000 | 内部就是 EA（size_guidance 为 0），会递归并带 `dir_name` 列 |
| `event_file_loader.LegacyEventFileLoader` 逐条迭代 | tensorboard | 7.7 s / 128 MB | 32.7 s / 135 MB | 500 000 | 慢在纯 Python 的 CRC 校验 |
| 自写 TFRecord 帧解析 + `event_pb2`（流式） | tensorboard（仅 proto） | **1.0 s / 89 MB** | **1.05 s / 93 MB** | 500 000 | 不校验 CRC |

几点说明：

- **为什么慢。** 没装 TF 时，TensorBoard 用 `tensorboard/compat/tensorflow_stub/pywrap_tensorflow.py` 里的 `PyRecordReader_New` 读记录，它会对每条记录**逐字节**用纯 Python 算 masked CRC32C（`crc_update` 是一个 Python for 循环）。所以耗时跟文件字节数成正比，不跟 scalar 点数成正比，B 里的图片字节几乎占满了时间。自写的解析器跳过 CRC，只按 `uint64 length | uint32 crc(length) | data | uint32 crc(data)` 的帧格式切片，所以 A 和 B 的耗时几乎一样。
- **蓄水池默认值。** `event_accumulator.DEFAULT_SIZE_GUIDANCE` 是 `SCALARS: 10000, IMAGES: 4, HISTOGRAMS: 1, TENSORS: 10, ...`，设成 0 表示全保留（`reservoir.Reservoir` 的 docstring 里写着 "If 0, all values will be kept"）。采样是 `random.Random(seed=0)` 加上 `always_keep_last=True`，结果是**不等间距**的点，画出来的曲线会失真，而且不会有任何提示。TensorBoard UI 自己默认每个 tag 只保留 1000 个 scalar 点（`data_ingester.DEFAULT_TENSOR_SIZE_GUIDANCE`），可以用 `--samples_per_plugin scalars=0` 改。
- **不装 TF 可以吗。** 可以。上面五种方法都在没装 TF 的 venv 里跑通了。`tbparse` 源码里写的是 `try: import tensorflow / except ImportError`，TF 只是可选依赖。
- 自写解析器的核心代码（完整脚本留在 tmp，不提交）：

```python
import struct
from tensorboard.compat.proto import event_pb2
def read_scalars(path):                      # path: 单个 events.out.tfevents.* 文件
    out = {}
    with open(path, "rb") as fh:
        while True:
            hdr = fh.read(12)                # uint64 len + uint32 masked_crc(len)
            if len(hdr) < 12: break
            (ln,) = struct.unpack("<Q", hdr[:8])
            body = fh.read(ln + 4)           # data + uint32 masked_crc(data)
            if len(body) < ln + 4: break     # 尾部截断（仍在写 / 同步到一半）
            ev = event_pb2.Event.FromString(body[:ln])
            for v in ev.summary.value:
                if v.HasField("simple_value"):
                    out.setdefault(v.tag, []).append((ev.step, ev.wall_time, v.simple_value))
    return out
```

截断测试：把 A 截到一半再多 7 个字节，`EventAccumulator` 和自写解析器都读出 251 627 个点，一致。events 文件是只追加的，以后可以按 `(路径, 大小, mtime)` 缓存，也可以记住上次读到的偏移做增量读，这是推断，未实测。

## 目录结构与 tag

**`torch.utils.tensorboard.SummaryWriter`**
- 不传 `log_dir` 时写到 `runs/<%b%d_%H-%M-%S>_<hostname><comment>`，**相对于当前工作目录**（见 `writer.py`）。**这和工作台的 `runs/<id>/` 撞名**：如果从仓库根目录运行而不传 `log_dir`，日志会落进工作台的 `runs/` 里。必须显式传 `log_dir=runs/<id>/tb/...`。
- 文件名是 `events.out.tfevents.<unix秒>.<host>.<pid>.<序号><filename_suffix>`。每实例化一个 writer，就多一个文件。
- `add_scalar(tag, v, step)`：tag 原样写入，写成 `simple_value`（plugin 是 `scalars`）。
- `add_scalars("loss", {"train": .., "val": ..})`：会**新建子目录** `loss_train/`、`loss_val/`，每个子目录一个 events 文件，tag 都叫 `loss`；父目录里只有一个 88 B 的空文件。实测对父目录跑 `EventAccumulator`，scalar tag 是 `[]`；`tbparse` 能递归读出来，并通过 `dir_name` 区分。**建议自己的循环不要用 `add_scalars`。**
- 默认 `flush_secs=120, max_queue=10`，所以训练中途同步下来的文件可能落后最多约 2 分钟。

**Lightning `TensorBoardLogger`**（`save_dir/name/version_N/`）
- 版本号取 `save_dir/name/` 下已有 `version_*` 的最大值加 1。`hparams.yaml` 与 events 文件放在同一目录，**只在文件不存在时写**。实测续训写进 `version_1` 后，`hparams.yaml` 仍是崩溃那次的内容。
- `self.log("train/loss", on_step=True, on_epoch=True)` 会产生 `train/loss_step` 和 `train/loss_epoch` 两个 tag。只 `on_step` 时 tag 就是原名。
- step 语义：`logger_connector.log_metrics` 不传 step 时，用 `fit_loop.epoch_loop._batches_that_stepped`，并**自动加一个 `epoch` tag**。实测 `log_every_n_steps=10` 时，step 是 9, 19, …；`epoch` 在同一个 step 上可能重复出现（例如 step 239 出现 3 次）。
- 上游 LoFTR（`train.py`：`TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)`）的 `ckpt_dir = Path(logger.log_dir) / 'checkpoints'`，也就是 **checkpoint 放在 version 目录里**。它直接调用 `logger.experiment.add_scalar(...)`：`train/*` 的 step 是 `global_step`，而 `val_{i}/avg_*` 和 `metrics_{i}/*` 的 **step 是 epoch 编号**，另外还有 `add_figure('train_match/...')` 和 `add_figure('val_match_{i}/...')`。**同一个文件里不同 tag 的 x 轴含义不同**，前端不能假设 step 可以跨 tag 对齐。

## 多次运行的合并

实测场景：`version_0` 是完整的一次运行；`version_1` 在 step 130 崩溃，最后一个 ckpt 在 step 100，然后用 `TensorBoardLogger(version=1)` 加 `ckpt_path=last.ckpt` 续训。结果 `version_1/` 里有**两个** events 文件：第一个的 step 是 9…129，第二个是 109…239。

- `EventAccumulator` 和 `tbparse` 都**不去重**：`train/loss_step` 读出 27 个点，109/119/129 各出现两次。TensorBoard UI（`/data/plugin/scalars/scalars?run=version_1`）返回的也是同样 27 个点，曲线会折返。原因是 `file_version=brain.Event:2` 时，EA 只在遇到 `SessionLog.START` 事件时才清理（`_CheckForRestartAndMaybePurge`），而 Lightning 不写这个事件。`SummaryWriter(purge_step=T)` 会写这个事件，TB 因此会隐藏 step ≥ T 的旧点。
- **建议规则（我们自己实现）**：同一目录下的文件按文件名里的时间戳排序；对每个 tag，后一个文件出现后，删掉前面文件中该 tag 里 step ≥ 后一个文件该 tag 最小 step 的点。这与 `purge_step` 和 EA 的 `_Purge(by_tags=True)` 语义相同，**必须按 tag 做**，因为 epoch 为 step 的 tag 和 global_step 为 step 的 tag 混在一起。
- **不同的 `version_N` 默认当成不同的运行**，并排显示，带版本名，不自动拼接。上游 LoFTR 续训时不指定 version，会新开一个 `version_{N+1}`，step 从 ckpt 处接着走。要不要把它拼到前一个版本后面，需要人来判断（这是推断，未实测上游脚本）。`wall_time` 只用来排序和显示，不作为 x 轴。

## 同步范围与体积

| 内容 | 体积（实测） | 是否同步 |
|---|---|---|
| 只含 scalar 的 events 文件 | 24.0 MiB / 50 万点 ≈ **50 B/点**；gzip 6.5 MiB | 是 |
| 同一文件再加 100 张图 + 150 个直方图 | 175.4 MiB；gzip 156.8 MiB（PNG 几乎压缩不动） | 看情况 |
| 服务器端抽出的 scalars：列式 JSON / CSV | 7.0 MiB（gz 2.3）/ 12.6 MiB（gz 2.6） | 替代方案 |
| `hparams.yaml` | 几十 B | 是 |
| `version_N/checkpoints/*.ckpt`（上游 LoFTR） | 模型大小 | **否**，要显式排除 |

按上游 Lightning 默认的 `log_every_n_steps=50` 换算，10 个 tag 跑 5 万步约 0.5 MB。这是外推值，图片仍然是主要体积来源。本次合成的图带噪声，接近最坏情况，真实的 matplotlib 匹配图通常更小，这点未实测。可选方案：

1. **自己的 finetune 循环**：开两个 `SummaryWriter`，分别写 `runs/<id>/tb/scalars/` 和 `runs/<id>/tb/media/`。平时只同步 `scalars/`，需要看图时再拉 `media/`。TensorBoard 用 `--logdir runs/<id>/tb` 打开时，会把它们显示成两个 run。
2. **上游 Lightning 代码**：图片和 scalar 在同一个文件里，无法按字节范围拆开，因为它们是逐条记录交错排列的。默认同步整个 `version_N/`，排除 `checkpoints/`。events 文件只追加，用 rsync 增量同步时每次只传新增部分。如果体积确实不可接受，就在服务器上用上面的解析器（只依赖 `tensorboard` 包）导出 `scalars.json` 再拉回来。代价是本地的「在 TensorBoard 中打开」就看不到图了。

## 在 TensorBoard 中打开

- **推荐用子进程**：`Popen([sys.executable, "-m", "tensorboard.main", "--logdir", d, "--host", "127.0.0.1", "--port", str(p)])`，其中 `p` 通过先 `bind(("127.0.0.1", 0))` 再关掉 socket 拿到。然后轮询 `http://127.0.0.1:p/data/runs`，就绪后返回 URL，由前端 `window.open`。实测对一个小 logdir，约 **3.7 s** 就绪，`terminate()` 后正常退出。用 `sys.executable` 可以保证启动的是工作台所在 venv 里的 tensorboard，不依赖 PATH。
- **生命周期**：工作台持有 `{logdir: (Popen, port)}`，同一个 logdir 复用同一个实例，退出时通过 `atexit` 统一 `terminate()`。如果工作台被强杀，Windows 上的子进程会成为孤儿并继续占着端口。由于每次都另选空闲端口，这不影响再次启动，这是推断。`tensorboard.manager.start()` 也能用：它按 cache key 复用实例，并把 info 文件写到 `%TEMP%\.tensorboard-info\`。但它调用的是 PATH 上的 `tensorboard` 可执行文件（可用 `TENSORBOARD_BINARY` 覆盖），没有显式调用简单。
- **不用进程内的 `program.TensorBoard().launch()`**：它在当前进程里起一个 daemon 线程运行 `serve_forever`，返回的只有 URL，没有停止它的接口。它还和工作台共用 GIL，加载大日志时要做 30 s 以上的纯 Python CRC 计算，会拖慢工作台自己的请求。这一点是推断，依据是上表 `LegacyEventFileLoader` 的耗时和 TB 走的是同一条读取路径。
- `--port` 不传时默认先试 6006，被占就往后找；传 `0` 由系统分配，但得从 stdout 解析实际端口，不如自己先选好端口。

## 已知的坑

- `EventAccumulator` 默认每个 tag 只保留 1 万个 scalar，并且是随机蓄水池采样；TB UI 默认只保留 1000 个。两者都**不会提示**丢了点。
- `EventAccumulator` 只读指定目录**这一层**的 events 文件，不递归，所以 `add_scalars` 的子目录会漏掉。
- 续训的重叠 step 在所有官方读取器和 TB UI 里都会重复出现，除非训练端用了 `purge_step`。
- Windows 上的 `tensorboard-data-server 0.7.2` 是 `py3-none-any` 包，没有附带二进制，所以 `--load_fast=auto` 只能退回 Python 读取路径。没装 TF 时每个字节都要过纯 Python CRC，175 MiB 的日志在 TB 里要约 30 s 以上才能加载完（推断，同上）。启动时打印的 "TensorFlow installation not found - running with reduced feature set" 可以忽略。
- 同一个 events 文件里，不同 tag 的 step 含义可能不同（上游 LoFTR 的 val 用 epoch 作 step）；Lightning 会自动加 `epoch` tag，并且可能在同一 step 重复写。
- 续训写进同一个 version 时，`hparams.yaml` 不会更新。
- 自写解析器不校验 CRC。文件损坏时可能读出垃圾值或抛出 protobuf `DecodeError`，需要 try/except 并在遇到的第一条坏记录处停止。

## 来源

- TensorBoard 源码（与本地安装的 2.21.0 一致）：[event_accumulator.py](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py)（`DEFAULT_SIZE_GUIDANCE`、`_MaybePurgeOrphanedData`、`_Purge`）、[reservoir.py](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/reservoir.py)、[event_file_loader.py](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_file_loader.py)、[tensorflow_stub/pywrap_tensorflow.py](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/tensorflow_stub/pywrap_tensorflow.py)（纯 Python CRC32C）、[data_ingester.py](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/data_ingester.py)（UI 的 scalars 为 1000）、[core_plugin.py](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/core/core_plugin.py)（`--port`、`--load_fast`、`--samples_per_plugin`）、[program.py](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/program.py)（`launch()`）、[manager.py](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/manager.py)（`start()`、info 目录）
- TFRecord 帧格式：[TensorFlow TFRecord 文档](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details)
- tbparse：[文档](https://tbparse.readthedocs.io/en/latest/)、[summary_reader.py](https://github.com/j3soon/tbparse/blob/master/tbparse/summary_reader.py)（基于 EA，`size_guidance` 为 0，TF 可选）
- PyTorch：[torch.utils.tensorboard 文档](https://docs.pytorch.org/docs/stable/tensorboard.html)、[writer.py](https://github.com/pytorch/pytorch/blob/main/torch/utils/tensorboard/writer.py)（默认 `log_dir`、`purge_step`、`add_scalars` 的 `fw_tag`）
- Lightning：[TensorBoardLogger 文档](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html)、[loggers/tensorboard.py](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/loggers/tensorboard.py)、[logger_connector.py](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py)（自动加 `epoch`、step 取 `_batches_that_stepped`）、[Logging 文档](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)
- 上游 LoFTR：[train.py](https://github.com/zju3dv/LoFTR/blob/master/train.py)、[lightning_loftr.py](https://github.com/zju3dv/LoFTR/blob/master/src/lightning/lightning_loftr.py)
- 实测脚本（未提交）：`C:\Users\yougu\.claude\jobs\f0982caf\tmp\{gen.py, gen_pl.py, bench.py, sizes.py, multi.py, launch.py}`
