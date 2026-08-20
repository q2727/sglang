# 02 · 配置参考:CLI 与环境变量

> 全部行号对应 `459ae93227`。**这份是查询手册**;要理解每个旋钮为什么存在,读 [01-architecture](01-architecture.md)。

## 1 · CLI:开启解耦所必需的

| Flag | 默认 | 说明 |
|---|---|---|
| `--decoupled-spec-role` | `"null"` | `null`(关)/ `verifier` / `drafter`。**它是"角色"而非"算法"**,与 `--speculative-algorithm` 正交 |
| `--decoupled-spec-bind-endpoint` | `None` | 本进程入站通道 bind 的地址(verifier 收结果、drafter 收控制) |
| `--decoupled-spec-connect-endpoints` | `None` | **JSON 数组**,按对端 rank 排序的对端 bind 地址。shell 里要用单引号包住 |
| `--decoupled-spec-rank` | `None` | 本进程在自己角色空间内的 rank |
| `--decoupled-spec-data-transport` | `"zmq"` | `zmq` / `cuda_ipc`。**控制平面永远走 zmq**;两侧必须设成同一个值 |
| `--spec-trace-dir` | `None` | 解耦 spec 的 trace 输出目录 |

前四项:三个 endpoint/rank 相关的**缺一个就报错**(`speculative_hook.py:213-222`)。

### 与解耦联动的投机参数

| Flag | 声明默认 | 解耦下的实际取值 |
|---|---|---|
| `--speculative-algorithm` | `None` | **必填**(用 `STANDALONE`);drafter 角色在校验后被清成 `None` |
| `--speculative-num-steps` | `None` | **必填**,即 K |
| `--speculative-fanout` | `None` | 解耦下自动填 **4**(`speculative_hook.py:323-324`);colocated 下给这个 flag 会直接报错 |
| `--speculative-eagle-topk` | `None` | 被**钉死为 1**(给 ≠1 会报错) |
| `--speculative-num-draft-tokens` | `None` | 被强制为 **K+1**;drafter 角色随后清成 `None` |
| `--cuda-graph-bs-decode` | `None` | drafter 上**必须覆盖 (K+1)×F**,见 [D29](05-pitfalls.md#d29) |
| `--max-mamba-cache-size` | `None` | 仅 hybrid(GDN)draft 模型需要 |

### 自动生效的角色默认值

| 项 | verifier | drafter | 出处 |
|---|---|---|---|
| `max_running_requests` | 64 | 512 | `:243-250`(drafter 需要更多 `req_to_token` 行放 transient backbone 与 (K+1)×F 分支链暂存) |
| `disable_radix_cache` | 不动 | **True** | `:271`(逐 commit 的截断释放走非 refcount-aware 的 allocator free) |
| `mamba_radix_cache_strategy` | 不动 | `"no_buffer"` | `:274` |
| `enable_mixed_chunk` | False | False | `:282-286` |
| `speculative_algorithm` | 保留 | **清成 None** | `:164-181`(drafter 的 ModelRunner 必须按纯 decode 引擎定尺,否则 decode graph 会按 num_draft_tokens 个位置回放、破坏枚举 harness 的 1-token 步) |
| `speculative_num_draft_tokens` | K+1 | **清成 None** | 同上(hybrid draft 上 mamba 池否则会按 `[层, slot, num_draft_tokens, heads, dk, dv]` fp32 分配,有用缓存尺寸下达数百 GB) |

### 直接拒绝的组合

`enable_dp_attention`、`dp_size != 1`、`pp_size != 1`——三者都会 `raise ValueError`。
page size **两侧都不受限**(drafter 引擎是 page-aware 的,verifier 从无 page 依赖)。

---

## 2 · 环境变量

51 个 `SGLANG_*DECOUPLED*` 变量全部有活跃读点(**无死变量**)。下面按作用分组,标注默认值。
✅ = 默认开,⛔ = 默认关。

<a id="prod-env"></a>

### 2.1 生产形态必开的五件套

标准 benchmark 用的 drafter 环境(**注意其中三个默认是关的**):

```bash
SGLANG_ENABLE_DECOUPLED_CHAIN_GRAPH=1      # ⛔ 默认关 → K 步链压成一张图
SGLANG_ENABLE_DECOUPLED_DRAFT_PRELAUNCH=1  # ⛔ 默认关 → extend 半程预发射
SGLANG_ENABLE_DECOUPLED_CHAIN_PRELAUNCH=1  # ⛔ 默认关 → 链回放也进预发射序列
SGLANG_ENABLE_DECOUPLED_DEVICE_PACK=1      # ✅ 默认开(显式写出以防回归)
SGLANG_DECOUPLED_ENUM_WAIT_MS=200          # ▸ 默认 200
```

> 前三个默认关是有意的("default off until gated"):它们是这条线上收益最大、也最容易出值路径 bug 的机制
> (见 [D28](05-pitfalls.md#d28))。**纯 attention draft 模型上引擎会自动把前两个关掉并告警。**

### 2.2 门与节拍(verifier)

| 变量 | 默认 | 作用 |
|---|---|---|
| `SGLANG_DECOUPLED_ENUM_WAIT_MS` | `200` ▸ | 每轮等块的上界(ms)。`0` = 纯异步不等 |
| `SGLANG_ENABLE_DECOUPLED_ADAPTIVE_GATE_WAIT` | `True` ✅ | 用实测到达时间定预算(见下方公式) |
| `SGLANG_ENABLE_DECOUPLED_STREAM_GATE` | `True` ✅ | C6 门做成 forward 流上的 host-func 节点,而非停 scheduler 线程 |
| `SGLANG_ENABLE_DECOUPLED_DOORBELL` | `False` ⛔ | 设备侧 `cuStreamWaitValue32` 门。注释写明 **driver 580.126.09 上验证为 UNSAFE** |
| `SGLANG_ENABLE_DECOUPLED_VERIFY_PREP_AHEAD` | `True` ✅ | 在等门之前先把 verify 里与块无关的那半准备好 |
| `SGLANG_ENABLE_DECOUPLED_SELECT_GRAPH` | `False` ⛔ | 把 select 也捕获成图(按 bs × read-slot 分桶) |

> **`ENUM_WAIT_MS` 不等于实际等待。** 自适应开启且采满 20 次到达后:
> `budget = min(ceiling, max(0.008, 4.0 × EWMA + 0.005))`——200ms 只是**上界与 bootstrap 值**。
> 日志里 `gate_budget_ms=8.0` 表示已落到公式地板,即块基本早就到了。

### 2.3 drafter 的轮结构与图

| 变量 | 默认 | 作用 |
|---|---|---|
| `SGLANG_ENABLE_DECOUPLED_FUSED_EXTEND` | `True` ✅ | advance + glue 合成一次批量 forward |
| `SGLANG_ENABLE_DECOUPLED_EXTEND_GRAPH` | `True` ✅ | 融合 extend 捕获成 `DRAFT_EXTEND_V2` 图(行 pad 到 W=2K+1) |
| `SGLANG_ENABLE_DECOUPLED_TOPK_GRAPH` | `True` ✅ | 猜测尾部(fused top-F + poison + branch select)捕获成图 |
| `SGLANG_ENABLE_DECOUPLED_FUSED_TOPK` | `True` ✅ | top-F 用一个融合 kernel 代替 scatter + `torch.topk`(约省 12 次发射) |
| `SGLANG_ENABLE_DECOUPLED_CHAIN_PLAN` | `True` ✅ | K 步 branch decode 预先 staging,循环里只重绑 token |
| `SGLANG_ENABLE_DECOUPLED_CHAIN_GRAPH` | `False` ⛔ | 整条 K 步链一张图(按行数分桶) |
| `SGLANG_ENABLE_DECOUPLED_CASE0_CHAIN_GRAPH` | `True` ✅ | case-0 链也走链图。**依赖链图,所以出厂默认下是惰性的** |
| `SGLANG_ENABLE_DECOUPLED_PREP_AHEAD` | `True` ✅ | 空窗里预建下一轮的 alloc 与 batch 骨架 |
| `SGLANG_ENABLE_DECOUPLED_PREADVANCE_FAST_ALLOC` | `True` ✅ | restage preadvance 的模板 fast path |
| `SGLANG_ENABLE_DECOUPLED_PINNED_H2D` | `True` ✅ | 小索引/长度张量走 pinned 异步 H2D |

### 2.4 预发射与值平面

| 变量 | 默认 | 作用 |
|---|---|---|
| `SGLANG_ENABLE_DECOUPLED_DRAFT_PRELAUNCH` | `False` ⛔ | extend 半程在 commit 门后预发射(bs==1 fast path) |
| `SGLANG_ENABLE_DECOUPLED_CHAIN_PRELAUNCH` | `False` ⛔ | 链回放也排进预发射序列(需链图 + 上一项) |
| `SGLANG_ENABLE_DECOUPLED_DEVICE_PACK` | `True` ✅ | 设备侧打包块 |
| `SGLANG_ENABLE_DECOUPLED_EARLY_JUDGE` | `True` ✅ | 微秒级 match kernel 在 dispatch 时判 commit。**`and device_pack`——关掉 DEVICE_PACK 会静默连带关掉它** |
| `SGLANG_ENABLE_DECOUPLED_FUSED_ARM_COW` | `True` ✅ | armed 分支头 COW 的 (src,dst) 表一次 kernel 建好 |
| `SGLANG_ENABLE_DECOUPLED_FUSED_PAGE_ROW` | `True` ✅ | preadvance 页表行一次 kernel 写完 |
| `SGLANG_ENABLE_DECOUPLED_SCATTER_CONSUME` | `False` ⛔ | 命中轮的 commit 走 scatter kernel 进 replay-fb 静态缓冲 |

### 2.5 枚举宽度

| 变量 | 默认 | 作用 |
|---|---|---|
| `SGLANG_ENABLE_DECOUPLED_ADAPTIVE_FANOUT` | `True` ✅ | 推送延迟超预算时自动减半有效 fanout。**仅当 `ENUM_WAIT_MS > 0` 且 `fanout > 1` 时生效**,预算 = `0.75 × wait_ms` |
| `SGLANG_ENABLE_DECOUPLED_DEAD_GUESS_EXCLUSION` | `True` ✅ | 各 accept case 的"死"猜测在 top-F 前被屏蔽(仅 glue fast-path 轮) |
| `SGLANG_ENABLE_DECOUPLED_PER_CASE_FANOUT` | `False` ⛔ | 逐 case 列预算:case 0→F、case K→min(2,F)、中间→1 |
| `SGLANG_DECOUPLED_PER_CASE_FANOUT_BUDGETS` | `""` ▸ | 显式逐 case 预算(如 `"4,1,1,2"`)。**它比上面的开关先被读,设了它就等于打开逐 case 预算** |

### 2.6 已封存的实验开关(默认关,勿在生产开)

`SGLANG_ENABLE_DECOUPLED_TOP1_PRERUN`、`SGLANG_ENABLE_DECOUPLED_BET_PREBUILD`(及其同族)——
投机式提前推送,机制全通但被判"空窗不免费",见 [07-open-questions S2](07-open-questions.md#s2)。
两者都**只在 zmq 数据面下可用**。

### 2.7 调试与观测(全部默认关)

| 变量 | 作用 |
|---|---|
| `SGLANG_DEBUG_DECOUPLED_HOST_BANDS` | drafter 引擎的零同步逐段 host 计时(mean/p90/max 汇总) |
| `SGLANG_DEBUG_DECOUPLED_VERIFY_PROFILE` | verifier 的轮时间线分解日志 |
| `SGLANG_DEBUG_DECOUPLED_SELECT_GRAPH_CHECK` | 逐轮 eager 重算对拍 select 图 |
| `SGLANG_DEBUG_DECOUPLED_BET` | miss 取证 dump(限流) |
| `SGLANG_TEST_DECOUPLED_LOOPBACK` | 单进程假 mesh + 脚本化 drafter(`"garbage"` / `"stale"`),**测正确性兜底用** |

### 2.8 与解耦联动的融合开关(不带 DECOUPLED 前缀)

| 变量 | 默认 | 说明 |
|---|---|---|
| `SGLANG_OPT_USE_FUSED_SILU_MUL_QUANT` | `True` ✅ | shared-expert 的位忠实 silu+quant 融合核 |
| `SGLANG_OPT_USE_FUSED_GDN_NORM_QUANT` | `False` ⛔ | GDN norm+quant 融合。**位一致但会掉接受率**,见 [D27](05-pitfalls.md#d27) |

---

## 3 · 三张速查表

### 复现 benchmark(照抄)
```bash
# drafter 侧
SGLANG_ENABLE_DECOUPLED_CHAIN_GRAPH=1 SGLANG_DECOUPLED_ENUM_WAIT_MS=200 \
SGLANG_ENABLE_DECOUPLED_DRAFT_PRELAUNCH=1 SGLANG_ENABLE_DECOUPLED_CHAIN_PRELAUNCH=1 \
SGLANG_ENABLE_DECOUPLED_DEVICE_PACK=1
# verifier 侧:不需要任何 env,全默认
```

<a id="bisect"></a>

### 出问题时的二分顺序
```bash
# 1. 先关预发射族(最可能出值路径问题)
SGLANG_ENABLE_DECOUPLED_DRAFT_PRELAUNCH=0 SGLANG_ENABLE_DECOUPLED_CHAIN_PRELAUNCH=0
# 2. 再关链图
SGLANG_ENABLE_DECOUPLED_CHAIN_GRAPH=0
# 3. 再关 extend/topk 图族
SGLANG_ENABLE_DECOUPLED_EXTEND_GRAPH=0 SGLANG_ENABLE_DECOUPLED_TOPK_GRAPH=0 \
SGLANG_ENABLE_DECOUPLED_FUSED_EXTEND=0 SGLANG_ENABLE_DECOUPLED_FUSED_TOPK=0
# 4. 全 eager(最慢但语义最直白)——若此时正确,问题一定在上面某一族
```
> **注意**:`prelaunch 关 + chain graph 开` 的组合在纯 attention draft 上会卡死,别二分到这个格子。

### 观测
```bash
SGLANG_DEBUG_DECOUPLED_HOST_BANDS=1     # drafter 逐段 host 计时
SGLANG_DEBUG_DECOUPLED_VERIFY_PROFILE=1 # verifier 轮时间线
```
