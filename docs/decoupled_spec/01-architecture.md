# 01 · 架构与设计原理

> 行号对应 `459ae93227`。每节的结构:**原理**(为什么需要它)→ **实现**(落在哪、怎么写)→ **契约**(不变量与坑)。
>
> 全文一个颜色约定:**GPU 侧**(kernel / graph / 显存)与 **host 侧**(Python 发射 / 轮询 / 跨进程)要分开看——
> 这套设计的全部要点,就是把 host 从关键路径上挪走。

---

## 1 · 出发点:colocated 的两个结构性耦合

投机解码的收益 = `接受长度 / 轮周期`。colocated(draft 与 target 同进程同 GPU 组)在两个方向同时受损:

- **资源耦合**:draft 被迫继承 target 的并行度。397B 开 TP4 时,0.8B 的 draft 也在 TP4 上跑——
  小模型的 allreduce 开销远超算力收益。实测 397B colocated **比不开投机还慢 24–31%**([04-results](04-results.md))。
- **串行耦合**:draft 与 verify 在同一条流上交替,轮周期 = draft + verify,两者永不重叠。

解耦把 drafter 放进独立进程独占一张卡,换来两个新问题:**跨进程的结果依赖**(drafter 要知道 verify 接受了几个
才能画下一轮)与**跨进程的传输延迟**。后面所有机制都是在消解这两个代价。

---

## 2 · 拓扑:两进程、两平面

### 原理
drafter 不是新造的服务,而是**一台完整的、未加修改的 sglang server 换掉事件循环**:
同一个 `Scheduler` 类、完整的 TpModelWorker / 内存池 / CUDA graph / attention backend,零新框架。
解耦行为全部由**组合**注入,不用继承也不用 mixin(仓库风格规则明令 avoid mixins)。

通信按性质分两条平面:**控制平面**(请求生命周期:开座、提交、关座、重播种)与
**数据平面**(每轮的枚举块)。两者共用一条 zmq mesh 但消息类型不同、排空顺序有硬约束。

### 实现

| 关注点 | 位置 |
|---|---|
| 角色分支(verifier 换 worker) | `managers/scheduler.py:800-808` |
| 角色分支(drafter 强制 `spec_algorithm=NONE`) | `managers/scheduler.py:354-361` |
| 角色分支(drafter 换事件循环) | `managers/scheduler.py:4560-4572` → `DecoupledDraftManager.run_loop` |
| manager 挂载(**仅 TP rank 0**) | `managers/scheduler.py:835-862` |
| 批结果钩子 | `managers/scheduler.py:3662-3663` |
| 传输抽象 | `speculative/decoupled_spec_transport.py`(每端 bind 一个 PULL、对每个对端 connect 一个 PUSH) |
| 线材 schema | `speculative/decoupled_spec_io.py`(`DraftSync` / `VerifyCommit` / `DraftClose` / `DraftEnumerationBufferBatch`) |
| 两侧 IPC 守护线程 | `speculative/{drafter,verifier}_ipc_thread.py` |

`verify_worker.py` 的 `VerifyWorker` 继承 `BaseVerifyWorker`(`base_verify_worker.py:26`,全文件 44 行):
verify-only worker 的 `draft_worker` 恒为 `None`,基类里所有 draft 感知的初始化钩子自然 no-op——
内存池、attention backend、CUDA graph 全归 target 的 `TpModelWorker`。

### 契约

- **endpoint 交叉**:`connect_endpoints[k]` 必须是 rank-k 对端的 `bind_endpoint`。
- **排空顺序(verifier)**:controls 先于 commits(`verifier_ipc_thread.py:171-182`)——
  一个请求的 `DraftSync` 必须先于它第一轮的 commit 上线。
- **着陆纪律(drafter)**:`land → sync → publish,一次一条`(`drafter_ipc_thread.py:223-234`)。
  批量 sync 后再发布会让 gated scatter 读到旧代——历史事故:200 轮里 195 轮作废。
- **只有 rank 0 拥有解耦平面**:门的结果依赖墙钟到达,复制的 TP 调度器各自等门会分叉成集合通信 desync。

---

## 3 · 核心思想:预测式枚举

### 原理
解耦后最朴素的做法是**响应式**:等 verify 结果到了再画下一轮。但这样轮周期 = verify + 传输 + draft,
串行链比 colocated 还长。本设计走**预测式**:verify 还没出结果时,drafter 就把结果的**所有可能取值**全画完。

可行的原因是结果空间小且可枚举——一轮 verify 后的状态由两个变量完全决定:

- **接受长度** `a ∈ {1..K+1}`,共 K+1 种(称 accept case)
- **bonus token**:理论上是整个词表,但真实分布高度集中,取 draft 的 **top-F** 覆盖大部分概率质量

于是 drafter 枚举一个 **(K+1) × F 的网格**,每格是"若 verify 接受 a 个、bonus 是第 f 个猜测"世界线下的一条 K 步链。
verifier 收到块后,等真实结果落地,**在 GPU 上选出匹配行**:命中则零等待;未命中则退化为一次普通 decode
(只收 1 token),**永远不会收错 token**。

每条链的解剖:**backbone**(各 case 共享的已提交前缀 KV)+ **glue**(该 case 假设被接受的 draft token,写进
carrier 的私有页行)+ **branch**(以 bonus 猜测为根的新 K 步链)。

### 契约
命中率是本形态的第一敏感参数,且**它与接受长度是同一个环的两个投影**(见 §6.3 的吸收态)。
实测:draft/target 一致性好的 pair(32B+0.6B)hit 0.97–0.985;差的(397B+0.8B)0.75–0.85。

---

## 4 · Verifier 侧

### 4.1 GPU select:一条纯函数里的全部判定

**原理**——判定的输入(上一轮的 `accept_lens`、`bonus_tokens`)本身就是 GPU 上的 verify 输出,
任何 D2H 同步都会拖停 verify 流,而那条流上还排着下一轮的 forward。所以判定必须全程在 GPU 上完成。

**实现**——`speculative/verify_worker.py:124` `select_enum_units`,三步张量运算:

```python
# verify_worker.py:153-171
gen_matches = stamps.eq(base_committed_lens.unsqueeze(1))  # ① 新鲜度:stamp 严格相等
fresh = gen_matches.any(dim=1)
gen_indices = gen_matches.to(torch.int64).argmax(dim=1)
cases = prev_accept_lens.clamp(min=0, max=num_cases - 1)   # ② 用上轮接受长度定位 case 行

case_units = units[batch_arange, gen_indices, cases]       # [bs, F, unit_width]
guesses = case_units[:, :, 0]        # unit 的第 0 元素 = 猜的 bonus = 匹配键
guess_matches = guesses.eq(bonus_tokens.unsqueeze(1))
hits = fresh & guess_matches.any(dim=1)                    # ③ 命中 = 新鲜 ∧ 猜中

fallback_units = bonus_tokens.unsqueeze(1).expand(bs, unit_width)
selected = torch.where(hits.unsqueeze(1), selected, fallback_units)
```

命中行直接就是 verify 的输入:`selected[:,0]` 是根(真 bonus),`selected[:,1:]` 是 K 个 draft。
**miss 行整行替换为 `[bonus, bonus, …]`**——根仍是真实 bonus,垃圾尾巴被 verify 逐 token 拒掉,
该轮退化为 1-token decode。**正确性从不依赖缓冲区内容,只有吞吐依赖。**

**契约**
- 函数是纯的(无副作用),被 `SGLANG_DEBUG_DECOUPLED_SELECT_GRAPH_CHECK` 用来逐轮对拍捕获图。
- hit 结果的 D2H 走 **pinned + async + event**(`_queue_select_hits`,`:491`):pageable D2H 会等整个设备,
  在被 gate 的流上会拖停 scheduler 线程。
- TP>1 时**只有 rank 0 select 后 broadcast**(`:502`),广播兼作屏障;这既是一致性要求也是防挂死。
- `clamp` 只是防设备越界的护栏,不是 miss 来源——case 错了自然猜不中、自然 fallback。

### 4.2 枚举块缓冲:stamp 就是版本号

**原理**——块的着陆(IPC 线程写显存)与消费(verify 流读显存)并发,却**不加锁**。
可行的原因是 stamp 纪律自带版本化:每个块携带"它是从哪个已提交长度枚举出来的",select 用**严格相等**匹配;
不同代块的 stamp 至少差一个 commit 步长,**写了一半的块永远不可能满足读者的等值判定**。
最坏的撕裂读只产生一次 miss。

**实现**——`speculative/decoupled_enum_buffer.py`,`gen_count = 3`(`:96`):

| 代 | 用途 |
|---|---|
| gen 0 / gen 1 | 两个最新的**真块**,轮转写(总是写"上次没写的那代") |
| gen 2 | **投机(prerun/bet)专用槽**,stamp 只在押注猜对时匹配,且从不推进轮转 |

为什么要两代真块:服务第 r 轮的块是两个 commit 之前枚举的,而第 r−1 轮的 commit 已经推来了下一个块——
只留一代,新块会覆盖 r 轮正要 select 的那个,每轮都变 staleness fallback。

**契约**
- 着陆走**私有 land stream**(pinned 暂存 → async H2D + scatter → event 同步后才返回),
  所以"已到达"蕴含"GPU 可见",gather 不需要跨流 fence。
- `reset_slot`(`:344-355`)**只把 stamp 写成 −1、并清 doorbell**,**不清 token 行**——
  留着的陈旧数据是安全的(等值判定永不匹配),清行反而是浪费。

### 4.3 C6 StreamGate:把截止时间从"调度器醒来"推迟到"GPU 消费"

**原理**——verifier 也有一处等待:块没到就发射 select 必然 miss。朴素做法是 scheduler 线程睡等,
但这把截止时间定在了 **host 发射时刻**,白白浪费"发射到 GPU 真正执行"之间的整段排队时间。

C6 把等待做成 **verify 流上的一个 `cudaLaunchHostFunc` 节点**:发射线程立刻返回、本轮的 select/verify
照常排进流里,只有 GPU 推进到门节点时才执行等待回调——**整个 launch span 都变成额外的到达窗口**。

**实现**——`speculative/decoupled_verify_manager.py:743-749`:

```python
if expected and self.arrival_wait_s > 0 and self._stream_gate is not None:
    budget_s = self._gate_budget_s()
    if self._stream_gate.enqueue(
        torch.cuda.current_stream(),
        lambda: self._stream_gate_wait(expected, budget_s),
    ):
        return          # 门已排进流;host 线程继续发射后续 kernel
    # 驱动拒绝节点:回退到 host gate
```

实测收益(记录在 `environ.py:547-551` 的注释里):H200 select hit **0.788 → 0.960**、
B200 0.781 → 0.845,吞吐同升,TEXT_EXACT 双绿。

**等待预算是自适应的**(`:648-666`):

```python
budget = min(ceiling, max(0.008, 4.0 * ewma + 0.005))
# ceiling = SGLANG_DECOUPLED_ENUM_WAIT_MS(默认 200ms),bootstrap(<20 次采样)与 anneal 时用
# EWMA 只采样「到达了的轮」实际等了多久,超时轮被删失 → desync 期永不撑大预算
# 4×(而非 2×):更紧的裁剪会喂饱连续超时隔离、塌掉命中率(历史失败形态)
# anneal:连续超时「恰好 == 2」时给一次 ceiling 停车,重新进入正确相位;只给一次
```

**契约**
- 回调里**不允许任何 CUDA API**(阻塞的 host func 不持有驱动停车状态,着陆 scatter/event/分配照常工作);
  异常被记录并吞掉,该轮自然退化为 fallback。
- **到达板是双道的**:真块走 **GEQ 道**(stamp 单调推进,drafter 的 merge 可能整代跳过,
  座位越过期望值后再等也无益);投机块走**精确匹配道**——bet 的 stamp 是全收假设 `base+K+1`,
  恒 ≥ 一切期望值,喂进 GEQ 道会让 verify 环自由跑进永久 staleness。
- 每 drafter 连续超时 3 次 → 隔离 5 秒;每座位连续超时 4 次 → `DraftSync` 重播种
  (overlap 下滑出 k 代的座位靠等待永远追不回来)。

### 4.4 提交回推

verify forward 一结束,结果就以 `EventedVerifyCommits` 交给 IPC 线程,线程用「`copy_done` event 已 query」
做就绪门(未 record 的 event 会假报 True——历史上读出过 `0x01010101` 的 accept_lens),
然后按**「账本跟结果走、不跟发送走」**推进每请求的已提交长度游标(`verifier_ipc_thread.py:293-362`):
被 resync floor 或负 token 守卫跳过的发送**也必须推进游标**,否则后续每个 commit 都归因到错误的绝对基点。

---

## 5 · Drafter 侧:host 发射设计(本文重点)

<a id="pipeline-model"></a>

### 5.1 两级流水线模型

**原理**——解耦后的系统是一条**两级 1:1 锁步流水线**:verifier 每消费一个块就生产一个 commit,
drafter 每消费一个 commit 就生产一个块。稳态轮周期由慢的一级决定:

```
T ≈ max(V + h,  D)
     └─────┘   └─ drafter 从收到 commit 到块可用的生产周期
       verify GPU 时间 + verifier host 消费/提交开销
```

三个实测事实决定了全部火力分配:

1. **D 的地板不是 draft 算力。** draft forward 本身只要 ~1.2 ms(0.8B,B200),但 D 实测 ≈7.4 ms——
   大头是**跨进程轮询与 host 准备**:commit 着陆、行/页表分配、kernel 发射、块推送。
   profile 佐证:verifier host 线程 **83% 的时间在 `scheduler.recv_requests` 轮询**(约每步 950 次),
   而 verify 的 GPU 段只有 2.10 ms。**优化 draft kernel 几乎无用,优化 host 发射链才有用。**
2. **V+h 与 D 谁大取决于 target。** 27B TP1 时 V+h < D,系统压在 D 地板上;
   397B TP4 时两级咬合,任何一侧 ±50µs 都会移动相位。
3. **地板对机器个体敏感。** 同代码同库栈的两台 B200:MTP(GPU 决定周期)持平,decoupled 差 7%。

### 5.2 轮结构

**原理**——轮分两类:**slow 轮**是生命周期事件(每请求 bootstrap 一次,构建 seat 与全网格,不值得优化);
**fast 轮**是稳态(>97%),路径上每微秒都算数。fast 轮的工作是把预算选中的格子画完:
推进已提交前缀(advance)→ 取 node-0 的 top-F 猜测 → 对每个 case 画 backbone 搭桥的 K 步分支链。

**实现**
- 行选择与 scatter 模板按预算向量预构建(`_build_fanout_variant`,`decoupled_draft_engine.py:4254`):
  选中的行**保持其全 F 池位置**(case c 列 f → 行 `c*F+f`),所以任何预算向量复用同一批 carrier 行,
  每轮只是转发其中一部分。
- **miss 轮塌进 `_case0_round`**(`:7260`):不重建 backbone、不做 glue,只在 case-0 行上跑
  **一条 f_live 行的 decode 链**;块再被 pad 回 (K+1)×F(死格子 guess 写 −1、chain 写 0)。
  这是 §6.3"给 case-0 加宽几乎免费"的机械原因。
- **锁步节拍**由 `_consumable_commit_len`(`decoupled_draft_manager.py:411-431`)维护:每轮恰好消费一个
  verify 轮的 delta,保证每一代块都被生产;积压超过 2 轮(overlap 的正常在途量)才合并跳代,
  每次跳代的代价是 verifier 一次 fallback。

**契约**
- 日志里的 `fast=` / `slow=` 是**每轮布尔**不是座位数(`:4083` / `:4096`),
  混合轮**两个都会 +1**;`slow=` 把 case-0 miss、forced_case0、真 bootstrap 三者合在一个数里。
- **所有计数器都是生命周期累计**,相邻两条日志要**做差**才是速率。

### 5.3 chain graph:K 步链一张图

**原理**——fast 轮的 K 步 branch 解码,eager 形态是 K 次「prepare → launch」的 host 往返;
每步几十个 kernel 发射加 metadata 准备,host 成本远超 GPU 计算本身。CUDA graph 把 K 步压成**一次发射**。
难点在于图要求静态形状与静态地址,而每轮的行数、页表、序列长度都在变。

**实现**——`speculative/decoupled_chain_graph.py`(390 行):
- 按 **`(rows, forked)`** 分桶(不只是 rows);
- 首次遇到某桶时**边执行边捕获**(`try_capture_and_run`,`:124`)——本轮的真实工作就发生在捕获里,不做牺牲轮;
- 此后同形状轮走 `stage → fire`:host 只把本轮的 plan 值拷进桶的静态缓冲(`_fill_meta`,`:99`——
  positions / seq_lens / req_rows / out_locs)。

**契约**(两条都是血的教训,写在代码注释里)
- **`max_rows` = decode graph 的 max bs**:attention backend 的静态 metadata 按它定尺,更多行会越界索引。
  链图一次要跑 (K+1)×F 行 → 越界即 bucket **永久禁用**,退化逐步链,**只打一条 INFO**。这就是 [D29](05-pitfalls.md#d29)。
- **池行必须经静态缓冲读入**:新请求落在新 carrier 行上,把请求 1 的张量烤进图会让请求 2 读旧行——
  实测 acc 3.95 → 1.0。

### 5.4 pre-launch / armed:把下一轮的发射藏进本轮空窗

**原理**——链图消掉了 K 步发射,fast 轮仍剩一段串行 host 工作:advance/glue 的 extend 前处理、
行与页表准备、chain metadata staging。它们都要等 commit 才知道参数——**但只差一个参数**。
下一轮的形状在本轮结束时几乎完全确定,唯一未知的是"verify 会接受几个"。

armed 机制在 drafter 的空窗(starved 95–99% 的等待期)里,**用假设值把下一轮完整准备并预发射**;
commit 到达时只需把"假设错的那部分"用 COW 修正,然后直接 `fire_armed`。
关键路径上的 host 工作从"一整轮准备"缩成"一次 COW + 一次图回放"。

**实现**——`_pre_launch_chain_armed`(`decoupled_draft_engine.py:1765-1992`)。三个工程要点:

- **armed-COW 单 kernel**(`build_armed_cow_locs`,`decoupled_fused_ops.py`):一次 launch 构建整批
  (src,dst) 拷贝位置表,替代逐请求 Python 装配。
- **假设填充的卫生**:armed 序列把 scatter 换成常量假设填充,零新行、零新页、零新图。
  毒值 clamp、影子槽播种、transient 行翻转是三类历史 bug 的修法——教训是
  **假设值既不能污染真值路径、也不能被真值路径污染**。
- **停在"预发射不预推送"**:更激进的 bet/prerun(把"押注全收"的块提前推给 verifier 的 gen-2 槽)
  机制全通、押中率 87–96%,但被判封存:bet 的 GPU 生产时间 ~1:1 折进轮周期,
  **"空窗免费"被证伪**(见 [07 S2](07-open-questions.md#s2))。

**契约**——armed 与链图**默认关闭**,生产形态显式开启(见 [02-configuration §2.1](02-configuration.md#prod-env));
纯 attention draft 模型上被 D28 守卫自动关断(§6.2)。

### 5.5 fused host ops

**原理**——armed 之外还剩几段"小而频"的 host 工作:chain metadata 逐值填充、页表行 prefix|extend 拼装、
extend 分配前处理。每段只有零点几毫秒,但都骑在 dispatch tail 上。修法一致:
**把 Python 侧的逐元素装配下沉为一次 triton kernel**。

**实现**——`decoupled_fused_ops.py`(1239 行,14 个 kernel)。三件套默认全开:
`chain_meta_fill`(标量 + 设备侧 case_of_row 向量,消掉两次 H2D)、
`fill_page_table_row`(行 = prefix|extend 一次填完,kernel 内 int64→int32)、
extend 的 preadvance 模板 fast-path。

**契约**——**热路径上的 triton kernel 必须做特化纪律**:runtime int 会按 `{==1, %16, other}` 隐式特化进
缓存键 → 运行中途反复编译。14 个 kernel 里有 3 个带 `do_not_specialize`,正是逐轮参数变化的那三个。
详见 [D26](05-pitfalls.md#d26)。另:**每条新 fast-path 必须带命中计数器并在日志可见**——
历史教训是门恒假时两腿全等、"全绿"实为"没跑"。

### 5.6 device pack 与 pinned ping-pong

块的打包(把网格行装进 wire 布局)在 GPU 上完成,经 **pinned 镜像 + CUDA event** 交给 IPC 线程:
push 提交后 host 立即返回,IPC 线程以「event 已 query」为就绪门,按 FIFO 头序保代序,1 秒卡头看门狗兜底。

**契约**:pinned 镜像必须 **per-seat ping-pong**——pack 提前一轮覆盖 host 正要读的镜像是历史 bug,
修法是消费时才翻面。另注:`EARLY_JUDGE` 的实际条件是 `flag and device_pack`,
**关掉 DEVICE_PACK 会静默连带关掉 EARLY_JUDGE**。

### 5.7 hybrid(GDN)状态编排

hybrid draft 模型(如 Qwen3.5-0.8B 的 GDN 层)除 KV 外还带每行的递归状态(conv + ssm)。
枚举要求"同一前缀的多个续写",而递归状态是破坏性更新的——所以状态槽必须显式编排,不能像 KV 一样靠页表共享。

引擎自持一个 mamba state arena:每 seat 一个持久槽(只被 advance 推进)、carrier 行**终生固定槽位**
(行→槽映射静态,可进图)、glue 槽保存各 case 消化 glue 后的状态。
extend graph 的**双平面契约**:full-attention 平面按统一宽度 W=2K+1 回放(pad 行重复末 token);
**GDN 平面经图内静态缓冲喂真实逐行长度,递归扫描永不碰 pad**。

这套编排是 armed 值路径的正确性根基——也是它在纯 attention 模型上出 [D28](05-pitfalls.md#d28) 的原因:
那类模型没有这层结构,armed 的假设填充走了未经验证的值路径。

### 5.8 观测:让 host 侧可见

**原理**——主战场在 host 侧,但 torch profiler(kineto)的回调是**线程本地**的:长命工作线程
(两个 IPC 线程、C6 门的驱动回调线程)在 trace 里完全隐形。没有可见性就没有优化。

**实现**——`utils/thread_band_recorder.py`:工作线程各自注册记录器
(`sgl-draft-ipc` / `sgl-verify-ipc` / `sgl-c6-gate-callback`),profiler 启动时打单调钟同步标记,
停止时把 band 以 `user_annotation` 事件**注入导出的 Chrome trace**——于是
`drafter_ipc.land_commit`、`verifier.c6_stream_gate_park` 这些纯 host 事件与 GPU kernel 出现在同一时间轴上。
引擎内另有零同步的逐段 host 计时(`SGLANG_DEBUG_DECOUPLED_HOST_BANDS`)。

---

## 6 · 数值与守卫纪律

### 6.1 verifier 的数值是承重墙

接受率 = draft 分布与 verifier 分布的一致性。verifier 哪怕亚量子扰动,边界 token 的贪心翻转 × 层数累积
→ accept 链被截断。实测:一个离线看只有 1.7% 元素翻桶的现成融合核,上线后 acc 3.21→2.55(−20%)。

修法是自写**位忠实**融合核(精确 fp32 silu + 显式一次 bf16 舍入 + 位精确 ceil-log2;
UE8M0 的 2 幂 scale 让量化除法无舍入)。**验收 = 逐位一致 + e2e TEXT_EXACT + hit/acc 不掉**。
完整故事见 [D27](05-pitfalls.md#d27)。

### 6.2 D28 守卫

armed/prelaunch 的值路径在**非 hybrid** draft 模型上算错内容。守卫按**池类型**判定(不是按架构名):

```python
# decoupled_draft_engine.py:912, 1196-1199
self._hybrid = isinstance(model_runner.req_to_token_pool, HybridReqToTokenPool)
self._d28_pure_attn_fallback = not self._hybrid and (
    envs.SGLANG_ENABLE_DECOUPLED_DRAFT_PRELAUNCH.get()
    or envs.SGLANG_ENABLE_DECOUPLED_CHAIN_GRAPH.get()
)
```

命中守卫时**只关** `_prelaunch_enabled` 与链图 runner 的构造(`:1206-1211`、`:1249-1252`);
`_chain_plan` 与 `_case0_chain_graph` 在它之前读取、保持开启,所以 eager 的逐步链循环照常跑。
守卫后 32B/235B 恢复满接受长度(3.94/3.96)。

### 6.3 case-0 吸收态与列预算

**原理**——miss 有自增强结构:miss → 只提交 1 个 bonus → 下一轮塌进 case 0,
而 case-0 轮只画 F 条链,这 F 个猜测是**唯一逃逸口**;更糟的是 case-0 的 bonus 天然难猜
(top-1 刚被 verify 拒,还被 dead-guess exclusion 显式屏蔽,真 bonus 必然是 rank-2 起)。
**一次 miss 把系统推进低逃逸概率的吸收态**——这解释了 hit 与 acc 为何在不同机器上同向移动。

**实现**——`_resolve_per_case_budgets`(`:4210`)。出厂形状曾是 `[1,1,2,4]`(case K 最宽),
实测是所有分配里最差;commit `459ae93227` 改为 **case 0 → F、case K → min(2,F)、中间 → 1**
(K=3/F=4 即 `[4,1,1,2]`)。等成本(8 行)拿到近似均匀 F=4(16 行)的接受长度,数据见
[04-results §5](04-results.md#budget-exp)。

**契约(三条读代码才知道的)**
1. 三重门的顺序是:`fanout < 2` → 显式预算串 → enable flag。
   **设了 `SGLANG_DECOUPLED_PER_CASE_FANOUT_BUDGETS` 就等于打开逐 case 预算**,不需要再开 flag。
2. **skew 只为全宽度安装**:自适应 fanout 一旦减半有效宽度,该宽度按均匀构建——**skew 被静默丢弃**。
3. wire 块始终是 (K+1)×F(未预算格子毒化),verifier 无感知;所以预算省的是 drafter 行数,
   **不是传输与 select 代价**(这正是 F=4 收益被轮周期抵消的原因,见 [07 O2](07-open-questions.md#o2))。

---

<a id="legacy-docs"></a>

## 7 · 相关既有文档的状态

| 文档 | 状态 |
|---|---|
| `speculative/decoupled_draft_tail_buffer_placement.md` | **已作废**。它的主题 `DraftTailBuffer` 及其数据平面消息在代码中已完全不存在(响应式逐 token 平面被枚举取代);其 C6 结论("门绝不与 device fence 耦合")也被现在的 StreamGate 推翻。仅两点仍有价值:值依赖的对账不该变成设备分支(这仍是 GPU-select-with-fallback 的理由)、EDGE-B 复用决策的出处 |
| `speculative/decoupled_overlap_panorama.md` | **可读但需带 caveat**。§1-12 是上游 overlap 调度器的全景图,机制描述仍成立;但全部行号已漂(基线落后 95 个 commit),且它列为"未来工作"的 read_done 门拓宽**已经落地** |
| `new-roadmap.md` | 计划文档,其中提到的 `SchedulerDecoupledVerifyMixin` **从未存在**——实际实现用的是组合 |

> 两份旧文档的正文没有删除(它们的推理过程仍有考古价值),但**任何 file:line 都不要直接采信**。
