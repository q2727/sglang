> **[SUPERSEDED · 2026-08]** 本文讨论的 `DraftTailBuffer` 与其响应式逐 token 数据平面
> **在当前代码中已完全不存在**(被枚举式数据平面取代),文中的 C6 结论("门绝不与 device fence 耦合")
> 也已被现在默认启用的 `StreamGate`(forward 流上的 host-func 节点)推翻。
> 保留本文是为其推理过程:值依赖的对账不应变成设备分支——这仍是"GPU select + fallback"的理由。
> **任何 file:line 都不要直接采信。** 当前设计见 [`docs/decoupled_spec/`](../../../../docs/decoupled_spec/README.md)。

# Decoupled DraftTailBuffer 放置权衡:CPU vs GPU

> 决策记录:decoupled verifier 的 `DraftTailBuffer` 放 CPU 还是 GPU,以及各自能拿到多少 overlap。
> 配套:[decoupled_overlap_panorama.md](decoupled_overlap_panorama.md)(overlap 运行时全景 + C1–C6 + verifier/drafter 接缝)。
> 溯源:标 `✓ main` 的引当前工作树;标 `fork@16c8cf4a` 的来自原型 PR #22520(`sisyphus111/sglang_dev`,不在 main);标 `1b` 的来自 `zhendonghua/sglang@decoupled-spec-1b-draft-tail-buffer`。
>
> **TL;DR**:**token tail buffer 留 CPU;把 valid-position / accept_len / committed(纯数值)做成 GPU device relay(= EDGE-B)。** GPU 化 token buffer 省的是 tiny H2D、碰不到最深的 C6、还要把便宜的 host 对账逻辑变成 D2H round-trip 或反-GPU kernel;真正解锁 overlap 的杠杆(EDGE-B 长度 relay)与 token 存哪**正交**。

---

## 0. 先厘清:两个正交的东西,别绑一起

讨论"buffer 放哪"最容易犯的错是把两样东西当成一体:

| 东西 | 内容 | 本质 |
|---|---|---|
| **A. token tail buffer** | drafter 流回的 draft token 序列 + **对账状态机** | 值依赖的 **host 控制流** |
| **B. 位置计数器** | `committed_len` / `accept_len` / `new_seq_lens` / valid-position | **纯数值**(= colocated 的 `new_seq_lens_buf` / EDGE-B) |

**真正的 overlap 杠杆是 B,不是 A。** 而 B 能否上 GPU 与 A 存哪**完全无关**——这份文档的核心结论就是:**A 留 CPU,B 上 GPU(device relay)。**

---

## 1. `DraftTailBuffer` 到底是什么:对账状态机,不是数据 buffer

它的核心工作不是"存 token",是**逐 token 值比较 + 分支决策**:
- **append**:`base_committed_len < can_accept_prefix_len → return "stale_base"`;逐 token contiguity 检查;协议守卫 `RuntimeError`(`1b:draft_tail_buffer.py:193/464-465`)。
- **consume**:前缀匹配 `while match: del tail[:matched]; committed_len += matched`;不匹配→清空 + 抬 `can_accept_prefix_len` + `pending_expected`(`1b:draft_tail_buffer.py:202-236`)。
- **snapshot**:per-req 用 `committed_len` 判 consumable tail(`1b:draft_tail_buffer.py:637`)。

这些是 **"CPU 读值 → 做决策 → 改控制流"**。`committed_len` 今天是 host int(`1b:draft_tail_buffer.py:38`)。**这就是"最大难点"的根源(见 §3)。**

写方 = 独立 daemon 线程 `DraftProxyThread`,写 hook = **网络到达**(`fork@16c8cf4a:draft_proxy.py:143/158`);读方 = scheduler 线程每 verify step 的 `get_draft_snapshots`(非破坏性,不推进 committed_len);两侧共享**一把 `threading.Condition`**(`fork@16c8cf4a:draft_tail_buffer.py` / `1b:301-302`)。

---

## 2. CPU 方案(现状)

### 架构
```
[drafter 进程] --ZMQ--> [DraftProxyThread(daemon)] --append(host list)--> DraftTailBuffer(CPU)
                                                                              │ threading.Condition
[scheduler 线程] --get_draft_snapshots(读拷贝)--> 组 verify batch ------------┘
```
- **对账逻辑**:host 控制流,O(1) list 操作,分支天然在 CPU 跑。
- **写读互斥**:一把 `threading.Condition`,非竞争 acquire 纳秒级,锁内一次 list copy 给出**无撕裂 snapshot**——**免费**。

### overlap 程度(verifier 引擎内)
- **能 overlap**:`forward N` (GPU) ∥ `pop_and_process(N-1)`(CPU:emit 用户 token + 发 `VerifyCommit_{N-1}`)。VerifyCommit 的 ZMQ send 异步(`fork@16c8cf4a:draft_proxy.py:82`)。
- **prep(over-alloc + read snapshot)**:依赖 host `committed_len`(accept 落 CPU、`pop_and_process` 里推进),所以**滞后一迭代**,用 committed_{N-1} 组 batch_{N};靠 `base_committed_len` 对齐(stale-base reject)吸收滞后。
- **净**:拿到 depth-1 的"处理上一轮结果 ∥ forward"这半;prep 的关键路径依赖 host committed,不能像 colocated 那样纯设备 future enqueue-ahead。

### 优缺点
- ✅ 对账逻辑在 CPU(它本来就是 host 控制流);写读互斥免费;实现简单、已存在。
- ❌ prep 依赖 host committed_len(host round-trip),verifier 引擎内 overlap 只到基本盘。

---

## 3. GPU 方案(假想)

### 架构(若强行做)
```
[drafter 进程] --ZMQ--> [IPC 线程(自建 CUDA stream)] --H2D append--> GPU tensor buffer
                                                                          │ 同步?(见下)
[forward stream] --verify kernel 读 [0, valid]-----------------------------┘
```

### 同步:append-only 可无锁(上一轮 PD 质疑纠正的点)
- **不是"一定要锁/串行化"**:若设计成 **append-only + valid-position 门**(SPSC lock-free)——写方只 append `[valid+1,…]`、从不改 `[0,valid]`,读方只读 `[0,valid]`,读方等 "`[0,valid]` H2D 完成" 的 completion 信号(event/flag)——**可无锁并发**,借的正是 PD 的 completion 模式(见 §5)。
- 备选 double-buffer/ping-pong:第二块 GPU buffer + host swap latch → 加 2x 显存 + 把想删的 host 同步加回来。

### ★ 最大难点:对账逻辑是 host 控制流,GPU 驱动不了分支
这是 GPU 化的**根本 mismatch**,比同步/收益/C6 都本质:
- **(a) 决策留 CPU + `.item()` D2H 拉值** → host round-trip,**比 CPU buffer 更慢**,token 存 GPU 白搭。
- **(b) 对账状态机重写成 GPU kernel**(逐 token 比较、stale 判断、prefix 匹配、pending 管理) → **巨大重写 + 反-GPU**(分支密集、序列依赖,warp divergence)。

即:**数据能上 GPU,逻辑上不去。**

### overlap 程度
- **省的**:draft-token 的 H2D——但 payload 是标量 int(`DraftTailStreamOutput{base_committed_len, new_token_pos, new_token}`,[decoupled_spec_io.py:124-147](decoupled_spec_io.py#L124) ✓ main),每 req 每 step 一个 int,**微小**。
- **不碰的**:**C6**——draft token 的**值**是另一进程经 ZMQ 送到 verifier **host** 的 Python int;ZMQ 到达 host 这步**没有 CUDA 原语**;H2D 是"到达之后、进程内"的事。GPU 化只改 token 存哪(WHERE),**不改它相对 verify step 何时可知(WHEN)**。C6 两方案一样。
- **不碰的**:真正的 host 串行大头 = snapshot+bind + `broadcast_pyobj`(TP 广播),和 token 存哪无关。
- **加的**:IPC 线程的 CUDA stream/context 管理 + H2D completion event。

### 优缺点
- ✅ 若纯数据流(无对账),append-only 可无锁——但 DraftTailBuffer 不是纯数据流。
- ❌ 对账逻辑上不去(§3 最大难点);省的微小;不碰 C6;加 CUDA 复杂度。

---

## 4. overlap 程度对比表

| 维度 | CPU 方案(现状) | GPU 方案(假想) |
|---|---|---|
| **对账逻辑**(stale-reject/prefix-match/consumable) | host 控制流,便宜,天然在 CPU | ⚠️ **驱动不了 GPU**:`.item()` D2H 或重写 kernel(反-GPU) |
| **IPC 写 vs verify 读互斥** | 免费 `threading.Condition`(纳秒) | 无 host 锁;append-only + completion 可无锁,或 double-buffer(2x 显存 + host latch) |
| **省下的传输** | — | tiny token H2D(标量 int) |
| **prep overlap** | 依赖 host committed_len,滞后一迭代,靠 base_committed 对齐吸收 | 若 token 存 GPU 但决策留 CPU,仍要 D2H 拉值 → 不改善 |
| **C6 跨进程到达门** | host latch | **一样**(ZMQ 到 host 无 CUDA,H2D 在到达之后) |
| **真正 host 串行大头**(snapshot+bind / broadcast_pyobj) | 在 | **一样在**(与 token 存哪无关) |
| **实现复杂度** | 低(已存在) | 高(多线程 CUDA + 对账逻辑重写/或 D2H) |

---

## 5. 正交的真杠杆:EDGE-B 长度/accept_len device relay(带例子)

**一句话:EDGE-B = 把这一轮 verify 产出的 `accept_len`/`new_seq_lens` 放进一块 GPU 常驻 tensor(`new_seq_lens_buf`),让下一轮 schedule 直接用它,而不是等这个值 D2H 回 CPU。**([overlap_utils.py:154-156/280/305-314](../managers/overlap_utils.py#L154) ✓ main)

**为什么它是"杠杆"**:§2 讲过 verifier prep 的痛点——下一轮 over-alloc 要知道"从哪个 position 开始"(= 上一轮 accept 后的 seq_lens),而这是 verify(forward) 的输出。若 schedule 等它的 **host 值**,就得等 forward 结束 + D2H → 串行。EDGE-B 让 schedule 改用它的 **GPU tensor**(device future):CPU 排 op 时只用 `req_pool_idx` 句柄、不碰具体值,值延到 forward 入口在 GPU 上解析 → schedule 能和 forward **并行**(prep-ahead)。

**数字例子**(req A 在 `req_to_token` 第 5 行 = `req_pool_idx=5`;`num_draft_tokens=4`):

```
轮 N — verify A:
  committed = 10                                  # A 已确定 10 个 token
  verify + eagle_sample: 接受 3 个候选 → accept_len = 3   # GPU 上的张量
  new_seq_lens = 10 + 3 = 13
  on_publish(new_seq_lens):  new_seq_lens_buf[5] = 13     # 写 GPU tensor + record publish_ready
                                                          # ★ 13 在 GPU 上算,不落 CPU

轮 N+1 — verify A 的下一段(schedule 阶段,与轮 N 的 forward 并行):
  要 over-alloc A 的下一个窗口 → 需要"从 position 13 开始"分配 4 个 slot
  ❌ 用 host 值: 等 accept_len D2H 落 CPU → committed=13(host int)
                但这要等轮 N forward 结束 + D2H → schedule 串行卡在 forward 之后
  ✅ EDGE-B:     schedule 用 req_pool_idx=5 句柄排 over-alloc op,
                起始 position 用 new_seq_lens_buf[5] 这个 GPU tensor(值延到 kernel 执行时=13);
                CPU 排 op 不等值 → schedule 与轮 N forward 并行
  forward 入口: resolve_seq_lens_cpu 用 publish_ready gate 拿 batch.seq_lens = new_seq_lens_buf[5] = 13
```

**关键 1:EDGE-B 不猜、不 rollback**(区别于 Idea 1)。它不假设"全接受",而是把**真实**的 accept 结果留在 GPU tensor,schedule 用句柄延迟解析 → 永远是真值,不需要事后回滚。这就是全文反复说的"位置变长用 future relay 吸收,不用 guess+rollback"。

**关键 2:这个 CUDA event 为什么不串行化**。`publish_ready` 是 verifier **进程内**的 event(verify 在本进程 GPU 算 accept_len、在 forward stream record)。下一轮 schedule 排 `wait_event(publish_ready)` op **立即返回**(CPU 不停),只是让 schedule stream 上后续 GPU op 排在它之后——**device 侧 ordering,不是 host block**([overlap_utils.py:274-279](../managers/overlap_utils.py#L274) ✓ main)。所以对同进程生产者它是"正确且非串行化"的门。(对比 §3:GPU token buffer 的写方是跨线程/跨进程的,event 就退化成串行 fence——这是两者的分水岭。)

### C6 门:按 PD 的 completion 信号建模(带例子)

EDGE-B 解决了"位置(seq_lens)",但还有一个跨进程的东西要等:**这一轮要验的 draft token 值,是 drafter 另一个进程经 ZMQ 送来的**。C6 门 = "launch verify forward 前,确认这些 token 到了"。

**它不该是 GPU 锁,该是 host 完成信号——照抄 PD**:
- **PD**:prefill 把 KV 写进 decode 的 GPU 内存,decode 不能直接读,要 poll 一个 host 标志 `KVPoll.Success`(写完才翻)→ Success 才准入([disaggregation/base/conn.py:79-84](../disaggregation/base/conn.py#L79) ✓ main)。
- **decoupled**:drafter 每个 `DraftTailStreamOutput` 带 `new_token_pos`(token 位置)+ `base_committed_len`(drafter 基于哪个 committed 产的)([decoupled_spec_io.py:128-142](decoupled_spec_io.py#L128) ✓ main)。C6 门就 poll:"buffer 里 A 的 draft tail 是否 valid 到我需要的 position?"——这是**连续版的 `KVPoll`**(PD 是 per-request 翻一次 Success,decoupled 是持续推进的 valid-position)。

**C6 门例子**(接上例,轮 N+1 要从 position 13 验 4 个):
```
门检查: buffer 里 A 的 tail 是否 valid 覆盖 [13, 17)?
  稳态(drafter 超前): drafter 早 draft 到 17+,tail 已到 → 门绿灯,直接 launch verify
  drafter 落后/冷启动:  tail 只到 14 → allow_partial 下只验 [13,14)(其余 padding、tail_len 截断),
                       或 allow_partial=False 下等 drafter 补到 17
```
稳态下门几乎总绿灯(drafter 超前 W 个 token),所以 C6 **不是每轮阻塞、是"buffer 不够时的兜底等待"**(和 §2 一致)。**门只做 host 到达判断,绝不与 GPU device fence 耦合**(否则重蹈 `verify_done` 覆辙)。

---

## 6. 净结论 + 落地顺序

**裁决:三层东西分开放。**

| 东西 | 放哪 | 具体怎么做 |
|---|---|---|
| **token tail buffer**(draft token + 对账状态机) | **CPU**(不动) | `DraftProxyThread` 继续 append 到 host list,`threading.Condition` 保一致 snapshot——它是 host 控制流(§3),GPU 驱动不了 |
| **位置数值**(accept_len / seq_lens) | **GPU tensor(EDGE-B)** | verify 后 `on_publish(new_seq_lens)` 写 `new_seq_lens_buf`;schedule 用 `req_pool_idx` 句柄 + device tensor over-alloc,不等 host——见 §5 例子 |
| **C6 跨进程到达门** | **host 完成信号** | verifier launch verify forward 前 poll "draft tail valid 到需要的 position?"(用 `base_committed_len/new_token_pos`,像 `KVPoll`),不发明 GPU 锁 |

一句话理由:**buffer 的价值在对账逻辑(host 控制流)、不在存 token;数据能上 GPU、逻辑上不去。而真正该上 GPU 的是纯数值(EDGE-B),它恰恰没有对账逻辑。**

**落地顺序(每步做什么 + 为什么这个顺序)**:
1. **先做 EDGE-B**(唯一真杠杆)。改动:verify 后把 `new_seq_lens` publish 进 GPU `new_seq_lens_buf`(colocated 已有这个接缝,decoupled verify 复用);schedule 的 over-alloc 改用 device tensor 起始 position(`req_pool_idx` 句柄)。效果:schedule 能与 verify forward 并行(prep-ahead),不等 host accept 值。**先做它,因为它收益最大、且独立于其它。**
2. **token tail buffer 留 host**——不动 `DraftProxyThread` + `threading.Condition`。**别搬 GPU**(§3/§7:省 tiny payload、撞对账 host 分支、不碰 C6)。
3. **加 C6 门**——一个 host latch:verifier launch verify forward 前 poll draft-tail-ready(`base_committed_len` 对齐)。稳态绿灯,drafter 落后时兜底等。
4. **最后(可选)**:只当 profiling 显示 `broadcast_pyobj`(TP snapshot 广播)成主导瓶颈时,才单独考虑把 snapshot 广播做成 device-tensor 广播——这是另一个优化,和"token buffer 上 GPU"是两回事,别混。

> **前置提醒**:verifier 引擎内 overlap 本就有限(verify-only 缺 draft_extend,可藏窗口只有 verify forward);decoupled 的主收益是**跨引擎 overlap**(verify N ∥ draft N+1),靠 draft-ahead 窗口 + 异步 IPC,不靠 verifier 引擎内 overlap。所以 buffer 放置的优化空间本身就不大——更不该在收益最小的地方(token buffer 上 GPU)投入最大的复杂度。

---

## 7. 三个想法评审(buffer 放置 / overlap 预取)+ codebase 相似问题对比

> 结论先行:三个想法都属「乐观预取/预分配 + 事后裁决」一族。codebase 已有干净先例,共性解法是——**预取/预分配可乐观、数据可上 device;但 commit-vs-discard 的「决策」与「对账逻辑」始终留 host(或做成 model-compute 的 device gather),从不做成 device 分支;位置变长用 future/length relay(EDGE-B)吸收,不用 guess+rollback。**

### 想法 1 — 乐观预分配 + rollback
- **本质**:假设全接受乐观调度 → verify → 回滚未全接受部分。
- **问题**:KV 半边其实已是现状的 over-alloc(reserve = committed + 2×window)+ 结束时整体回收(`eagle_utils.py:784/794/798`;注释 `:792-794`;commit `schedule_batch.py:2651/2079`;`pop_overallocated_kv_cache` `:1067`)。但「乐观推进 committed/seq_lens + 回滚」不干净:污染 read snapshot、污染 VerifyCommit、破坏 verifier 单调性。
- **裁决**:位置变长应由 **EDGE-B device future** 吸收,不用 guess+rollback。KV 半边已实现,控制半边不要碰。

### 想法 2 — DraftTailBuffer 上 GPU + 专用 H2D writer 线程
- **本质**:把 host 的 draft-tail buffer 搬上 GPU + 独立写线程。
- **问题**:DraftTailBuffer 是**值依赖的 host 对账状态机**(stale-base reject / prefix match / consumable 判定 / protocol guard,全是 host 控制流)。GPU 张量驱动不了 Python 分支:要么 `.item()` D2H(更糟),要么改写成反-GPU 的分支密集 kernel。且 payload 极小、不触碰 C6。
- **裁决**:不值得。真正的杠杆是 **EDGE-B(length / accept_len device relay)**,与 token buffer 正交。

### 想法 3 — GPU 增量 buffer(预取下一轮 verify input + 条件提交)
- **本质**:GPU buffer 记录 snapshot 之后到达的 draft token;verify 后按是否全接受,决定 append 进下一轮 verify input(full-accept)或丢弃(diverge)。= 预取 + 条件提交。
- **最直接先例**:colocated `draft_extend_for_decode`(verify 后预取下一轮 draft 首层输入,`eagle_worker_v2.py:939-943`)。关键差异:draft_extend 是 **MODEL-COMPUTE 预取**——全接受 vs diverge 的条件被表达成一条 **device gather index**(`select_index = arange + accept_lens - 1`,`:835`,使用 `:904/:908`),diverge 由「取被接受前缀最后一行播种下一轮」天然处理,无 host 分支;而 Idea 3 是 **RECONCILIATION-BUFFER 拼接**,靠 host 分支决定 valid/splice/discard。另一先例 FutureMap 用 pool-index 无条件覆写(`overlap_utils.py:316/326`)+ device length gather `new_seq_lens_buf[fi]`(`:280`)吸收变长,同样无 commit 分支。
- **问题**:
  - (a) 没去掉 host 依赖——「append/丢」仍判 `accept_len == window`(host 值),只搬了 token **存储**上 GPU,**控制流留 host**,撞回 Idea 2 的最大难点。
  - (b) 增量 buffer 装的仍是 draft token + 那套 host 对账状态机,Idea 2 的难点原样继承。
  - (c) 「增量」在现状 CPU buffer 已隐含:drafter 持续 append,下一轮 snapshot 按 host `committed_len` 自然取到;显式化+GPU+条件预取几乎没加东西。更关键,按不对称性——全接受 `accept ≡ window`(确定,host 早知,无需猜),diverge 请求下一轮无法 verify 新 draft(须等 drafter 跨进程 redraft = C6,无 CUDA 原语跨进程)——**要优化的「条件」在两条路径上都是空的**,退化成 append 一段确定续写,而这段续写 KV 已由 over-alloc 预留、位置由 EDGE-B relay 吸收。
- **裁决**:**不是真改进**。draft_extend 与 FutureMap 已是「预取 + 条件提交」的干净版(条件活在 GPU index / pool-index 覆写里)。verifier 正解 = **device gather / EDGE-B length relay**(照抄 `select_index`@`:835` 与 `new_seq_lens_buf[fi]`@`:280`),而非带 host commit 分支的 GPU 增量 buffer。VERIFIER 无 draft_extend 的 GPU 窗口,其 commit 仍会 gate 在 host accept_length + host C6 latch 上。

### codebase 相似问题与解法对比表

| 模式 | 机制(file:line) | commit-vs-discard 决策位置 | 对三个想法的启示 |
|---|---|---|---|
| KV over-alloc + release rejected | 滚动预留 `committed + double_alloc`(`eagle_utils.py:784/794/798`,注释 `:792-794`);commit=host int `+=1`/`=seq_len`(`schedule_batch.py:2651/2079`);结束整体回收 `pop_overallocated_kv_cache`(`:1067`) | host 标量(committed int) | Idea1 KV 半边已实现;变长靠肥预留吸收,非 guess+rollback |
| PD KV 预取 + completion | `KVPoll{Failed=0…Success=4}`(`disaggregation/base/conn.py:79-84`);`Success` 才准入(`decode.py:1721`),`Failed` 中止(`:671/:1690`) | host 轮询枚举 | 跨进程完成只能 host 门控 → 对应 Idea3 的 C6 latch |
| radix 增量 append | `[:kv_committed_len]` 切片后 `insert`(`radix_cache.py:443/451/465`),投机尾巴从不插入 | host 标量切片 | 与 Idea3「丢弃」最像的先例:丢弃=host slice,非 device 分支 |
| mamba checkpoint pool | int8 caching:`store_from_active`/`load_to_active`(`mamba_checkpoint_pool.py:245/253`,alloc/free `:229/:232`),由 radix hit/miss 驱动 | host(radix hit/miss) | checkpoint→restore 由 host 事件驱动;非 Phase-7 rollback ring(main 无) |
| grammar rollback | DFS `accept_token`(`spec_utils.py:401`)→ `rollback(1)`(`:416`) | 纯 host python 状态机 | 乐观 advance→rollback 全是 host 控制流;device 张量驱动不了——Idea3 host 分支不可上 device 的反例 |
| launch-ahead + FutureMap | device 上 stash EDGE-A(`overlap_utils.py:316/326`)+ publish EDGE-B(`:305/:309`),下一轮 resolve(`:69/:91`)+ gated D2H(`:263/:280/:298`);overlap loop `scheduler.py:1553`,copy_done gate `:3267` | 无 discard;pool-index 无条件覆写,变长靠 device length gather | 可借数据布局(pool-index device buffer);不可借条件式 device commit——它只 relay+resolve 真值。verifier 应扩展 EDGE-A/B,而非新造带 host 分支的 GPU 增量 buffer |

### 共性 takeaway
凡做 commit-or-discard 的先例,**决策一律落 host 标量**(committed int / poll 枚举 / rollback 计数 / radix hit-miss);唯一把数据留 device 的 FutureMap **拒绝把它做成条件**,改用 pool-index 无条件 relay + device length gather 吸收位置变长。因此 decoupled verifier 的正确形态是 **device gather / EDGE-B length relay**,不是「GPU 预取 buffer + device commit 分支」。三个想法里,KV 半边(Idea1)已落地,token buffer 上 GPU(Idea2)与 GPU 增量 buffer(Idea3)均不推进;把力气投在 EDGE-B。
