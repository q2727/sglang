# 解耦投机解码(Decoupled Speculative Decoding)

drafter 与 verifier 跑在**独立进程、独立 GPU** 上的投机解码。drafter 不等 verify 结果,
而是把结果的所有可能取值**预测式枚举**成一个 (K+1)×F 的网格推给 verifier,
verifier 在 GPU 上选出匹配的那一行——命中则零等待,未命中则退化为普通 decode,永不出错。

**实测**(B200,Qwen3.5-397B-A17B-FP8 + Qwen3.5-0.8B,bs=1):

| 形态 | decode 吞吐 | 相对无投机 |
|---|---|---|
| MTP(target 自带头) | 484.2 tok/s | ⚠️ 2.47× |
| **decoupled(本工作)** | **430.8 tok/s** | ⚠️ **2.20×** |
| 无投机 | 196.1 tok/s | 1.00× |
| colocated STANDALONE | 146.9 tok/s | ⚠️ 0.75×(负优化) |

**decoupled = MTP 的 89%、colocated 的 2.93×**(同盒同期,可直接比)。
三个 model pair、三台机器上 decoupled 全面胜过 colocated。

> ⚠️ 无投机 196.1 采自**另一台** B200(原盒没跑无 spec 腿),所以"相对无投机"这一列是跨盒的,只作量级参考
> ——本文档集自己的规矩是跨盒不可比([04 §0.4](04-results.md#caliber))。
> 需要同盒证据时看新盒:无 spec 196.1 vs colocated 141.2(同盒同期),比值 0.72×,负优化的结论不依赖跨盒。
> 另注:decoupled 用 5 张卡(TP4 verifier + 1 张 drafter),其余形态 4 张;
> 按**每卡**归一,2.93× 折算为 2.34×([04 §0.5](04-results.md#caliber))。

---

## 该读哪一篇

| 你想做的事 | 读这篇 |
|---|---|
| 看不懂 seat / carrier / glue / stamp 这些词 | **[01-architecture §0 术语表](01-architecture.md)** — 先看这个,两分钟 |
| 理解设计与取舍 | **[01-architecture](01-architecture.md)** — 原理→实现→契约,逐机制展开 |
| 查一个 flag / 环境变量 | **[02-configuration](02-configuration.md)** — 51 个 env + 全部 CLI 的速查表 |
| 自己跑一遍 benchmark | **[03-benchmarking](03-benchmarking.md)** — 可照抄的命令 + 判读纪律 |
| 看我们测出了什么 | **[04-results](04-results.md)** — 全部数据与出处 |
| 避免重复踩坑 | **[05-pitfalls](05-pitfalls.md)** — 8 类坑,每条附可外推的判据 |
| 在新机器上从零复现 | **[06-reproduction](06-reproduction.md)** — 带验收判据的 runbook |
| 接着往下做 | **[07-open-questions](07-open-questions.md)** — OPEN / 已封存 / 未验证 |

**最短路径**:想快速判断这套东西值不值得用 → 读本页 + [04-results §1](04-results.md)。
想上手改代码 → [01](01-architecture.md) → [06 §3 冒烟](06-reproduction.md) → [05](05-pitfalls.md)。

---

## 30 秒理解设计

```
verifier 进程(target, TP=N)                    drafter 进程(draft, TP=1, 独占卡)
┌────────────────────────────┐                 ┌────────────────────────────┐
│ TARGET_VERIFY forward      │  ── commit ──▶  │ 枚举 (K+1)×F 网格          │
│   + GPU select(选中匹配行)│   accept_len    │   每格 = 一条 K 步 draft 链│
│ C6 门:host-func 节点       │   + bonus       │ armed:空窗里预发射下一轮   │
│   排在 verify 流上          │ ◀── 枚举块 ──   │ chain graph:K 步一张图     │
└────────────────────────────┘                 └────────────────────────────┘
        两条 zmq 平面:控制(生命周期)+ 数据(每轮块与 commit)
```

三个关键设计:

1. **预测式枚举**——不等结果,把结果的所有可能取值都画出来。miss 只损吞吐不损正确性
   (fallback 行的根是真 bonus,垃圾尾巴被 verify 逐 token 拒掉)。
2. **一切等待都在 host 车道**——两个进程的 GPU 之间没有任何直接同步。C6 门做成 forward 流上的
   `cudaLaunchHostFunc` 节点,把块的到达截止时间从"调度器醒来"推迟到"GPU 真正消费",
   整个 launch span 变成额外的到达窗口。
3. **优化目标是 host 发射链,不是 draft kernel**——draft forward 只占轮周期的 ~16%,
   地板在 commit 着陆、行/页表准备、kernel 发射这串 host 工作上。armed 预发射、chain graph、
   fused ops 全都在打这个目标。

---

## 代码地图

| 模块 | 行数 | 职责 |
|---|---|---|
| `speculative/decoupled_draft_engine.py` | 8479 | drafter 枚举引擎(轮结构、网格、armed、状态编排) |
| `speculative/decoupled_verify_manager.py` | 980 | verifier 侧总控(C6 门、到达板、隔离/重播种) |
| `speculative/verify_worker.py` | — | verify-only worker + GPU select |
| `speculative/decoupled_draft_manager.py` | 728 | drafter 事件循环、锁步节拍、自适应 fanout |
| `speculative/decoupled_fused_ops.py` | 1239 | 14 个 host 装配下沉 kernel |
| `speculative/decoupled_chain_graph.py` | 390 | K 步链的 CUDA graph |
| `speculative/decoupled_enum_buffer.py` | — | 枚举块的显存缓冲(三代 + stamp 版本化) |
| `speculative/decoupled_spec_io.py` | — | 线材 schema |
| `speculative/decoupled_spec_transport.py` | — | zmq mesh 抽象 |
| `speculative/{drafter,verifier}_ipc_thread.py` | — | 两侧 IPC 守护线程 |
| `speculative/decoupled_stream_gate.py` | 97 | C6 host-func 门 |
| `utils/thread_band_recorder.py` | 227 | 把工作线程的 host band 注入 profiler trace |

> 同目录下的 `decoupled_draft_tail_buffer_placement.md` 与 `decoupled_overlap_panorama.md` 是**更早的设计记录**,
> 其主题与结论已被本文档集取代或需带 caveat 阅读,详见 [01-architecture §7](01-architecture.md#legacy-docs)。

---

## 数据与资产

| 资产 | 位置 |
|---|---|
| 逐腿原始 JSONL | `~/Desktop/b200_backup_0813/{newbox,oldbox}/bench/`、`h200/bench_h200.tgz` |
| benchmark harness | **`benchmark/decoupled_spec/bench_matrix.py`(在本仓库里)** |
| profile traces | `~/Desktop/h200_prof397/`(397B dec vs MTP)、`~/Desktop/b200_prof27/`(27B) |
| 机器重建手册 | `~/Desktop/b200_restore_kit/RESTORE.md` |
