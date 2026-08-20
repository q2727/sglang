# 05 · 坑录:这条线上摔过的坑

> 每条的格式:**症状 → 根因 → 修法 → 可外推的规则**。
> 最后一栏才是真正的资产——症状会过时,判据不会。

## 目录

| 编号 | 一句话 | 类别 |
|---|---|---|
| [D26](#d26--triton-runtime-int-的隐式特化) | 热路径 triton kernel 运行中途反复编译 | CUDA/编译 |
| [D27](#d27--verifier-的数值是承重墙) | 亚量子数值漂移被 spec 环放大成 −20% 接受率 | 数值 |
| [D28](#d28--纯-attention-draft-模型的-armed-值路径) | 机制全绿但 draft 内容错 | 值路径 |
| [D29](#d29--链图行数超上限静默降级) | 改 F 让吞吐掉 5× 且只打一条 INFO | CUDA graph |
| [E11](#e11--profiler-的两个陷阱) | profiler 默认参数膨胀 170×;扰动不对称 | 观测 |
| [B1](#b1--colocated-在大-target-上的-oom) | colocated 8k prefill 崩 CUDA OOM | 资源 |
| [M1](#m1--验收口径纪律) | 首跑偏低、单跑离群、跨 K 不可比 | 测量 |
| [M2](#m2--新机器的三个静默失效前置) | 无外网时整轮 benchmark 吐 null | 环境 |

---

<a id="d26"></a>

## D26 · triton runtime int 的隐式特化

**症状**:融合 kernel 上线后正确性全绿、p90 band 变好,但 **mean band 反涨**(H200 arm_chain
mean 0.825→1.368 而 p90 0.555→0.486);B200 上一条腿渐进塌进永久 fallback 环(273→151→148→141,acc→1.01),
同构型的其他腿却正常。

**根因**:triton 对每个 runtime int 参数按 `{==1, %16==0, other}` 自动特化进缓存键。新 kernel 有 4 个
逐轮变化的 int + 5 档表宽 → 几十种组合,每种首次遇到即编译 ~100 ms。**mean 被污染而 p90 干净**
(p90 只反映近期环)。每次编译 stall 都推迟 gate 入流,容忍窗口小的机器一次 stall 就被 fallback 环俘获且自持。

**修法**(commit `b6da986ca3`):① `@triton.jit(do_not_specialize=[所有逐轮 host int])`;
② 表宽这类"引擎常量"降为 runtime 参数,别用 constexpr 吃进缓存键;③ init 时对唯一必要的
constexpr 维度全档 prewarm。

> **规则**:任何加进逐轮 host 路径的 triton kernel,上线清单 = 枚举它的全部编译维度 →
> `do_not_specialize` 压掉逐轮 int → init 全档预热。**判读 A/B 时 mean 与 p90 背离 = 首编译/间歇 stall 的指纹**,
> 先查编译面再谈机制。

---

<a id="d27"></a>

## D27 · verifier 的数值是承重墙

**症状**:shared-expert 的 `silu_and_mul+quant` 换成现成融合核(离线看只有 1.7% 元素亚量子翻桶、
scale 全等),上线后 acc 3.21→2.55(**−20%**)、TEXT_EXACT=False。dense 服务里这种量化噪声完全无感。

**根因**:投机解码的接受率 = draft 分布与 verifier 分布的一致性。verifier 数值哪怕亚量子扰动,
边界 token 的贪心选择翻转率 × 层数累积 → accept 链被截断。定罪谱系:
① v2 融合分支 `bf16(silu_fp32(gate)) * up_bf16`(bf16 乘法 + 双重舍入)≠ 非融合的 fp32 乘法单次舍入;
② dsv4 的核偏离更大(3.6%);③ 非融合对与 fp64 真值量化逐位一致 = 基准。

**修法**:自写位忠实 triton 核——silu 精确 fp32(非 fast-math exp)+ fp32 乘法 +
**显式一次 bf16 舍入**(复刻中间张量)+ 位精确 ceil-log2。**UE8M0 的 2 幂 scale 是关键盟友**:
量化除法变成无舍入的指数移位,act 位一致 ⇒ 全管道位一致。

**两条加料**:
- **归约几何入契约**:值路径含归约时,同一公式在 1D grid+4 warps 与 2D tile+1 warp 下归约树不同。
  randn 探针踩不到 ulp 边界,真实深层激活会翻(实测 q_bad=14@layer37)。GDN 融合核最终是对原核几何的**逐字转写**。
- **位一致 ≠ 可以上线**:GDN 融合核位一致,但 e2e acc 仍 3.21→2.55(hit −8pp),轮周期不变 →
  verify 侧 −50µs 级提速本身就会移动环相位。该融合 **default OFF**。

> **规则**:给 verify 侧上任何融合 kernel,验收 = 与被替换路径**逐位一致**(离线 q/s bitexact + e2e TEXT_EXACT),
> "数值差不多"不合格;**且验收口径必须含 `hit_rate/acc`,不能只看墙钟**。

---

<a id="d28"></a>

## D28 · 纯 attention draft 模型的 armed 值路径

**症状**:32B/235B pair(draft = Qwen3-0.6B,纯 full-attention)机制层全绿——select hit 0.91–0.98、
prelaunch fast 比例 >99%、零 traceback——但 acc 卡在 **1.54**(同模型 colocated 3.98),node-0 猜测 ~75% 错。

**二分**:eager/host 全关腿恢复 3.88;extend graph 族无罪;`全开−chain graph` 仍 1.54;
`全开−topk graph` 2.04。**嫌疑收敛到 armed/prelaunch 化的 extend 消费路径**(该机制为 hybrid drafter 特化开发,
纯 attn 值路径从未验证)。

**守卫**(commit `80c468daae`,已默认):引擎构造期检测非 hybrid draft 模型 → 自动关 pre-launch + chain graph,
**保留** extend/fused/topk 三张图,打 WARNING。守卫后 32B/235B 恢复满接受长度(3.94/3.96),吞吐 +156%/+158%。
对 hybrid drafter 是构造期短路的严格 no-op。

**root cause 仍 OPEN**。破案需要与 armed 兼容的 in-model 双算取证——注意 **sync 型探针会与 armed 机构死锁**,
`prelaunch 关 + chain 开` 的组合也卡死。

> **规则**:为某个模型族特化开发的值路径机制,换族时必须**重验值**而不只是重验机制。
> "机制全绿"(命中率、计数器、无异常)与"值正确"是两件事。

---

<a id="d29"></a>

## D29 · 链图行数超上限静默降级

**症状**:F 从 2 抬到 4,命中率与接受长度都如预期上升(hit 0.750→0.799、acc 2.47→2.69),
**吞吐却从 280 塌到 54.5**;drafter 轮 1.8→**46.7 ms**、idle 5.3→0.57、starved 98%→1%。

**根因**:`ChainGraphRunner.max_rows = server_args.cuda_graph_config.decode.max_bs`,而链图一次要跑
**(K+1)×F 行**。drafter 一直按 `--cuda-graph-bs-decode 1 2 4 8`(max 8)启动:
K3F2 = 8 行**恰好卡满**(历史全绿纯属巧合),K3F4=16 / K5F2=12 / K5F4=24 **全部越界** →
bucket 记进 `_failed_rows` **永久禁用**,退化成逐步链(每轮 K 次 host 发射)。

**判读陷阱**:越界只打**一条 INFO**,没有 WARNING、没有计数器;drafter 日志的 `fast=` 计的是
host fast-path 不是图命中,所以常规日志面上完全看不出来。

**修法**:按配置算 `(K+1)*F` 生成 graph bs 列表(本轮用 `1 2 4 8 12 16 24 32`)。
对照实验两腿必须同列表——实测 K3F2 在宽/窄列表下逐项复现(267/280.7 vs 265.7/280.5),宽列表本身无副作用。

> **规则**:任何改 K 或 F 的实验,**先算链图行数 (K+1)×F 并确认它在 drafter 的 decode graph bs 列表里**;
> 扫描类实验开跑前先跑一条越界探针(`grep "exceed the decode graph" drafter.log`)。
> 更一般地:**枚举维度的参数扫描会同时改变图的形状面,别默认图还在。**

---

<a id="e11"></a>

## E11 · profiler 的两个陷阱

1. **默认参数膨胀 CPU 侧 ~170×**:`/start_profile` 必须传 `with_stack:false, record_shapes:false`,
   否则被测系统掉到 2.5 tok/s,量出来的是 profiler 自己。
2. **扰动对 decoupled 不对称**:开 profiler 时 MTP 的 acc 纹丝不动(3.42→3.42),decoupled 的从
   2.40 崩到 1.09(hit→0.20/0.36)。因为 decoupled 的接受长度是**时序耦合量**(块必须在消费点前到达且 stamp 匹配),
   任何拖慢 host 的东西都转化为 miss。

> **规则**:decoupled 的 trace **只读结构**(band 形状、比例、依赖关系),
> **绝对吞吐/acc 一律引用未开 profiler 的腿**。另:profile 过的 server 必须重启再测性能。

---

<a id="b1"></a>

## B1 · colocated 在大 target 上的 OOM

**症状**:`colo_235b` 的 serving-8k 与 gsm8k 腿连续四次被 SIGKILL,日志尾部是 NCCL TCPStore shutdown
(误导性表象),真因在更前面:`CUDA out of memory. Tried to allocate 564.00 MiB. GPU 0 has ... 353.06 MiB is free`。

**根因**:colocated 把 draft 模型挤进 target 的 TP4 组,`--mem-fraction-static 0.85` 下 8k prefill 的
激活峰值溢出。decoupled 免疫(draft 在独立进程独立卡)。

**修法**:`--mem-fraction-static 0.80` + boot 后沉降 120 s。

> **规则**:诊断 SIGKILL 别停在日志尾部的 NCCL 噪声,往前找第一条 `Scheduler hit an exception`。
> 另:**这不只是配置问题,是 colocated 的结构性运维劣势**——同样的 pair、同样的负载,decoupled 不需要调这个旋钮。

---

<a id="m1"></a>

## M1 · 验收口径纪律

| 陷阱 | 数据 | 纪律 |
|---|---|---|
| 首跑偏低 | one_batch rep1 系统性比 rep2-4 低 8–13% | 至少 2 次,判精细差异用 3–4 次取稳态 |
| gsm8k 单跑离群 | 曾测出 −17% 的假差异,复验不复现 | 判大差必复跑;两次复测互差应 <1% |
| 跨 K 不可比 | 不同 K 的 verify 窗口 kernel 形状不同 → 数值差 → 输出 token 数变(K5 多输出 3.2%) | 判 K/F **只认固定输出长度的 one_batch** |
| acc_server 是累计值 | 三个配置在 serving-8k 上都报 4.000(理论上限) | 用 one_batch 的 acc_length |
| seed 影响横比 | H200 轮各 config 的 serving seed 由时间戳派生 → 看到的不是同一批 prompt | 对照实验固定 seed |
| 新 fast-path 可能根本没跑 | 历史事故:门恒假,两腿全等,"全绿"实为"没跑" | **每条新 fast-path 必须带命中计数器并在日志可见** |

---

<a id="m2"></a>

## M2 · 新机器的三个静默失效前置

在无外网的盒子上(2026-08 重建的 B200 即是),下面三件事会让整轮 benchmark 悄悄失效:

1. **`random` 数据集要从 HF 下 ShareGPT 采样 token id** → 所有 serving / one_batch 腿吐 **null**。
   修:中转 `ShareGPT_V3_unfiltered_cleaned_split.json`(672 MB / 94145 条)并给两个 bench 客户端加 `--dataset-path`。
2. **gsm8k 的 `test.jsonl`** 默认从 raw.githubusercontent 下载 → 该腿整体失效。修:预置到仓库根。
3. **HF model id 解析**需要联网 → server 起不来。修:模型落成纯目录,启动用绝对路径;并设
   `HF_HUB_OFFLINE=1` 让缺文件快速报错而不是挂在超时上。

> **规则**:接手新机器先跑三件事——`getent hosts huggingface.co`(判外网)、
> `du` 增量(判"集群在下载"这类承诺是否为真)、以及一条最小 serving 腿(判数据集依赖)。
> **别信承诺,信增量。**

---

## 附:两条方法论

- **估 sync/stall 代价看它排在流上什么位置,不是这个操作本身多贵。** 算子内部数据依赖 sync 是固定 µs 级;
  排在长队列后面的阻塞式操作没有自身量级,只有位置。
- **一个观测对应多个机制时,不能从观测反推机制。** 每波改动留 env kill switch 做 A/B(不用 git stash),
  让机制可以被单独关掉验证。
