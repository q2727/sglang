# 07 · 未竟事项与已封存分支

分三类:**OPEN**(有明确下一步)、**SEALED**(做过、判定为不采用、附判定依据)、**UNVERIFIED**(设计上成立但没测过)。
封存项写清依据是为了不被重复挖坟。

## OPEN · 有明确下一步

### O1 · D28 root cause(优先级最高的正确性债)
armed / pre-launch 的值路径在纯 attention draft 模型上算错内容(见 [05-pitfalls D28](05-pitfalls.md#d28))。
守卫已兜住吞吐,但机制本身仍是黑箱。
**下一步**:与 armed 兼容的 in-model 双算取证——同一轮内同时跑 armed 路径与 eager 路径、逐张量对拍。
**已知障碍**:sync 型探针会与 armed 机构死锁;`prelaunch 关 + chain 开` 的组合也卡死,所以不能用简单二分。

<a id="o2"></a>

### O2 · wire 宽度跟随预算
逐 case 列预算省的是 drafter 的分支行数,但 wire block 仍是固定的 (K+1)×F(未预算格子毒化)。
所以**块宽的传输与 select 代价没有下降**——这正是 F=4 的接受长度收益被轮周期涨幅抵消的直接原因
(见 [04-results §5](04-results.md#budget-exp))。
**下一步**:让 wire 宽度也跟着预算走,需要 verifier 侧的 select 能读变宽块(当前 select 假设固定 stride)。

### O3 · `stale_ct` 遥测
`fresh` 在 select 里算出后即被丢弃(`verify_worker.py` 的 `hits = fresh & guess_matches.any(dim=1)`)。
把它一并入队就能**零 GPU 成本**地把"时序 miss"与"内容 miss"彻底分开,当前只能靠上界算术推断。
**注意**:verify 侧任何改动都有环相位敏感风险(D27 加料),必须在非测量期落地并按 hit/acc 口径验收。

### O4 · bs>1 推广(用户既定方向)
现有全部优化只在 bs=1 上验证。已知的 bs>1 语义问题:
- `round_ct` 累加的是**座位行数**不是轮数,`hit_rate` 是 per-row 而非 per-round;
- gate 是**全或无**(一个落后座位让整轮记一次超时,而其他座位的块早已到达);
- `sync_wait_timeout_ct` 按 gate 调用计、`round_ct` 按座位行计,**两者量纲不同,比值在 bs>1 时无意义**。

### O5 · host 轮询地板本身
轮周期的下界 D≈7.4 ms 是跨进程轮询底(不是 draft 算力,draft forward 仅 ~1.2 ms)。
这是 decoupled 相对 MTP 的最后一段结构性差距,也是"换更快的卡收益递减"的原因
(见 [04-results §1](04-results.md#b200-397b) 的两盒对照)。

### O6 · 待修的小债
- `need_topk` 一次性闩锁应加 assert:首个 stash 缺 `topk_p` 会让 topk 中继永久静默降级。
- 两个 stale test 仍断言已被上游移除的 spec-overlap force-off。

---

## SEALED · 判定为不采用(附依据)

### S1 · cuda_ipc 数据面
**做到哪**:三处死锁修复后可用,峰值 441.8 tok/s / acc 3.36,超过当时的 zmq 峰值 430.6。
**为什么封存**:①用户裁决沿用 zmq;②次请求起 GPU stamps 与 arrival board 漂移(首请求正常,
之后 hit→0.147),嫌疑是 `reset_slot` 默认流与 land 私有流的竞态,未破案;
③功能降级——CUDA IPC 的行头没有 `speculative` 标志位,所以 evented push / prerun / bet 三条路都不可用。
**代码仍在**(`cuda_ipc_enum_transport.py`,CI 有注册测试),可用 `--decoupled-spec-data-transport cuda_ipc` 选中。

<a id="s2"></a>

### S2 · prerun / bet(投机式提前推送)
**做到哪**:押注"verify 全收"并把块提前推给 verifier 的 gen-2 槽,机制全通、押中率 87–96%、正确性全绿。
**为什么封存**:bet 的 GPU 生产时间 ~1:1 折进轮周期(extend-only 形态就 −8%),
**"空窗免费"被证伪**——它的 host 发射段拉长 dispatch tail,gate 晚于 commit 入流(profile 铁证)。
armed 停在"预发射不预推送"这一档,是实测的收益/风险平衡点。

### S3 · GDN norm+quant 融合
**做到哪**:位一致修成(逐字转写归约几何),离线 q/s 逐位相等。
**为什么 default OFF**:带图 e2e acc 3.21→2.55、hit −8pp,而轮周期不变——
verify 侧 −50µs 级提速本身移动环相位。**留了 OPEN 问题:gate pacing 为何不吸收这个提速。**

### S4 · doorbell(设备侧 `cuStreamWaitValue32` 门)
代码在(`decoupled_doorbell.py`),但环境变量注释里写明 **"verified UNSAFE on driver 580.126.09"**,默认关。
C6 host-func 门是当前方案。

---

## UNVERIFIED · 设计上成立但没测过

| 项 | 状态 |
|---|---|
| **PP(pipeline parallel)** | 参数校验层直接拒绝(`pp_size != 1` 报错),未验证 |
| **DP / dp-attention** | 参数校验层直接拒绝,设计上已否决(座位到 rank 的映射与 DP 冲突) |
| **1:N(多 drafter)** | 代码路径存在(`pool_idx % num_drafters` 路由、per-drafter 隔离),benchmark 未覆盖 |
| **纯 linear attention 模型** | overlap 审计时发现池配置器除零(SGLang 全局问题,非本线),起不来 |
| **MLA + paged attention** | 启动即 raise,未适配 |
| **bs>1 全形态** | 见 O4 |

---

## 给接手者的三条建议

1. **先复现,再改**。按 [06-reproduction](06-reproduction.md) 跑通 32B pair(最快、最稳、无外网依赖最少),
   确认拿到 dec≈405 / colo≈308(B200)或 dec≈299 / colo≈224(H200)量级,再动任何代码。
2. **改 verify 侧要格外小心**。那一侧的每个 ±50µs 都可能移动环相位,验收口径必须含 `hit_rate` 和 `acc`,
   只看墙钟会漏判(D27 加料是血的教训)。
3. **改 drafter 侧优先看 host,不是 kernel**。draft forward 只占轮周期的 ~16%;
   真正的地板在 commit 着陆、行/页表准备、kernel 发射这串 host 工作上(见 [01-architecture §5.1](01-architecture.md#pipeline-model))。
