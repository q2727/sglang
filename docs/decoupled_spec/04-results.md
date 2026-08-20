# 04 · 实测结果与出处

> 每张表都标了**出处文件**,全部可回溯到原始记录(备份在 `~/Desktop/b200_backup_0813/`)。
> 读表前先读 [§0 口径](#caliber),否则容易读错。

<a id="caliber"></a>

## 0 · 口径(先读这一节)

### 0.1 三个吞吐口径

| 口径 | 定义 | 陷阱 |
|---|---|---|
| **one_batch decode** | `batch_size * output_len / (latency - last_ttft)`(`python/sglang/benchmark/one_batch_server.py:776`),**分母扣掉 prefill** | 是纯 decode 吞吐,与 serving 的 e2e 吞吐系统性不同,不可混比 |
| **serving** | 总输出 token / 整个测试墙钟(含 prefill、请求间隙、调度) | 输入越长与 one_batch 差距越大 |
| **gsm8k** | 总输出 token / 墙钟,200 题 5-shot temperature 0 | **分子分母都会随配置变**(见 0.3),判 K/F 不能用它 |

### 0.2 聚合规则(本文档全局统一)

**每个 one_batch 单元格 n=2,表里给出两次读数,并以 run2 作为稳态估计。**

理由是实测的:17 条 one_batch 腿里,run1 与 run2 的偏差只有 **2 条**超过 3%——
`oldbox dec_397b`(−8.5%)与 `h200 dec_397b`(−10.7%),**都是 hybrid drafter 的 decoupled 397B pair**;
其余 15 条(含全部 colocated / MTP / 32B / 235B)两次读数都在 ±3.5% 内。
这不是通用的"首跑效应",而是**这一对特有的环相位建立过程**(gate 的 EWMA 需要 20 次到达才离开 bootstrap、
自适应 fanout 有冷却窗口)。旁证:同配置的 4 次重复实验稳态是 289.8 tok/s,与 run2(288.2)相符,
与均值(272.8)不符——**用均值会系统性低估 decoupled,且只低估它**。

> 早期版本的本文档对旗舰数字取 run2、对分母取均值,是不一致的;现已统一。

### 0.3 acc 与正确性

- `acc length` 用 **one_batch 的 `acc_length`**(每腿重启,较干净)。
  serving 腿的 `acc_server` 是**自启动累计值**,会顶到理论上限(多个配置在 serving-8k 上都报 4.000),不可用。
- 不同 K 的 verify 窗口 kernel 形状不同 → 亚量子数值差 → 翻转个别贪心选择 → **输出 token 数本身在变**
  (实测 K5 比 K3 多输出 3.2%)。所以 **gsm8k 吞吐跨 K 不可比**。

### 0.4 跨盒不可比

三台机器的库栈(transformers 5.12.1/5.8.1、flashinfer 0.6.15/0.6.12、deep_gemm 0.1.5/0.1.3)、
attention backend(B200 `trtllm_mha` / H200 `fa3`)、NUMA 拓扑都不同。
**每张表内部自洽,跨表只比形态间的相对关系。**

### 0.5 资源口径

decoupled 用 **N+1 张卡**(verifier TP=N + drafter 独占 1 张),colocated / MTP / 无 spec 用 **N 张**。
下面所有比值都是**吞吐比,不是每卡吞吐比**。397B(TP4)的场景下 decoupled 多用 25% 的卡:
按每卡归一,`decoupled 2.93× colocated` 折算为 **2.34×**,`2.20× 无 spec` 折算为 **1.76×**。
选型时用哪个口径取决于你的约束是"这批卡的总吞吐"还是"这个请求的延迟"。

---

<a id="b200-397b"></a>

## 1 · B200 · 397B + 0.8B(权威行 = 原盒 zdhua-parallel-dev)

Qwen3.5-397B-A17B-FP8(TP4, GPU0-3)+ Qwen3.5-0.8B(TP1, GPU4)· K=3 F=2 · bs=1
> 出处:`oldbox/bench/matrix_b200_results.jsonl`(spec 三形态)+ `newbox/bench/b200_nospec_397b.txt`(无 spec 行)

| 形态 | one_batch(run1 / **run2**) | acc | serving 1k/2k/8k | TPOT 1k | gsm8k | 正确率 |
|---|---|---|---|---|---|---|
| **MTP**(target 自带头) | 484.1 / **484.2** | 3.60 | 434.7 / 423.2 / 474.2 | 2.15 ms | 338.9 | 0.98 |
| **decoupled** | 394.2 / **430.8** | 3.21 | 329.2 / 327.1 / 435.3 | 2.90 ms | 271.7 | 0.98 |
| colocated STANDALONE | 150.7 / **146.9** | 3.47 | 128.9 / 126.8 / 160.9 | 7.57 ms | 115.5 | 0.98 |

**无 spec 基线 196.1(one_batch)/ 190.0(serving-1k)/ 168.3(gsm8k),正确率 0.975 —— 采自新盒**
(原盒未跑无 spec 腿)。因此下面带 ⚠️ 的比值是**跨盒**的,只作量级参考;同盒比值不带标记。

| 比较 | 值 |
|---|---|
| decoupled / MTP | **89.0%**(one_batch)· 80.2%(gsm8k) |
| decoupled / colocated | **2.93×**(one_batch)· 2.35×(gsm8k) |
| ⚠️ decoupled / 无 spec(跨盒) | ≈2.20× |
| ⚠️ colocated / 无 spec(跨盒) | ≈**0.75×**,即**负优化** |

> colocated 是负优化这一条**不依赖跨盒**:新盒上无 spec(196.1)与 colocated(141.2)是**同盒同期**采的,
> 比值 0.72×,方向一致。

### 同一矩阵在新盒(zdhua-parallel)的读数
> 出处:`newbox/bench/matrix_b200_results.jsonl`

| 形态 | one_batch(run1 / **run2**) | acc | gsm8k |
|---|---|---|---|
| mtp_397b | 487.2 / **487.2** | 3.60 | 334.7 |
| dec_397b | 403.4 / **401.6** | 3.21 | 255.6 |
| colo_397b | 146.2 / **141.2** | 3.47 | 110.2 |
| 无 spec | 195.9 / **196.3** | — | 168.3 |

**两盒对照是一条机制证据,不是噪声**:同代码、同库栈、同 flags 下,MTP 几乎持平(484.2 vs 487.2),
decoupled 差 **7.3%**(430.8 vs 401.6)。原因见 [01-architecture §5.1](01-architecture.md#pipeline-model):
MTP 的轮周期由 GPU 决定,decoupled 的压在 host 跨进程轮询地板上,继承 host/内存子系统的个体差异。
**推论:部署 decoupled 时挑盒子有真实收益;跨盒回归判定必须同盒。**

---

## 2 · B200 · 其余 pair(新盒)
> 出处:`newbox/bench/matrix_b200_results.jsonl`;`colo_235b` 的 8k/gsm8k 两腿出处 `newbox/bench/b200_colo235_{fix2,srv8k80}.txt`

| pair | 形态 | one_batch(run1 / **run2**) | acc | serving 1k/2k/8k | gsm8k | 正确率 |
|---|---|---|---|---|---|---|
| 235B+0.6B | **decoupled** | 427.5 / **427.8** | 3.96 | 330.8 / 378.2 / 365.0 | 241.6 | 0.965 |
| 235B+0.6B | colocated | 331.6 / **331.6** | 3.99 | 279.7 / 304.8 / 303.0 | 211.3 | 0.965 |
| 32B+0.6B | **decoupled** | 406.1 / **404.7** | 3.94 | 267.6 / 321.9 / 336.4 | 273.4 | 0.89 |
| 32B+0.6B | colocated | 307.7 / **307.9** | 3.97 | 230.8 / 259.9 / 269.7 | 232.4 | 0.89 |

decoupled 优势:235B **+29%**(one_batch)/ +14%(gsm8k);32B **+31%** / +18%。

> `colo_235b` 的 serving-8k 与 gsm8k 腿曾四次崩在 CUDA OOM(colocated 把 draft 挤进 target 的 TP4 组,
> 8k prefill 激活峰值溢出;decoupled 免疫),最终用 `--mem-fraction-static 0.80` + boot 后沉降 120 s 取得。
> 见 [B1](05-pitfalls.md#b1)。

---

## 3 · H200 全矩阵(7 配置 × 6 腿 = 42 腿,零 crash)
> 出处:`h200/bench_h200.tgz` → `bench/matrix_h200_results.jsonl`(注意去重,见下方警告)

8×H200 · `fa3` · 其余同 B200

| 形态 | one_batch(run1 / **run2**) | acc | serving 1k/2k/8k | TPOT 1k | gsm8k | 正确率 |
|---|---|---|---|---|---|---|
| mtp_397b | 375.8 / **375.9** | 3.42 | 379.4 / 350.5 / 384.2 | 2.46 ms | 276.7 | 0.95 |
| **dec_397b** | 257.4 / **288.2** | 2.52 | 311.5 / 299.9 / 302.8 | 3.01 ms | 228.5 | 0.95 |
| colo_397b | 108.9 / **108.2** | 2.85 | 102.6 / 115.0 / 98.3 | 9.53 ms | 88.8 | 0.95 |
| **dec_235b** | 352.4 / **352.8** | 3.96 | 334.7 / 276.9 / 295.2 | 2.83 ms | 195.4 | 0.96 |
| colo_235b | 201.7 / **198.6** | 3.99 | 177.9 / 192.9 / 183.9 | 5.38 ms | 133.6 | 0.96 |
| **dec_32b** | 299.1 / **299.2** | 3.94 | 217.8 / 265.4 / 252.2 | 4.52 ms | 205.0 | 0.89 |
| colo_32b | 223.6 / **224.1** | 3.98 | 181.4 / 191.5 / 189.7 | 5.42 ms | 169.7 | 0.89 |

decoupled 优势:397B **2.66×**、235B **+77%**、32B **+34%**(one_batch)。

**正确性判词**:同一 target 下所有 spec 形态的 gsm8k 正确率逐一持平(32B 双形态 0.89;397B 三形态 0.95;
235B 双形态 0.96)——投机解码语义无损。

> ⚠️ **JSONL 有 54 行而非 42 行**:`dec_32b`/`dec_235b` 各有 6 行是 D28 守卫落地**前**的历史行
> (acc≈1.5、吞吐 117/136)。读表必须按 `(config, leg)` 取最新 `ts`,否则会把两个配置低估 2.5×。

### 接受长度的三段分解(397B,同 target)

| 形态 | acc | 相对上一行的损失来源 |
|---|---|---|
| MTP 头 | 3.42 | — |
| colocated(0.8B) | 2.85 | −0.57 = draft 模型质量 |
| decoupled(同 0.8B) | 2.52 | −0.33 = 枚举 miss 放大 |

H200 上 decoupled 与 MTP 的差距约 63% 来自 draft 质量、37% 来自枚举惩罚。
**B200 上这个分解完全不同**(colo 3.47 / dec 3.21,draft 质量只差 0.13)——
**"draft 质量是主要天花板"是 H200 的局部结论,不可外推。**

### 每步耗时:差距不在开销

| 箱 | 形态 | acc | tok/s | 每步 = acc/tok/s |
|---|---|---|---|---|
| H200 | MTP | 3.42 | 375.9 | 9.10 ms |
| H200 | decoupled | 2.52 | 288.2 | **8.75 ms** |
| B200(旧盒) | MTP | 3.60 | 484.2 | 7.43 ms |
| B200(旧盒) | decoupled | 3.21 | 430.8 | **7.45 ms** |

**decoupled 的轮周期与 MTP 相当甚至更短**(H200 上短 4%,B200 上持平)——
尽管 MTP 每轮还要多跑一次 in-line draft。**全部差距都在接受长度上**,这正是
[01-architecture §5.1](01-architecture.md#pipeline-model) 的流水线模型所预测的:
decoupled 的轮周期压在 host 轮询地板上,而不是 GPU 计算上。

---

## 4 · K/F 扫描

链图行数 = (K+1)×F,必须放得进 drafter 的 `--cuda-graph-bs-decode`,否则数据无效(见 [D29](05-pitfalls.md#d29))。
下表均为修复后的有效数据。

### B200 397B pair
> 出处:`newbox/bench/matrix_b200_results.jsonl`(`kf_397b_*` 配置)

| 配置 | serving-1k | gsm8k | gsm8k acc | gsm8k hit |
|---|---|---|---|---|
| K3F2 | 292.8 | 249.2 | 2.79 | 0.776 |
| **K3F4** | **308.5** | **253.6** | 2.88 | **0.842** |
| K5F2 | 258.9 | 246.9 | 3.42 | 0.709 |
| K5F4 | 262.0 | 247.2 | 3.56 | 0.768 |

### H200 397B pair
> 出处:H200 箱 `bench/h200_kf_result.txt`(备份于 `h200/bench_h200.tgz`)

| 配置 | one_batch | acc | gsm8k | gsm8k hit | drafter 轮 |
|---|---|---|---|---|---|
| K3F2 | 273.9 | 2.49 | 227.5 | 0.809 | 1.1 ms |
| K3F4 | 280.5 | 2.66 | 228.2 | 0.847 | 1.2 ms |
| K5F2 | 272.4 | 3.02 | 214.4 | 0.670 | 2.8 ms |
| K5F4 | 255.0 | 3.18 | 205.9 | 0.732 | 2.9 ms |
| K3F4(链图越界,**无效**) | 54.5 | 2.69 | — | 0.799 | **46.7 ms** |

**结论**:K=3 胜;F=4 略优于 F=2(B200 +5.3% serving、H200 +2.4% one_batch),但 K/F 都只是在
"接受长度"与"轮周期"之间换汇率。**主矩阵两盒都用 K3F2 以保持跨盒可比,K3F4 作为推荐调优档。**

### K 的最优值依赖负载(B200 32B pair)
> 出处:`newbox/bench/b200_fsweep_result.txt`、`newbox/bench/b200_gsm_recheck.txt`

| 口径 | K3F2 | K5F2 |
|---|---|---|
| one_batch | **405.9** | 257.8 |
| gsm8k(两次复测) | 272.9 / 273.4 | **298.1 / 299.1** |

同一个 **K5F2** 配置在两种负载下的 select hit:one_batch regime **0.733** vs gsm8k regime **0.811**。
gsm8k 的真实文本(数学推理,格式化程度高)让 draft 与 target 一致性更好 → 长链吃得满。
(作为对照,K3F2 在 one_batch regime 的 hit 是 0.975——它本来就没有提升空间。)

colocated 同样受益(gsm8k 248.8@K5 vs 232.4@K3),所以**解耦对 colocated 的优势对调参鲁棒**
(K=3 时 +17.6%,K=5 时 +20%)。

---

<a id="budget-exp"></a>

## 5 · 逐 case 列预算实验(H200 397B,K=3,4 次重复稳态)
> 出处:H200 箱 `bench/h200_budget_repeat.txt`、`bench/h200_case0_result.txt`、`bench/h200_default_skew.txt`

| 逐 case 预算 | drafter 行数 | one_batch | acc | select hit |
|---|---|---|---|---|
| 旧出厂 `[1,1,2,4]` | 8 | 280.7* | 2.48 | 0.714 |
| 均匀 F=2 | 8 | 289.8 | 2.52 | 0.733 |
| 均匀 F=4 | 16 | 296.3 | 2.66 | 0.806 |
| `[4,1,1,4]` | 10 | **298.6** | 2.64 | 0.775 |
| **`[4,1,1,2]`(新默认形状)** | **8** | 297.3 | 2.64 | 0.771 |

\* 单次采集,其余为 4 次重复稳态(rep1 同样丢弃);它在两个口径下都是最差,结论方向不受影响。

**命中率随 case-0 列数严格单调**;`[4,1,1,2]` 用与均匀 F=2 相同的 8 行拿到近似均匀 F=4(16 行)的接受长度。
→ 这是**分配问题不是预算问题**,已落地为新默认形状(commit `459ae93227`,旋钮仍默认关)。

### 全腿复核:收益只在 one_batch 兑现
> 出处:H200 箱 `bench/h200_tuned_397b.txt`(调优腿)与 `bench/h200_default_397b.txt`(同 seed 默认腿)

| 口径 | 默认 K3F2 | 调优 K3F4+case0 skew | Δ |
|---|---|---|---|
| one_batch | 288.5 | **297.3** | **+3.1%** |
| acc | 2.52 | 2.64 | +4.8% |
| serving 1k/2k/8k | 303.0 / 278.4 / 357.7 | 303.7 / 281.8 / 356.9 | +0.2 / +1.2 / −0.2% |
| gsm8k | 229.6 | 228.5 | −0.5% |

**接受长度 +4.8% 只在固定输出长度的 one_batch 上兑现成 +3.1%,serving 与 gsm8k 全平。**
机制发现扎实,端到端收益温和且口径相关——两件事要分开说。

---

## 6 · overlap 调度开关 A/B(H200,bs=1)
> 出处:H200 箱 `bench/ovl_ab.sh`、`bench/ovl_nospec_ab.sh` 的结果文件

| 口径 | 397B ON | 397B OFF | 32B ON | 32B OFF |
|---|---|---|---|---|
| **无 spec** one_batch | **156.8** | 139.2 | **84.4** | 81.9 |
| **无 spec** serving-1k | 152.8 | 136.7 | 84.1 | 81.5 |
| **无 spec** gsm8k | 136.0 | 123.9 | 83.2 | 81.2 |
| spec(colocated)one_batch | 225.4 | 224.7 | 108.8 | 109.5 |
| spec(colocated)gsm8k | 171.4 | 173.2 | 102.6 | 105.9 |

- **无 spec 下 overlap 有一致正收益**:397B +10~12%、32B +3%(step 越短 CPU 占比越大)
- **spec v2 下开关差异 ≤±5% 且无一致方向**——这是形态属性(draft/verify 已在 worker 内流水线化,
  scheduler CPU 无处可藏),不是回归。两侧正确性全等。

---

## 7 · profile 资产

| 资产 | 位置 | 内容 |
|---|---|---|
| 397B dec vs MTP | `~/Desktop/h200_prof397/` | 9 个 trace(dec verifier TP0-3 + dec drafter + mtp verifier TP0-3)+ README |
| 27B(TP1)+0.8B | `~/Desktop/b200_prof27/` | verifier + drafter 双侧 30 步 |

关键 band 读数(H200 397B,30 步):

| band | decoupled | MTP |
|---|---|---|
| `step[TARGET_VERIFY bs=1]` GPU 段 | 2.096 ms | 2.135 ms |
| `step[DRAFT_EXTEND_V2]` + `draft` | 无 | 3.58 + 0.90 ms |
| 轮周期(profiler 下) | 8.78 ms | 10.95 ms |
| `scheduler.recv_requests` 占 host 时间 | 83% | 88% |

**判读警告**:开 profiler 时 decoupled 的 acc 从 2.40 崩到 1.09,MTP 纹丝不动(3.42→3.42)。
decoupled 的接受长度是**时序耦合量**。**trace 只读结构,吞吐/acc 一律以未开 profiler 的腿为准。**
