# 04 · 实测结果与出处

> 全部数字来自逐腿 JSONL 原始记录,备份在 `~/Desktop/b200_backup_0813/`
> (`{newbox,oldbox}/bench/matrix_b200_results.jsonl`、`h200/bench_h200.tgz`)。
> 读表前请先读 [§0 口径](#0-口径先读这一节),否则容易读错。

## 0 · 口径(先读这一节)

| 口径 | 定义 | 陷阱 |
|---|---|---|
| **one_batch decode** | `bench_one_batch_server` 的 output throughput = `batch_size * output_len / (latency - last_ttft)`,**分母扣掉了 prefill**,是纯 decode 吞吐 | 与 serving 的 e2e 吞吐系统性不同,不可混比 |
| **serving** | `bench_serving` 的 Output token throughput = 总输出 token / 整个测试墙钟(含 prefill、请求间隙、调度) | 输入越长与 one_batch 差距越大 |
| **gsm8k** | `总输出 token / 墙钟`,200 题 5-shot temperature 0 | **分子分母都会随配置变**:不同 K 的 verify 窗口 kernel 形状不同 → 亚量子数值差 → 翻转个别贪心选择 → 输出 token 数变了。**判 K/F 只认 one_batch** |
| **acc length** | server 端 `avg_spec_accept_length`,**自启动累计**,不是单腿统计 | 单腿归因有邻腿污染;one_batch 的 acc_length 同源但因每腿重启而较干净 |
| **稳态** | one_batch 首跑系统性比后续低 8–13%(B200 惯性) | 单跑数字不可用,至少 2 次、建议 3–4 次取后段 |

**跨盒不可比**:三台机器的库栈(transformers 5.12.1/5.8.1、flashinfer 0.6.15/0.6.12、deep_gemm 0.1.5/0.1.3)、attention backend(B200 trtllm_mha / H200 fa3)、NUMA 拓扑都不同。**每张表内部自洽,跨表只比形态间的相对关系。**

---

<a id="b200-397b"></a>

## 1 · B200 · 397B + 0.8B(权威行 = 原盒 zdhua-parallel-dev)

Qwen3.5-397B-A17B-FP8(TP4, GPU0-3)+ Qwen3.5-0.8B(TP1, GPU4)· K=3 F=2 · bs=1

| 形态 | one_batch | acc | serving 1k/2k/8k | gsm8k | 正确率 | vs 无 spec |
|---|---|---|---|---|---|---|
| **MTP**(target 自带头) | 484.1 | 3.60 | 434.7 / 423.2 / 474.2 | 338.9 | 0.98 | **2.47×** |
| **decoupled** | **430.8** | 3.21 | 329.2 / 327.1 / 435.3 | 271.7 | 0.98 | **2.20×** |
| 无 spec(新盒值) | 196.1 | — | 190.0 | 168.3 | 0.975 | 1.00× |
| colocated STANDALONE | 148.8 | 3.47 | 128.9 / 126.8 / 160.9 | 115.5 | 0.98 | **0.76×** |

- decoupled = **MTP 的 89.0%**(one_batch)/ 80.2%(gsm8k),= **colocated 的 2.90×**
- **colocated 在 397B 上是负优化**:三个口径全部低于无 spec 基线
- one_batch 取 run2(run1 = 394.2,首跑带,按纪律丢弃)

### 同一矩阵在新盒(zdhua-parallel)的读数

| 形态 | one_batch | acc | gsm8k |
|---|---|---|---|
| mtp_397b | 487.2 | 3.60 | 334.7 |
| dec_397b | 402.5 | 3.21 | 255.6 |
| colo_397b | 143.7 | 3.47 | 110.2 |
| 无 spec | 196.1 | — | 168.3 |

**两盒对照是一条机制证据,不是噪声**:同代码、同库栈、同 flags 下,MTP 几乎持平(484.1 vs 487.2),
decoupled 差 7%(430.8 vs 402.5)。原因见 [01-architecture §5.1](01-architecture.md#pipeline-model):
MTP 的轮周期由 GPU 决定,decoupled 的压在 host 跨进程轮询地板上,继承 host/内存子系统的个体差异。
**推论:部署 decoupled 时挑盒子有真实收益;跨盒回归判定必须同盒。**

---

## 2 · B200 · 其余 pair(新盒)

| pair | 形态 | one_batch | acc | serving 1k/2k/8k | gsm8k | 正确率 |
|---|---|---|---|---|---|---|
| 235B+0.6B | **decoupled** | **427.6** | 3.96 | 330.8 / 378.2 / 365.0 | 241.6 | 0.965 |
| 235B+0.6B | colocated | 331.6 | 3.99 | 279.7 / 304.8 / 303.0 | 211.3 | 0.965 |
| 32B+0.6B | **decoupled** | **405.4** | 3.94 | 267.6 / 321.9 / 336.4 | 273.4 | 0.89 |
| 32B+0.6B | colocated | 307.8 | 3.97 | 230.8 / 259.9 / 269.7 | 232.4 | 0.89 |

decoupled 优势:235B **+29%**(one_batch)/ +14%(gsm8k);32B **+32%** / +18%。

> `colo_235b` 的 serving-8k 与 gsm8k 腿曾四次崩在 CUDA OOM(colocated 把 draft 挤进 target 的 TP4 组,
> 8k prefill 激活峰值溢出;decoupled 免疫)。最终用 `--mem-fraction-static 0.80` + boot 后沉降 120s 取得。
> 这本身是 colocated 的运维劣势,见 [05-pitfalls](05-pitfalls.md)。

---

## 3 · H200 全矩阵(7 配置 × 6 腿 = 42 腿,零 crash)

8×H200 · fa3 · 其余同 B200

| 形态 | one_batch | acc | serving 1k/2k/8k | gsm8k | 正确率 |
|---|---|---|---|---|---|
| mtp_397b | 375.9 | 3.42 | 379.4 / 350.5 / 384.2 | 276.7 | 0.95 |
| **dec_397b** | 272.8 | 2.50 | 311.5 / 299.9 / 302.8 | 228.5 | 0.95 |
| colo_397b | 108.6 | 2.90 | 102.6 / 115.0 / 98.3 | 88.8 | 0.95 |
| **dec_235b** | **352.6** | 3.96 | 334.7 / 276.9 / 295.2 | 195.4 | 0.96 |
| colo_235b | 200.2 | 3.99 | 177.9 / 192.9 / 183.9 | 133.6 | 0.96 |
| **dec_32b** | **299.2** | 3.94 | 217.8 / 265.4 / 252.2 | 205.0 | 0.89 |
| colo_32b | 223.9 | 3.98 | 181.4 / 191.5 / 189.7 | 169.7 | 0.89 |

**正确性判词**:同一 target 下所有 spec 形态的 gsm8k 正确率逐一持平(32B 三形态 0.89;397B 三形态 0.95;
235B 双形态 0.96)——投机解码语义无损。

> **JSONL 有 54 行而非 42 行**:`dec_32b`/`dec_235b` 各有 12 行是 D28 守卫落地**前**的历史行
> (acc≈1.5、吞吐 117/136)。读表必须按 `(config, leg)` 取最新 `ts`,否则会把两个配置低估 2.5×。

### 接受长度的三段分解(397B,同 target)

| 形态 | acc | 相对上一行的损失来源 |
|---|---|---|
| MTP 头 | 3.42 | — |
| colocated(0.8B) | 2.90 | −0.52 = draft 模型质量 |
| decoupled(同 0.8B) | 2.50 | −0.40 = 枚举 miss 放大 |

H200 上 decoupled 与 MTP 的差距 57% 来自 draft 质量、43% 来自枚举惩罚。
**B200 上这个分解完全不同**(colo 3.47 / dec 3.21,draft 质量只差 0.13)——
**"draft 质量是主要天花板"是 H200 的局部结论,不可外推。**

---

## 4 · K/F 扫描

链图行数 = (K+1)×F,必须放得进 drafter 的 `--cuda-graph-bs-decode`,否则数据无效(见 [D29](05-pitfalls.md#d29))。
下表均为修复后的有效数据。

### B200 397B pair

| 配置 | serving-1k | gsm8k | gsm8k acc | gsm8k hit |
|---|---|---|---|---|
| K3F2 | 292.8 | 249.2 | 2.79 | 0.776 |
| **K3F4** | **308.5** | **253.6** | 2.88 | **0.842** |
| K5F2 | 258.9 | 246.9 | 3.42 | 0.709 |
| K5F4 | 262.0 | 247.2 | 3.56 | 0.768 |

### H200 397B pair

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

| 口径 | K3F2 | K5F2 |
|---|---|---|
| one_batch | **405.9** | 257.8 |
| gsm8k(两次复测) | 272.9 / 273.4 | **298.1 / 299.1** |
| 同配置 select hit | one_batch regime 0.733 | gsm8k regime **0.811** |

gsm8k 的真实文本(数学推理,格式化程度高)让 draft 与 target 一致性更好 → 长链吃得满。
colocated 同样受益(gsm8k 248.8@K5 vs 232.4@K3),所以**解耦对 colocated 的优势对调参鲁棒**
(K=3 时 +17.6%,K=5 时 +20%)。

---

<a id="budget-exp"></a>

## 5 · 逐 case 列预算实验(H200 397B,K=3,4 次重复稳态)

| 逐 case 预算 | drafter 行数 | one_batch | acc | select hit |
|---|---|---|---|---|
| 旧出厂 `[1,1,2,4]` | 8 | 280.7* | 2.48 | 0.714 |
| 均匀 F=2 | 8 | 289.8 | 2.52 | 0.733 |
| 均匀 F=4 | 16 | 296.3 | 2.66 | 0.806 |
| `[4,1,1,4]` | 10 | **298.6** | 2.64 | 0.775 |
| **`[4,1,1,2]`(新默认形状)** | **8** | 297.3 | 2.64 | 0.771 |

\* 单次采集,其余为 4 次重复稳态;它在两个口径下都是最差,结论方向不受影响。

**命中率随 case-0 列数严格单调**;`[4,1,1,2]` 用与均匀 F=2 相同的 8 行拿到近似均匀 F=4(16 行)的接受长度。
→ 这是**分配问题不是预算问题**,已落地为新默认形状(commit `459ae93227`,旋钮仍默认关)。

### 全腿复核:收益只在 one_batch 兑现

同 seed 同协议,默认 K3F2 vs 调优 K3F4+case0 skew:

| 口径 | 默认 | 调优 | Δ |
|---|---|---|---|
| one_batch | 288.5 | **297.3** | **+3.1%** |
| acc | 2.52 | 2.64 | +4.8% |
| serving 1k/2k/8k | 303.0 / 278.4 / 357.7 | 303.7 / 281.8 / 356.9 | +0.2 / +1.2 / −0.2% |
| gsm8k | 229.6 | 228.5 | −0.5% |

**接受长度 +4.8% 只在固定输出长度的 one_batch 上兑现成 +3.1%,serving 与 gsm8k 全平。**
机制发现扎实,端到端收益温和且口径相关——两件事要分开说。

---

## 6 · overlap 调度开关 A/B(H200,bs=1)

| 口径 | 397B ON | 397B OFF | 32B ON | 32B OFF |
|---|---|---|---|---|
| **无 spec** one_batch | **156.8** | 139.2 | **84.4** | 81.9 |
| **无 spec** serving-1k | 152.8 | 136.7 | 84.1 | 81.5 |
| **无 spec** gsm8k | 136.0 | 123.9 | 83.2 | 81.2 |
| spec(colocated)one_batch | 225.4 | 224.7 | 108.8 | 109.5 |
| spec(colocated)gsm8k | 171.4 | 173.2 | 102.6 | 105.9 |

- **无 spec 下 overlap 有一致正收益**:397B +10~12%、32B +3%(step 越短 CPU 占比越大)
- **spec v2 下开关差异 ≤±5% 且无一致方向**——这是形态属性(draft/verify 已在 worker 内流水线化,
  scheduler CPU 无处可藏),不是回归。两侧正确性全等,ON/OFF 状态有日志判据背书。

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
