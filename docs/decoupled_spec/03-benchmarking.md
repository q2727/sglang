# 03 · 如何测:命令、协议与判读纪律

> 目标:任何人在一台新机器上照抄本文就能得到可比的数字。
> 结果数据见 [04-results](04-results.md);从零搭环境见 [06-reproduction](06-reproduction.md)。

## 1 · 四种被测形态

| 形态 | 是什么 | 进程数 | 用途 |
|---|---|---|---|
| **decoupled** | 本工作:drafter 独立进程独占卡 | 2 | 被测主体 |
| **colocated STANDALONE** | 上游形态:draft 与 target 同进程同 GPU 组 | 1 | 主要对照 |
| **MTP** | target 自带的 MTP 头 | 1 | 上界参照 |
| **无 spec** | 同模型去掉全部 spec flags | 1 | **零点**——没有它无法判断某形态是不是负优化 |

> **无 spec 基线不是可选项。** 397B 上 colocated 比无 spec 还慢 24–31%;
> 只比 decoupled/colocated 会得出"decoupled 快 2.9×"这种正确但误导的结论。

---

## 2 · 服务端启动命令

以下用 `$V`(target 路径)、`$D`(draft 路径)占位。B200 用 `--attention-backend trtllm_mha`,
H200 用 `fa3`。**同一次对比里所有形态必须用同一个 backend、同一个 page-size。**

<a id="dec-launch"></a>

### 2.1 decoupled(两个进程,先起 verifier 再起 drafter)

verifier —— target,占 GPU 0-3(TP4;单卡模型改 `--tp 1` 且 `CUDA_VISIBLE_DEVICES=0`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path $V --port 33700 \
  --tp 4 --attention-backend trtllm_mha --page-size 64 \
  --mem-fraction-static 0.85 --disable-prefill-cuda-graph --skip-server-warmup \
  --speculative-algorithm STANDALONE --speculative-draft-model-path $D \
  --speculative-num-steps 3 --speculative-fanout 2 \
  --decoupled-spec-role verifier --decoupled-spec-rank 0 \
  --decoupled-spec-data-transport zmq \
  --decoupled-spec-bind-endpoint ipc:///tmp/bm_v \
  --decoupled-spec-connect-endpoints '["ipc:///tmp/bm_d"]'
```

drafter —— draft 模型,独占 GPU 4:

```bash
env SGLANG_ENABLE_DECOUPLED_CHAIN_GRAPH=1 \
    SGLANG_DECOUPLED_ENUM_WAIT_MS=200 \
    SGLANG_ENABLE_DECOUPLED_DRAFT_PRELAUNCH=1 \
    SGLANG_ENABLE_DECOUPLED_CHAIN_PRELAUNCH=1 \
    SGLANG_ENABLE_DECOUPLED_DEVICE_PACK=1 \
    CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server \
  --model-path $D --port 33701 \
  --mem-fraction-static 0.6 --skip-server-warmup \
  --attention-backend trtllm_mha --page-size 64 --max-mamba-cache-size 1024 \
  --speculative-algorithm STANDALONE --speculative-draft-model-path $D \
  --speculative-num-steps 3 --speculative-fanout 2 \
  --cuda-graph-bs-decode 1 2 4 8 --max-running-requests 2048 \
  --decoupled-spec-role drafter --decoupled-spec-rank 0 \
  --decoupled-spec-data-transport zmq \
  --decoupled-spec-bind-endpoint ipc:///tmp/bm_d \
  --decoupled-spec-connect-endpoints '["ipc:///tmp/bm_v"]'
```

四个要点:
- **endpoint 交叉**:一侧的 `bind` 是另一侧的 `connect`。`connect_endpoints` 是 **JSON 数组**,
  在 shell 里要用单引号包住(`'["ipc:///tmp/bm_d"]'`),否则 shlex 会拆坏。
- **`--speculative-draft-model-path` 两侧都要给**:verifier 用它定枚举形状,drafter 用它当自己的模型。
- **`--max-mamba-cache-size 1024` 仅 hybrid draft 模型需要**(如 Qwen3.5-0.8B);纯 attention draft 去掉。
- **`--cuda-graph-bs-decode` 的最大值必须 ≥ (K+1)×F**,否则链图静默降级(见 [D29](05-pitfalls.md#d29))。
  K3F2 → 8 够用;K3F4 → 需要 16;K5F4 → 需要 24。保险起见扫描时统一用 `1 2 4 8 12 16 24 32`。

### 2.2 colocated STANDALONE(单进程)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path $V --port 33700 \
  --tp 4 --attention-backend trtllm_mha --page-size 64 \
  --mem-fraction-static 0.85 --disable-prefill-cuda-graph --skip-server-warmup \
  --speculative-algorithm STANDALONE --speculative-draft-model-path $D \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

> 大 target + 长输入时可能 CUDA OOM(draft 被挤进 TP 组,8k prefill 激活峰值溢出)。
> 降到 `--mem-fraction-static 0.80` 并在 boot 后沉降 120s 再压测。见 [B1](05-pitfalls.md#b1)。

### 2.3 MTP(单进程,draft 路径 = target 自己)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path $V --port 33700 \
  --tp 4 --attention-backend trtllm_mha --page-size 64 \
  --mem-fraction-static 0.85 --disable-prefill-cuda-graph --skip-server-warmup \
  --speculative-algorithm EAGLE --speculative-draft-model-path $V \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

`--speculative-algorithm EAGLE` + draft 路径指回 target 本身,arch 会自动重映射到 MTP 头。

### 2.4 无 spec 基线

同 2.2 但删掉全部四个 `--speculative-*` flag。

---

## 3 · 客户端命令(三种腿)

### 3.1 one_batch —— decode-only 吞吐(判 K/F 与 kernel 改动的**唯一**可信口径)

```bash
python3 -m sglang.bench_one_batch_server \
  --base-url http://127.0.0.1:33700 --model-path $V \
  --batch-size 1 --input-len 1024 --output-len 1024 --skip-warmup \
  --dataset-path /path/to/sharegpt_v3.json      # 无外网时必须给
```

它报的 `output throughput` 是 **decode-only**:

```python
# python/sglang/benchmark/one_batch_server.py:776
output_throughput = batch_size * output_len / (latency - last_ttft)
```

分母显式扣掉了 prefill,所以它与 `bench_serving` 的 e2e 吞吐系统性不同(如 colo_32b 224 vs 181),
**两者不可混比**。输出长度固定 1024 → 分子恒定 → 这是唯一"分子不随配置漂移"的口径。

### 3.2 serving —— e2e 吞吐

```bash
python3 -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port 33700 \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1.0 --num-prompts 10 --max-concurrency 1 \
  --seed 20260812 --disable-tqdm \
  --dataset-path /path/to/sharegpt_v3.json
```

三档输入:1024 / 2048 / 8192(8192 档用 `--num-prompts 6`)。
**`--random-range-ratio 1.0` 表示长度不抖动**(语义与直觉相反,别设 0)。
**seed 必须在所有对照配置间固定**,否则各 config 看到的不是同一批 prompt。

### 3.3 gsm8k —— 真实文本锚点 + 正确性

```bash
python3 benchmark/gsm8k/bench_sglang.py \
  --num-questions 200 --parallel 1 --num-shots 5 \
  --host 127.0.0.1 --port 33700
```

`--host` 只要主机名,**不要带 `http://`**。它同时给出 accuracy(正确性判据)和吞吐。
无外网时需预置 `test.jsonl` 到仓库根(脚本默认 `--data-path test.jsonl`)。

---

## 4 · 测量协议(harness 里编码的规矩)

矩阵驱动脚本:`~/Desktop/b200_backup_0813/newbox/bench/bench_matrix.py`(**打过 5 个补丁的终版**,
比 restore kit 里的原件新)。它对每个配置做的事:

1. `kill_all()` → 清残留进程 **+ 删 stale ipc socket**(`/tmp/bm_v`、`/tmp/bm_d`)
2. 起 verifier(+ drafter),`wait_health` 轮询 `/health`(超时 2400s;boot 日志出现
   `kill_process_tree called` 即判失败快速退出)
3. **drafter 存活门**:sleep 8 后检查 drafter 进程仍在。
   *为什么必要*:drafter 死掉时 verifier 会安静地全程走 fallback,产出**看起来完全正常**的数字。
4. warmup:一次 `/generate` 短生成 + 一次**弃测**的 1k/256 one_batch
5. one_batch × 2(rep1 按纪律视为热身)
6. serving 三档,**每档前 `flush_cache`**,seed 固定
7. gsm8k,跑前 `flush_cache`
8. 每腿写一行 JSONL:吞吐、acc、TTFT/TPOT、**该腿末尾的 select/drafter 日志行**、
   `d28_guard` 计数、traceback 计数

> 逐腿保存 select/drafter 日志行是这套 harness 最有价值的设计:
> 事后所有"命中率是多少、是不是走了守卫、drafter 有没有饿死"的追问都能回溯,不用重跑。

### 判读纪律(违反任何一条,数字就不可信)

| 规矩 | 原因 |
|---|---|
| one_batch 至少 2 次,判精细差异用 3–4 次取稳态 | rep1 系统性低 8–13% |
| gsm8k 大差异必须复跑 | 单跑可出 15% 级离群;两次复测应互差 <1% |
| 判 K/F **只认 one_batch** | 不同 K 的输出 token 数会变,gsm8k 分子分母同时漂 |
| acc 用 one_batch 的 `acc_length` | `acc_server` 是自启动累计值,serving 腿上会顶到理论上限 |
| profile 过的 server 必须重启再测性能 | profiler 残留影响 |
| 对照两腿必须同 seed、同 graph bs 列表、同 backend | 否则比的不是同一件事 |
| 新 fast-path 必须带命中计数器 | 门恒假时两腿全等,"全绿"实为"没跑" |

---

## 5 · profile 采集

```bash
curl -s http://127.0.0.1:33700/start_profile -H "Content-Type: application/json" \
  -d '{"output_dir":"/path/prof","num_steps":30,"activities":["CPU","GPU"],
       "with_stack":false,"record_shapes":false}'
# drafter 侧同样打一份(端口 33701),双侧对齐才能看跨进程节拍
```

- **`with_stack:false, record_shapes:false` 是硬性要求**:默认值把 CPU 侧膨胀 ~170×。
- 采集前先跑一次弃测 + 一次稳态 one_batch,让 profile 窗口落在稳态。
- drafter 侧建议加 `SGLANG_DEBUG_DECOUPLED_HOST_BANDS=1`,把 host 分段计时打进 trace。
- 导出后用 `gzip -t` 验完整性(截断的 trace 会静默半读)。
- **判读警告**:开 profiler 时 decoupled 的 acc 会崩(2.40→1.09)而 MTP 不受影响。
  **trace 只读结构,吞吐/acc 一律引用未开 profiler 的腿。** 见 [E11](05-pitfalls.md#e11)。
