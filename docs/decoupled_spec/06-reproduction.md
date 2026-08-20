# 06 · 从零复现:新环境 runbook

> 目标:一台什么都没有的机器 → 拿到与 [04-results](04-results.md) 同量级的数字。
> 按顺序做,每步都有**验收判据**;判据不过就停下,别往下走。

## 0 · 机器要求

| 项 | 最低 | 说明 |
|---|---|---|
| GPU | 2 张(单卡 target)或 5 张(TP4 target) | verifier 用 `--tp N`,drafter 恒独占 1 张 |
| 显存 | 按模型 | 397B-FP8 需 TP4;32B-FP8 单卡够 |
| 盘 | 模型大小 + 30% | 397B ≈ 379 GiB、235B ≈ 223 GiB、32B ≈ 32 GiB |
| 外网 | **先测** | `getent hosts huggingface.co`;无外网见 §2b |

**最省事的起手式是 32B pair**(Qwen3-32B-FP8 + Qwen3-0.6B):单卡 target、32 GiB 权重、
无 hybrid 状态、最稳。先用它跑通全链路,再上大模型。

---

## 1 · 代码

```bash
cd /sgl-workspace
git clone https://github.com/zhendonghua/sglang.git sglang
cd sglang && git checkout decoupled-spec-e2e
git log --oneline -1     # 应为 459ae93227 或更新
python3 -c "import sglang; print(sglang.__file__)"
```

**验收**:最后一行必须指向 `/sgl-workspace/sglang/python/sglang/__init__.py`。
若指向 site-packages,你测的是**另一份代码**——这是最隐蔽的整轮作废原因。

---

## 2 · 模型

### 2a · 有外网
```bash
export HF_HOME=/scratch/$USER/hf_cache
hf download Qwen/Qwen3-32B-FP8
hf download Qwen/Qwen3-0.6B
```

### 2b · 无外网(2026-08 的 B200 即是此情形)

三件事都要做,漏一件就会**静默**失效:

```bash
# ① 模型:从一台有外网的机器流式中转(不落中转机的盘),3 路并发最优
#    清单:curl -sL "https://huggingface.co/api/models/<repo>/tree/main?recursive=1"
#    然后逐文件 curl | ssh box 'cat > f.part && 校验大小 && mv'
#    落成纯目录 /scratch/$USER/models/<name>/,启动时用绝对路径(绕开 HF id 解析)

# ② gsm8k 数据集(否则该腿整体失效)
curl -sL -o /sgl-workspace/sglang/test.jsonl \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
wc -l /sgl-workspace/sglang/test.jsonl     # 应为 1319

# ③ ShareGPT(否则所有 serving/one_batch 腿吐 null —— random 数据集要从 HF 采样 token id)
#    672 MB:huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
#    → /scratch/$USER/sharegpt_v3.json,并给两个 bench 客户端加 --dataset-path

export HF_HUB_OFFLINE=1   # 让缺文件快速报错,而不是挂在 hub 超时上
```

**验收**:逐文件与 HF manifest 对账(size 全等),不要只看 `du -sh`。

---

## 3 · 冒烟:先证明这套东西能跑

用 32B pair 起一对(命令见 [03-benchmarking §2.1](03-benchmarking.md#dec-launch),
把 `--tp 4`/`CUDA_VISIBLE_DEVICES=0,1,2,3` 换成 `--tp 1`/`CUDA_VISIBLE_DEVICES=0`,
drafter 用 GPU 1,纯 attention draft 去掉 `--max-mamba-cache-size`)。

```bash
curl -s http://127.0.0.1:33700/health && echo " verifier OK"
curl -s http://127.0.0.1:33701/health && echo " drafter OK"
curl -s http://127.0.0.1:33700/generate -H "Content-Type: application/json" \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}'
```

**四条验收判据**(全过才算跑通):

1. 两个 `/health` 都返回 200;
2. 生成结果语义正常(不是乱码——乱码说明值路径有问题,不是性能问题);
3. verifier 日志里出现 `decoupled enum select:` 且 **hit_rate > 0.7**;
4. drafter 日志里出现 `decoupled drafter rounds:` 且 **starved 在 90% 以上**
   (说明 drafter 在等 verifier,即流水线正常;若 starved 很低,drafter 成了瓶颈,先查 [D29](05-pitfalls.md#d29))。

> 纯 attention draft 模型(0.6B)会看到一条 WARNING:
> `pre-launch + chain graph disabled for pure-attention draft models`。**这是预期行为**,见 [D28](05-pitfalls.md#d28)。

---

## 4 · 单腿基准

```bash
python3 -m sglang.bench_one_batch_server \
  --base-url http://127.0.0.1:33700 --model-path $V \
  --batch-size 1 --input-len 1024 --output-len 1024 --skip-warmup \
  --dataset-path /scratch/$USER/sharegpt_v3.json
```

跑 **3 次**,取后两次。**参考量级**(32B pair):

| 机器 | decoupled | colocated |
|---|---|---|
| B200 | ~405 tok/s / acc 3.94 | ~308 / 3.97 |
| H200 | ~299 / 3.94 | ~224 / 3.98 |

差 10% 以内算复现成功。差得多的排查顺序:
① `acc` 是否 ~3.9(低于 3 说明值路径有问题,不是性能问题)→
② `hit_rate` 是否 >0.9 → ③ drafter 的 `avg_ms` 是否 <5ms(远大于说明链图没生效,查 D29)→
④ 是不是首跑(rep1 系统性低 8–13%)。

---

## 5 · 全矩阵

矩阵驱动脚本在备份里:`~/Desktop/b200_backup_0813/newbox/bench/bench_matrix.py`
(**打过 5 个补丁的终版**,比 restore kit 里的原件新)。

```bash
cp bench_matrix.py /sgl-workspace/bench/
cd /sgl-workspace/bench

# 跑之前先改三处:
#   - HF_ENV(HF_HOME 路径、HF_HUB_OFFLINE)
#   - 五个模型常量(无外网时换成绝对路径)
#   - DATASET_ARG(指向本地 sharegpt json)

python3 bench_matrix.py b200 dec_32b colo_32b     # 先跑最小子集
python3 bench_matrix.py b200                      # 全队列(约 8-12 小时)
```

结果落 `matrix_b200_results.jsonl`,每腿一行(含该腿末尾的 select/drafter 日志行)。

**读表纪律**:按 `(config, leg)` 取最新 `ts` 去重——历史行会混在同一文件里
(H200 那份 54 行里就有 12 行是守卫前的旧值,直接聚合会把两个配置低估 2.5×)。

---

## 6 · 常见故障速查

| 现象 | 第一嫌疑 | 怎么确认 |
|---|---|---|
| 所有 serving/one_batch 腿 `null` | ShareGPT 数据集缺失 | 看 harness 输出里的 `LocalEntryNotFoundError` |
| gsm8k 腿失效 | `test.jsonl` 缺失 | 同上,报下载失败 |
| server 起不来、报 HF 解析错 | 无外网 + 用了 HF model id | 改绝对路径 |
| 吞吐只有预期的 1/5,drafter 轮 40ms+ | [D29](05-pitfalls.md#d29) 链图越界 | `grep "exceed the decode graph" drafter.log` |
| acc 卡在 1.5 左右 | [D28](05-pitfalls.md#d28)(旧版本)或值路径 bug | 看有没有守卫 WARNING;按 [02 §3 二分顺序](02-configuration.md#bisect)关机制 |
| 数字看着正常但明显偏低 | drafter 已死,verifier 全程 fallback | drafter 日志有没有在涨;**这就是 harness 加存活门的原因** |
| colocated 大模型长输入崩 | [B1](05-pitfalls.md#b1) CUDA OOM | 往前找第一条 `Scheduler hit an exception`,别停在尾部 NCCL 噪声 |
| 两次跑差 8% | 首跑效应 / 机器个体 | 多跑几次;跨机器比较必须同盒 |

---

## 7 · 正确性回归(改代码后必跑)

```bash
# 单进程假 mesh,不需要两张卡,专门验证 select 的兜底语义
SGLANG_TEST_DECOUPLED_LOOPBACK=garbage python3 -m sglang.launch_server ...   # 喂垃圾块
SGLANG_TEST_DECOUPLED_LOOPBACK=stale   python3 -m sglang.launch_server ...   # 喂陈旧块
```
两种注入下**输出必须与不开投机完全一致**——这直接验证"miss 只损吞吐、不损内容"这条核心契约。

单元测试:`test/registered/unit/spec/test_verify_worker_select.py`(select 的命中/陈旧/猜错/越界 case)、
`test/registered/unit/spec/test_decoupled_enum_buffer.py`(缓冲代轮转与 reset_slot)。

**改了 verify 侧的任何东西**,除了上面这些,还必须看 `hit_rate` 与 `acc` 有没有掉——
verify 侧 ±50µs 的时序变化就足以移动环相位,墙钟看不出来。见 [D27](05-pitfalls.md#d27)。
