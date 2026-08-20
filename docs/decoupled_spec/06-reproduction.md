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

### 软件前置(数字就是在这套栈上取的)

| 组件 | B200 轮 | H200 轮 | 备注 |
|---|---|---|---|
| torch | 2.11.0+cu130 | 2.11.0+cu130 | `python/pyproject.toml` 钉 `torch==2.11.0` |
| flashinfer | 0.6.15.post1 | 0.6.12 | attention backend 依赖 |
| transformers | 5.12.1 | 5.8.1 | 影响 config 解析与 chat template |
| deep_gemm | 0.1.5.post1 | 0.1.3 | 影响 MoE 数值路径 |
| sgl-kernel | 0.4.5 | 0.4.5 | 编译扩展 |
| NVIDIA driver | 580.126.09 | — | doorbell 在此版本被判 UNSAFE(见 [07 S4](07-open-questions.md#s4)) |

**这几个版本不是摆设**:[§0.4](04-results.md#caliber) 把它们列为跨盒不可比的原因之一,
数值路径的融合核也对 deep_gemm 版本敏感([D27](05-pitfalls.md#d27))。跨版本复现时,
数字对不上先怀疑这里。

**最省事的起手式是 32B pair**(Qwen3-32B-FP8 + Qwen3-0.6B):单卡 target、32 GiB 权重、
无 hybrid 状态、最稳。先用它跑通全链路,再上大模型。

---

## 1 · 代码

```bash
cd /sgl-workspace
git clone https://github.com/zhendonghua/sglang.git sglang     # 见下方"拿不到这个 fork"
cd sglang && git checkout decoupled-spec-e2e
git log --oneline -1     # 应为 459ae93227 或更新
```

> **拿不到这个 fork 时**:这是个人 fork,不保证长期可访问。本工作的代码全部在
> `decoupled-spec-e2e` 分支上(关键 commit:`b6da986ca3` armed-COW kernel、`6048dece7d` 页表行 kernel、
> `2e130ce07e` host band、`568be6d2b2`/`dd002e31ff` 位忠实融合、`80c468daae` D28 守卫、
> `459ae93227` case-0 预算)。若 fork 不可达,向仓库所有者索取 `git bundle`
> (`git bundle create decoupled.bundle decoupled-spec-e2e`)——重建 B200 时就是这么恢复的。

### 1b · 安装(不做这一步,后面全部跑不起来)

**推荐:用仓库自带的镜像**,它把 CUDA / torch / flashinfer / sgl-kernel 都装好了:

```bash
# 仓库根有 docker/Dockerfile(基于 nvidia/cuda:*-cudnn-devel-ubuntu24.04)
docker build -f docker/Dockerfile -t sglang-decoupled .
docker run --gpus all --ipc=host --shm-size 32g -v /sgl-workspace:/sgl-workspace -it sglang-decoupled
```

**或者在已有 CUDA 环境里装**:

```bash
cd /sgl-workspace/sglang
pip install -e "python[all]"        # 装 sglang 本体与依赖
pip install sgl-kernel               # 编译扩展;或 cd sgl-kernel && make build
```

**验收**(三条都要过):

```bash
python3 -c "import sglang; print(sglang.__file__)"
#   必须指向 /sgl-workspace/sglang/python/sglang/__init__.py
#   指向 site-packages 说明你测的是另一份代码——这是最隐蔽的整轮作废原因
python3 -c "import sgl_kernel, torch, flashinfer; print(torch.__version__, flashinfer.__version__)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

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

### 2c · 后面所有命令都依赖这三个变量

**无论走 2a 还是 2b,都要导出**(后续 §3/§4/§5 直接引用):

```bash
# 走 2a(有外网、用 HF 缓存)时,模型可以直接用 hub id:
export V=Qwen/Qwen3-32B-FP8
export D=Qwen/Qwen3-0.6B
# 走 2b(无外网、纯目录)时,用绝对路径:
# export V=/scratch/$USER/models/Qwen3-32B-FP8
# export D=/scratch/$USER/models/Qwen3-0.6B

# ShareGPT 两条路径都需要(random 数据集要拿它采样 token id)。
# 有外网时首次运行会自动下载到 HF 缓存,也可以显式指定一份:
export SHAREGPT=/scratch/$USER/sharegpt_v3.json
```

---

## 3 · 冒烟:先证明这套东西能跑

用 32B pair 起一对(命令见 [03-benchmarking §2.1](03-benchmarking.md#dec-launch),
把 `--tp 4`/`CUDA_VISIBLE_DEVICES=0,1,2,3` 换成 `--tp 1`/`CUDA_VISIBLE_DEVICES=0`,
drafter 用 GPU 1,纯 attention draft 去掉 `--max-mamba-cache-size`)。

**起服时把两侧日志都接出来**(后面的判据与排查都要 grep 它们):

```bash
... python3 -m sglang.launch_server ... 2>&1 | tee /tmp/verifier.log &
... python3 -m sglang.launch_server ... 2>&1 | tee /tmp/drafter.log &
```

```bash
curl -s http://127.0.0.1:33700/health && echo " verifier OK"
curl -s http://127.0.0.1:33701/health && echo " drafter OK"
# 注意 max_new_tokens 要够大:两条统计日志每 200 轮才打一次,
# 16 个 token 只有 4-5 轮,日志根本不会出现,会让健康系统"验收失败"
curl -s http://127.0.0.1:33700/generate -H "Content-Type: application/json" \
  -d '{"text":"Write a detailed essay about the history of computing.",
       "sampling_params":{"max_new_tokens":1024,"temperature":0}}'
```

**四条验收判据**(全过才算跑通):

1. 两个 `/health` 都返回 200;
2. 生成结果语义正常(不是乱码——乱码说明值路径有问题,不是性能问题);
3. `grep "decoupled enum select:" /tmp/verifier.log` 有输出,且 **hit_rate 落在合理带内**:
   32B/235B pair(纯 attn draft)**> 0.9**;397B pair(hybrid draft)**> 0.7** 即正常
   ——实测健康值 0.75–0.85,见 [04 §4](04-results.md)。低于这个带才需要排查;
4. `grep "decoupled drafter rounds:" /tmp/drafter.log` 有输出,且 **starved 在 90% 以上**
   (说明 drafter 在等 verifier,即流水线正常;若 starved 很低,drafter 成了瓶颈,先查 [D29](05-pitfalls.md#d29))。

> 两条日志都是**每 200 轮打一次**(`decoupled_verify_manager.py:968`、`decoupled_draft_manager.py:505`),
> 且**计数是生命周期累计**——要算速率得把相邻两条做差。没看到日志时先加大生成量,别急着 debug。

> 纯 attention draft 模型(0.6B)会看到一条 WARNING:
> `pre-launch + chain graph disabled for pure-attention draft models`。**这是预期行为**,见 [D28](05-pitfalls.md#d28)。

---

## 4 · 单腿基准

```bash
python3 -m sglang.benchmark.one_batch_server \
  --base-url http://127.0.0.1:33700 --model-path $V \
  --batch-size 1 --input-len 1024 --output-len 1024 --skip-warmup \
  --dataset-path $SHAREGPT
```
> `python3 -m sglang.bench_one_batch_server` 也能用,但那是个 23 行的兼容 shim,会打 FutureWarning。

跑 **3 次**,取后两次。**参考量级**(32B pair):

| 机器 | decoupled | colocated |
|---|---|---|
| B200 | ~405 tok/s / acc 3.94 | ~308 / 3.97 |
| H200 | ~299 / 3.94 | ~224 / 3.98 |

差 10% 以内算复现成功。差得多的排查顺序:
① `acc` 是否 ~3.9(低于 3 说明值路径有问题,不是性能问题)→
② `hit_rate` 是否 >0.9 → ③ drafter 的 `avg_ms` 是否 <5ms(远大于说明链图没生效,查 D29)→
④ 是不是首跑(**仅 decoupled 397B pair** 的 rep1 会低 8–11%,其余配置 rep1≈rep2,见 [04 §0.2](04-results.md#caliber))。

> ⚠️ **32B pair 跑通 ≠ 全部机制跑通。** 0.6B 是纯 attention draft,会命中 [D28 守卫](05-pitfalls.md#d28),
> 自动关掉 **pre-launch 与 chain graph**——也就是 [01 §5.3](01-architecture.md) / [§5.4](01-architecture.md) 讲的两个核心机制
> 在这条 on-ramp 上**根本没被执行**。要验证它们,必须用 hybrid draft 的 pair(如 Qwen3.5-0.8B),
> 并确认 drafter 日志里**没有** `pure-attention draft models` 那条 WARNING。

---

## 5 · 全矩阵

矩阵驱动脚本**已在仓库里**:`benchmark/decoupled_spec/bench_matrix.py`。
全部路径通过环境变量覆盖,默认值即发布数字所用的配置:

```bash
mkdir -p /sgl-workspace/bench
export SGLBENCH_ROOT=/sgl-workspace/sglang      # 仓库根(默认即此)
export SGLBENCH_DIR=/sgl-workspace/bench        # 结果与日志落点
export SGLBENCH_MODEL_DIR=/scratch/$USER/models # 纯目录模型;走 hub id 时留空
export SGLBENCH_SHAREGPT=$SHAREGPT
export SGLBENCH_OFFLINE=1                       # 无外网时设 1
# 可选:SGLBENCH_ATTN(默认 b200→trtllm_mha / 其它→fa3)、SGLBENCH_VPORT / SGLBENCH_DPORT

cd /sgl-workspace/sglang
python3 benchmark/decoupled_spec/bench_matrix.py b200 dec_32b colo_32b   # 先跑最小子集
python3 benchmark/decoupled_spec/bench_matrix.py b200                    # 全队列(约 8-12 小时)
```

**参数**:第一个位置参数是 box class(`b200` / `h200`,决定 attention backend 与队列顺序);
后面可选若干 config 名做子集过滤,缺省为 `all`。可用的 config 名:
`dec_397b` `colo_397b` `mtp_397b` `dec_32b` `colo_32b` `dec_235b` `colo_235b`
`kf_397b_k3f2` `kf_397b_k3f4` `kf_397b_k5f2` `kf_397b_k5f4`。

结果落 `$SGLBENCH_DIR/matrix_<box>_results.jsonl`,每腿一行(含该腿末尾的 select/drafter 日志行);
server 日志落 `$SGLBENCH_DIR/matrix_logs/<config>_{v,d}.log`。

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
| 数字看着正常但明显偏低、hit_rate 不涨 | **两侧 K/F 不一致**(最阴的一种) | `grep "enumeration block dims differ" /tmp/verifier.log`;两条启动命令的 `--speculative-num-steps` 与 `--speculative-fanout` 逐字对比 |
| 数字看着正常但明显偏低 | drafter 已死,verifier 全程 fallback | drafter 日志有没有在涨;**这就是 harness 加存活门的原因** |
| colocated 大模型长输入崩 | [B1](05-pitfalls.md#b1) CUDA OOM | 往前找第一条 `Scheduler hit an exception`,别停在尾部 NCCL 噪声 |
| 两次跑差 8% | 首跑效应 / 机器个体 | 多跑几次;跨机器比较必须同盒 |

---

## 7 · 正确性回归(改代码后必跑)

### 7.1 自动化测试(最省事,先跑这个)

```bash
cd /sgl-workspace/sglang
# 纯逻辑单测,不需要 GPU:
python3 -m pytest test/registered/unit/spec/test_verify_worker_select.py \
                  test/registered/unit/spec/test_decoupled_enum_buffer.py \
                  test/registered/unit/spec/test_decoupled_spec_io.py \
                  test/registered/unit/spec/test_decoupled_spec_transport.py \
                  test/registered/unit/spec/test_decoupled_spec_hook.py \
                  test/registered/unit/spec/test_decoupled_spec_ipc_integration.py -q
# 端到端环回(需要 1 张 GPU,单进程假 mesh):
python3 -m pytest test/registered/spec/test_decoupled_spec_loopback.py -q
# CUDA IPC 数据面(需要 2 张 GPU;仅在你要用 cuda_ipc 时才跑):
python3 -m pytest test/registered/spec/test_decoupled_cuda_ipc_transport.py -q
```

`test_verify_worker_select.py` 覆盖 select 的命中 / 陈旧 stamp / 猜错 / case 越界 / 首猜优先 /
上一代服务 / 混合 batch 独立性;`test_decoupled_spec_loopback.py` 是下面手工环回的自动化版本。

### 7.2 手工环回(想亲眼看兜底行为时)

`SGLANG_TEST_DECOUPLED_LOOPBACK` 在**单进程**内起一个假 mesh + 脚本化 drafter,
但**仍然要给三个 endpoint/rank 参数**(它们先被 `server_args` 校验、之后才被环回替换掉):

```bash
SGLANG_TEST_DECOUPLED_LOOPBACK=garbage \
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path $V --port 33700 --tp 1 --attention-backend fa3 --page-size 64 \
  --mem-fraction-static 0.85 --skip-server-warmup \
  --speculative-algorithm STANDALONE --speculative-draft-model-path $D \
  --speculative-num-steps 3 --speculative-fanout 2 \
  --decoupled-spec-role verifier --decoupled-spec-rank 0 \
  --decoupled-spec-data-transport zmq \
  --decoupled-spec-bind-endpoint ipc:///tmp/lb_v \
  --decoupled-spec-connect-endpoints '["ipc:///tmp/lb_d"]'
# 再跑一遍,把 garbage 换成 stale
```

两种注入下**输出必须与不开投机完全一致**——这直接验证"miss 只损吞吐、不损内容"这条核心契约。

**改了 verify 侧的任何东西**,除了上面这些,还必须看 `hit_rate` 与 `acc` 有没有掉——
verify 侧 ±50µs 的时序变化就足以移动环相位,墙钟看不出来。见 [D27](05-pitfalls.md#d27)。
