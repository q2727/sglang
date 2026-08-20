#!/usr/bin/env python3
"""Decoupled-vs-colocated speculative decoding benchmark matrix (bs=1).

One box, sequential configs. For each config: boot server pair (or single
colocated server), health-wait, then run the leg suite:
  - bench_one_batch_server 1k/1k (1 discarded warmup + 2 measured)
  - bench_serving random, input {1k,2k,8k} x output 1k, concurrency 1,
    fresh seed per leg, flush_cache between legs
  - gsm8k 200 questions (bench_sglang.py method), parallel 1
Collects JSONL rows into RESULTS. Acc length from /server_info
(avg_spec_accept_length) + decoupled verifier/drafter log lines.

Usage: python3 bench_matrix.py <box: b200|h200> <config names... | all>
"""

import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request

BOX = sys.argv[1]
ONLY = sys.argv[2:] if len(sys.argv) > 2 else ["all"]

# Everything below is overridable by environment so the harness runs on a box
# that is not the one it was written on. Defaults reproduce the published runs.
ROOT = os.environ.get("SGLBENCH_ROOT", "/sgl-workspace/sglang")
BENCH = os.environ.get("SGLBENCH_DIR", "/sgl-workspace/bench")
RESULTS = f"{BENCH}/matrix_{BOX}_results.jsonl"
LOGD = f"{BENCH}/matrix_logs"
os.makedirs(LOGD, exist_ok=True)

# Attention backend per box class: sm100 (B200) wants trtllm_mha, H200 fa3.
ATTN = os.environ.get("SGLBENCH_ATTN", "trtllm_mha" if BOX == "b200" else "fa3")

# HF_HOME matters only when models are given as hub ids; with SGLBENCH_MODEL_DIR
# set, every model resolves to a plain directory and the hub is never touched.
HF_ENV = {"HF_HOME": os.environ.get("SGLBENCH_HF_HOME", "/root/.cache/huggingface")}
if os.environ.get("SGLBENCH_OFFLINE", "0") == "1":
    HF_ENV["HF_HUB_OFFLINE"] = "1"

# Models: hub ids by default; a local staging dir wins when set (the offline
# path -- see docs/decoupled_spec/06-reproduction.md).
M397 = "Qwen/Qwen3.5-397B-A17B-FP8"
D08 = "Qwen/Qwen3.5-0.8B"
M32 = "Qwen/Qwen3-32B-FP8"
M235 = "Qwen/Qwen3-235B-A22B-FP8"
D06 = "Qwen/Qwen3-0.6B"

_LOCAL = os.environ.get("SGLBENCH_MODEL_DIR", "")
if _LOCAL:
    M397 = f"{_LOCAL}/Qwen3.5-397B-A17B-FP8"
    D08 = f"{_LOCAL}/Qwen3.5-0.8B"
    M32 = f"{_LOCAL}/Qwen3-32B-FP8"
    M235 = f"{_LOCAL}/Qwen3-235B-A22B-FP8"
    D06 = f"{_LOCAL}/Qwen3-0.6B"

VPORT = int(os.environ.get("SGLBENCH_VPORT", "33700"))
DPORT = int(os.environ.get("SGLBENCH_DPORT", "33701"))

# Fixed per-input-length seeds: every config must see identical prompts.
SEEDS = {1024: 20260812, 2048: 20260813, 8192: 20260814}

# The random dataset samples token ids out of ShareGPT. Point both bench
# clients at a staged copy; required on a box without hub egress, harmless
# otherwise (SGLBENCH_SHAREGPT="" restores the download-on-demand behavior).
_SHAREGPT = os.environ.get("SGLBENCH_SHAREGPT", "")
DATASET_ARG = f" --dataset-path {_SHAREGPT}" if _SHAREGPT else ""

DEC_DRAFTER_ENV = {
    "SGLANG_ENABLE_DECOUPLED_CHAIN_GRAPH": "1",
    "SGLANG_DECOUPLED_ENUM_WAIT_MS": "200",
    "SGLANG_ENABLE_DECOUPLED_DRAFT_PRELAUNCH": "1",
    "SGLANG_ENABLE_DECOUPLED_CHAIN_PRELAUNCH": "1",
    "SGLANG_ENABLE_DECOUPLED_DEVICE_PACK": "1",
}


def sh(cmd, timeout=None, env=None, log=None):
    e = dict(os.environ)
    e.update(HF_ENV)
    if env:
        e.update(env)
    f = open(log, "ab") if log else subprocess.DEVNULL
    try:
        return subprocess.run(
            cmd if isinstance(cmd, list) else shlex.split(cmd),
            cwd=ROOT,
            env=e,
            stdout=f if log else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    finally:
        if log:
            f.close()


def sh_out(cmd, timeout=None, env=None):
    e = dict(os.environ)
    e.update(HF_ENV)
    if env:
        e.update(env)
    r = subprocess.run(
        cmd if isinstance(cmd, list) else shlex.split(cmd),
        cwd=ROOT,
        env=e,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return (r.stdout or "") + (r.stderr or "")


def spawn(cmd, env, log):
    e = dict(os.environ)
    e.update(HF_ENV)
    e.update(env or {})
    f = open(log, "ab")
    return subprocess.Popen(
        shlex.split(cmd),
        cwd=ROOT,
        env=e,
        stdout=f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def kill_all():
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    time.sleep(12)
    for sock in ("/tmp/bm_v", "/tmp/bm_d"):
        try:
            os.unlink(sock)
        except OSError:
            pass


def http(path, port=VPORT, timeout=10, post=False, body=None):
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(
        url,
        data=(
            json.dumps(body).encode() if body is not None else (b"" if post else None)
        ),
        headers={"Content-Type": "application/json"},
        method="POST" if (post or body is not None) else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def wait_health(port, boot_log, timeout_s=2400):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            http("/health", port=port, timeout=5)
            return True
        except Exception:
            pass
        try:
            if "kill_process_tree called" in open(boot_log).read()[-4000:]:
                return False
        except Exception:
            pass
        time.sleep(10)
    return False


def flush_cache():
    try:
        http("/flush_cache", post=True, timeout=30)
    except Exception:
        pass
    time.sleep(2)


def server_acc():
    try:
        info = json.loads(http("/server_info", timeout=30))
        for st in info.get("internal_states", []):
            v = st.get("avg_spec_accept_length")
            if v is not None:
                return round(float(v), 3)
    except Exception:
        pass
    return None


def grep_last(path, pat):
    try:
        hits = [ln for ln in open(path, errors="ignore") if pat in ln]
        return hits[-1].strip() if hits else None
    except Exception:
        return None


def emit(row):
    row["box"] = BOX
    row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("EMIT", json.dumps(row, ensure_ascii=False), flush=True)


def verifier_cmd(model, tp, gpus, spec):
    base = (
        f"python3 -m sglang.launch_server --model-path {model} --port {VPORT} "
        f"--tp {tp} --attention-backend {ATTN} --page-size 64 "
        f"--mem-fraction-static 0.85 --disable-prefill-cuda-graph "
        f"--skip-server-warmup"
    )
    return base + " " + spec, {"CUDA_VISIBLE_DEVICES": gpus}


def drafter_cmd(model, gpu, k, f, hybrid):
    mamba = "--max-mamba-cache-size 1024 " if hybrid else ""
    # Chain graph needs one bucket per (K+1)*F rows; without it the bucket is
    # disabled and the drafter round blows up ~26x (measured H200 K3F4).
    graph_bs = "1 2 4 8" if (k + 1) * f <= 8 else "1 2 4 8 12 16 24 32"
    cmd = (
        f"python3 -m sglang.launch_server --model-path {model} --port {DPORT} "
        f"--mem-fraction-static 0.6 --skip-server-warmup "
        f"--attention-backend {ATTN} --page-size 64 {mamba}"
        f"--speculative-algorithm STANDALONE --speculative-draft-model-path {model} "
        f"--speculative-num-steps {k} --speculative-fanout {f} "
        f"--cuda-graph-bs-decode {graph_bs} --max-running-requests 2048 "
        f"--decoupled-spec-role drafter --decoupled-spec-rank 0 "
        f"--decoupled-spec-data-transport zmq "
        f"--decoupled-spec-bind-endpoint ipc:///tmp/bm_d "
        "--decoupled-spec-connect-endpoints '[\"ipc:///tmp/bm_v\"]'"
    )
    env = dict(DEC_DRAFTER_ENV)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return cmd, env


def dec_spec(draft, k, f):
    return (
        f"--speculative-algorithm STANDALONE --speculative-draft-model-path {draft} "
        f"--speculative-num-steps {k} --speculative-fanout {f} "
        f"--decoupled-spec-role verifier --decoupled-spec-rank 0 "
        f"--decoupled-spec-data-transport zmq "
        f"--decoupled-spec-bind-endpoint ipc:///tmp/bm_v "
        "--decoupled-spec-connect-endpoints '[\"ipc:///tmp/bm_d\"]'"
    )


def colo_spec(draft, k):
    return (
        f"--speculative-algorithm STANDALONE --speculative-draft-model-path {draft} "
        f"--speculative-num-steps {k} --speculative-eagle-topk 1 "
        f"--speculative-num-draft-tokens {k + 1}"
    )


def mtp_spec(model, k):
    return (
        f"--speculative-algorithm EAGLE --speculative-draft-model-path {model} "
        f"--speculative-num-steps {k} --speculative-eagle-topk 1 "
        f"--speculative-num-draft-tokens {k + 1}"
    )


def make_configs():
    cfgs = {}

    def dec(name, target, draft, tp, vg, dg, k, fo, hybrid_draft):
        cfgs[name] = {
            "kind": "decoupled",
            "target": target,
            "draft": draft,
            "k": k,
            "f": fo,
            "tp": tp,
            "verifier": verifier_cmd(target, tp, vg, dec_spec(draft, k, fo)),
            "drafter": drafter_cmd(draft, dg, k, fo, hybrid_draft),
        }

    def single(name, kind, target, draft, tp, vg, k):
        spec = colo_spec(draft, k) if kind == "colocated" else mtp_spec(target, k)
        cfgs[name] = {
            "kind": kind,
            "target": target,
            "draft": draft,
            "k": k,
            "f": None,
            "tp": tp,
            "verifier": verifier_cmd(target, tp, vg, spec),
            "drafter": None,
        }

    for k, fo in [(3, 2), (3, 4), (5, 2), (5, 4)]:
        dec(f"kf_397b_k{k}f{fo}", M397, D08, 4, "0,1,2,3", "4", k, fo, True)
    dec("dec_397b", M397, D08, 4, "0,1,2,3", "4", 3, 2, True)
    dec("dec_32b", M32, D06, 1, "0", "1", 3, 2, False)
    dec("dec_235b", M235, D06, 4, "0,1,2,3", "4", 3, 2, False)
    single("colo_397b", "colocated", M397, D08, 4, "0,1,2,3", 3)
    single("colo_32b", "colocated", M32, D06, 1, "0", 3)
    single("colo_235b", "colocated", M235, D06, 4, "0,1,2,3", 3)
    single("mtp_397b", "mtp", M397, None, 4, "0,1,2,3", 3)
    return cfgs


def parse_one_batch(out):
    row = {}
    for key, pat in [
        ("latency_s", r"latency: ([\d.]+)"),
        ("input_tps", r"input throughput: ([\d.]+)"),
        ("decode_tps", r"output throughput: ([\d.]+)"),
        ("ttft_s", r"last_ttft: ([\d.]+)"),
        ("acc_length", r"acc_length: ([\d.]+)"),
    ]:
        m = re.search(pat, out)
        row[key] = float(m.group(1)) if m else None
    return row


def parse_serving(out):
    row = {}
    for key, pat in [
        ("request_tps", r"Request throughput \(req/s\):\s+([\d.]+)"),
        ("input_tps", r"Input token throughput \(tok/s\):\s+([\d.]+)"),
        ("output_tps", r"Output token throughput \(tok/s\):\s+([\d.]+)"),
        ("total_tps", r"Total token throughput \(tok/s\):\s+([\d.]+)"),
        ("mean_ttft_ms", r"Mean TTFT \(ms\):\s+([\d.]+)"),
        ("mean_tpot_ms", r"Mean TPOT \(ms\):\s+([\d.]+)"),
        ("mean_e2e_ms", r"Mean E2E Latency \(ms\):\s+([\d.]+)"),
    ]:
        m = re.search(pat, out)
        row[key] = float(m.group(1)) if m else None
    return row


def parse_gsm8k(out):
    row = {}
    for key, pat in [
        ("accuracy", r"Accuracy: ([\d.]+)"),
        ("latency_s", r"Latency: ([\d.]+)"),
        ("output_tps", r"Output throughput: ([\d.]+)"),
    ]:
        m = re.search(pat, out)
        row[key] = float(m.group(1)) if m else None
    return row


def dec_log_stats(name):
    vlog, dlog = f"{LOGD}/{name}_v.log", f"{LOGD}/{name}_d.log"
    return {
        "d28_guard": sum(
            1
            for ln in open(dlog, errors="ignore")
            if "pure-attention draft models" in ln
        ),
        "select_line": grep_last(vlog, "decoupled enum select:"),
        "drafter_line": grep_last(dlog, "decoupled drafter rounds:"),
        "tb_v": sum(
            1
            for ln in open(vlog, errors="ignore")
            if "Traceback" in ln or "CUDA error" in ln
        ),
    }


def run_config(name, cfg, legs):
    print(f"===== CONFIG {name} ({cfg['kind']}) legs={legs}", flush=True)
    kill_all()
    vlog = f"{LOGD}/{name}_v.log"
    open(vlog, "w").close()
    vcmd, venv = cfg["verifier"]
    spawn(vcmd, venv, vlog)
    dproc = None
    if cfg["drafter"]:
        dlog = f"{LOGD}/{name}_d.log"
        open(dlog, "w").close()
        dcmd, denv = cfg["drafter"]
        dproc = spawn(dcmd, denv, dlog)
    if not wait_health(VPORT, vlog):
        emit({"config": name, "leg": "BOOT", "status": "FAIL"})
        tail = ""
        try:
            tail = open(vlog, errors="ignore").read()[-1500:]
        except Exception:
            pass
        print(f"BOOT FAIL {name}\n{tail}", flush=True)
        kill_all()
        return False
    time.sleep(8)
    if dproc is not None and dproc.poll() is not None:
        emit(
            {
                "config": name,
                "leg": "BOOT",
                "status": "FAIL",
                "reason": f"drafter exited rc={dproc.returncode}",
            }
        )
        print(f"DRAFTER DEAD {name}", flush=True)
        kill_all()
        return False
    # warmup: short gen + one discarded 1k/256 one-batch pass
    try:
        http(
            "/generate",
            body={
                "text": "Hello, my name is",
                "sampling_params": {"max_new_tokens": 32, "temperature": 0},
            },
            timeout=600,
        )
    except Exception as exc:
        print("warmup gen failed:", exc, flush=True)
    sh(
        f"python3 -m sglang.bench_one_batch_server --base-url http://127.0.0.1:{VPORT} "
        f"--model-path {cfg['target']} --batch-size 1 --input-len 1024 "
        f"--output-len 256 --skip-warmup" + DATASET_ARG,
        timeout=1200,
    )

    if "one_batch" in legs:
        for i in (1, 2):
            out = sh_out(
                f"python3 -m sglang.bench_one_batch_server "
                f"--base-url http://127.0.0.1:{VPORT} --model-path {cfg['target']} "
                f"--batch-size 1 --input-len 1024 --output-len 1024 --skip-warmup"
                + DATASET_ARG,
                timeout=1800,
            )
            row = {"config": name, "leg": f"one_batch_run{i}", **parse_one_batch(out)}
            if cfg["kind"] == "decoupled":
                row.update(dec_log_stats(name))
            emit(row)

    if "serving" in legs:
        for ilen in (1024, 2048, 8192):
            flush_cache()
            nprompt = 6 if ilen == 8192 else 10
            seed = SEEDS[ilen]
            out = sh_out(
                f"python3 -m sglang.bench_serving --backend sglang "
                f"--host 127.0.0.1 --port {VPORT} --dataset-name random "
                f"--random-input-len {ilen} --random-output-len 1024 "
                f"--random-range-ratio 1.0 --num-prompts {nprompt} "
                f"--max-concurrency 1 --seed {seed} --disable-tqdm" + DATASET_ARG,
                timeout=5400,
            )
            row = {
                "config": name,
                "leg": f"serving_in{ilen}",
                "seed": seed,
                "num_prompts": nprompt,
                **parse_serving(out),
            }
            row["acc_server"] = server_acc()
            if cfg["kind"] == "decoupled":
                row.update(dec_log_stats(name))
            emit(row)
            if row.get("output_tps") is None:
                print("SERVING PARSE MISS:", out[-1200:], flush=True)

    if "gsm8k" in legs:
        flush_cache()
        out = sh_out(
            f"python3 {ROOT}/benchmark/gsm8k/bench_sglang.py "
            f"--num-questions 200 --parallel 1 --num-shots 5 "
            f"--host 127.0.0.1 --port {VPORT}",
            timeout=7200,
        )
        row = {"config": name, "leg": "gsm8k_200", **parse_gsm8k(out)}
        row["acc_server"] = server_acc()
        if cfg["kind"] == "decoupled":
            row.update(dec_log_stats(name))
        emit(row)
        if row.get("accuracy") is None:
            print("GSM8K PARSE MISS:", out[-1200:], flush=True)

    kill_all()
    return True


def main():
    cfgs = make_configs()
    if BOX == "b200":
        order = [
            ("kf_397b_k3f2", ["gsm8k", "serving1k"]),
            ("kf_397b_k3f4", ["gsm8k", "serving1k"]),
            ("kf_397b_k5f2", ["gsm8k", "serving1k"]),
            ("kf_397b_k5f4", ["gsm8k", "serving1k"]),
            ("dec_397b", ["one_batch", "serving", "gsm8k"]),
            ("colo_397b", ["one_batch", "serving", "gsm8k"]),
            ("mtp_397b", ["one_batch", "serving", "gsm8k"]),
            ("dec_32b", ["one_batch", "serving", "gsm8k"]),
            ("colo_32b", ["one_batch", "serving", "gsm8k"]),
            ("dec_235b", ["one_batch", "serving", "gsm8k"]),
            ("colo_235b", ["one_batch", "serving", "gsm8k"]),
        ]
    else:
        order = [
            ("colo_32b", ["one_batch", "serving", "gsm8k"]),
            ("dec_32b", ["one_batch", "serving", "gsm8k"]),
            ("colo_397b", ["one_batch", "serving", "gsm8k"]),
            ("mtp_397b", ["one_batch", "serving", "gsm8k"]),
            ("dec_397b", ["one_batch", "serving", "gsm8k"]),
            ("colo_235b", ["one_batch", "serving", "gsm8k"]),
            ("dec_235b", ["one_batch", "serving", "gsm8k"]),
        ]
    for name, legs in order:
        if ONLY != ["all"] and name not in ONLY:
            continue
        # kf sweep: serving 1k only
        if "serving1k" in legs:
            cfg = cfgs[name]
            print(f"===== CONFIG {name} (kf sweep)", flush=True)
            kill_all()
            vlog = f"{LOGD}/{name}_v.log"
            open(vlog, "w").close()
            vcmd, venv = cfg["verifier"]
            spawn(vcmd, venv, vlog)
            dlog = f"{LOGD}/{name}_d.log"
            open(dlog, "w").close()
            dcmd, denv = cfg["drafter"]
            spawn(dcmd, denv, dlog)
            if not wait_health(VPORT, vlog):
                emit({"config": name, "leg": "BOOT", "status": "FAIL"})
                kill_all()
                continue
            time.sleep(8)
            sh(
                f"python3 -m sglang.bench_one_batch_server "
                f"--base-url http://127.0.0.1:{VPORT} --model-path {cfg['target']} "
                f"--batch-size 1 --input-len 1024 --output-len 256 --skip-warmup"
                + DATASET_ARG,
                timeout=1200,
            )
            flush_cache()
            seed = SEEDS[1024]
            out = sh_out(
                f"python3 -m sglang.bench_serving --backend sglang "
                f"--host 127.0.0.1 --port {VPORT} --dataset-name random "
                f"--random-input-len 1024 --random-output-len 1024 "
                f"--random-range-ratio 1.0 --num-prompts 10 "
                f"--max-concurrency 1 --seed {seed} --disable-tqdm" + DATASET_ARG,
                timeout=5400,
            )
            row = {
                "config": name,
                "leg": "serving_in1024",
                "seed": seed,
                **parse_serving(out),
            }
            row["acc_server"] = server_acc()
            row.update(dec_log_stats(name))
            emit(row)
            flush_cache()
            out = sh_out(
                f"python3 {ROOT}/benchmark/gsm8k/bench_sglang.py "
                f"--num-questions 200 --parallel 1 --num-shots 5 "
                f"--host 127.0.0.1 --port {VPORT}",
                timeout=7200,
            )
            row = {"config": name, "leg": "gsm8k_200", **parse_gsm8k(out)}
            row["acc_server"] = server_acc()
            row.update(dec_log_stats(name))
            emit(row)
            kill_all()
        else:
            run_config(name, cfgs[name], legs)
    print("MATRIX_DONE", flush=True)


if __name__ == "__main__":
    main()
