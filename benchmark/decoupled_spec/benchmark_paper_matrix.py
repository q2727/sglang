#!/usr/bin/env python3
"""Run one frozen paper dataset with resumable B=1 requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


DATASETS = (
    "gsm8k",
    "math500",
    "aime25",
    "mbpp",
    "humaneval",
    "lcb",
    "mt-bench",
    "alpaca",
    "arena-hard-v2",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    for ordinal, row in enumerate(read_jsonl(path)):
        if "text" in row:
            text = row["text"]
        elif row.get("turns"):
            text = row["turns"][0]
        else:
            raise ValueError(f"{path} row {ordinal} has no prompt text")
        prompts.append(
            {
                "index": row.get("index", ordinal),
                "source_index": row.get("source_index"),
                "text": text,
            }
        )
    return prompts


def validate_resume(
    records: list[dict], prompts: list[dict], dataset: str, label: str
) -> None:
    if len(records) > len(prompts):
        raise ValueError("Output has more records than the frozen dataset")
    for ordinal, record in enumerate(records):
        prompt = prompts[ordinal]
        prompt_hash = hashlib.sha256(prompt["text"].encode()).hexdigest()
        if record.get("dataset") != dataset or record.get("label") != label:
            raise ValueError(f"Record {ordinal} belongs to a different run")
        if record.get("dataset_index") != prompt["index"]:
            raise ValueError(f"Record {ordinal} does not match the frozen prompt")
        if record.get("prompt_sha256") != prompt_hash:
            raise ValueError(f"Record {ordinal} has a different prompt hash")


def generate(
    url: str, input_ids: list[int], max_new_tokens: int, timeout: float
) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(
            {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                    "skip_special_tokens": False,
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    begin = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    latency_s = time.perf_counter() - begin
    meta = result["meta_info"]
    output_text = result["text"]
    spec_stats = {
        key: meta[key]
        for key in (
            "spec_accept_rate",
            "spec_accept_length",
            "spec_accept_token_num",
            "spec_draft_token_num",
            "spec_verify_ct",
        )
        if key in meta
    }
    return {
        "latency_s": latency_s,
        "completion_tokens": int(meta["completion_tokens"]),
        "output_text_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
        **spec_stats,
    }


def summarize(
    output_path: Path, label: str, dataset: str, expected_requests: int
) -> dict:
    records = read_jsonl(output_path)
    latencies = [float(record["latency_s"]) for record in records]
    tokens = sum(int(record["completion_tokens"]) for record in records)
    rounds = sum(int(record.get("spec_verify_ct", 0)) for record in records)
    accepted = sum(
        int(record.get("spec_accept_token_num", 0)) for record in records
    )
    drafted = sum(int(record.get("spec_draft_token_num", 0)) for record in records)
    summary = {
        "label": label,
        "dataset": dataset,
        "requests": len(records),
        "expected_requests": expected_requests,
        "complete": len(records) == expected_requests,
        "total_tokens": tokens,
        "latency_sum_s": sum(latencies),
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "throughput_tok_s": tokens / sum(latencies) if latencies else None,
    }
    records_with_accept_length = [
        record for record in records if record.get("spec_accept_length") is not None
    ]
    if rounds and records_with_accept_length:
        weighted_rounds = sum(
            int(record.get("spec_verify_ct", 0))
            for record in records_with_accept_length
        )
        weighted_tau = (
            sum(
                float(record["spec_accept_length"])
                * int(record.get("spec_verify_ct", 0))
                for record in records_with_accept_length
            )
            / weighted_rounds
            if weighted_rounds
            else None
        )
        rate_rounds = sum(
            int(record.get("spec_verify_ct", 0))
            for record in records_with_accept_length
            if record.get("spec_accept_rate") is not None
        )
        weighted_acceptance_rate = (
            sum(
                float(record["spec_accept_rate"])
                * int(record.get("spec_verify_ct", 0))
                for record in records_with_accept_length
                if record.get("spec_accept_rate") is not None
            )
            / rate_rounds
            if rate_rounds
            else None
        )
        summary.update(
            {
                "verify_rounds": rounds,
                "acceptance_rate": weighted_acceptance_rate,
                "tau": weighted_tau,
            }
        )
        if accepted or drafted:
            summary.update(
                {
                    "accepted_draft_tokens": accepted,
                    "drafted_tokens": drafted,
                }
            )
    output_path.with_suffix(".jsonl.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = load_prompts(args.dataset_dir / f"{args.dataset}.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.output)
    validate_resume(records, prompts, args.dataset, args.label)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be non-negative")
    for warmup_index in range(args.warmup_requests):
        prompt = prompts[warmup_index % len(prompts)]
        input_text = prompt["text"]
        if args.apply_chat_template:
            input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        result = generate(
            args.url, input_ids, args.max_new_tokens, args.request_timeout
        )
        print(
            f"[warmup {warmup_index + 1}/{args.warmup_requests}] "
            f"input={len(input_ids)} output={result['completion_tokens']} "
            f"latency={result['latency_s']:.3f}s",
            flush=True,
        )

    with args.output.open("a") as output:
        for ordinal in range(len(records), len(prompts)):
            prompt = prompts[ordinal]
            input_text = prompt["text"]
            if args.apply_chat_template:
                input_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": input_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
                )
            input_ids = tokenizer.encode(input_text, add_special_tokens=False)
            if len(input_ids) > args.max_input_tokens:
                raise ValueError(
                    f"{args.dataset}[{prompt['index']}] has {len(input_ids)} "
                    f"tokens, exceeding {args.max_input_tokens}"
                )
            result = generate(
                args.url, input_ids, args.max_new_tokens, args.request_timeout
            )
            record = {
                "label": args.label,
                "ordinal": ordinal,
                "dataset": args.dataset,
                "dataset_index": prompt["index"],
                "source_index": prompt["source_index"],
                "prompt_sha256": hashlib.sha256(prompt["text"].encode()).hexdigest(),
                "prompt_tokens": len(input_ids),
                **result,
            }
            output.write(json.dumps(record) + "\n")
            output.flush()
            summary = summarize(args.output, args.label, args.dataset, len(prompts))
            print(
                f"[{ordinal + 1}/{len(prompts)}] {args.dataset}"
                f"[{prompt['index']}] input={len(input_ids)} "
                f"output={result['completion_tokens']} "
                f"latency={result['latency_s']:.3f}s "
                f"aggregate={summary['throughput_tok_s']:.2f}tok/s",
                flush=True,
            )

    summary = summarize(args.output, args.label, args.dataset, len(prompts))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
