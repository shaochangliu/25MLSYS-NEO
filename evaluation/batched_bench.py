"""
Synthetic benchmark for SwiftLLM / NEO.

This script mirrors the llama.cpp batched-bench workload shape, but keeps the
NEO scheduling and pipeline decisions unchanged. The user specifies:
  - prompt length (PP)
  - output length (TG)
  - global batch size (number of prompts)

The script generates synthetic prompts instead of reading a dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
neo_dir = os.path.dirname(cur_dir)

if neo_dir not in sys.path:
    sys.path.insert(0, neo_dir)

import swiftllm  # pylint: disable=import-error
from swiftllm.model_config import LlamaModelConfig

DEFAULT_PROMPT_TOKEN = 10


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_library_path(model_path: str, library_path: str | None) -> str:
    if library_path is not None:
        return library_path

    model_name = os.path.basename(model_path).lower()
    if "llama-3" in model_name or "llama3" in model_name:
        library_name = "libpacpu-llama3_8b-tp1.so"
    elif "llama-2" in model_name or "llama2" in model_name:
        library_name = "libpacpu-llama2_7b-tp1.so"
    else:
        raise ValueError(
            f"Cannot infer pacpu library from model path {model_path!r}. Pass --library-path explicitly."
        )
    return os.path.join(neo_dir, "pacpu", "build", library_name)

    
def build_engine_config(
    model_path: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    profile_dir: str,
    *,
    block_size: int = 16,
    gpu_mem_utilization: float = 0.99,
    num_gpu_blocks_override: int = -1,
    swap_space: int = 20,
    max_tokens_in_batch: int,
    library_path: str | None = None,
    tensor_parallel_degree: int = 1,
) -> swiftllm.EngineConfig:
    model_config = LlamaModelConfig.load_from_model_path(model_path)
    max_seq_len = input_len + output_len
    if max_seq_len > int(model_config.max_position_embeddings):
        raise ValueError(
            f"input_len + output_len = {max_seq_len} exceeds max_model_len = {model_config.max_position_embeddings}"
        )

    max_blocks_per_seq = math.ceil(max_seq_len / block_size)

    library_path = infer_library_path(model_path, library_path)

    return swiftllm.EngineConfig(
        model_path=model_path,
        use_dummy=False,
        block_size=block_size,
        gpu_mem_utilization=gpu_mem_utilization,
        num_gpu_blocks_override=num_gpu_blocks_override,
        swap_space=swap_space,
        max_seqs_in_block_table=num_prompts,
        max_blocks_per_seq=max_blocks_per_seq,
        max_batch_size=min(256, num_prompts),
        max_tokens_in_batch=max_tokens_in_batch,
        library_path=library_path,
        profile_result_path=profile_dir,
        tensor_parallel_degree=tensor_parallel_degree,
        disable_partial_offl=False,
        always_use_gpu=False,
        extra_layer_for_cprf=False,
    )


def compute_perf_summary(
    perf_results,
    num_layers: int,
    num_prompts: int,
    input_len: int,
    output_len: int,
    total_wall_time_s: float,
) -> dict:
    total_input_tokens = num_prompts * input_len
    total_output_tokens = num_prompts * output_len
    total_tokens = total_input_tokens + total_output_tokens

    rps = num_prompts / total_wall_time_s if total_wall_time_s > 0 else 0.0
    total_throughput = total_tokens / total_wall_time_s if total_wall_time_s > 0 else 0.0

    prefill_time_s = 0.0
    decode_time_s = 0.0
    for result in perf_results:
        prefill_time_s += float(result.avg_pref_time) * num_layers / 1000.0
        decode_time_s += float(result.avg_gdec_time + result.avg_cdec_time) * num_layers / 1000.0

    prefill_throughput = total_input_tokens / prefill_time_s if prefill_time_s > 0 else 0.0
    decode_throughput = total_output_tokens / decode_time_s if decode_time_s > 0 else 0.0

    return {
        "num_prompts": num_prompts,
        "input_len": input_len,
        "output_len": output_len,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_wall_time_s": total_wall_time_s,
        "rps": rps,
        "total_throughput_tok_per_s": total_throughput,
        "prefill_time_s": prefill_time_s,
        "prefill_throughput_tok_per_s": prefill_throughput,
        "decode_time_s": decode_time_s,
        "decode_throughput_tok_per_s": decode_throughput,
        "num_perf_records": len(perf_results),
    }


async def run_case(
    model_path: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    *,
    library_path: str | None = None,
    block_size: int = 16,
    gpu_mem_utilization: float = 0.99,
    num_gpu_blocks_override: int = -1,
    swap_space: int = 20,
    max_tokens_in_batch: int,
    tensor_parallel_degree: int = 1,
) -> dict:
    profile_dir = os.path.join(neo_dir, "profile_results") + os.sep

    engine_config = build_engine_config(
        model_path,
        num_prompts,
        input_len,
        output_len,
        profile_dir,
        block_size=block_size,
        gpu_mem_utilization=gpu_mem_utilization,
        num_gpu_blocks_override=num_gpu_blocks_override,
        swap_space=swap_space,
        max_tokens_in_batch=max_tokens_in_batch,
        library_path=library_path,
        tensor_parallel_degree=tensor_parallel_degree,
    )

    engine = swiftllm.AsyncEngine(engine_config)
    loop_task = None
    start_wall = 0.0
    end_wall = 0.0
    perf_results = []

    try:
        await engine.initialize_async()
        engine.executor.turn_on_perf_monitor()

        loop_task = asyncio.create_task(engine.start_all_event_loops())

        prompts = [[DEFAULT_PROMPT_TOKEN] * input_len for _ in range(num_prompts)]
        raw_requests = [swiftllm.RawRequest(prompt, output_len) for prompt in prompts]

        start_wall = time.perf_counter()
        await asyncio.gather(*(engine.add_request_and_wait(raw_request) for raw_request in raw_requests))
        end_wall = time.perf_counter()
        perf_results = engine.executor.turn_off_perf_monitor_and_flush_results()
    finally:
        if loop_task is not None:
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)
        torch.cuda.empty_cache()

    summary = compute_perf_summary(
        perf_results,
        engine.model_config.num_layers,
        num_prompts,
        input_len,
        output_len,
        end_wall - start_wall,
    )

    summary.update({
        "model": os.path.basename(model_path),
        "model_path": model_path,
        "library": os.path.basename(engine_config.library_path),
        "block_size": engine_config.block_size,
        "max_batch_size": engine_config.max_batch_size,
        "max_tokens_in_batch": engine_config.max_tokens_in_batch,
        "max_seqs_in_block_table": engine_config.max_seqs_in_block_table,
        "max_blocks_per_seq": engine_config.max_blocks_per_seq,
        "gpu_mem_utilization": engine_config.gpu_mem_utilization,
    })
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic batched benchmark for SwiftLLM / NEO")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the HF model directory")
    parser.add_argument("--num-prompts", type=int, required=True, help="Global batch size")
    parser.add_argument("--input-len", type=int, required=True, help="Prompt length (PP)")
    parser.add_argument("--output-len", type=int, required=True, help="Generation length (TG)")
    parser.add_argument("--config", type=str, default=None, help="Optional NEO evaluation config file for overrides")
    parser.add_argument("--block-size", type=int, default=16, help="PagedAttention block size (keep the same as NEO paper)")
    parser.add_argument("--max-tokens-in-batch", type=int, default=2048, help="User-provided token budget per NEO batch (keep the same as llama.cpp)")
    parser.add_argument("--gpu-mem-utilization", type=float, default=0.99, help="Primary memory budget knob for inference")
    parser.add_argument("--num-gpu-blocks-override", type=int, default=-1, help="Optional override for the profiled GPU block count")
    parser.add_argument("--swap-space", type=int, default=20, help="Swap space in GB")
    parser.add_argument("--library-path", type=str, default=None, help="Override CPU kernel library path from config")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save the benchmark summary as JSON")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    if args.config is not None:
        config = load_config(args.config)
        if args.library_path is None:
            args.library_path = config.get("library_path") or (
                os.path.join(neo_dir, "pacpu", "build", config["library"])
                if config.get("library")
                else None
            )
        args.block_size = int(config.get("block_size", args.block_size))
        args.gpu_mem_utilization = float(config.get("gpu_memory_utilization", args.gpu_mem_utilization))
        args.num_gpu_blocks_override = int(config.get("num_gpu_blocks_override", args.num_gpu_blocks_override))
        args.swap_space = int(config.get("swap_space", args.swap_space))

    summary = await run_case(
        args.model_path,
        args.num_prompts,
        args.input_len,
        args.output_len,
        library_path=args.library_path,
        block_size=args.block_size,
        max_tokens_in_batch=args.max_tokens_in_batch,
        gpu_mem_utilization=args.gpu_mem_utilization,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        swap_space=args.swap_space,
    )

    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))