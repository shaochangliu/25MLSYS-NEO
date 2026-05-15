"""
Minimal pacpu CPU decode test.

Runs torch.ops.pacpu.paged_attention_cpu with a single short sequence.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import json
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
neo_dir = os.path.dirname(cur_dir)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal pacpu CPU decode test")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the HF model directory")
    parser.add_argument("--library-path", type=str, default=None, help="Path to libpacpu-*.so")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length (<= block size is best)")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations to run")
    return parser.parse_args()


def load_model_config(model_path: str) -> dict:
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    library_path = infer_library_path(args.model_path, args.library_path)
    torch.ops.load_library(library_path)

    model_config = load_model_config(args.model_path)
    num_layers = model_config["num_hidden_layers"]
    num_q_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config.get("num_key_value_heads", num_q_heads)
    hidden_size = model_config["hidden_size"]
    head_dim = hidden_size // num_q_heads
    softmax_scale = head_dim ** -0.5

    block_size = 16

    if head_dim != 128:
        raise ValueError(f"Expected head_dim=128 for pacpu, got {head_dim}")

    if args.seq_len < 1 or args.seq_len > block_size:
        raise ValueError(f"seq_len must be in [1, {block_size}] for this test")

    batch_size = 1
    num_cpu_blocks = 1
    max_blocks_per_seq = 1
    max_seqs_in_block_table = 1

    q = torch.randn((batch_size, num_q_heads, head_dim), dtype=torch.float16, device="cpu", pin_memory=True)
    k = torch.randn((batch_size, num_kv_heads, head_dim), dtype=torch.float16, device="cpu", pin_memory=True)
    v = torch.randn((batch_size, num_kv_heads, head_dim), dtype=torch.float16, device="cpu", pin_memory=True)

    k_cache = torch.zeros(
        (num_layers, num_cpu_blocks, num_kv_heads, block_size, head_dim),
        dtype=torch.float16,
        device="cpu",
        pin_memory=True,
    )
    v_cache = torch.zeros_like(k_cache)

    block_table = torch.zeros(
        (max_seqs_in_block_table, max_blocks_per_seq),
        dtype=torch.int32,
        device="cpu",
    )
    block_table[0, 0] = 0

    o = torch.zeros((batch_size, num_q_heads, head_dim), dtype=torch.float32, device="cpu", pin_memory=True)

    seq_ids = [0]
    seq_lengths = [args.seq_len]
    cur_layer = 0
    for i in range(args.iters):
        start_ms = time.perf_counter() * 1e3
        torch.ops.pacpu.paged_attention_cpu(
            cur_layer,
            softmax_scale,
            seq_ids,
            seq_lengths,
            q,
            k,
            v,
            k_cache,
            v_cache,
            block_table,
            o,
        )
        end_ms = time.perf_counter() * 1e3
        print(f"iter {i}: ok, dur_ms={end_ms - start_ms:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
