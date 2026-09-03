# Copyright (c) 2026, Google LLC. All rights reserved.
"""Standalone TPU trainer & verification driver for NeMo Speech models using torchtpu."""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.losses.tpu_ctc_loss import tpu_ctc_loss


def init_tpu_environment():
    """Initializes torch_tpu runtime and distributed process group if applicable."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    device = torch.device("tpu")

    if world_size == 1:
        try:
            from torch_tpu._internal.device._device_module import TpuDeviceModule
            if not TpuDeviceModule.is_initialized():
                print("[Rank 0] Initializing PJRT runtime for single chip...")
                TpuDeviceModule._init_runtime_options()
        except Exception as e:
            print(f"[Rank 0] Warning during single-chip PJRT init: {e}")
    else:
        if not dist.is_initialized():
            print(f"[Rank {rank}] Initializing torch.distributed with backend='tpu_dist' (world_size={world_size})...")
            dist.init_process_group(backend="tpu_dist", rank=rank, world_size=world_size)
            dist.barrier()

    return device, rank, world_size, local_rank


def build_small_conformer_config(vocab_size=32, d_model=176, n_layers=4, n_heads=4, pos_emb_max_len=100):
    """Creates a clean Hydra configuration for a small Conformer CTC model."""
    cfg = OmegaConf.create({
        "sample_rate": 16000,
        "preprocessor": {
            "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
            "sample_rate": 16000,
            "normalize": "per_feature",
            "window_size": 0.025,
            "window_stride": 0.01,
            "window": "hann",
            "features": 80,
            "n_fft": 512,
            "frame_splicing": 1,
            "dither": 0.0,
            "pad_to": 0,
        },
        "encoder": {
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "feat_in": 80,
            "feat_out": -1,
            "n_layers": n_layers,
            "d_model": d_model,
            "subsampling": "striding",
            "subsampling_factor": 4,
            "subsampling_conv_channels": d_model,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": n_heads,
            "att_context_size": [-1, -1],
            "xscaling": True,
            "untie_biases": True,
            "pos_emb_max_len": pos_emb_max_len,
            "conv_kernel_size": 31,
            "conv_norm_type": "layer_norm",
            "dropout": 0.0,
            "dropout_emb": 0.0,
            "dropout_att": 0.0,
        },
        "decoder": {
            "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
            "feat_in": d_model,
            "num_classes": vocab_size,
            "vocabulary": [str(i) for i in range(vocab_size)],
        },
        "optim": {
            "name": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }
    })
    return cfg


def build_static_dataloader(batch_size, feat_dim, time_steps, target_len, vocab_size, device):
    """Generates statically-shaped synthetic spectrogram batches to prevent XLA recompilation."""
    # Static spectrogram shape: (B, D, T)
    features = torch.randn(batch_size, feat_dim, time_steps, device=device, dtype=torch.bfloat16)
    feature_lengths = torch.full((batch_size,), time_steps, device=device, dtype=torch.long)
    # Static targets shape: (B, L)
    targets = torch.randint(0, vocab_size, (batch_size, target_len), device=device, dtype=torch.long)
    target_lengths = torch.full((batch_size,), target_len, device=device, dtype=torch.long)

    while True:
        yield features, feature_lengths, targets, target_lengths


def apply_block_compilation(model, compile_mode="layer"):
    """Applies torch.compile(backend='tpu', dynamic=False) to encoder blocks."""
    if compile_mode == "none":
        print("Compilation disabled (running in eager mode).")
        return model

    if compile_mode == "layer":
        print("Applying block-level torch.compile(backend='tpu', dynamic=False) to encoder layers...")
        for i, layer in enumerate(model.encoder.layers):
            model.encoder.layers[i] = torch.compile(
                layer, backend="tpu", dynamic=False
            )
    elif compile_mode == "model":
        print("Applying whole-encoder torch.compile(backend='tpu', dynamic=False)...")
        model.encoder = torch.compile(model.encoder, backend="tpu", dynamic=False)
    else:
        raise ValueError(f"Unknown compile_mode: {compile_mode}")
    return model


def train_step(model, batch, optimizer, use_tpu_ctc=False, use_graph_split=False, dtype=torch.bfloat16):
    """Single forward + backward + optimizer step."""
    features, feature_lengths, targets, target_lengths = batch
    optimizer.zero_grad()

    with torch.autocast(device_type="tpu", dtype=dtype):
        encoded, encoded_len = model.encoder(audio_signal=features, length=feature_lengths)
        log_probs = model.decoder(encoder_output=encoded)

        if use_tpu_ctc:
            loss = tpu_ctc_loss(log_probs, targets)
        else:
            # Standard PyTorch CTCLoss fallback (expects time-first log_probs: T, B, C)
            log_probs_t = log_probs.float().transpose(0, 1)
            loss = nn.functional.ctc_loss(
                log_probs_t,
                targets,
                encoded_len,
                target_lengths,
                blank=log_probs.shape[-1] - 1,
                reduction="mean",
                zero_infinity=True,
            )

    if use_graph_split:
        import torch_tpu._internal.sync as tpu_sync
        tpu_sync.synchronize(loss, wait=False)

    loss.backward()
    optimizer.step()
    return loss


def main():
    parser = argparse.ArgumentParser(description="NeMo Speech TPU Trainer & Verification")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--time_steps", type=int, default=400)
    parser.add_argument("--target_len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--compile_mode", type=str, default="layer", choices=["none", "layer", "model"])
    parser.add_argument("--use_tpu_ctc", action="store_true", help="Use Pallas/JAX CTC loss custom op")
    parser.add_argument("--use_graph_split", action="store_true", help="Split fwd and bwd XLA graphs")
    parser.add_argument("--d_model", type=int, default=176)
    parser.add_argument("--n_layers", type=int, default=4)
    args = parser.parse_args()

    device, rank, world_size, local_rank = init_tpu_environment()
    torch.manual_seed(42 + rank)

    cfg = build_small_conformer_config(
        vocab_size=32, d_model=args.d_model, n_layers=args.n_layers, n_heads=4, pos_emb_max_len=args.time_steps // 4
    )
    if rank == 0:
        print(f"Building EncDecCTCModel (d_model={args.d_model}, n_layers={args.n_layers})...")

    model = EncDecCTCModel(cfg=cfg)
    model = model.to(device=device, dtype=torch.bfloat16)

    # Use foreach optimizer implementation (fused CUDA optimizers are unsupported on TPU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)

    model = apply_block_compilation(model, compile_mode=args.compile_mode)

    dataloader = build_static_dataloader(
        batch_size=args.batch_size,
        feat_dim=80,
        time_steps=args.time_steps,
        target_len=args.target_len,
        vocab_size=32,
        device=device,
    )

    if rank == 0:
        print(f"Starting training loop for {args.steps} steps (compile_mode={args.compile_mode})...")

    total_time = 0.0
    total_frames = 0

    for step, batch in enumerate(dataloader):
        if step >= args.steps:
            break

        t0 = time.perf_counter()
        loss = train_step(
            model,
            batch,
            optimizer,
            use_tpu_ctc=args.use_tpu_ctc,
            use_graph_split=args.use_graph_split,
        )
        # Synchronize device execution for accurate step timing
        loss_val = loss.cpu().item()
        step_time = time.perf_counter() - t0

        frames = args.batch_size * args.time_steps * world_size
        if step >= args.warmup_steps:
            total_time += step_time
            total_frames += frames

        if rank == 0:
            tag = "[COMPILE / WARMUP]" if step < args.warmup_steps else "[STEADY]"
            print(f"Step {step:02d} {tag} | Loss: {loss_val:.4f} | Step Time: {step_time*1000:.2f} ms | FPS: {frames/step_time:.1f}")

    if rank == 0 and total_time > 0:
        avg_fps = total_frames / total_time
        avg_step_ms = (total_time / (args.steps - args.warmup_steps)) * 1000
        print("=" * 60)
        print(f"Verification Complete!")
        print(f"Average Steady-State Step Time: {avg_step_ms:.2f} ms")
        print(f"Average Throughput: {avg_fps:.1f} audio frames/sec")
        print("=" * 60)

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
