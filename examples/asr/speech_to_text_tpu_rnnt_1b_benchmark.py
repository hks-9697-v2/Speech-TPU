# Copyright (c) 2026, Google LLC. All rights reserved.
"""
TPU v6e Benchmark & FLOP Validation Driver for 1.1B Customized Conformer-Transducer (RNN-T).

Supports:
- Block-level compilation (--compile_mode layer)
- Whole-model compilation (--compile_mode full) fusing Encoder + Decoder + Joint + Custom JAX RNN-T Loss
- Custom JAX/Pallas RNN-T Loss (--use_tpu_rnnt_loss)
- Activation Checkpointing / Rematerialization (--use_remat) for long audio sequences (30s) at BS=4
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist

from nemo.collections.asr.modules import ConformerEncoder, RNNTDecoder, RNNTJoint
from nemo.collections.asr.losses.tpu_rnnt_loss import tpu_rnnt_loss


def init_tpu_environment():
    """Initializes torch_tpu runtime and distributed process group."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    device = torch.device("tpu")

    if world_size == 1:
        try:
            from torch_tpu._internal.device._device_module import TpuDeviceModule
            if not TpuDeviceModule.is_initialized():
                TpuDeviceModule._init_runtime_options()
        except Exception:
            pass
    else:
        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist", rank=rank, world_size=world_size)
            dist.barrier()

    return device, rank, world_size, local_rank


class ConformerTransducer1B(nn.Module):
    """Wrapper combining 42-layer ConformerEncoder, 2-layer RNNTDecoder, RNNTJoint, and optional fused RNNT Loss."""

    def __init__(
        self,
        d_model=1024,
        n_layers=42,
        n_heads=8,
        pred_hidden=640,
        pred_layers=2,
        vocab_size=4096,
        max_len=400,
        use_tpu_rnnt_loss=True,
        use_remat=False,
    ):
        super().__init__()
        self.use_tpu_rnnt_loss = use_tpu_rnnt_loss
        self.use_remat = use_remat
        self.encoder = ConformerEncoder(
            feat_in=80,
            n_layers=n_layers,
            d_model=d_model,
            subsampling="striding",
            subsampling_factor=4,
            subsampling_conv_channels=d_model,
            ff_expansion_factor=4,
            self_attention_model="rel_pos",
            n_heads=n_heads,
            pos_emb_max_len=max_len,
            conv_kernel_size=31,
            conv_norm_type="layer_norm",
            dropout=0.0,
            dropout_emb=0.0,
            dropout_att=0.0,
        )
        self.encoder.use_remat = use_remat
        self.decoder = RNNTDecoder(
            prednet={"pred_hidden": pred_hidden, "pred_rnn_layers": pred_layers, "dropout": 0.0},
            vocab_size=vocab_size,
        )
        self.joint = RNNTJoint(
            jointnet={
                "encoder_hidden": d_model,
                "pred_hidden": pred_hidden,
                "joint_hidden": pred_hidden,
                "activation": "relu",
                "dropout": 0.0,
            },
            num_classes=vocab_size,
            vocabulary=[str(i) for i in range(vocab_size)],
        )

    def _single_joint_and_loss(self, enc_i, dec_i, tgt_i):
        joint_i = self.joint(encoder_outputs=enc_i, decoder_outputs=dec_i)
        if self.use_tpu_rnnt_loss:
            return tpu_rnnt_loss(joint_i, tgt_i[:, : joint_i.shape[2] - 1])
        else:
            return joint_i.float().mean()

    def _joint_and_loss(self, encoded, decoder_out, targets):
        joint_out = self.joint(encoder_outputs=encoded, decoder_outputs=decoder_out)
        if self.use_tpu_rnnt_loss:
            return tpu_rnnt_loss(joint_out, targets[:, : joint_out.shape[2] - 1])
        else:
            return joint_out.float().mean()

    def forward(self, features, feature_lengths, targets, target_lengths):
        encoded, encoded_len = self.encoder(audio_signal=features, length=feature_lengths)
        decoder_out, _, _ = self.decoder(targets=targets, target_length=target_lengths)
        if self.use_remat and self.training:
            B = encoded.shape[0]
            losses = []
            for i in range(B):
                loss_i = torch.utils.checkpoint.checkpoint(
                    self._single_joint_and_loss,
                    encoded[i : i + 1],
                    decoder_out[i : i + 1],
                    targets[i : i + 1],
                    use_reentrant=False,
                )
                losses.append(loss_i)
            loss = torch.stack(losses).mean()
        else:
            loss = self._joint_and_loss(encoded, decoder_out, targets)
        return loss


def compute_exact_rnnt_flops(batch_size, time_steps, target_len, d_model=1024, n_layers=42, pred_hidden=640, vocab_size=4096):
    """
    Calculates exact theoretical floating-point operations (FLOPs) per training step
    (Forward + Backward = 3x Forward FLOPs) for the 1.1B Conformer-Transducer model.
    """
    t_enc = time_steps // 4  # 4x striding subsampling
    d_ff = d_model * 4

    # 1. Conformer Encoder FLOPs per sample (Forward)
    ff_flops_per_layer = 2 * (2 * 2 * t_enc * d_model * d_ff)
    attn_proj_flops = 5 * (2 * t_enc * d_model * d_model)
    attn_mat_flops = 4 * (t_enc ** 2) * d_model
    conv_flops = 6 * t_enc * (d_model ** 2) + 2 * t_enc * d_model * 31

    enc_fwd_flops_per_sample = n_layers * (ff_flops_per_layer + attn_proj_flops + attn_mat_flops + conv_flops)

    # 2. RNN-T Decoder FLOPs per sample (2-layer LSTM)
    dec_fwd_flops_per_sample = 2 * target_len * (8 * (pred_hidden ** 2))

    # 3. RNN-T Joint Network FLOPs per sample
    joint_proj_flops = 2 * t_enc * d_model * pred_hidden + 2 * target_len * pred_hidden * pred_hidden
    joint_lattice_flops = 2 * t_enc * target_len * pred_hidden * vocab_size
    joint_fwd_flops_per_sample = joint_proj_flops + joint_lattice_flops

    total_fwd_flops_per_sample = enc_fwd_flops_per_sample + dec_fwd_flops_per_sample + joint_fwd_flops_per_sample
    total_step_flops_per_sample = 3 * total_fwd_flops_per_sample
    total_step_flops_per_device = batch_size * total_step_flops_per_sample

    return {
        "enc_fwd_gflops": enc_fwd_flops_per_sample / 1e9,
        "dec_fwd_gflops": dec_fwd_flops_per_sample / 1e9,
        "joint_fwd_gflops": joint_fwd_flops_per_sample / 1e9,
        "sample_step_tflops": total_step_flops_per_sample / 1e12,
        "device_step_tflops": total_step_flops_per_device / 1e12,
    }


def main():
    parser = argparse.ArgumentParser(description="1.1B Conformer-Transducer TPU v6e Benchmark")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-Device Batch Size (PDBS)")
    parser.add_argument("--audio_duration_sec", type=float, default=15.0, help="Audio duration per sample in seconds")
    parser.add_argument("--target_len", type=int, default=120, help="Target token sequence length U")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--compile_mode", type=str, default="full", choices=["none", "layer", "full"])
    parser.add_argument("--use_tpu_rnnt_loss", action="store_true", default=True)
    parser.add_argument("--use_remat", action="store_true", help="Enable activation checkpointing (remat) for Encoder and Joint lattice")
    parser.add_argument("--use_graph_split", action="store_true", default=True)
    args = parser.parse_args()

    device, rank, world_size, local_rank = init_tpu_environment()
    torch.manual_seed(42 + rank)

    time_steps = int(args.audio_duration_sec * 100)
    t_enc = time_steps // 4

    if rank == 0:
        print("=" * 80)
        print(f"TPU v6e Benchmark: 1.1B Customized Conformer-Transducer (compile_mode={args.compile_mode}, remat={args.use_remat})")
        print("=" * 80)
        print(f"Hardware: {world_size}x Google TPU v6e | Precision: BF16")
        print(f"Per-Device Batch Size (PDBS): {args.batch_size} | Global Batch Size (GBS): {args.batch_size * world_size}")
        print(f"Audio Duration: {args.audio_duration_sec}s ({time_steps} frames -> {t_enc} encoded frames) | Target Tokens: {args.target_len}")
        print(f"Loss Function: {'Custom Pallas/JAX RNN-T Loss' if args.use_tpu_rnnt_loss else 'Lattice Mean Substitute'}")

    model = ConformerTransducer1B(
        d_model=1024,
        n_layers=42,
        n_heads=8,
        pred_hidden=640,
        pred_layers=2,
        vocab_size=4096,
        max_len=t_enc,
        use_tpu_rnnt_loss=args.use_tpu_rnnt_loss,
        use_remat=args.use_remat,
    )

    if rank == 0:
        enc_p = sum(p.numel() for p in model.encoder.parameters())
        dec_p = sum(p.numel() for p in model.decoder.parameters())
        joint_p = sum(p.numel() for p in model.joint.parameters())
        tot_p = enc_p + dec_p + joint_p
        flop_stats = compute_exact_rnnt_flops(
            batch_size=args.batch_size,
            time_steps=time_steps,
            target_len=args.target_len,
        )
        print("-" * 80)
        print(f"Total Model Parameters: {tot_p:,} ({tot_p/1e9:.4f} B)")
        print(f"Total Step FLOPs (Fwd+Bwd) per TPU device (PDBS={args.batch_size}): {flop_stats['device_step_tflops']:.3f} TFLOPs")
        print("=" * 80)

    model = model.to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)

    compiled_joint_loss = None
    if args.compile_mode == "layer":
        if rank == 0:
            print("Compiling 42 ConformerEncoder layers with torch.compile(backend='tpu', dynamic=False)...")
        for i, layer in enumerate(model.encoder.layers):
            model.encoder.layers[i] = torch.compile(layer, backend="tpu", dynamic=False)
    elif args.compile_mode == "full":
        if args.use_remat and args.batch_size >= 4:
            if rank == 0:
                print("Compiling Encoder, Decoder, and per-sample Joint+Loss with torch.compile(backend='tpu', dynamic=False) for micro-batched remat...")
            model.encoder = torch.compile(model.encoder, backend="tpu", dynamic=False)
            model.decoder = torch.compile(model.decoder, backend="tpu", dynamic=False)
            compiled_joint_loss = torch.compile(model._single_joint_and_loss, backend="tpu", dynamic=False)
        else:
            if rank == 0:
                print("Compiling FULL ConformerTransducer1B model (Encoder+Decoder+Joint+Loss) with torch.compile(backend='tpu', dynamic=False)...")
            model = torch.compile(model, backend="tpu", dynamic=False)

    features = torch.randn(args.batch_size, 80, time_steps, device=device, dtype=torch.bfloat16)
    feature_lengths = torch.full((args.batch_size,), time_steps, device=device, dtype=torch.long)
    targets = torch.randint(0, 4096, (args.batch_size, args.target_len), device=device, dtype=torch.long)
    target_lengths = torch.full((args.batch_size,), args.target_len, device=device, dtype=torch.long)

    total_time = 0.0
    measured_steps = 0

    for step in range(args.steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()

        if args.use_remat and args.batch_size >= 4:
            with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                encoded, encoded_len = model.encoder(audio_signal=features, length=feature_lengths)
                decoder_out, _, _ = model.decoder(targets=targets, target_length=target_lengths)

            enc_detached = encoded.detach().requires_grad_(True)
            dec_detached = decoder_out.detach().requires_grad_(True)
            joint_fn = compiled_joint_loss if compiled_joint_loss is not None else model._single_joint_and_loss

            total_loss_val = 0.0
            for i in range(args.batch_size):
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    loss_i = joint_fn(
                        enc_detached[i : i + 1].clone(),
                        dec_detached[i : i + 1].clone(),
                        targets[i : i + 1].clone(),
                    ) / args.batch_size
                loss_i.backward()
                total_loss_val += loss_i.detach().cpu().item()

            torch.autograd.backward([encoded, decoder_out], [enc_detached.grad, dec_detached.grad])
            optimizer.step()
            loss_val = total_loss_val
        else:
            with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                loss = model(features, feature_lengths, targets, target_lengths)

            if args.use_graph_split:
                import torch_tpu._internal.sync as tpu_sync
                tpu_sync.synchronize(loss, wait=False)

            loss.backward()
            optimizer.step()
            loss_val = loss.cpu().item()

        step_time = time.perf_counter() - t0

        if step >= args.warmup_steps:
            total_time += step_time
            measured_steps += 1

        if rank == 0:
            tag = "[WARMUP/COMPILE]" if step < args.warmup_steps else "[STEADY]"
            print(f"Step {step:02d} {tag} | Step Time: {step_time:.3f} s | Loss: {loss_val:.4f}")

    if rank == 0 and measured_steps > 0:
        avg_step_time = total_time / measured_steps
        peak_tflops_v6e = 918.0
        device_step_tflops = flop_stats["device_step_tflops"]
        realized_tflops_per_device = device_step_tflops / avg_step_time
        mfu_percent = (realized_tflops_per_device / peak_tflops_v6e) * 100.0
        audio_sec_per_sec_per_device = (args.batch_size * args.audio_duration_sec) / avg_step_time
        audio_sec_per_sec_global = audio_sec_per_sec_per_device * world_size

        print("=" * 80)
        print(f"BENCHMARK SUMMARY (compile_mode={args.compile_mode}, remat={args.use_remat}, PDBS={args.batch_size}, Audio={args.audio_duration_sec}s)")
        print("=" * 80)
        print(f"Steady-State Step Time:          {avg_step_time:.3f} s")
        print(f"Theoretical TFLOPs / Step / TPU: {device_step_tflops:.2f} TFLOPs")
        print(f"Realized TFLOP/s per Device:     {realized_tflops_per_device:.2f} TFLOP/s")
        print(f"Model FLOPs Utilization (MFU):   {mfu_percent:.2f} %")
        print(f"Audio Throughput per TPU Chip:   {audio_sec_per_sec_per_device:.1f} audio sec / sec / TPU")
        print(f"Global Audio Throughput (4 TPUs):{audio_sec_per_sec_global:.1f} audio sec / sec")
        print("=" * 80)

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
