# Copyright (c) 2026, Google LLC. All rights reserved.
"""JAX RNN-T Loss implementation for NeMo Speech on TPU via torch_tpu."""

import logging
import torch

try:
    import jax
    import jax.numpy as jnp
    from torch_tpu._internal import pallas

    def rnnt_loss_forward_jax(logits: jax.Array, labels: jax.Array) -> jax.Array:
        """Computes RNN-T forward loss along anti-diagonals using jax.lax.scan.

        Args:
            logits: logit tensor, shape (B, T, U, V)
            labels: label tensor, shape (B, U-1), int32

        Returns:
            Scalar loss.
        """
        B, T, U, V = logits.shape
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        blank_idx = V - 1

        blank_lp = log_probs[..., blank_idx]
        target_indices = jnp.expand_dims(labels, axis=(1, 3))
        target_lp = jnp.take_along_axis(log_probs[:, :, :-1, :], target_indices, axis=-1).squeeze(-1)
        target_lp = jnp.pad(target_lp, ((0, 0), (0, 0), (0, 1)), constant_values=-1e30)

        init_alpha = jnp.full((B, U), -1e30, dtype=jnp.float32)
        init_alpha = init_alpha.at[:, 0].set(0.0)

        def scan_step(alpha_prev, d):
            u_idx = jnp.arange(U)
            t_idx = d - u_idx
            valid_mask = (t_idx >= 0) & (t_idx < T)
            safe_t = jnp.clip(t_idx, 0, T - 1)
            t_minus_1 = jnp.clip(t_idx - 1, 0, T - 1)
            lp_blank = blank_lp[:, t_minus_1, u_idx]
            from_blank = alpha_prev + lp_blank

            u_minus_1 = jnp.clip(u_idx - 1, 0, U - 1)
            lp_target = target_lp[:, safe_t, u_minus_1]
            alpha_prev_shifted = jnp.roll(alpha_prev, shift=1, axis=1)
            alpha_prev_shifted = alpha_prev_shifted.at[:, 0].set(-1e30)
            from_target = alpha_prev_shifted + lp_target

            new_alpha = jnp.logaddexp(from_blank, from_target)
            new_alpha = jnp.where((d == 0) & (u_idx == 0), 0.0, new_alpha)
            new_alpha = jnp.where(valid_mask, new_alpha, -1e30)
            return new_alpha, None

        final_alpha, _ = jax.lax.scan(scan_step, init_alpha, jnp.arange(T + U - 1))
        total_log_prob = final_alpha[:, U - 1] + blank_lp[:, T - 1, U - 1]
        return -jnp.mean(total_log_prob)

    def rnnt_loss_backward_jax(
        logits: jax.Array, labels: jax.Array, d_loss: jax.Array
    ) -> jax.Array:
        """Computes RNN-T loss gradient w.r.t logits using JAX autodiff."""

        def surrogate(logits_local):
            return rnnt_loss_forward_jax(logits_local, labels) * d_loss

        grad_logits = jax.grad(surrogate)(logits)
        return grad_logits.astype(logits.dtype)

    rnnt_loss_fwd = pallas.jax_op(
        "nemo_asr::jax_rnnt_loss_fwd",
        rnnt_loss_forward_jax,
    )
    rnnt_loss_bwd = pallas.jax_op(
        "nemo_asr::jax_rnnt_loss_bwd",
        rnnt_loss_backward_jax,
    )

    def _setup_context(ctx, inputs, output):
        del output
        logits, labels = inputs
        ctx.save_for_backward(logits, labels)

    def _backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        grad_logits = rnnt_loss_bwd(logits, labels, grad_output)
        return grad_logits, None

    torch.library.register_autograd(
        "nemo_asr::jax_rnnt_loss_fwd",
        _backward,
        setup_context=_setup_context,
    )

    def tpu_rnnt_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Entry point for TPU RNN-T loss."""
        labels_int32 = labels.to(torch.int32)
        return rnnt_loss_fwd(logits, labels_int32)

except ImportError as e:
    logging.warning(f"Failed to initialize JAX/Pallas TPU RNN-T loss: {e}")

    def tpu_rnnt_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("tpu_rnnt_loss requires jax and torch_tpu._internal.pallas")
