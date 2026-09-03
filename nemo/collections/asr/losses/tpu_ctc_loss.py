# Copyright (c) 2026, Google LLC. All rights reserved.
"""JAX CTC Loss implementation for NeMo Speech on TPU via torch_tpu."""

import logging
import torch

try:
    import jax
    import jax.numpy as jnp
    import optax
    from torch_tpu._internal import pallas

    def ctc_loss_forward_jax(logits: jax.Array, labels: jax.Array) -> jax.Array:
        """Computes CTC loss using optax.

        Args:
            logits: logit tensor, shape (B, T, C)
            labels: label tensor, shape (B, N)

        Returns:
            Scalar loss.
        """
        vocab_size = logits.shape[-1]

        # Ignore index for padding (matches PyTorch's default -100 or 0 depending on mask)
        ignore_mask = labels < 0

        safe_labels = jnp.where(ignore_mask, 0, labels)
        safe_labels = (safe_labels % vocab_size).astype(jnp.int32)
        label_paddings = ignore_mask.astype(jnp.float32)

        logit_paddings = jnp.zeros(logits.shape[:-1], dtype=jnp.float32)

        loss_per_seq = optax.losses.ctc_loss(
            logits, logit_paddings, safe_labels, label_paddings, blank_id=vocab_size - 1
        )
        return jnp.mean(loss_per_seq)

    def ctc_loss_backward_jax(
        logits: jax.Array, labels: jax.Array, d_loss: jax.Array
    ) -> jax.Array:
        """Computes CTC loss gradient w.r.t logits using JAX autodiff."""

        def surrogate(logits_local):
            return ctc_loss_forward_jax(logits_local, labels) * d_loss

        grad_logits = jax.grad(surrogate)(logits)
        return grad_logits

    ctc_loss_fwd = pallas.jax_op(
        "nemo_asr::jax_ctc_loss_fwd",
        ctc_loss_forward_jax,
    )
    ctc_loss_bwd = pallas.jax_op(
        "nemo_asr::jax_ctc_loss_bwd",
        ctc_loss_backward_jax,
    )

    def _setup_context(ctx, inputs, output):
        del output
        logits, labels = inputs
        ctx.save_for_backward(logits, labels)

    def _backward(ctx, d_loss):
        logits, labels = ctx.saved_tensors
        grad_logits = ctc_loss_bwd(logits, labels, d_loss)
        return grad_logits, None

    ctc_loss_fwd.register_autograd(_backward, setup_context=_setup_context)
    _HAS_TPU_CTC = True

except Exception as e:
    logging.warning("JAX CTC loss custom op not registered (expected if not on TPU): %s", e)
    ctc_loss_fwd = None
    ctc_loss_bwd = None
    _HAS_TPU_CTC = False


def tpu_ctc_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """PyTorch entry point for JAX CTC loss on TPU.

    Args:
        logits: logit tensor, shape (B, T, C)
        targets: label tensor, shape (B, N)

    Returns:
        Scalar loss.
    """
    if not _HAS_TPU_CTC or ctc_loss_fwd is None:
        raise RuntimeError("JAX CTC loss custom op was not registered or torch_tpu is unavailable.")
    # Cast targets to int32 to match JAX TPU default integer type (i32)
    return ctc_loss_fwd(logits, targets.to(torch.int32))
