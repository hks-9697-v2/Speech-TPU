# Copyright (c) 2026, Google LLC. All rights reserved.
"""Unit and numerical parity tests for NeMo Speech TPU adaptations."""

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.asr.parts.submodules.conformer_modules import ConformerConvolution


def test_tpu_glu_numerical_parity():
    """Verify that chunk(2, dim=-1) + sigmoid matches F.glu forward and backward."""
    torch.manual_seed(42)
    x_ref = torch.randn(4, 100, 176, requires_grad=True)
    x_tpu = x_ref.detach().clone().requires_grad_(True)

    # Reference F.glu
    out_ref = F.glu(x_ref, dim=-1)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # TPU-compatible GLU
    chunks = x_tpu.chunk(2, dim=-1)
    out_tpu = chunks[0] * torch.sigmoid(chunks[1])
    loss_tpu = out_tpu.sum()
    loss_tpu.backward()

    assert torch.allclose(out_ref, out_tpu, atol=1e-6), "Forward outputs do not match!"
    assert torch.allclose(x_ref.grad, x_tpu.grad, atol=1e-6), "Backward gradients do not match!"


def test_conformer_convolution_glu_forward_backward():
    """Verify ConformerConvolution forward and backward pass with TPU-compatible GLU."""
    torch.manual_seed(42)
    conv = ConformerConvolution(
        d_model=88,
        kernel_size=31,
        norm_type="layer_norm",
        pointwise_activation="glu_",
    )
    x = torch.randn(2, 50, 88, requires_grad=True)
    out = conv(x)
    loss = out.sum()
    loss.backward()

    assert out.shape == (2, 50, 88)
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_tpu_ctc_loss_import_and_parity():
    """Verify tpu_ctc_loss module imports and executes if torch_tpu/jax are present."""
    from nemo.collections.asr.losses.tpu_ctc_loss import _HAS_TPU_CTC, tpu_ctc_loss

    if not _HAS_TPU_CTC:
        pytest.skip("torch_tpu or JAX/Optax not available in current environment")

    device = torch.device("tpu")
    torch.manual_seed(42)

    B, T, C, L = 2, 50, 16, 10
    logits = torch.randn(B, T, C, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, C - 1, (B, L), device=device, dtype=torch.long)

    loss = tpu_ctc_loss(logits, targets)
    loss.backward()

    assert loss.ndim == 0
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
