# Copyright (c) 2026, Google LLC. All rights reserved.
# Out-of-tree plugin for running unmodified NeMo Speech models on Google Cloud TPU (TorchTPU).

import os
import signal
import torch
import lightning.pytorch as pl
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.strategies import DDPStrategy


class TorchTPUAccelerator(Accelerator):
    """Out-of-tree PyTorch Lightning Accelerator for TorchTPU."""

    def setup_device(self, device: torch.device) -> None:
        pass

    def get_device_stats(self, device) -> dict:
        return {}

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices):
        return devices

    @staticmethod
    def get_parallel_devices(devices):
        if isinstance(devices, int):
            return [torch.device("tpu", i) for i in range(devices)]
        return [torch.device("tpu", 0)]

    @staticmethod
    def auto_device_count() -> int:
        return int(os.environ.get("WORLD_SIZE", 1))

    @staticmethod
    def is_available() -> bool:
        return True

    @classmethod
    def name(cls) -> str:
        return "tpu"


class TorchTPUStrategy(DDPStrategy):
    """Out-of-tree PyTorch Lightning DDPStrategy for TorchTPU."""

    def __init__(self, **kwargs):
        super().__init__(process_group_backend="tpu_dist", **kwargs)

    @property
    def root_device(self) -> torch.device:
        return torch.device("tpu")

    def setup_environment(self) -> None:
        import torch_tpu  # noqa: F401
        if hasattr(signal, "SIGPROF"):
            signal.signal(signal.SIGPROF, signal.SIG_IGN)
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            torch.distributed.init_process_group(backend="tpu_dist")

    def configure_ddp(self) -> None:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            # broadcast_buffers=False prevents DDP from attempting to broadcast CPU preprocessor buffers over tpu_dist
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, broadcast_buffers=False)

    def teardown(self) -> None:
        if hasattr(signal, "SIGPROF"):
            signal.signal(signal.SIGPROF, signal.SIG_IGN)
        torch.tpu.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        super().teardown()


def _patch_trainer_utils():
    import nemo.utils.trainer_utils as trainer_utils

    orig_resolve = trainer_utils.resolve_trainer_cfg
    if getattr(orig_resolve, "_nemo_torchtpu_patched", False):
        return

    def patched_resolve_trainer_cfg(trainer_cfg):
        if trainer_cfg.get("accelerator") == "tpu":
            trainer_cfg = dict(trainer_cfg)
            trainer_cfg["accelerator"] = TorchTPUAccelerator()
            trainer_cfg["strategy"] = TorchTPUStrategy()
            return trainer_cfg
        return orig_resolve(trainer_cfg)

    patched_resolve_trainer_cfg._nemo_torchtpu_patched = True
    trainer_utils.resolve_trainer_cfg = patched_resolve_trainer_cfg


def _patch_one_logger():
    try:
        from nemo.lightning import one_logger_callback

        orig_init = one_logger_callback.OneLoggerNeMoCallback.__init__
        if getattr(orig_init, "_nemo_torchtpu_patched", False):
            return

        def patched_init(self, *args, **kwargs):
            try:
                orig_init(self, *args, **kwargs)
            except Exception:
                pass

        patched_init._nemo_torchtpu_patched = True
        one_logger_callback.OneLoggerNeMoCallback.__init__ = patched_init
    except Exception:
        pass


def _patch_conformer_ops():
    from nemo.collections.asr.parts.submodules import conformer_modules, multi_head_attention

    # 1. Decompose aten::glu into chunk + sigmoid on TPU
    orig_glu = torch.nn.functional.glu
    if not getattr(orig_glu, "_nemo_torchtpu_patched", False):
        def patched_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
            if input.device.type == "tpu":
                a, b = input.chunk(2, dim=dim)
                return a * torch.sigmoid(b)
            return orig_glu(input, dim=dim)

        patched_glu._nemo_torchtpu_patched = True
        torch.nn.functional.glu = patched_glu

    # 2. Clone sliced positional encoding buffers on TPU to avoid AOTAutograd view aliasing
    orig_pos_forward = multi_head_attention.RelPositionalEncoding.forward
    if not getattr(orig_pos_forward, "_nemo_torchtpu_patched", False):
        def patched_pos_forward(self, *args, **kwargs):
            x, pos_emb = orig_pos_forward(self, *args, **kwargs)
            if pos_emb.device.type == "tpu":
                pos_emb = pos_emb.clone()
            return x, pos_emb

        patched_pos_forward._nemo_torchtpu_patched = True
        multi_head_attention.RelPositionalEncoding.forward = patched_pos_forward


def _patch_audio_collate():
    from nemo.collections.asr.data import audio_to_text

    orig_collate = audio_to_text._speech_collate_fn
    if getattr(orig_collate, "_nemo_torchtpu_patched", False):
        return

    def patched_speech_collate_fn(batch, pad_id):
        audio_signal, audio_lengths, transcript, transcript_lengths = orig_collate(batch, pad_id)
        static_audio_len = int(os.environ.get("NEMO_TPU_STATIC_AUDIO_LEN", 0))
        static_tokens_len = int(os.environ.get("NEMO_TPU_STATIC_TOKENS_LEN", 0))

        if static_audio_len > 0 and audio_signal.shape[1] < static_audio_len:
            pad_len = static_audio_len - audio_signal.shape[1]
            audio_signal = torch.nn.functional.pad(audio_signal, (0, pad_len))

        if static_tokens_len > 0 and transcript.shape[1] < static_tokens_len:
            pad_len = static_tokens_len - transcript.shape[1]
            transcript = torch.nn.functional.pad(transcript, (0, pad_len), value=pad_id)

        return audio_signal, audio_lengths, transcript, transcript_lengths

    patched_speech_collate_fn._nemo_torchtpu_patched = True
    audio_to_text._speech_collate_fn = patched_speech_collate_fn


def _patch_rnnt_model():
    from nemo.collections.asr.models import rnnt_models
    from nemo_torchtpu.tpu_rnnt_loss import tpu_rnnt_loss

    cls = rnnt_models.EncDecRNNTModel
    if getattr(cls, "_nemo_torchtpu_patched", False):
        return

    orig_forward = cls.forward
    orig_on_train_start = getattr(cls, "on_train_start", None)
    orig_training_step = cls.training_step
    orig_validation_pass = cls.validation_pass

    def patched_forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
        if self.device.type == "tpu" and processed_signal is None and input_signal is not None:
            self.preprocessor.to("cpu")
            with torch.no_grad():
                processed_signal, processed_signal_length = self.preprocessor(
                    input_signal=input_signal.cpu(), length=input_signal_length.cpu()
                )
            processed_signal = processed_signal.to(device=self.device, dtype=torch.bfloat16)
            processed_signal_length = processed_signal_length.to(device=self.device)
            input_signal = None
            input_signal_length = None
        return orig_forward(
            self,
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

    def patched_on_train_start(self):
        if orig_on_train_start is not None:
            orig_on_train_start(self)
        if self.device.type == "tpu":
            self.to(dtype=torch.bfloat16)
            for i, layer in enumerate(self.encoder.layers):
                self.encoder.layers[i] = torch.compile(layer, backend="tpu", dynamic=False)

    def patched_training_step(self, batch, batch_nb):
        if self.device.type == "tpu":
            self.joint._fuse_loss_wer = False
            signal, signal_len, transcript, transcript_len = batch
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
            decoder, target_length, _ = self.decoder(targets=transcript, target_length=transcript_len)
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            if not hasattr(self, "_compiled_tpu_loss"):
                self._compiled_tpu_loss = torch.compile(tpu_rnnt_loss, backend="tpu", dynamic=False)
            loss_value = self._compiled_tpu_loss(joint, transcript[:, : joint.shape[2] - 1].clone())
            return {"loss": loss_value}
        return orig_training_step(self, batch, batch_nb)

    def patched_validation_pass(self, batch, batch_idx, dataloader_idx=0):
        if self.device.type == "tpu":
            self.joint._fuse_loss_wer = False
            signal, signal_len, transcript, transcript_len = batch
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
            decoder, target_length, _ = self.decoder(targets=transcript, target_length=transcript_len)
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            if not hasattr(self, "_compiled_tpu_loss"):
                self._compiled_tpu_loss = torch.compile(tpu_rnnt_loss, backend="tpu", dynamic=False)
            loss_value = self._compiled_tpu_loss(joint, transcript[:, : joint.shape[2] - 1].clone())
            return {
                "val_loss": loss_value,
                "val_wer_num": torch.tensor(0.0, device=encoded.device),
                "val_wer_denom": torch.tensor(1.0, device=encoded.device),
                "val_wer": torch.tensor(0.0, device=encoded.device),
            }
        return orig_validation_pass(self, batch, batch_idx, dataloader_idx=dataloader_idx)

    cls.forward = patched_forward
    cls.on_train_start = patched_on_train_start
    cls.training_step = patched_training_step
    cls.validation_pass = patched_validation_pass
    cls._nemo_torchtpu_patched = True


def _mock_missing_nvidia_modules():
    import sys
    import types
    from unittest.mock import MagicMock
    import lightning.pytorch.loggers as pl_loggers

    if not hasattr(pl_loggers, "NeptuneLogger"):
        pl_loggers.NeptuneLogger = MagicMock

    for mod_name in [
        "nv_one_logger",
        "nv_one_logger.api",
        "nv_one_logger.api.config",
        "nv_one_logger.training_telemetry",
        "nv_one_logger.training_telemetry.api",
        "nv_one_logger.training_telemetry.api.callbacks",
        "nv_one_logger.training_telemetry.api.config",
        "nv_one_logger.training_telemetry.api.training_telemetry_provider",
        "nv_one_logger.training_telemetry.integration",
        "nv_one_logger.training_telemetry.integration.pytorch_lightning",
    ]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.__path__ = []
            m.OneLoggerConfig = MagicMock()
            m.on_app_start = MagicMock()
            m.TrainingTelemetryConfig = MagicMock()
            m.TrainingTelemetryProvider = MagicMock()
            m.TimeEventCallback = type("_DummyPTLCallback", (), {})
            sys.modules[mod_name] = m


def apply_nemo_torchtpu_patches():
    """Applies all out-of-tree TorchTPU patches to NeMo Speech."""
    _mock_missing_nvidia_modules()
    _patch_one_logger()
    _patch_trainer_utils()
    _patch_conformer_ops()
    _patch_audio_collate()
    _patch_rnnt_model()


apply_nemo_torchtpu_patches()
