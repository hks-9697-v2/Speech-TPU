# Copyright (c) 2026, Google LLC. All rights reserved.
# Entry point for running unmodified NeMo scripts with out-of-tree TorchTPU patches:
# Usage: torchrun --nproc_per_node=4 -m nemo_torchtpu examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py [args...]

import runpy
import sys
import nemo_torchtpu  # noqa: F401 - triggers apply_nemo_torchtpu_patches()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m nemo_torchtpu <path_to_nemo_script.py> [hydra_args...]")
        sys.exit(1)

    script_path = sys.argv[1]
    # Shift sys.argv so the target script sees its own path as sys.argv[0]
    sys.argv = sys.argv[1:]
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
