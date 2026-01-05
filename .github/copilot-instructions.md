# Copilot instructions for contributors and AI agents

Goal: help an AI coding agent become productive quickly in this repository.

- **Big picture**: this repo implements and benchmarks ECG baseline-wander removal.
  - Signal processing core: [models/model_proposed/v37_standalone.py](models/model_proposed/v37_standalone.py#L1)
  - Comparative ML methods: DAE in [models/model_DAE](models/model_DAE) and UNet in [models/model_UNet](models/model_UNet)
  - I/O + utilities: [common/io_wfdb.py](common/io_wfdb.py#L1), [common/utils.py](common/utils.py#L1), [common/config.py](common/config.py#L1)
  - Experiment orchestration: scripts under [scripts/](scripts) (train_DAE.py, run_benchmark.py, run_mitdb_nstdb_experiment.py)

- **Why things are organized this way**: common/ centralizes dataset, sampling and metric logic so benchmarks across different model implementations stay comparable. models/* contain interchangeable denoising backends invoked by the runner in scripts/run_benchmark.py.

- **Key conventions and patterns to follow**:
  - Data splits are record-wise (not window-wise). See [common/dataset_split.py](common/dataset_split.py#L1) and `common/splits.json` — NEVER mix windows from the same record across train/val/test.
  - Standard project sample-rate is 250 Hz. Use `_resample_to_target` in [common/utils.py](common/utils.py#L1) when converting signals.
  - DAE training/inference uses per-window min-max normalization (window len 101, radius 50). See [scripts/train_DAE.py](scripts/train_DAE.py#L1) and [scripts/run_benchmark.py](scripts/run_benchmark.py#L1) for normalization + overlap-add inference.
  - UNet inference uses windowing (default win=512, hop=512) and supports `minmax_by_noisy` or `none`. See UNet runner in [scripts/run_benchmark.py](scripts/run_benchmark.py#L1).
  - Checkpoint loading is defensive: code accepts either raw state_dict or dict wrappers and strips `module.` prefixes. See `_load_dae_model` / `_load_unet_model` in [scripts/run_benchmark.py](scripts/run_benchmark.py#L1).

- **Developer workflows / common commands** (run from repository root):

  Install deps (project has a requirements file):

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r setup/requirements.txt
  ```

  Quick reproduction examples:

  - Run the proposed method experiment (MITDB + NSTDB):
    ```bash
    python scripts/run_mitdb_nstdb_experiment.py
    ```

  - Run unified benchmark (proposed / dae / unet):
    ```bash
    python scripts/run_benchmark.py --methods proposed,dae,unet --preset paper
    ```

  - Train the DAE baseline used for comparison:
    ```bash
    python scripts/train_DAE.py --epochs_pre 10 --epochs_fine 20
    ```

- **Project-specific gotchas** (things an agent should not change lightly):
  - `common/config.py` contains the single source of truth for default paths and parameters (FS_DEFAULT, DURATION_SEC_DEFAULT, etc.). If changing defaults, update dependent scripts and document the change.
  - Many scripts expect PhysioNet WFDB files; paths in config are absolute defaults. Prefer passing paths through CLI flags (e.g., `--split_path`, `--out_dir`) rather than changing code.
  - Splits are saved to `common/splits.json` and used by training/eval scripts—regenerate with `common/dataset_split.create_splits_json()` if needed.
  - `models/model_proposed/v37_standalone.py` is highly algorithmic and numerically sensitive; favor small, testable refactors and include visual checks (the file includes a demo block under `if __name__ == '__main__':`).

- **Good first edits for an AI agent**:
  - Add unit tests around small, pure functions (e.g., `fast_percentile_filter`, `load_mitdb_wfdb`) to guard refactors.
  - Add a small README fragment referencing dataset download instructions (PhysioNet) and how to place files under `data/`.
  - Make path handling robust: prefer CLI-overrides and make `common/config.py` defaults relative or clearly documented.

If anything here is unclear or you want more coverage of a particular area (e.g., model internals, training hyperparams, or deployment guidance), tell me which section to expand.
