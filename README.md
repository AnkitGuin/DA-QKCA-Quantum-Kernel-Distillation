# DA-QKCA: Quantum-Kernel Distillation Research

Research code for studying quantum-kernel teachers and classical student models, including feature selection, kernel construction, training, and evaluation. The experiments use exact classical statevector simulation; this repository does not claim quantum hardware advantage.

The paper manuscript and peer-review materials are not part of this code release. A paper title and citation will be added after the publication record is settled.

## Install

Use Python 3.11 or newer:

```powershell
python -m pip install -e ".[dev]"
```

## Smoke run

```powershell
$env:PYTHONPATH = 'src'
python scripts/run_smoke.py --config configs/smoke.yaml
```

Run the core unit tests with:

```powershell
python -m pytest
```

## Project structure

- `src/daqkca/` contains model, data, quantum-kernel, training, and evaluation code.
- `configs/smoke.yaml` provides a small correctness run.
- `scripts/` contains experiment, baseline, plotting, gradient-check, and locked-test commands.
- `data/` documents the synthetic data generator and its limitations. No third-party dataset is bundled.

## Research status

This is research software. Results depend on the selected configuration and environment. Read the configuration and data documentation before interpreting an experiment. The included code implements classical simulation and does not execute quantum hardware circuits.

## Citation

Please cite the software release using `CITATION.cff`. The paper citation will be added when the paper is publicly released.
