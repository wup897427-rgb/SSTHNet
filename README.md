# SSTHNet

Core research code for multi-target intraoperative physiological prediction
using multiscale routine monitoring inputs.

SSTHNet uses 22 monitoring sources over 120-, 300-, and 600-second histories.
The input consists of 660 numerical summaries, represented as 66 tokens.
One shared-expert model produces logits for seven targets at 5, 10, and
15 minutes, giving 21 target-horizon outputs. The final model contains no
learned router or private experts.

## Release Scope

This is a selected core-code release, not the complete experimental archive.
It includes the model, cached-input training routines, single-history and
structural analyses, ranking metrics, patient-clustered bootstrap,
validation-intercept recalibration, synthetic tests, and an LR-660 comparator.

Dataset download, cleaning, feature extraction, label construction and
patient-split generation code are excluded. Patient data, individual
predictions, training caches, logs and trained weights are also excluded.
Readers must independently obtain appropriately prepared inputs. This
repository does not claim end-to-end reconstruction of the study dataset.

## Installation

The original training environment used Python 3.10 and PyTorch 2.2.2 with
CUDA 11.8. For that CUDA version:

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

For CPU-only use, install the corresponding official CPU PyTorch wheel.
No tree-model library is required by the released SSTHNet code.

## Model Interface

Run from the repository root:

```python
import torch
from ssthnet import make_model

model = make_model().eval()
with torch.inference_mode():
    logits, auxiliary = model(torch.zeros(2, 3, 22, 10))
    probabilities = logits.sigmoid()  # [batch, 7 targets, 3 horizons]
```

This example uses random parameters and synthetic input. For study inference,
load the matching checkpoint including its fold-specific scaler buffers.
The ten summaries and their ordering must match the original input contract.
See [the input contract](docs/INPUT_CONTRACT.md).

## Research Entry Points

For one-fold training on already prepared arrays, without historical selection
records, use the new core entry point:

```bash
python tools/train_from_arrays.py --development-features /path/development.npy --development-groups /path/groups.npz --validation-features /path/validation.npy --validation-reference /path/validation_reference.npz --seed 1 --fold 0 --device cpu --out /path/new_training_output
```

It reuses the released original training loop at the fixed final recipe
(15 epochs), with input/source hashes and a separate provenance record.
Use the same output directory for resuming the same seed/fold; use a distinct
output directory for a different seed/fold. It does not run model selection,
test evaluation or dataset construction. This adapter has synthetic tests;
it has not been used to regenerate the study's final weights.

| File | Purpose |
| --- | --- |
| `scripts/run_ssthnet_training_optimization.py` | Historical validation-only training/selection procedure |
| `scripts/run_ssthnet_history_window_ablations.py` | Matched single-history training |
| `scripts/evaluate_ssthnet_history_windows.py` | Final and single-history evaluation |
| `scripts/run_ssthnet_ablations.py` | Earlier epoch-5 structural analyses |
| `scripts/run_ssthnet_joint_vs_independent.py` | Earlier joint-versus-independent-target analysis |
| `scripts/evaluate_ssthnet_final_clinical.py` | Overall/current-normal comparison and calibration |
| `scripts/bootstrap_ssthnet_five_seed_ablations.py` | Paired patient-clustered bootstrap utilities |
| `baselines/lr660/lr660.py` | LR-660 comparator |

The historical entry points in the table retain frozen-input and provenance checks.
They require the separately prepared arrays and experiment records described
in [the execution notes](docs/REPRODUCIBILITY.md); they are not fresh-clone
one-command dataset-to-paper workflows. Do not disable their hash checks to
make unrelated input files pass. The model and synthetic tests run without
the study data.

## Model Versions

The final analysis uses fixed seeds 1-5, folds 0-3, constant weighted training
(`A_constant_weighted`), non-EMA epoch 15. Four-fold probabilities are averaged
within each seed; metrics are then averaged across seeds. This is not a
five-seed probability ensemble or selection of the best seeds.

Earlier structural and joint-versus-independent analyses use epoch 5 after
six training epochs. Their results belong to a separate supporting protocol.
See `configs/final_model.json` and `configs/supporting_epoch5.json`.

## Checks

```bash
python tools/check_release.py
python -m unittest discover -s tests -v
python -m unittest discover -s scripts -p test_evaluate_ssthnet_final_clinical.py -v
```

`SOURCE_MANIFEST.json` records the selected source files, hashes and extracted
helper functions. Function extraction preserves the original bodies.

## Availability and Terms

Source data are available from [VitalDB](https://vitaldb.net/) under its
applicable access and use terms. No patient-level data are redistributed here.
Trained weights and the complete experiment archive are not distributed in
this release. No additional availability-by-request commitment is made here.

The authors have not yet selected a software licence. Public visibility alone
does not grant an open-source licence. See [release decisions](docs/RELEASE_DECISIONS.md).

Research software; not a clinically validated deployment system.
