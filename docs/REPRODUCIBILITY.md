# Execution and Provenance

## Available Without Study Data

The public model factory, architecture inspection and synthetic tests execute
without access to the study dataset. Metric, loss and calibration helpers are
also available for independently prepared compatible input.

## Historical Experiment Routines

`tools/train_from_arrays.py` provides a separate cached-input interface for
one seed/fold at the fixed final recipe. It uses the original training loop,
fits scalers on the fold's training subset, and never accepts a test input.
Prepared arrays and correct patient-separated fold assignment remain the
caller's responsibility. This new adapter has not regenerated paper results.

The scripts retain their historical folder names so imports remain traceable.
`scripts/run_lightgbm660.py` is reduced to shared array I/O and metric helpers:
it contains no LightGBM model fitting. The long CA84 training module contains
only the original scaler and class-weight functions. The CA75 modules contain
only threshold and metric functions. No dataset-building implementation is
included in these compatibility modules.

Historical training/selection routines require prepared input files and
frozen experiment records under the expected `reports/` directories. The
epoch-15 selection runner also expects the earlier five-seed validation
predictions and provenance. The final clinical evaluation requires final
predictions, Clinical LR predictions, current-normal state and patient mapping.
Those assets are intentionally excluded. A fresh clone alone cannot recreate
the study tables. The JSON configs document the study recipe; existing
historical CLIs do not automatically load them.

Extracted modules have different source hashes from their original full
modules. Their original function bodies are retained, but historic protocol
hashes will not match the reduced files. Original freeze/resume checks must
remain intact; these scripts document and expose the procedures rather than
silently impersonating the original frozen working directory.

`SOURCE_MANIFEST.json` records original and release hashes plus each extraction.
`.gitattributes` disables automatic line-ending conversion to preserve these
hashes across Windows and Linux checkouts.
No patient outputs or historical completion files are shipped as if regenerated
by the reduced code. Runtime checks use synthetic fixtures only.

## Final Versus Supporting Protocols

The final main and history-window evaluations use non-EMA epoch 15 and fixed
seeds 1-5. Seed-level metrics follow four-fold probability averaging. Earlier
structural/joint analyses use epoch 5 after six training epochs. Their outputs
must be reported separately. Current-normal filtering changes evaluation
eligibility, not the trained model. Calibration intercepts are fitted on
validation data and remain fixed during test bootstrap.
