# Validation

Checked on 2026-11-05 in the existing Python 3.10 / PyTorch 2.2.2+cu118
research environment, using a disposable directory and synthetic fixtures.

- 69 unit/regression tests passed; no failures or errors.
- Public factory: 155271 trainable parameters, no router/private experts,
  finite `[batch, 7, 3]` logits, scaler-buffer state-dictionary roundtrip.
- Public and historical model initialisation matched for identical seeds.
- Extracted metric, threshold, scaler and class-weight helpers passed tests.
- Cached-array training adapter passed input/provenance and fixed-recipe tests.
- Historical final-clinical, history-window, training/selection,
  joint-versus-independent and LR-660 tests passed on synthetic fixtures.
- Release source hashes, Python syntax and excluded-asset checks passed.

No study model was retrained, and no real study dataset was rebuilt for this
check. Synthetic benchmark tests do not constitute completed study speed
measurements. The cached-array adapter test mocks the training call; original
training-loop regression tests exercise small synthetic optimisation steps.
