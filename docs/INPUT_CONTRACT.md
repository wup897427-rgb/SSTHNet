# Prepared Input Contract

The repository consumes existing numerical arrays. Dataset acquisition,
cleaning, causal resampling, feature construction, labels and patient/fold
assignment are outside this selected release.

## Model Input

Model tensors have shape `[batch, 3, 22, 10]`; the history axis is ordered
120, 300, 600 seconds. The ten summaries, in order, are current valid value,
mean, standard deviation, minimum, maximum, linear slope, valid fraction,
direct-observation fraction, quality-warning fraction, and log1p observation
age. Source ordering is listed in `configs/source_order.json`. A matching scaler
is stored in each trained model's state dictionary.

## Existing Training Files

The historical scripts read features `[N, 3, 220]` and grouped arrays:
`representative_index [G]`, `group_fold [G]`, `labels [G, 7, 3]`, and
`masks [G, 7, 3]`. Development folds are numbered 0-3. Eligible labels must
be binary. Masks determine eligibility; missing labels are not negatives.

Outer reference arrays contain `global_position`, `case_id`, `anchor_seconds`,
`horizons`, `labels [N, 7]`, and `masks [N, 7]`; patient identity must also
be available for patient-clustered uncertainty. Case identity is not a
substitute for patient identity. These arrays and identifiers are not bundled.

The target order is hypotension, low depth of anaesthesia, heart rate out of
range, hypoxaemia, monitor-recorded low temperature, hypocapnia, and sustained
elevated peak airway pressure. The horizon order is 5, 10, 15 minutes.

Current-normal evaluation additionally requires the independently established
observable/normal state at the anchor. An imputed current value must not be
used to infer observability.
