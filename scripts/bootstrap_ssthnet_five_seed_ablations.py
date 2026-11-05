#!/usr/bin/env python3
"""Patient-paired bootstrap of existing five-seed SSTHNet ablation predictions.

No training, threshold refitting, model selection, or prediction regeneration.
Scores are sorted once; exact weighted AP/AUROC/F1 are computed at score ties.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import numpy as np

VARIANTS = ('ssthnet', 'no_source_interaction', 'no_scale_interaction', 'independent_queries')
METRICS = ('Macro_AP', 'Macro_AUROC', 'Macro_F1')
TASKS = ('hypotension', 'low_depth_of_anesthesia', 'heart_rate_out_of_range', 'hypoxemia',
         'monitor_recorded_low_temperature', 'hypocapnia', 'sustained_elevated_peak_airway_pressure')
SEEDS = (1, 2, 3, 4, 5)
IDENTITY = ('global_position', 'case_id', 'anchor_seconds', 'horizons', 'labels', 'masks')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        aliases = {'horizons': ('horizons', 'horizon_minutes'), 'labels': ('labels', 'y'),
                   'masks': ('masks', 'm'), 'probability': ('probability', 'probabilities')}
        result = {}
        for key in (*IDENTITY, 'probability', 'subject_id'):
            name = next((k for k in aliases.get(key, (key,)) if k in z.files), None)
            if name:
                result[key] = np.asarray(z[name])
        return result


def check_identity(reference, candidate, name):
    for key in IDENTITY:
        require(key in candidate and key in reference, name + ': missing identity ' + key)
        if key == 'labels':
            valid = reference['masks'] > 0
            equal = np.array_equal(reference[key][valid], candidate[key][valid])
        else:
            equal = np.array_equal(reference[key], candidate[key])
        require(equal, name + ': mismatched ' + key)


def map_patients(reference, mapping):
    require('subject_id' in mapping, 'Patient mapping lacks subject_id; do not substitute case_id')
    positions = mapping['global_position']
    require(len(np.unique(positions)) == len(positions), 'Mapping global positions are not unique')
    order = np.argsort(positions)
    slots = np.searchsorted(positions[order], reference['global_position'])
    require(bool((slots < len(order)).all()), 'Prediction rows missing from patient mapping')
    indices = order[slots]
    require(np.array_equal(positions[indices], reference['global_position']), 'Prediction rows missing from patient mapping')
    aligned = {k: np.asarray(v)[indices] for k, v in mapping.items() if k in (*IDENTITY, 'subject_id')}
    check_identity(reference, aligned, 'patient mapping')
    subject = aligned['subject_id'].astype(str)
    require(not any(s.strip().lower() in ('', 'nan', 'none', 'null', '-1') for s in subject), 'Missing patient identifiers')
    names, inverse = np.unique(subject, return_inverse=True)
    cases = reference['case_id']
    sorted_rows = np.argsort(cases, kind='stable')
    require(not np.any((cases[sorted_rows][1:] == cases[sorted_rows][:-1]) &
                       (inverse[sorted_rows][1:] != inverse[sorted_rows][:-1])), 'One case maps to multiple patients')
    return inverse.astype(np.int32), len(names)


def prepare_cell(labels, score, patients, threshold):
    y = np.asarray(labels, dtype=np.uint8)
    p = np.asarray(score, dtype=np.float64)
    require(len(y) > 0 and set(np.unique(y)) == {0, 1}, 'Original cell must contain both classes')
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid eligible probability')
    require(math.isfinite(threshold) and 0 <= threshold <= 1, 'Invalid frozen threshold')
    order = np.argsort(-p, kind='stable')
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    return (np.asarray(patients, dtype=np.int32)[order], y[order], starts.astype(np.int32),
            int(np.count_nonzero(sorted_p >= threshold)))


def weighted_metrics(cell, patient_counts):
    subjects, y, starts, cutoff = cell
    w = np.asarray(patient_counts[subjects], dtype=np.float64)
    group_positive = np.add.reduceat(w * y, starts)
    group_negative = np.add.reduceat(w * (1 - y), starts)
    positive = float(group_positive.sum())
    negative = float(group_negative.sum())
    if positive <= 0 or negative <= 0:
        return np.full(3, np.nan)
    tp = np.cumsum(group_positive)
    total = tp + np.cumsum(group_negative)
    precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
    ap = float(np.sum(precision * group_positive) / positive)
    auc = float(np.sum(group_negative * (tp - .5 * group_positive)) / (positive * negative))
    true_positive = float(np.sum(w[:cutoff] * y[:cutoff]))
    predicted_positive = float(w[:cutoff].sum())
    f1 = 2 * true_positive / (positive + predicted_positive)
    return np.asarray([ap, auc, f1])


def draw(base_seed, replicate, patients):
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, replicate, 1]))
    counts = np.bincount(rng.integers(0, patients, size=patients), minlength=patients)
    rng_seed = np.random.default_rng(np.random.SeedSequence([base_seed, replicate, 2]))
    seed_slots = rng_seed.integers(0, 5, size=5)
    return counts, seed_slots


def evaluate_replicate(cache, support, patients, base_seed, replicate):
    counts, slots = draw(base_seed, replicate, patients)
    # Preserve the SAME 21-cell estimand for every configuration. A draw that
    # loses either class in any cell is recorded as unestimable, never re-drawn.
    valid = all(np.sum(pos * counts) > 0 and np.sum(neg * counts) > 0 for pos, neg in support)
    metrics = np.full((5, 4, 3), np.nan)
    if valid:
        for seed in range(5):
            for model in range(4):
                metrics[seed, model] = np.mean([weighted_metrics(c, counts) for c in cache[seed][model]], axis=0)
        require(np.isfinite(metrics).all(), 'Supported bootstrap draw produced nonfinite metrics')
    return metrics, slots, valid


def load_inputs(root, mapping_path):
    helper_path = root / 'scripts/summarize_ssthnet_five_seed_ablations.py'
    spec = importlib.util.spec_from_file_location('bootstrap_five_seed_summary', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original = root / 'reports/ssthnet_single_expert_matched_ablations_v1'
    extra = root / 'reports/ssthnet_single_expert_extra_seeds_v1'
    runs, _, hashes, contract = helper.load_runs([original, extra])
    hashes['summary_script'] = sha256(helper_path)
    hashes['patient_mapping'] = sha256(mapping_path)
    mapping = read_npz(mapping_path)
    reference = read_npz(original / 'ssthnet_seed02/TEST_PREDICTIONS.npz')
    require(set(IDENTITY) <= set(reference), 'Reference metadata incomplete')
    require(len(np.unique(reference['global_position'])) == len(reference['global_position']), 'Reference positions are not unique')
    require(np.isin(reference['masks'], (0, 1)).all(), 'Nonbinary masks')
    require(np.isin(reference['horizons'], (5, 10, 15)).all(), 'Unexpected horizon')
    valid = reference['masks'] > 0
    require(np.isin(reference['labels'][valid], (0, 1)).all(), 'Nonbinary eligible labels')
    inverse, patients = map_patients(reference, mapping)
    del mapping
    selected = [(valid[:, t] & (reference['horizons'] == h)) for t in range(7) for h in (5, 10, 15)]
    support, support_rows = [], []
    for cell_slot, sel in enumerate(selected):
        t, hslot = divmod(cell_slot, 3)
        y = reference['labels'][sel, t]
        pos = np.bincount(inverse[sel][y == 1], minlength=patients)
        neg = np.bincount(inverse[sel][y == 0], minlength=patients)
        support.append((pos, neg))
        support_rows.append(dict(task_id=TASKS[t], horizon_minutes=(5, 10, 15)[hslot],
                                 eligible_rows=int(sel.sum()), positive_rows=int(y.sum()),
                                 positive_patients=int(np.count_nonzero(pos)), negative_patients=int(np.count_nonzero(neg))))
    cache = [[None for _ in VARIANTS] for _ in SEEDS]
    point = np.zeros((5, 4, 3))
    for s, seed in enumerate(SEEDS):
        folder = original if seed == 2 else extra
        validation_lock = json.loads((folder / 'VALIDATION_LOCK.json').read_text())
        hashes[str(folder / 'VALIDATION_LOCK.json')] = sha256(folder / 'VALIDATION_LOCK.json')
        for m, variant in enumerate(VARIANTS):
            name = f'{variant}_seed{seed:02d}'
            prediction_path = folder / name / 'TEST_PREDICTIONS.npz'
            threshold_path = folder / name / 'VALIDATION_THRESHOLDS.csv'
            print('PREPARE', name, flush=True)
            hashes[str(prediction_path)] = sha256(prediction_path)
            hashes[str(threshold_path)] = sha256(threshold_path)
            require(hashes[str(threshold_path)] == validation_lock['thresholds_sha256'][name], 'Frozen validation thresholds changed: ' + name)
            prediction = read_npz(prediction_path)
            check_identity(reference, prediction, name)
            require(prediction['probability'].shape == reference['labels'].shape, 'Probability shape mismatch')
            threshold_rows = read_csv(threshold_path)
            thresholds = {(r['task_id'], int(r['horizon_minutes'])): float(r['threshold']) for r in threshold_rows}
            require(len(threshold_rows) == len(thresholds) == 21, 'Incomplete or duplicate frozen thresholds')
            cells = []
            for c, sel in enumerate(selected):
                t, hslot = divmod(c, 3)
                cells.append(prepare_cell(reference['labels'][sel, t], prediction['probability'][sel, t], inverse[sel],
                                          thresholds[(TASKS[t], (5, 10, 15)[hslot])]))
            cache[s][m] = cells
            point[s, m] = np.mean([weighted_metrics(c, np.ones(patients)) for c in cells], axis=0)
            expected = np.array([float(runs[('test', variant, seed)]['equal_cell_macro_' + metric])
                                 for metric in ('auprc', 'auroc', 'f1')])
            require(np.allclose(point[s, m], expected, rtol=0, atol=1e-10),
                    'Recomputed point metrics differ from frozen report: ' + name + ' / ' + str(point[s, m] - expected))
            del prediction
    print('PREFLIGHT_COMPLETE', 'patients=' + str(patients), 'rows=' + str(len(inverse)), 'models=20 cells_per_model=21', flush=True)
    return dict(cache=cache, support=support, support_rows=support_rows, patients=patients,
                rows=len(inverse), point=point, hashes=hashes, training_contract=contract)


def summarize(point, fixed, crossed, valid):
    require(int(valid.sum()) >= 100, 'Fewer than 100 estimable draws; do not issue intervals')
    point_mean = point.mean(axis=0)
    rows = []
    for design, bootstrap in (('patient_clustered_fixed_five_seeds', fixed), ('crossed_seed_by_patient', crossed)):
        values = bootstrap[valid]
        for m, variant in enumerate(VARIANTS[1:], start=1):
            for metric, name in enumerate(METRICS):
                difference = values[:, 0, metric] - values[:, m, metric]
                low, high = np.quantile(difference, [.025, .975])
                # Three exploratory AP component comparisons form ONE family.
                # Bonferroni 98.333...% per-comparison percentile intervals.
                adjusted_low, adjusted_high = np.quantile(difference, [.05 / 6, 1 - .05 / 6]) if metric == 0 else (None, None)
                enough_support = float(valid.mean()) >= .9
                assessment = 'not_assessed_secondary_metric'
                if metric == 0:
                    assessment = ('support_limited_no_formal_claim' if not enough_support else
                                  'supports_positive_contribution_on_this_test' if adjusted_low > 0 else
                                  'favours_ablation_on_this_test' if adjusted_high < 0 else 'inconclusive')
                rows.append(dict(design=design, comparison='ssthnet_minus_' + variant, metric=name,
                                 ssthnet_point=float(point_mean[0, metric]), ablation_point=float(point_mean[m, metric]),
                                 delta=float(point_mean[0, metric] - point_mean[m, metric]),
                                 delta_pp=100 * float(point_mean[0, metric] - point_mean[m, metric]),
                                 ci95_lower_pp=100 * float(low), ci95_upper_pp=100 * float(high),
                                 ap_bonferroni_ci_lower_pp=None if adjusted_low is None else 100 * float(adjusted_low),
                                 ap_bonferroni_ci_upper_pp=None if adjusted_high is None else 100 * float(adjusted_high),
                                 attempted_replicates=len(valid), estimable_replicates=int(valid.sum()),
                                 estimable_fraction=float(valid.mean()), assessment=assessment))
    return rows


def verify_metric_engine():
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    rng = np.random.default_rng(187)
    y = np.r_[0, 1, rng.integers(0, 2, size=78)]
    p = rng.choice([.05, .2, .5, .8, .95], size=len(y))
    patients = np.arange(len(y), dtype=np.int32) % 9
    cell = prepare_cell(y, p, patients, .5)
    for _ in range(20):
        counts = rng.integers(0, 4, size=9)
        weights = counts[patients]
        if weights[y == 1].sum() == 0 or weights[y == 0].sum() == 0:
            continue
        expected = [average_precision_score(y, p, sample_weight=weights),
                    roc_auc_score(y, p, sample_weight=weights),
                    f1_score(y, p >= .5, sample_weight=weights, zero_division=0)]
        require(np.allclose(weighted_metrics(cell, counts), expected, rtol=0, atol=1e-12), 'Metric engine differs from sklearn')
    print('METRIC_ENGINE_SKLEARN_CHECK_PASS', flush=True)


def run(args):
    root = args.project_root.resolve()
    mapping = (args.patient_mapping or root / 'reports/phase6bja2t_v7_simple_clinical_comparators_one_time_test_v1/BJA2T_TEST_PREDICTIONS.npz').resolve()
    output = (args.out or root / 'reports/ssthnet_five_seed_paired_bootstrap_v1').resolve()
    require(args.replicates >= 100 and args.chunk_size > 0 and args.workers > 0, 'Invalid bootstrap settings')
    verify_metric_engine()
    data = load_inputs(root, mapping)
    protected = [root / 'reports/ssthnet_single_expert_matched_ablations_v1',
                 root / 'reports/ssthnet_single_expert_extra_seeds_v1',
                 root / 'reports/ssthnet_single_expert_5seed_summary_v1']
    require(all(output != p and p not in output.parents and output not in p.parents for p in protected), 'Output overlaps original results')
    require(output != mapping and output not in mapping.parents, 'Output overlaps patient mapping')
    contract = dict(version='ssthnet_five_seed_paired_bootstrap_v1', script_sha256=sha256(Path(__file__)),
                    input_sha256=data['hashes'], patients=data['patients'], rows=data['rows'],
                    seeds=list(SEEDS), variants=list(VARIANTS), metrics=list(METRICS),
                    population='previously observed aligned internal test; posthoc analysis',
                    estimand='mean of per-seed equal-21-cell metrics; never average seed probabilities',
                    replicates=args.replicates, base_seed=args.base_seed, chunk_size=args.chunk_size,
                    resampling='same patient counts and same seed resamples paired across all configurations',
                    missing_cell_policy='record unsupported draw, no redraw, exclude whole draw from all configurations/metrics',
                    ap_multiple_comparisons='three comparisons; Bonferroni 98.333333 percent per-comparison percentile intervals',
                    principal_uncertainty='crossed_seed_by_patient; fixed_five_seeds as sensitivity',
                    threshold_refitting=False, training=False, model_selection=False,
                    numpy_version=np.__version__)
    fingerprint = canonical_hash(contract)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'RUNNING.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        protocol = output / 'protocol.json'
        if protocol.exists():
            require(json.loads(protocol.read_text()) == contract, 'Bootstrap inputs/configuration changed; preserve existing output')
        else:
            save_json(protocol, contract)
        if (output / 'COMPLETE.json').exists():
            require(json.loads((output / 'COMPLETE.json').read_text())['input_fingerprint'] == fingerprint, 'Completion identity mismatch')
            print('BOOTSTRAP_ALREADY_COMPLETE', output)
            return
        save_csv(output / 'PATIENT_CELL_SUPPORT.csv', data['support_rows'])
        point_rows = [dict(seed_index=s, variant=v, **{metric: float(data['point'][si, mi, k]) for k, metric in enumerate(METRICS)})
                      for si, s in enumerate(SEEDS) for mi, v in enumerate(VARIANTS)]
        save_csv(output / 'VERIFIED_SEED_POINT_ESTIMATES.csv', point_rows)
        chunks = output / 'chunks'
        chunks.mkdir(exist_ok=True)
        all_fixed, all_crossed, all_valid, all_missing = [], [], [], []
        start_time = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(1, args.replicates + 1, args.chunk_size):
                end = min(start + args.chunk_size - 1, args.replicates)
                path = chunks / f'BOOTSTRAP_{start:06d}_{end:06d}.npz'
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        require(str(z['input_fingerprint'].item()) == fingerprint, 'Chunk identity mismatch')
                        require(np.array_equal(z['replicate'], np.arange(start, end + 1)), 'Chunk replicate mismatch')
                        fixed, crossed, valid = z['fixed'].copy(), z['crossed'].copy(), z['valid'].copy()
                        missing = z['missing_cells'].copy()
                        require(fixed.shape == crossed.shape == (end - start + 1, 4, 3), 'Chunk geometry changed')
                        require(valid.shape == (end - start + 1,), 'Chunk validity shape changed')
                        require(np.isfinite(fixed[valid]).all() and np.isfinite(crossed[valid]).all(), 'Invalid saved supported metrics')
                else:
                    def calculate(replicate):
                        metrics, slots, supported = evaluate_replicate(data['cache'], data['support'], data['patients'], args.base_seed, replicate)
                        counts, _ = draw(args.base_seed, replicate, data['patients'])
                        missing = [int(i) for i, (p, n) in enumerate(data['support']) if np.sum(p * counts) == 0 or np.sum(n * counts) == 0]
                        return metrics.mean(axis=0), metrics[slots].mean(axis=0), supported, missing
                    records = list(pool.map(calculate, range(start, end + 1)))
                    fixed = np.stack([r[0] for r in records])
                    crossed = np.stack([r[1] for r in records])
                    valid = np.asarray([r[2] for r in records], dtype=bool)
                    missing = np.zeros((len(records), 21), dtype=bool)
                    for i, record in enumerate(records):
                        missing[i, record[3]] = True
                    temporary = path.with_name(path.name + '.tmp')
                    with temporary.open('wb') as f:
                        np.savez_compressed(f, replicate=np.arange(start, end + 1), fixed=fixed, crossed=crossed,
                                            valid=valid, missing_cells=missing, input_fingerprint=np.asarray(fingerprint))
                    temporary.replace(path)
                all_fixed.append(fixed)
                all_crossed.append(crossed)
                all_valid.append(valid)
                all_missing.append(missing)
                print('BOOTSTRAP_PROGRESS', f'{end}/{args.replicates}',
                      'estimable=' + str(sum(int(v.sum()) for v in all_valid)),
                      'elapsed_minutes=%.1f' % ((time.monotonic() - start_time) / 60), flush=True)
                save_json(output / 'PROGRESS.json', dict(completed=end, attempted=args.replicates,
                          estimable=sum(int(v.sum()) for v in all_valid), input_fingerprint=fingerprint))
        fixed, crossed, valid = np.concatenate(all_fixed), np.concatenate(all_crossed), np.concatenate(all_valid)
        results = summarize(data['point'], fixed, crossed, valid)
        save_csv(output / 'PAIRED_BOOTSTRAP_SUMMARY.csv', results)
        support_rows = [dict(**row, unsupported_draws=int(np.concatenate(all_missing)[:, i].sum()))
                        for i, row in enumerate(data['support_rows'])]
        save_csv(output / 'BOOTSTRAP_CELL_SUPPORT.csv', support_rows)
        save_json(output / 'COMPLETE.json', dict(status='COMPLETE_POSTHOC_PAIRED_BOOTSTRAP', input_fingerprint=fingerprint,
                  attempted_replicates=args.replicates, estimable_replicates=int(valid.sum()),
                  support_fraction=float(valid.mean()), training=False, intervals_conditioned_on_all_21_cells_estimable=True,
                  caution='Posthoc reused-test analysis; bootstrap intervals do not establish external validity, equivalence or clinical benefit'))
        for row in results:
            if row['design'] == 'crossed_seed_by_patient' and row['metric'] == 'Macro_AP':
                print(row['comparison'], 'delta=%.4f pp' % row['delta_pp'],
                      'CI95=[%.4f, %.4f]' % (row['ci95_lower_pp'], row['ci95_upper_pp']),
                      'AP_BONFERRONI_CI=[%.4f, %.4f]' % (row['ap_bonferroni_ci_lower_pp'], row['ap_bonferroni_ci_upper_pp']), row['assessment'], flush=True)
        print('SSTHNET_PAIRED_BOOTSTRAP_COMPLETE', output, flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--patient-mapping', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--base-seed', type=int, default=20261015)
    parser.add_argument('--chunk-size', type=int, default=25)
    parser.add_argument('--workers', type=int, default=2)
    run(parser.parse_args())
