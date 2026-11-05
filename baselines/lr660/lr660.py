"""L2 logistic baseline. Select on validation only; evaluate in a separate command."""
import argparse
import csv
import hashlib
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, brier_score_loss)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_split(path):
    """Separate .npy arrays allow memory-mapped 660-column features."""
    path = Path(path).resolve()
    meta = json.loads(path.read_text(encoding='utf-8'))
    required = ('X', 'y', 'mask', 'patient_id', 'row_id', 'feature_names', 'cell_names')
    missing = set(required) - set(meta)
    if missing:
        raise ValueError('Missing manifest keys: ' + str(sorted(missing)))
    arrays = {}
    hashes = {'manifest': digest(path)}
    for key in ('X', 'y', 'mask', 'patient_id', 'row_id', 'current_normal'):
        if key in meta:
            source = (path.parent / meta[key]).resolve()
            arrays[key] = np.load(source, mmap_mode='r', allow_pickle=False)
            hashes[key] = digest(source)
    x, y, mask = arrays['X'], arrays['y'], arrays['mask']
    if x.ndim != 2 or x.shape[1] != 660 or y.shape != (len(x), 21) or mask.shape != y.shape:
        raise ValueError('Require X=(N,660), y=(N,21), mask=(N,21). No guessed reshaping.')
    if len(meta['feature_names']) != 660 or len(set(meta['feature_names'])) != 660:
        raise ValueError('Require 660 unique ordered feature names')
    if len(meta['cell_names']) != 21 or len(set(meta['cell_names'])) != 21:
        raise ValueError('Require 21 unique ordered target-horizon cell names')
    for key in ('patient_id', 'row_id'):
        if arrays[key].shape != (len(x),):
            raise ValueError(key + ' must have shape (N,)')
    if not np.isin(mask, [0, 1]).all() or not np.isin(y[np.asarray(mask, dtype=bool)], [0, 1]).all():
        raise ValueError('Mask and evaluable labels must be binary')
    for key in ('patient_id', 'row_id'):
        ids = arrays[key].astype(str)
        if any(v.strip().lower() in ('', 'nan', 'none') for v in ids):
            raise ValueError('Missing ' + key)
    if len(set(arrays['row_id'].astype(str))) != len(x):
        raise ValueError('row_id is not unique; export one shared anchor row with 21 labels')
    if 'current_normal' in arrays:
        if arrays['current_normal'].shape != y.shape or not np.isin(arrays['current_normal'], [0, 1]).all():
            raise ValueError('current_normal must be binary (N,21), including observability')
    return meta, arrays, hashes


def check_compatible(a, b):
    for key in ('feature_names', 'cell_names'):
        if a[key] != b[key]:
            raise ValueError('Different ordering: ' + key)


def patient_set(data):
    return set(data['patient_id'].astype(str))


def preprocess_fit(x):
    medians = np.zeros(660, dtype=float)
    for j in range(660):
        col = np.asarray(x[:, j], dtype=float)
        finite = col[np.isfinite(col)]
        medians[j] = np.median(finite) if len(finite) else 0.0
    filled = preprocess_fill(x, medians)
    scaler = StandardScaler().fit(filled)
    return medians, scaler, scaler.transform(filled)


def preprocess_fill(x, medians):
    filled = np.array(x, dtype=np.float64, copy=True)
    for j in range(660):
        bad = ~np.isfinite(filled[:, j])
        filled[bad, j] = medians[j]
    return filled


def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


def train(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty. Existing runs will not be overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    dm, dev, dh = load_split(args.development)
    vm, val, vh = load_split(args.validation)
    check_compatible(dm, vm)
    if patient_set(dev) & patient_set(val):
        raise ValueError('Development and validation patients overlap')
    grid = sorted(set(float(c) for c in args.c_grid.split(',')))
    if not grid or not all(np.isfinite(c) and c > 0 for c in grid):
        raise ValueError('C must be finite and positive')
    protocol = {'model': 'LR-660-L2-LBFGS', 'C_grid': grid, 'class_weight': args.class_weight,
                'selection': 'maximum validation AP per cell; exact ties prefer smaller C',
                'threshold': 'maximum validation F1; exact ties prefer higher threshold',
                'development': dh, 'validation': vh, 'test_loaded': False,
                'sklearn': sklearn.__version__, 'python': platform.python_version(),
                'max_iter': args.max_iter, 'tol': args.tol, 'threads': args.threads,
                'cell_names': dm['cell_names'], 'feature_names': dm['feature_names']}
    write_json(out / 'protocol.json', protocol)
    medians, scaler, xd = preprocess_fit(dev['X'])
    xv = scaler.transform(preprocess_fill(val['X'], medians))
    models, thresholds, audit, selected = [], [], [], []
    for k, name in enumerate(dm['cell_names']):
        md = np.asarray(dev['mask'][:, k], dtype=bool)
        mv = np.asarray(val['mask'][:, k], dtype=bool)
        yd, yv = np.asarray(dev['y'][md, k], dtype=int), np.asarray(val['y'][mv, k], dtype=int)
        if len(np.unique(yd)) != 2 or len(np.unique(yv)) != 2:
            raise ValueError(name + ': development and validation each need both classes; no test-based fallback')
        best = None
        for c in grid:
            model = LogisticRegression(C=c, solver='lbfgs', max_iter=args.max_iter, tol=args.tol,
                                       class_weight=None if args.class_weight == 'none' else 'balanced')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(xd[md], yd)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            converged = converged and np.isfinite(model.coef_).all()
            p = model.predict_proba(xv[mv])[:, 1]
            ap = float(average_precision_score(yv, p))
            audit.append({'cell': name, 'C': c, 'validation_AP': ap,
                          'converged': bool(converged), 'iterations': int(model.n_iter_.max())})
            write_csv(out / 'validation_grid.csv', audit)
            print(f'{name} C={c:g} val_AP={ap:.6f} converged={converged}', flush=True)
            if not converged:
                raise RuntimeError('Candidate did not converge. Increase max_iter in a fresh run; test has not been read.')
            if best is None or ap > best[0]:
                best = (ap, model, p, c)
        threshold = choose_threshold(yv, best[2])
        models.append(best[1]); thresholds.append(threshold)
        selected.append({'cell': name, 'C': best[3], 'validation_AP': best[0], 'threshold': threshold,
                         'development_n': int(md.sum()), 'development_positive': int(yd.sum()),
                         'validation_n': int(mv.sum()), 'validation_positive': int(yv.sum())})
    bundle = {'models': models, 'thresholds': thresholds, 'medians': medians, 'scaler': scaler,
              'metadata': dm, 'development_patients': sorted(patient_set(dev)),
              'validation_patients': sorted(patient_set(val)), 'sklearn': sklearn.__version__}
    joblib.dump(bundle, out / 'frozen_model.joblib', compress=3)
    write_csv(out / 'selected_validation.csv', selected)
    write_json(out / 'FROZEN.json', {'model_sha256': digest(out / 'frozen_model.joblib'),
                                   'protocol_sha256': digest(out / 'protocol.json'), 'test_loaded': False})
    print('FROZEN: training complete. Test evaluation requires a separate command.', flush=True)


def evaluate(args):
    run, out = Path(args.model_dir), Path(args.out)
    frozen = json.loads((run / 'FROZEN.json').read_text())
    if digest(run / 'frozen_model.joblib') != frozen['model_sha256'] or digest(run / 'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('Frozen model or protocol changed')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Evaluation output must be empty')
    # Load only your own trusted model bundle; joblib files can execute code.
    bundle = joblib.load(run / 'frozen_model.joblib')
    if bundle['sklearn'] != sklearn.__version__:
        raise ValueError('Use the same scikit-learn version as training')
    meta, data, hashes = load_split(args.test)
    check_compatible(bundle['metadata'], meta)
    if patient_set(data) & (set(bundle['development_patients']) | set(bundle['validation_patients'])):
        raise ValueError('Test patients overlap development or validation')
    out.mkdir(parents=True, exist_ok=True)
    p = np.empty((len(data['X']), 21), dtype=np.float64)
    for start in range(0, len(p), args.batch_size):
        stop = start + args.batch_size
        x = bundle['scaler'].transform(preprocess_fill(data['X'][start:stop], bundle['medians']))
        for k, model in enumerate(bundle['models']):
            p[start:stop, k] = model.predict_proba(x)[:, 1]
    threshold = np.asarray(bundle['thresholds'])
    rows, summary = [], []
    for population in ['overall'] + (['current_normal'] if 'current_normal' in data else []):
        mask = np.asarray(data['mask'], dtype=bool).copy()
        if population == 'current_normal': mask &= np.asarray(data['current_normal'], dtype=bool)
        cells = []
        for k, name in enumerate(meta['cell_names']):
            eligible = mask[:, k]; y = np.asarray(data['y'][eligible, k], dtype=int); prob = p[eligible, k]
            pred = prob >= threshold[k]
            both = len(np.unique(y)) == 2
            row = {'population': population, 'cell': name, 'n': len(y), 'positive': int(y.sum()),
                   'AP': float(average_precision_score(y, prob)) if both else None,
                   'AUROC': float(roc_auc_score(y, prob)) if both else None,
                   'F1': float(f1_score(y, pred, zero_division=0)) if len(y) else None,
                   'Precision': float(precision_score(y, pred, zero_division=0)) if len(y) else None,
                   'Recall': float(recall_score(y, pred, zero_division=0)) if len(y) else None,
                   'Brier': float(brier_score_loss(y, prob)) if len(y) else None,
                   'threshold': float(threshold[k])}
            cells.append(row); rows.append(row)
        aggregate = {'population': population, 'AP_AUROC_supported_cells': sum(r['AP'] is not None for r in cells)}
        for metric in ('AP', 'AUROC', 'F1', 'Precision', 'Recall', 'Brier'):
            vals = [r[metric] for r in cells]
            aggregate['Macro_' + metric] = float(np.mean(vals)) if all(v is not None for v in vals) else None
        y = np.asarray(data['y'])[mask].astype(int); pred = (p >= threshold)[mask]
        aggregate['Micro_F1'] = float(f1_score(y, pred, zero_division=0)) if len(y) else None
        summary.append(aggregate)
    write_csv(out / 'test_21cell_metrics.csv', rows)
    write_csv(out / 'test_macro_metrics.csv', summary)
    np.savez_compressed(out / 'test_predictions.npz', probabilities=p, y=data['y'], mask=data['mask'],
                        patient_id=data['patient_id'], row_id=data['row_id'], thresholds=threshold,
                        cell_names=np.asarray(meta['cell_names']))
    write_json(out / 'EVALUATION.json', {'test_sources': hashes, 'frozen': frozen,
               'threshold_policy': 'per-cell validation F1, not fixed 0.5 or alarm episode thresholds',
               'unsupported_cells': 'No 21-cell macro metric if any required cell is unsupported',
               'CI': 'Not calculated; patient-level paired inference requires aligned SSTHNet predictions'})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--development', required=True); fit.add_argument('--validation', required=True)
    fit.add_argument('--out', required=True)
    fit.add_argument('--c-grid', default='0.001,0.01,0.1,1,10,100')
    fit.add_argument('--class-weight', choices=['none', 'balanced'], default='none')
    fit.add_argument('--max-iter', type=int, default=3000); fit.add_argument('--tol', type=float, default=1e-6)
    fit.add_argument('--threads', type=int, default=4)
    test = sub.add_parser('evaluate')
    test.add_argument('--model-dir', required=True); test.add_argument('--test', required=True)
    test.add_argument('--out', required=True); test.add_argument('--batch-size', type=int, default=8192)
    test.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or (hasattr(args, 'batch_size') and args.batch_size < 1):
        parser.error('threads and batch-size must be positive')
    with threadpool_limits(limits=args.threads):
        (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
