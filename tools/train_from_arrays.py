"""Train one final-recipe fold using independently prepared numerical arrays."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
import run_ssthnet_training_optimization as training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development_features", "development_groups", "validation_features", "validation_reference"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    loaded, sources = training.original.modules(root)
    helper = loaded["io"]
    inputs = {key: getattr(args, key).resolve() for key in training.INPUT_KEYS}
    output = args.out.resolve()
    for path in inputs.values():
        if not path.is_file():
            parser.error("Missing prepared input: " + str(path))
        if output == path or output in path.parents:
            parser.error("Output must not contain or replace input files")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Use cpu or an explicitly selected CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable")
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    x, groups = helper.development(inputs)
    validation_x, reference = helper.outer(inputs, "validation")
    job = dict(candidate="A_constant_weighted", seed=args.seed, fold=args.fold, epochs=15)
    protocol = dict(
        version="ssthnet_core_array_training_v1", job=job,
        role="final fixed recipe; no hyperparameter/epoch selection or test evaluation",
        inputs={key: dict(path=str(path), sha256=helper.sha256(path)) for key, path in inputs.items()},
        sources={key: helper.sha256(path) for key, path in sources.items()},
        entry_sha256=helper.sha256(Path(__file__)),
        training_runner_sha256=helper.sha256(Path(training.__file__)),
        shared_runner_sha256=helper.sha256(Path(training.shared.__file__)),
        ablation_runner_sha256=helper.sha256(Path(training.original.__file__)),
        torch_version=torch.__version__, numpy_version=training.np.__version__,
        sklearn_version=training.sklearn.__version__,
    )
    fingerprint = training.shared.boot.canonical_hash(protocol)
    output.mkdir(parents=True, exist_ok=True)
    record = output / f"CORE_PROTOCOL_seed{args.seed:02d}_fold{args.fold}.json"
    if record.exists():
        if json.loads(record.read_text()) != protocol:
            raise ValueError("Inputs or implementation changed; use a different output directory")
    else:
        helper.save_json(record, protocol)
    context = loaded, x, groups, validation_x, reference, protocol, fingerprint
    training.train_job(context, output, job, device)


if __name__ == "__main__":
    main()
