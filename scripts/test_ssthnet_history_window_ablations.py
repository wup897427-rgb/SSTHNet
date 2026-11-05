"""Synthetic correctness tests for history-window input ablations."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import run_ssthnet_history_window_ablations as run


ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.loaded, _ = run.opt.original.modules(ROOT)

    def model(self, seconds):
        run.opt.original.seed_everything(123)
        return run.opt.original.make_model(run.window_model_base(self.loaded['models'],seconds),'ssthnet')

    def test_output_heads_tokens_and_parameter_matching(self):
        for seconds in run.WINDOWS:
            model = self.model(seconds).eval()
            x = torch.randn(2,3,22,10)
            self.assertEqual(model._physiology_tokens(x).shape,(2,22,64))
            self.assertEqual(model(x)[0].shape,(2,7,3))
            self.assertEqual(len(model.task_heads),7)
            self.assertEqual(len(model.private_experts),0)
            self.assertIsNone(model.router)
            self.assertEqual(sum(p.numel() for p in model.parameters()),155271)
            self.assertEqual(model.effective_input_features,220)

    def test_excluded_histories_cannot_change_predictions_even_with_nan(self):
        for seconds in run.WINDOWS:
            model = self.model(seconds).eval()
            x = torch.randn(2,3,22,10)
            changed = x.clone()
            for slot in range(3):
                if slot != run.WINDOWS.index(seconds):
                    changed[:,slot] = float('nan')
            with torch.no_grad():
                a, b = model(x)[0], model(changed)[0]
            self.assertTrue(torch.equal(a,b),seconds)
            self.assertTrue(torch.isfinite(b).all())

    def test_no_input_or_embedding_gradient_from_other_histories(self):
        for seconds in run.WINDOWS:
            model = self.model(seconds).eval()
            x = torch.randn(2,3,22,10,requires_grad=True)
            model(x)[0].square().sum().backward()
            selected = run.WINDOWS.index(seconds)
            self.assertGreater(float(x.grad[:,selected].abs().sum()),0)
            for slot in range(3):
                if slot != selected:
                    self.assertEqual(float(x.grad[:,slot].abs().sum()),0)
                    self.assertEqual(float(model.scale_embedding.grad[slot].abs().sum()),0)

    def test_unchanged_module_initial_values_and_initialization_rng_stream(self):
        run.opt.original.seed_everything(123)
        full = run.opt.original.make_model(self.loaded['models'],'ssthnet')
        full_next = torch.rand(5)
        for seconds in run.WINDOWS:
            single = self.model(seconds)
            for key,value in full.state_dict().items():
                self.assertTrue(torch.equal(value,single.state_dict()[key]),key)
            self.assertTrue(torch.equal(full_next,torch.rand(5)))

    def test_adapter_does_not_modify_imported_model_module(self):
        base = self.loaded['models'].SSTHMoEAblation
        fake = (self.loaded,None,None,None,None,{},'parent-hash')
        adapted = run.window_context(fake,300,'window-hash')
        self.assertIs(self.loaded['models'].SSTHMoEAblation,base)
        self.assertIsNot(adapted[0],self.loaded)
        self.assertEqual(adapted[-1],'window-hash')
        with self.assertRaises(RuntimeError):
            run.window_model_base(self.loaded['models'],60)

    def test_paired_validation_aggregation_and_reference(self):
        helper = self.loaded['io']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root/'original'
            reference = dict(global_position=np.arange(6),case_id=np.arange(6),anchor_seconds=np.arange(6),
                             horizons=np.repeat([5,10,15],2),labels=np.tile([0,1],3)[:,None].repeat(7,axis=1),
                             masks=np.ones((6,7),np.uint8))
            context = (self.loaded,None,None,None,reference,
                       dict(inputs=dict(validation_reference=dict(sha256='reference'))),'parent')
            candidate = run.opt.CANDIDATES[0]
            for seed in run.SEEDS:
                directory = baseline/f'joint_seed{seed:02d}'
                directory.mkdir(parents=True)
                helper.save_npz(directory/'VALIDATION_PREDICTIONS.npz',**reference,
                                probability=np.where(reference['labels'],.8,.2).astype(np.float32))
                for seconds in run.WINDOWS:
                    fingerprint = run.opt.shared.boot.canonical_hash(dict(parent='window-contract',window_seconds=seconds))
                    for fold in range(4):
                        directory = run.opt.job_dir(root/f'window{seconds}',candidate,seed,fold)
                        directory.mkdir(parents=True)
                        helper.save_npz(directory/'epoch05_validation.npz',
                                        probability=np.where(reference['labels'],.9,.1).astype(np.float32),
                                        contract_hash=np.array(fingerprint),validation_reference_sha256=np.array('reference'))
            selected = dict(kind='original_epoch5',candidate=candidate,evaluation_epoch=5,root=str(baseline))
            run.summarize(context,root,selected,'window-contract')
            summary = run.opt.shared.boot.read_csv(root/'VALIDATION_SUMMARY.csv')
            self.assertEqual(len(summary),4)
            self.assertEqual([r['inputs'] for r in summary],['660','220','220','220'])
            self.assertEqual(len(run.opt.shared.boot.read_csv(root/'VALIDATION_21CELL_RESULTS.csv')),420)
            self.assertEqual(len(run.opt.shared.boot.read_csv(root/'VALIDATION_PAIRED_SEED_DIFFERENCES.csv')),15)
            self.assertFalse(json.loads((root/'COMPLETE.json').read_text())['test_files_opened'])


class ReferenceTests(unittest.TestCase):
    def test_requires_completed_parent_and_frozen_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'has not finished'):
                run.resolve_reference(parent,parent/'old')
            (parent/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE_FIVE_SEED_VALIDATION_TEST_NOT_RUN',
                configuration=run.opt.CANDIDATES[1],epoch=10,test_files_opened=False)))
            (parent/'SELECTION.json').write_text(json.dumps(dict(proceed_to_confirmation=True,
                candidate=run.opt.CANDIDATES[1],epoch=10)))
            selected=run.resolve_reference(parent,parent/'old')
            self.assertEqual((selected['candidate'],selected['evaluation_epoch'],selected['training_epochs']),
                             (run.opt.CANDIDATES[1],10,10))
            self.assertEqual(len(run.jobs(selected)),60)
            self.assertEqual({j['epochs'] for j in run.jobs(selected)},{10})
            self.assertEqual({j['seed'] for j in run.jobs(selected)},{1,2,3,4,5})

    def test_no_gain_keeps_original_epoch5_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            (parent/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE_NO_VALIDATION_GAIN_ORIGINAL_RETAINED',
                                                               test_files_opened=False)))
            (parent/'SELECTION.json').write_text(json.dumps(dict(proceed_to_confirmation=False)))
            selected=run.resolve_reference(parent,parent/'old')
            self.assertEqual((selected['training_epochs'],selected['evaluation_epoch']),(6,5))
            self.assertEqual(selected['candidate'],run.opt.CANDIDATES[0])


if __name__ == '__main__':
    unittest.main()
