"""Small synthetic tests; no real patient data and no test-partition access."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_ssthnet_training_optimization as run


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_learning_rates_and_resume_independent_formula(self):
        for epoch in range(1, 21):
            self.assertEqual(run.learning_rate(run.CANDIDATES[0], epoch), 2e-4)
            self.assertEqual(run.learning_rate(run.CANDIDATES[1], epoch),
                             run.learning_rate(run.CANDIDATES[2], epoch))
        rates = [run.learning_rate(run.CANDIDATES[1], e) for e in range(1, 21)]
        self.assertAlmostEqual(rates[0], 2e-4)
        self.assertAlmostEqual(rates[-1], 2e-5)
        self.assertTrue(all(a > b for a, b in zip(rates, rates[1:])))
        with self.assertRaises(RuntimeError):
            run.learning_rate(run.CANDIDATES[0], 21)

    def test_class_weight_changes_only_in_candidate_c(self):
        original = np.arange(1, 22, dtype=np.float32).reshape(7, 3)
        for candidate in run.CANDIDATES[:2]:
            np.testing.assert_array_equal(run.class_weights(candidate, original), original)
        np.testing.assert_array_equal(run.class_weights(run.CANDIDATES[2], original), np.ones((7, 3)))

    def test_bounded_training_budget_and_fixed_confirmation(self):
        jobs = run.screening_jobs()
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({(j['candidate'], j['seed'], j['fold']) for j in jobs}), 24)
        self.assertEqual({j['epochs'] for j in jobs}, {20})
        confirmation = run.confirmation_jobs(dict(candidate=run.CANDIDATES[1], epoch=10))
        self.assertEqual(len(confirmation), 12)
        self.assertEqual({j['seed'] for j in confirmation}, {3, 4, 5})
        self.assertEqual({j['epochs'] for j in confirmation}, {10})
        with self.assertRaises(RuntimeError):
            run.confirmation_jobs(dict(candidate=run.CANDIDATES[1], epoch=7))

    def test_only_development_and_validation_paths_returned(self):
        helper = SimpleNamespace(sources=lambda _: {key: key for key in (*run.INPUT_KEYS, 'test_features', 'test_reference')})
        self.assertEqual(set(run.input_paths(helper, ROOT)), set(run.INPUT_KEYS))

    def test_selection_uses_only_four_epochs_and_ap_not_auroc(self):
        rows = [dict(candidate=c, epoch=e, average_precision=.5, auroc=.9)
                for c in run.CANDIDATES for e in range(1, 21)]
        next(r for r in rows if r['candidate']==run.CANDIDATES[2] and r['epoch']==7)['average_precision'] = .99
        next(r for r in rows if r['candidate']==run.CANDIDATES[0] and r['epoch']==10)['average_precision'] = .6
        next(r for r in rows if r['candidate']==run.CANDIDATES[1] and r['epoch']==15)['auroc'] = .99
        result = run.choose(rows, .55)
        self.assertEqual((result['candidate'],result['epoch']), (run.CANDIDATES[0],10))
        self.assertTrue(result['proceed_to_confirmation'])
        self.assertFalse(run.choose(rows,.6)['proceed_to_confirmation'])
        tied = [dict(candidate=c, epoch=e, average_precision=.6) for c in run.CANDIDATES for e in run.SELECT_EPOCHS]
        result = run.choose(tied, .5)
        self.assertEqual((result['candidate'],result['epoch']), (run.CANDIDATES[0],5))
        with self.assertRaises(RuntimeError):
            run.choose(tied[:-1], .5)

    def test_fourfold_probabilities_are_averaged_before_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fold, value in enumerate([.2,.4,.6,.8]):
                path = run.job_dir(root,run.CANDIDATES[0],1,fold)
                path.mkdir(parents=True)
                np.savez(path/'epoch05_validation.npz', probability=np.full((6,7),value,np.float32),
                         contract_hash=np.array('frozen'), validation_reference_sha256=np.array('reference'))
            result = run.average_folds(root,run.CANDIDATES[0],1,5,'frozen','reference')
            np.testing.assert_allclose(result,.5)
            with self.assertRaises(RuntimeError):
                run.average_folds(root,run.CANDIDATES[0],1,5,'changed','reference')


class RealModelSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.loaded, _ = run.original.modules(ROOT)

    def context(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(8,3,220)).astype(np.float32)
        groups = dict(representative_index=np.arange(8), group_fold=np.arange(8)%4,
                      labels=np.broadcast_to((np.arange(8)%2)[:,None,None],(8,7,3)).copy(),
                      masks=np.ones((8,7,3),np.uint8))
        reference = dict(global_position=np.arange(6), case_id=np.arange(6), anchor_seconds=np.arange(6),
                         horizons=np.repeat([5,10,15],2), labels=np.tile([0,1],3)[:,None].repeat(7,axis=1),
                         masks=np.ones((6,7),np.uint8))
        val_x = rng.normal(size=(6,3,220)).astype(np.float32)
        protocol = dict(inputs=dict(validation_reference=dict(sha256='synthetic-reference')))
        return (dict(self.loaded), x, groups, val_x, reference, protocol, 'synthetic-contract')

    def test_final_model_and_all_21_diagnostic_metrics(self):
        context = self.context()
        model = run.original.make_model(self.loaded['models'],'ssthnet')
        self.assertEqual(sum(p.numel() for p in model.parameters()),155271)
        probability = run.predict(model,context[3],context[4],torch.device('cpu'))
        self.assertEqual(probability.shape,(6,7))
        rows = run.diagnostic_rows(self.loaded['io'],context[4],probability)
        self.assertEqual(len(rows),21)
        self.assertTrue(all(np.isfinite(list(run.macro(rows).values()))))

    def test_interrupted_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = self.context()
            job = dict(candidate=run.CANDIDATES[1],seed=1,fold=0,epochs=2)
            run.train_job(context,root/'fresh',job,torch.device('cpu'))
            helper = context[0]['io']
            real_save = helper.save_csv
            calls = [0]

            def interrupt(path,rows):
                real_save(path,rows)
                calls[0] += 1
                if calls[0] == 1:
                    raise RuntimeError('synthetic interruption after durable epoch 1')

            with patch.object(helper,'save_csv',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    run.train_job(context,root/'resumed',job,torch.device('cpu'))
            run.train_job(context,root/'resumed',job,torch.device('cpu'))
            a = run.original.load_checkpoint(run.job_dir(root/'fresh',job['candidate'],1,0)/'latest.pt')
            b = run.original.load_checkpoint(run.job_dir(root/'resumed',job['candidate'],1,0)/'latest.pt')
            for key in a['model_state']:
                self.assertTrue(torch.equal(a['model_state'][key],b['model_state'][key]),key)
            path = run.job_dir(root/'resumed',job['candidate'],1,0)
            run.verify_complete(helper,path,'synthetic-contract',job)
            with (path/'epoch02_validation.npz').open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(RuntimeError,'Completed output changed'):
                run.verify_complete(helper,path,'synthetic-contract',job)


if __name__ == '__main__':
    unittest.main()
