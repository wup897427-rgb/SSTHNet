import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('lr660',Path(__file__).with_name('lr660.py'))
lr=importlib.util.module_from_spec(spec);spec.loader.exec_module(lr)


class SmokeTest(unittest.TestCase):
    def make_split(self, base, name, n, offset):
        folder=base/name;folder.mkdir()
        rng=np.random.default_rng(offset)
        x=rng.normal(size=(n,660)).astype('float32')
        y=np.column_stack([(x[:,j]+rng.normal(size=n)*0.3>0).astype('int8') for j in range(21)])
        mask=np.ones_like(y,dtype=bool);mask[0,0]=False;y[0,0]=-1
        arrays={'X':x,'y':y,'mask':mask,'patient_id':np.arange(n)+offset,'row_id':np.arange(n)+offset,
                'current_normal':np.ones_like(mask)}
        meta={'feature_names':[f'feature_{j}' for j in range(660)],'cell_names':[f'cell_{j}' for j in range(21)]}
        for key,arr in arrays.items():np.save(folder/(key+'.npy'),arr);meta[key]=key+'.npy'
        lr.write_json(folder/'manifest.json',meta)
        return folder/'manifest.json'

    def test_end_to_end_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            dev=self.make_split(base,'development',80,1000)
            val=self.make_split(base,'validation',60,2000)
            args=argparse.Namespace(out=str(base/'model'),development=str(dev),validation=str(val),
                 c_grid='0.01,0.1',class_weight='none',max_iter=1000,tol=1e-6,threads=1)
            with lr.threadpool_limits(limits=1):lr.train(args)
            # No test files existed until after training and model freezing.
            self.assertTrue((base/'model/FROZEN.json').exists())
            test=self.make_split(base,'test',60,3000)
            ev=argparse.Namespace(model_dir=str(base/'model'),test=str(test),out=str(base/'evaluation'),batch_size=17)
            with lr.threadpool_limits(limits=1):lr.evaluate(ev)
            with np.load(base/'evaluation/test_predictions.npz') as z:
                self.assertEqual(z['probabilities'].shape,(60,21))
                self.assertTrue(np.isfinite(z['probabilities']).all())
            ev.out=str(base/'bad_overlap');ev.test=str(dev)
            with self.assertRaisesRegex(ValueError,'overlap'):lr.evaluate(ev)
            ev.out=str(base/'bad_schema');ev.test=str(test)
            meta=json.loads(test.read_text());meta['feature_names'].reverse();lr.write_json(test,meta)
            with self.assertRaisesRegex(ValueError,'ordering'):lr.evaluate(ev)
            with self.assertRaisesRegex(ValueError,'empty'):lr.train(args)
            (base/'model/protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):lr.evaluate(ev)

    def test_threshold(self):
        y=np.array([0,0,1,1]);p=np.array([0.1,0.2,0.7,0.9])
        self.assertEqual(lr.choose_threshold(y,p),0.7)


if __name__=='__main__':unittest.main()
