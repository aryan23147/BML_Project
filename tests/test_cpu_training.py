import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from approaches.ucb import Appr
from approaches.utils import BayesianSGD
from networks.mlp_ucb import Net


class CpuTrainingTest(unittest.TestCase):
    def test_train_epoch_uses_configured_cpu(self):
        args = SimpleNamespace(
            device='cpu', inputsize=[1, 2, 2], taskcla=[(0, 2)],
            samples=1, sbatch=4, lr=0.01, nhid=4, nlayers=1,
            rho=-3, sig1=0.0, sig2=6.0, pi=0.25,
            arch='mlp', nepochs=1, output='', checkpoint='.',
            experiment='mnist2', num_tasks=1,
        )
        model = Net(args).to(args.device)
        approach = Appr(model, args)
        approach.optimizer = BayesianSGD(approach.update_lr(0))
        before = model.fc1.weight_mu.detach().clone()
        images = torch.randn(8, 1, 2, 2)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

        approach.train_epoch(0, images, targets)

        self.assertEqual(model.fc1.weight_mu.device.type, 'cpu')
        self.assertFalse(torch.equal(before, model.fc1.weight_mu))
        loss, accuracy = approach.eval(0, images, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)


if __name__ == '__main__':
    unittest.main()
