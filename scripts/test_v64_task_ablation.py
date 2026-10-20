from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from src.v64_task_ablation import active_tasks, activate_heads, mode_loss, split_inner


class AblationTests(unittest.TestCase):
    def test_purge_and_minimum(self):
        dates = np.arange(1000)+1
        ends = np.repeat((dates+60)[:, None], 3, 1)
        self.assertIsNone(split_inner(np.arange(371), ends, dates))
        split = split_inner(np.arange(613), ends, dates)
        self.assertEqual(len(split['fit']), 427)
        self.assertLess(ends[split['fit']].max(), dates[split['early'][0]])

    def test_no_invalid_mode(self):
        with self.assertRaises(ValueError):
            active_tasks('best_V')

    def test_single_task_equal_average_joint_loss(self):
        pred = {t: torch.zeros(4, 3, 2, requires_grad=True) for t in ['asset', 'sleeve']}
        target = {t: torch.ones_like(v)*.1 for t, v in pred.items()}
        masks = {t: torch.ones_like(v, dtype=torch.bool) for t, v in pred.items()}
        groups = {t: torch.zeros(2, dtype=torch.long) for t in pred}
        joint = mode_loss(pred, target, masks, groups, 'joint')
        asset = mode_loss(pred, target, masks, groups, 'asset_only')
        sleeve = mode_loss(pred, target, masks, groups, 'sleeve_only')
        torch.testing.assert_close(joint, (asset+sleeve)/2)
        sleeve.backward()
        self.assertEqual(float(pred['asset'].grad.abs().sum()), 0.)
        self.assertGreater(float(pred['sleeve'].grad.abs().sum()), 0.)

    def test_inactive_head_not_decayed(self):
        model = nn.Module()
        model.heads = nn.ModuleDict({t: nn.Linear(3, 3) for t in ['asset', 'sleeve']})
        old = {k: v.clone() for k, v in model.heads['asset'].state_dict().items()}
        opt = torch.optim.AdamW(activate_heads(model, 'sleeve_only'), weight_decay=.1)
        model.heads['sleeve'](torch.ones(2, 3)).sum().backward()
        opt.step()
        for k, v in model.heads['asset'].state_dict().items():
            torch.testing.assert_close(old[k], v, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
