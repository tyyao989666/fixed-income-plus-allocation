import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.cuda_finite_check import noncompacting_finite_check, module


class FiniteTests(unittest.TestCase):
    def test_predicate_equivalence(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for values in [[1., 2., 3.], [float("nan"), 2., float("inf")]]:
                x = torch.tensor(values, device=device)
                for mask in [[False, False, False], [False, True, False], [True, True, True]]:
                    m = torch.tensor(mask, device=device)
                    self.assertEqual(torch.isfinite(x[m]).all().item(), torch.where(m, torch.isfinite(x), True).all().item())

    def test_scope_and_outputs(self):
        from src import v45_forecast as engine
        import numpy as np
        cfg = engine.configuration(Path(__file__).resolve().parents[1] / "config/v45_research.yaml")
        core = module.MaskSafeCore(120, 3, 2, np.array([0, 1, 2, 3, 4]), 3, cfg).eval()
        x = torch.randn(2, 120, 5, 3, requires_grad=True)
        m = torch.ones_like(x)
        macro = torch.randn(2, 120, 2)
        original = module.MaskSafeCore.encode_context
        before = core.encode_context(x, m, macro)
        grad_before = torch.autograd.grad(sum(y.sum() for y in before), x)[0]
        with noncompacting_finite_check():
            after = core.encode_context(x, m, macro)
            grad_after = torch.autograd.grad(sum(y.sum() for y in after), x)[0]
        self.assertIs(module.MaskSafeCore.encode_context, original)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(grad_before, grad_after, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
