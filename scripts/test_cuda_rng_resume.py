import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.cuda_rng_resume import host_rng_states, cuda_rng_restore


class RestoreTests(unittest.TestCase):
    def test_bytes_preserved(self):
        state = torch.arange(256, dtype=torch.uint8)[::2]
        actual = host_rng_states([state])[0]
        self.assertTrue(torch.equal(actual, state))
        self.assertEqual(actual.device.type, "cpu")
        self.assertTrue(actual.is_contiguous())

    def test_invalid_rejected(self):
        for bad in [torch.zeros(4), torch.zeros((2, 2), dtype=torch.uint8), None]:
            with self.assertRaises(ValueError):
                host_rng_states([bad])

    def test_scoped_restore(self):
        with patch.object(torch.cuda, "set_rng_state_all") as original:
            with cuda_rng_restore():
                torch.cuda.set_rng_state_all([torch.tensor([1, 2], dtype=torch.uint8)])
            self.assertIs(torch.cuda.set_rng_state_all, original)
            self.assertTrue(torch.equal(original.call_args.args[0][0], torch.tensor([1, 2], dtype=torch.uint8)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_sequence_exact(self):
        saved = torch.cuda.get_rng_state_all()
        try:
            expected = torch.rand(32, device="cuda")
            with cuda_rng_restore():
                torch.cuda.set_rng_state_all([s.to("cuda") for s in saved])
            actual = torch.rand(32, device="cuda")
            self.assertTrue(torch.equal(expected, actual))
        finally:
            torch.cuda.set_rng_state_all(saved)


if __name__ == "__main__":
    unittest.main()
