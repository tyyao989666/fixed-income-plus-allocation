import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src import windows_checkpoint_io as io


class CheckpointTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"model.pt";io.atomic_checkpoint({"x":torch.tensor([1.,2.])},p)
            self.assertTrue(torch.equal(torch.load(p,weights_only=True)["x"],torch.tensor([1.,2.])))
            self.assertEqual(len(list(Path(folder).iterdir())),1)

    def test_transient_denial(self):
        original=io.os.replace;count=[0]
        def transient(a,b):
            count[0]+=1
            if count[0]<3:raise PermissionError("fixture sharing violation")
            original(a,b)
        with tempfile.TemporaryDirectory() as folder,patch.object(io.os,"replace",side_effect=transient),patch.object(io.time,"sleep"):
            io.atomic_checkpoint({"x":1},Path(folder)/"model.pt")
        self.assertEqual(count[0],3)

    def test_permanent_denial_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(io.os,"replace",side_effect=PermissionError("fixture")) as replace,patch.object(io.time,"sleep"):
            with self.assertRaises(PermissionError):io.atomic_checkpoint({"x":1},Path(folder)/"model.pt")
            self.assertEqual(replace.call_count,8)
            self.assertEqual(list(Path(folder).iterdir()),[])


if __name__=="__main__":unittest.main()
