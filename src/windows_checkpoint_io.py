"""Bounded Windows sharing-violation retries around atomic torch checkpoints."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import time

import torch


def atomic_checkpoint(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle=tempfile.NamedTemporaryFile(prefix=path.name+".",suffix=".tmp",dir=path.parent,delete=False)
    temporary=Path(handle.name);handle.close()
    try:
        torch.save(payload,temporary)
        for attempt in range(8):
            try:
                os.replace(temporary,path)
                return
            except PermissionError:
                if attempt==7:raise
                time.sleep(min(.05*(2**attempt),2.))
    finally:
        if temporary.exists():temporary.unlink()


@contextmanager
def checkpoint_io(engine):
    previous=engine._atomic_save
    engine._atomic_save=atomic_checkpoint
    try:yield
    finally:engine._atomic_save=previous
