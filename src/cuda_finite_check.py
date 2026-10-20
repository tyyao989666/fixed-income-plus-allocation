"""Equivalent finite-input validation without CUDA boolean-index compaction."""
import ast
from contextlib import contextmanager
import inspect
import textwrap

from src import v49_masked_forecast as module


@contextmanager
def noncompacting_finite_check():
    original = module.MaskSafeCore.encode_context
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected = ast.parse("torch.isfinite(local_values[observed]).all()", mode="eval").body
    replacement = ast.parse("torch.where(observed, torch.isfinite(local_values), True).all()", mode="eval").body

    class Replace(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            if ast.dump(node) == ast.dump(expected):
                self.count += 1
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)

    change = Replace()
    tree = change.visit(tree)
    if change.count != 1:
        raise RuntimeError("Finite-check source changed; refusing an unreviewed patch")
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), vars(module), namespace)
    module.MaskSafeCore.encode_context = namespace["encode_context"]
    try:
        yield
    finally:
        module.MaskSafeCore.encode_context = original
