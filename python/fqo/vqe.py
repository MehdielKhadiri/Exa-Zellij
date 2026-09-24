"""Optional existing VQE evaluator, loaded only when first used.

Install the ``vqe`` extra and configure its simulator/cache as described in the
quantum guide. One molecular session is active per process, as in the native
embedded adapter. ``init`` returns the parameter dimension.
"""
from importlib import import_module
from importlib.util import find_spec


def _bridge():
    if find_spec("fqo._vqe") is not None:
        return import_module("fqo._vqe.vqe_bridge")
    # Source-tree development: the original modules live beside fqo/.
    return import_module("vqe_bridge")


def init(molname="LiH", basis="STO-3G"):
    return _bridge().init(molname, basis)


def eval(theta):
    return _bridge().eval(theta)


def eval_batch(rows):
    return _bridge().eval_batch(rows)
