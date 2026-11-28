"""Fixed-model, original-validation-only two-view explanation extension."""
import os
import sys

# The original train.py established this before importing torch. This standalone
# package deliberately does not import the trainer, so retain that prerequisite
# here, before any submodule can create a CUDA/cuBLAS context.
_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
_torch = sys.modules.get("torch")
if _workspace is None and _torch is not None and _torch.cuda.is_initialized():
    raise RuntimeError("Start the explanation CLI in a fresh process; its original cuBLAS workspace must be set before CUDA initialization")
if os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
    raise RuntimeError("This frozen explanation protocol requires CUBLAS_WORKSPACE_CONFIG=:4096:8; an existing different value is not overwritten")
