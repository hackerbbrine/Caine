"""
caine/gpu.py — GPU acceleration shim
=====================================
Provides ``xp``: a CuPy array namespace when a CUDA GPU is available,
falling back transparently to NumPy.

Usage (everywhere in the caine package):
    from caine.gpu import xp, to_numpy, to_device, GPU_AVAILABLE

    arr = xp.zeros(100)          # GPU or CPU array
    result = xp.dot(a, b)        # computed on whichever is available
    py_arr = to_numpy(arr)       # always a numpy ndarray
"""

import numpy as _np

GPU_AVAILABLE: bool = False
xp = _np   # default: numpy

try:
    import cupy as _cp
    # Verify a GPU device is actually available
    _cp.cuda.Device(0).use()
    _cp.zeros(1)          # smoke test
    xp = _cp
    GPU_AVAILABLE = True
    print("[gpu] CuPy available — computations will run on GPU.")
except Exception:
    pass   # CuPy not installed or no GPU; silently stay on numpy


def to_numpy(arr) -> _np.ndarray:
    """Convert a CuPy or NumPy array to a NumPy ndarray."""
    if GPU_AVAILABLE and isinstance(arr, xp.ndarray):
        return _cp.asnumpy(arr)
    return _np.asarray(arr)


def to_device(arr: _np.ndarray):
    """Move a NumPy array to the active device (GPU if available, else no-op)."""
    if GPU_AVAILABLE:
        return xp.asarray(arr)
    return arr
