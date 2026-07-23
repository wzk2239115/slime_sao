"""Patch helpers loaded by sitecustomize (run_env.sh puts this dir first in PYTHONPATH).

Three compatibility fixes for running slime image packages on older host glibc:

1. Triton 3.5 cache.py: default_cache_dir etc. not importable (lazy module).
2. torch_memory_saver: hook_mode_preload.abi3.so needs GLIBCXX_3.4.32. Force
   "torch" mode (no LD_PRELOAD).
3. sglang_router: reinstalled from PyPI in setup_env.sh (manylinux wheel).
"""
import os
import sys


# --- Patch 1: triton default_cache_dir / default_dump_dir / default_override_dir ---
try:
    import triton.runtime.cache as _tc
    from pathlib import Path

    _home = lambda: os.getenv("TRITON_HOME", Path.home())

    if not hasattr(_tc, 'default_cache_dir'):
        _tc.default_cache_dir = lambda: os.path.join(_home(), ".triton", "cache")
    if not hasattr(_tc, 'default_dump_dir'):
        _tc.default_dump_dir = lambda: os.path.join(_home(), ".triton", "dump")
    if not hasattr(_tc, 'default_override_dir'):
        _tc.default_override_dir = lambda: os.path.join(_home(), ".triton", "override")
except Exception:
    pass


# --- Patch 2: torch_memory_saver force "torch" mode (no LD_PRELOAD) ---
# hook_mode_preload.abi3.so needs GLIBCXX_3.4.32 which host libstdc++ lacks.
# "torch" mode loads the hook via python import instead of LD_PRELOAD.
try:
    from torch_memory_saver import torch_memory_saver as _tms
    _tms.hook_mode = "torch"
except Exception:
    pass
