"""Patch helpers loaded by sitecustomize (run_env.sh puts this dir first in PYTHONPATH).

Two issues with running slime image's packages on older host glibc:

1. Triton 3.5's cache.py defines default_cache_dir/default_dump_dir/
   default_override_dir but from-import fails (triton lazy module loading).
   sglang needs all three. Inject them.

2. sglang_router_rs.abi3.so (Rust) needs GLIBC_2.38 which host doesn't have.
   Fix: reinstall sglang_router from PyPI (manylinux wheel, glibc 2.17+).
   See setup_env.sh for the reinstall step. No fake needed.
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
