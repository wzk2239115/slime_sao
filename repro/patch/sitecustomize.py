"""Patch helpers loaded by sitecustomize (run.sh puts this dir first in PYTHONPATH).

glibc upgraded to 2.39 + GLIBCXX 3.4.33 — all preload/native libs work now.
Only triton cache fix remains (sglang imports names that triton 3.5 lazifies).
"""
import os
import sys


# --- Triton default_cache_dir / default_dump_dir / default_override_dir ---
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
