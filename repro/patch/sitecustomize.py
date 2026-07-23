"""Patch helpers loaded by sitecustomize (run_env.sh puts this dir first in PYTHONPATH).

Triton 3.5's cache.py defines default_cache_dir/default_dump_dir/default_override_dir
but from-import fails (triton uses lazy module loading that only exposes part of
the API). sglang needs all three, so we inject them at python startup.
"""
import os

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
