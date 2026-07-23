"""Patch helpers loaded by sitecustomize (run_env.sh puts this dir first in PYTHONPATH)."""
import os

# Triton 3.5: cache.py defines default_cache_dir() but it's not importable
# (triton's __init__.py likely interferes). sglang needs it, so we inject it.
try:
    import triton.runtime.cache as _tc
    if not hasattr(_tc, 'default_cache_dir'):
        from pathlib import Path
        _tc.default_cache_dir = lambda: os.path.join(
            os.getenv("TRITON_HOME", Path.home()), ".triton", "cache")
except Exception:
    pass
