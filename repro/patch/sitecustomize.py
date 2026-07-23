"""Patch helpers loaded by sitecustomize (run_env.sh puts this dir first in PYTHONPATH).

Three issues with running slime image's packages on older host glibc:

1. Triton 3.5's cache.py defines default_cache_dir/default_dump_dir/
   default_override_dir but from-import fails (triton lazy module loading).
   sglang needs all three. Inject them.

2. sglang_router_rs.abi3.so (Rust) needs GLIBC_2.38 which host doesn't have.
   Fake the whole sglang_router package; eval-only mode (num_rollout=0)
   doesn't actually use the router. For multi-engine training this patch
   will break router functionality - replace with a host-compatible build.
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


# --- Patch 2: fake sglang_router (Rust .so needs GLIBC_2.38, host has 2.35) ---
class _FakeModule:
    """Fake module that returns dummy objects for any attribute/call."""
    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        return _FakeModule()
    def __call__(self, *args, **kwargs):
        return _FakeModule()
    def __repr__(self):
        return '<fake sglang_router (GLIBC_2.38 unavailable)>'

# sglang_router_rs.abi3.so needs GLIBC_2.38; host doesn't have it.
# Unconditionally fake - host can never load image's Rust .so.
# eval-only mode (num_rollout=0, single engine) doesn't need the router.
for _m in ['sglang_router', 'sglang_router.launch_router',
           'sglang_router.mini_lb', 'sglang_router.router_args',
           'sglang_router.sglang_router_rs']:
    sys.modules[_m] = _FakeModule()
