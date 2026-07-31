"""Unsupported research code, split out of the lean ``ltpred`` core.

Modules here implement estimators and diagnostics that are benchmarked
explorations rather than part of the supported package: they are importable
as ``research.<module>`` from a repository checkout (the repo root must be on
``sys.path``; ``pip install`` does not ship them), their APIs may change
without notice, and their docstrings keep their own caveats. Code that
becomes part of the core estimation path graduates into ``ltpred`` proper.
"""
