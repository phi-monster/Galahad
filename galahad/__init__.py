"""Measurement battery and evaluation servers.

Deliberately empty of imports: ``galahad.battery`` orchestrates the eval scripts as
subprocesses and pulls in no heavy dependencies, so it can be imported (and ``--dry-run``
executed) on a machine with no GPU and no simulator installed.
"""

__all__ = ["battery"]
