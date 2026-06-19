"""Make the ``rl`` package importable when running ``pytest`` without install.

Placing this file at the repository root causes pytest to prepend the root to
``sys.path``, so ``import rl`` works even if the package has not been installed
with ``pip install -e .``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
