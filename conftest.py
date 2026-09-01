"""Make the rl package importable when running pytest without installing it.

Placing this file at the repository root makes pytest prepend the root to
sys.path, so "import rl" works even when the package has not been installed
with "pip install -e .".
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
