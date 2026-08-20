"""
Make the modules in src/ importable from tests without an install step.

Keeps test imports flat (`from linear_search import LinearIndex`) and means
`pytest` just works from the repo root against the local source tree.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
