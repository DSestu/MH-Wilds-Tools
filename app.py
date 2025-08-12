"""
Thin wrapper to run the Gradio app on environments that execute `python app.py`
from the repository root (e.g., Hugging Face Spaces).

It ensures the `src/` directory is on `sys.path` and imports the real app
from the installed package module `MH_Wilds_tools.app`.
"""

from __future__ import annotations

import os
import sys


def _ensure_src_on_sys_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_sys_path()

# Import the actual Gradio Blocks (`demo`) and the CLI `main()` from the package.
from MH_Wilds_tools.app import demo as demo  # noqa: E402  (import after sys.path tweak)
from MH_Wilds_tools.app import main as main  # noqa: E402


if __name__ == "__main__":
    # Running locally with `python app.py`
    main()
