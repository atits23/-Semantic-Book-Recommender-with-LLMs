"""Execute notebooks in order using this environment; optional developer check."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".cache/jupyter_runtime"))
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        print(f"Executing {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        nbformat.write(notebook, path)
        print(f"Passed {path.name}", flush=True)


if __name__ == "__main__":
    main()
