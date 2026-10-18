from pathlib import Path

import nbformat


def test_notebook_is_valid_and_python_cells_compile():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "jev.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            # Exclude Colab pip magics from the plain-Python syntax check.
            source = "\n".join(
                line for line in cell.source.splitlines()
                if not line.lstrip().startswith("%")
            )
            compile(source, f"jev.ipynb cell {index}", "exec")
