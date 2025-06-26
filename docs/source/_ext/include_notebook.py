"""
Custom Sphinx directive to include notebook-generated .rst files.

Usage:
.. include_notebook:: some_notebook.ipynb
.. include_notebook:: /path/from/source/root/notebook.ipynb
"""

import os
from pathlib import Path
from docutils.parsers.rst.directives.misc import Include

class IncludeNotebook(Include):
    def run(self):
        user_path = self.arguments[0]
        including_file = Path(self.state.document.current_source)

        # Correct root
        source_root = Path(__file__).resolve().parents[2] / "source"
        generated_root = source_root / "_generated"

        # Resolve the notebook .ipynb path
        if user_path.startswith("/"):
            notebook_path = source_root / user_path.lstrip("/")
        else:
            notebook_path = including_file.parent / user_path

        # Compute _generated path
        try:
            relative_notebook_path = notebook_path.relative_to(source_root)
        except ValueError as e:
            raise self.error(f"[include_notebook] ❌ Could not resolve relative path: {notebook_path} (from {source_root})")

        generated_rst_path = generated_root / relative_notebook_path.with_suffix(".rst")
        resolved_path = generated_rst_path.resolve()

        print(f"[include_notebook] Including: {resolved_path}")

        if not resolved_path.exists():
            raise self.error(f"[include_notebook] ❌ File not found: {resolved_path}")

        self.arguments[0] = str(resolved_path)
        return super().run()
def setup(app):
    app.add_directive("include_notebook", IncludeNotebook)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }