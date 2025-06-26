import subprocess
from pathlib import Path
import re

# Configuration
SRC_NOTEBOOKS = Path(__file__).resolve().parent.parent
DST_GENERATED = Path(__file__).resolve().parent.parent / "_generated"
GITHUB_REPO = "PrincetonUniversity/PsyNeuLink"  # replace with your GitHub username/repo
BRANCH = "main"
NOTEBOOK_BRANCH = "doc/additional_tutorials" # "notebooks-gh"


def colab_badge(nb_path: Path):
    """
    Generate a Colab badge pointing to the correct GitHub location of the generated notebook.

    Args:
        nb_path: Path relative to source/, e.g., Path("introductory_material/quickstart/notebooks/quickstart.ipynb")

    Returns:
        str: The reStructuredText for embedding the Colab badge.
    """
    colab_path = f"docs/source/_generated/colab_notebooks/{nb_path.as_posix()}"

    url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{NOTEBOOK_BRANCH}/{colab_path}"
    return f"""
.. raw:: html

   <a href="{url}" target="_blank">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
   </a>

"""


def generate_rst_from_ipynb():
    for ipynb_file in SRC_NOTEBOOKS.rglob("*.ipynb"):
        if "_generated" in ipynb_file.parts:
            continue

        rel_path = ipynb_file.relative_to(SRC_NOTEBOOKS)  # e.g. ch1/foo.ipynb
        output_dir = DST_GENERATED / rel_path.parent  # e.g. docs/_generated/ch1/
        output_basename = ipynb_file.stem  # "foo"

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run([
                "jupyter", "nbconvert",
                "--to", "rst",
                "--execute",
                "--output", output_basename,
                "--output-dir", str(output_dir),
                str(ipynb_file)
            ], check=True)
        except subprocess.CalledProcessError as e:
            subprocess.run([
                "jupyter", "nbconvert",
                "--to", "rst",
                "--output", output_basename,
                "--output-dir", str(output_dir),
                str(ipynb_file)
            ], check=True)

        # Prepend Colab badge
        rst_file = output_dir / f"{output_basename}.rst"
        badge = colab_badge(rel_path)

        with open(rst_file, "r+", encoding="utf-8") as f:
            content = f.read()
            content = convert_double_backticks_to_single(content)

            f.seek(0)
            f.write(badge + content)


def convert_double_backticks_to_single(content):
    """
    Example:
        >>> convert_double_backticks_to_single("Use ``Linear`` to create a linear layer.")
        'Use `Linear` to create a linear layer.'

        >>> convert_double_backticks_to_single("Use ``Linear <linear>`` and ``ReLU`` for activation.")
        'Use `Linear <liner>` and `ReLU` for activation.'

    """
    # Only convert short, one-word literals (like ``Linear``) to `Linear`
    return re.sub(r'``([^`]+?)``', r'`\1`', content)


if __name__ == "__main__":
    print("[generate_notebook_rst] Converting notebooks to RST...")
    generate_rst_from_ipynb()
    print("[generate_notebook_rst] Done.")
