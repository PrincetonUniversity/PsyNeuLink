import nbformat
import re
from bs4 import BeautifulSoup
from pathlib import Path

# === Config ===
SRC_NOTEBOOKS = Path(__file__).resolve().parent.parent # adjust if needed
HTML_BUILD_DIR = Path(__file__).resolve().parent.parent.parent / 'build/html/_generated'
DST_NOTEBOOKS = Path(__file__).resolve().parent.parent / '_generated/colab_notebooks'

BASE_DOC_URL = "https://princetonuniversity.github.io/PsyNeuLink"

def make_absolute_href(href: str, base_url: str) -> str:
    """Convert relative href to absolute using base URL."""
    return re.sub(r'^(\.\./)+', f'{base_url}/', href)

def extract_link_map(html_file: Path) -> dict:
    """Extract map like {'Linear': 'https://.../TransferFunctions.html#...'} from an HTML doc."""
    if not html_file.exists():
        return {}

    soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")
    symbol_links = {}

    for tag in soup.find_all("a", class_="reference internal"):
        inner = tag.find("code", class_="xref")
        if inner:
            symbol = inner.get_text(strip=True)
            href = tag.get("href")
            if symbol and href:
                absolute = make_absolute_href(href, BASE_DOC_URL)
                symbol_links[symbol] = absolute
    return symbol_links

def patch_notebook(ipynb_path: Path, out_path: Path, symbol_map: dict):
    nb = nbformat.read(ipynb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = re.sub(
                r'`([^`<>]+?)\s*<[^<>]+?>`',
                lambda m: (
                    f"[{m.group(1)}]({symbol_map[m.group(1).strip()]})"
                    if m.group(1).strip() in symbol_map else m.group(0)
                ),
                cell.source
            )

            # Then: Handle `Label` → [Label](url)
            cell.source = re.sub(
                r'`([A-Za-z_][A-Za-z0-9_]*)`',
                lambda m: (
                    f"[{m.group(1)}]({symbol_map[m.group(1)]})"
                    if m.group(1) in symbol_map else m.group(0)
                ),
                cell.source
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)

if __name__ == "__main__":
    for ipynb_file in SRC_NOTEBOOKS.rglob("*.ipynb"):
        print('Processing:', ipynb_file)
        if "_generated" in ipynb_file.parts:
            continue

        rel_path = ipynb_file.relative_to(SRC_NOTEBOOKS)
        html_path = HTML_BUILD_DIR / rel_path.with_suffix(".html")
        out_path = DST_NOTEBOOKS / rel_path

        print(f"Processing: {rel_path}")
        symbol_map = extract_link_map(html_path)
        patch_notebook(ipynb_file, out_path, symbol_map)
