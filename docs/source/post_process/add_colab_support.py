"""
This script is called by `make html` to:
- Inject a Colab badge and download into built HTML tutorial pages
- Rewrite relative links for images in downloadable .ipynb files to point to
  GitHub raw URLs that work in Colab or when downloaded

It only modifies build/html/tutorials/*.html and *.ipynb.
This (for now) only works when links to images are in the form `![alt_text](relative/path/to/image.png)`.

This allows tutorial source notebooks to remain clean and portable.
"""

from pathlib import Path
from bs4 import BeautifulSoup
import nbformat
import re

# --- CONFIGURATION ---
BUILD_ROOT = Path('build/html/')

SEARCH_DIRS = ['tutorials']

GITHUB_ORG = "PrincetonUniversity"
GITHUB_REPO = "PsyNeuLink"
GITHUB_BRANCH = "gh-pages"

RAW_IMAGE_BASE = f"https://raw.githubusercontent.com/{GITHUB_ORG}/{GITHUB_REPO}/refs/heads/master"

COLAB_URL_BASE = f"https://colab.research.google.com/github/{GITHUB_ORG}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/"

modified_files = []

for search_dir in SEARCH_DIRS:
    build_folder = BUILD_ROOT / search_dir
    if not build_folder.exists() or not build_folder.is_dir():
        print(f"Build folder {build_folder} does not exist. Skipping.")
        continue

    for html_file in build_folder.rglob("*.html"):
        ipynb_path = html_file.with_suffix(".ipynb")

        if not ipynb_path.exists():
            continue  # skip if no corresponding notebook file

        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        rel_ipynb_path = html_file.relative_to(BUILD_ROOT).with_suffix(".ipynb")

        colab_target = COLAB_URL_BASE + str(rel_ipynb_path)

        notebook_name = rel_ipynb_path.name

        badge_html = f'''
        <p>
          <a class="reference external"href="{colab_target}" target="_blank">
            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
          </a>
          <a href="{notebook_name}" download>
            ⬇ Download
          </a>
        </p>
        '''

        main_div = soup.find("div", class_="main-content")

        if not soup.body.find("img", {"alt": "Open in Colab"}):
            if not main_div:
                soup.body.insert(0, BeautifulSoup(badge_html, "html.parser"))
            else:
                main_div.insert(0, BeautifulSoup(badge_html, "html.parser"))
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(str(soup))
            modified_files.append(f"✔ Injected Colab badge into: {html_file.name}")

    # --- PATCH .ipynb FILES ---
    for ipynb_file in build_folder.rglob("*.ipynb"):
        nb = nbformat.read(ipynb_file, as_version=4)
        changed = False

        for cell in nb.cells:
            if cell.cell_type == "markdown":
                def replace_image(match):
                    alt_text, rel_path = match.group(1), match.group(2)
                    # Skip if already an absolute URL
                    if rel_path.startswith("http://") or rel_path.startswith("https://"):
                        return match.group(0)

                    # Path to image relative to the notebook
                    image_path = ipynb_file.parent / rel_path
                    try:
                        # Path to image relative to docs root
                        image_rel_to_build = image_path.relative_to(Path("build/html"))
                        image_path_source = 'docs/source/' / image_rel_to_build
                        github_raw_url = f"{RAW_IMAGE_BASE}/{image_path_source}"
                        return f"![{alt_text}]({github_raw_url})"
                    except ValueError:
                        return match.group(0)  # If not under docs/, don't modify


                new_source = re.sub(
                    r"!\[(.*?)\]\((.*?)\)",
                    replace_image,
                    cell.source
                )

                if new_source != cell.source:
                    cell.source = new_source
                    changed = True

        if changed:
            nbformat.write(nb, ipynb_file)
            modified_files.append(f"✔ Rewrote image paths in: {ipynb_file.relative_to(BUILD_ROOT)}")
