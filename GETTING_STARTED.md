# Getting Started with PsyNeuLink (Local Development)

This guide walks you through setting up a local development environment
for PsyNeuLink using the provided Makefile.

## Prerequisites

- **Python 3.8+** (check with `python3 --version`)
- **pip** (bundled with Python 3.4+)
- **make** (pre-installed on macOS/Linux; on Windows use WSL or Git Bash)
- **Git** (to clone the repository)

## Quick Start

```bash
# 1. Clone the repository (if you haven't already)
git clone https://github.com/PrincetonUniversity/PsyNeuLink.git
cd PsyNeuLink

# 2. Create a virtual environment and install PsyNeuLink
make install

# 3. Activate the virtual environment
source .venv/bin/activate
```

That's it — PsyNeuLink is now installed in editable mode and ready to use.

## Step-by-Step Walkthrough

### 1. Create the virtual environment

```bash
make venv
```

This creates a `.venv/` directory with an isolated Python environment and
upgrades pip, setuptools, and wheel to their latest versions.

### 2. Install PsyNeuLink

Choose the install target that matches your use case:

| Command                | What you get                                           |
|------------------------|--------------------------------------------------------|
| `make install`         | Core PsyNeuLink (editable install)                     |
| `make install-dev`     | Core + dev/test tools (pytest, linting, etc.)          |
| `make install-tutorial`| Core + tutorial dependencies (Jupyter, matplotlib)     |
| `make install-all`     | Core + dev + tutorial (everything)                     |

Each target automatically creates the virtual environment if it doesn't
exist yet, so you can skip `make venv` and go straight to, e.g.,
`make install-dev`.

### 3. Activate the virtual environment

After installation, activate the environment in your shell:

```bash
source .venv/bin/activate
```

Your prompt will change to show `(.venv)`, confirming the environment is
active. From here you can run `python`, `pytest`, `jupyter`, etc. using
the installed packages.

To deactivate later:

```bash
deactivate
```

### 4. Verify the installation

```bash
make info
```

This prints the Python version and installed PsyNeuLink package details.

You can also test it interactively:

```bash
source .venv/bin/activate
python -c "import psyneulink as pnl; print(pnl.__version__)"
```

## Working with Notebooks

### Your own notebooks

The `notebooks/` directory is the place to put your own notebooks.
A starter notebook (`Getting Started.ipynb`) is included to verify your
installation and demonstrate basic usage.

```bash
make jupyter
```

This opens Jupyter in the `notebooks/` directory. Click
**Getting Started.ipynb** to begin, or create a new notebook from there.

### The official tutorial

```bash
make tutorial
```

This opens the PsyNeuLink tutorial notebook (`tutorial/PsyNeuLink Tutorial.ipynb`)
directly in your browser.

## Running Tests

```bash
make test
```

This installs dev dependencies (if needed) and runs the test suite with
pytest. Tests run in parallel by default via pytest-xdist.

## Customization

You can override the Python interpreter or virtual environment location:

```bash
# Use a specific Python version
make PYTHON=python3.11 install

# Use a custom venv directory
make VENV_DIR=~/envs/pnl install
```

## Cleaning Up

```bash
# Remove just the virtual environment
make clean-venv

# Remove venv + all build artifacts (__pycache__, .egg-info, etc.)
make clean
```

## Troubleshooting

- **`python3: command not found`** — Ensure Python 3.8+ is installed and
  on your PATH. On some systems you may need to use
  `make PYTHON=python install`.

- **`make: command not found`** — Install make via your package manager
  (`xcode-select --install` on macOS, `sudo apt install make` on Ubuntu).

- **Permission errors during install** — Never use `sudo` with the
  Makefile. The virtual environment is user-local and should not require
  elevated privileges.

- **Dependency conflicts** — Run `make clean && make install` to start
  fresh.

For additional help, email psyneulinkhelp@princeton.edu or file an issue
at https://github.com/PrincetonUniversity/PsyNeuLink/issues.
