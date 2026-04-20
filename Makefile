# PsyNeuLink Makefile
# Handles virtual environment setup, installation, and common dev tasks.

SHELL := /bin/bash
VENV_DIR ?= .venv
PYTHON ?= python3
PIP := $(VENV_DIR)/bin/pip
PYTHON_VENV := $(VENV_DIR)/bin/python
JUPYTER := $(VENV_DIR)/bin/jupyter

.PHONY: help venv install install-dev install-tutorial install-all \
        jupyter test clean-venv clean info

help: ## Show this help message
	@echo "PsyNeuLink Development Makefile"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Configuration:"
	@echo "  VENV_DIR=$(VENV_DIR)  (override with: make VENV_DIR=path/to/venv <target>)"
	@echo "  PYTHON=$(PYTHON)      (override with: make PYTHON=python3.11 <target>)"

# --- Virtual environment ---

venv: $(VENV_DIR)/bin/activate ## Create virtual environment

$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel
	@echo ""
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo "Activate it with: source $(VENV_DIR)/bin/activate"

# --- Installation targets ---

install: venv ## Install PsyNeuLink (editable) into the venv
	$(PIP) install -e .

install-dev: venv ## Install PsyNeuLink with dev/test dependencies
	$(PIP) install -e ".[dev]"

install-tutorial: venv ## Install PsyNeuLink with tutorial dependencies (includes jupyter)
	$(PIP) install -e ".[tutorial]"

install-all: venv ## Install PsyNeuLink with all optional dependencies
	$(PIP) install -e ".[dev,tutorial]"

# --- Running ---

jupyter: install-tutorial ## Launch Jupyter notebook server
	$(JUPYTER) notebook

tutorial: install-tutorial ## Open the PsyNeuLink tutorial notebook
	$(JUPYTER) notebook "tutorial/PsyNeuLink Tutorial.ipynb"

# --- Testing ---

test: install-dev ## Run the test suite
	$(VENV_DIR)/bin/pytest tests/

# --- Cleanup ---

clean-venv: ## Remove the virtual environment
	rm -rf $(VENV_DIR)

clean: clean-venv ## Full clean (venv + build artifacts)
	rm -rf build/ dist/ *.egg-info psyneulink.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

# --- Info ---

info: venv ## Show installed PsyNeuLink version and Python info
	@$(PYTHON_VENV) --version
	@$(PIP) show psyneulink 2>/dev/null || echo "PsyNeuLink is not installed yet. Run: make install"
