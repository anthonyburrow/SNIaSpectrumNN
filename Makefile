.PHONY: help venv install test clean

PYTHON := python
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
PYTEST_FLAGS := -sv
PYTORCH_FLAGS := --index-url https://download.pytorch.org/whl/cu130

IN_VENV := $(shell python -c 'import sys; print(int(sys.prefix != sys.base_prefix))')

help:
	@echo "Available targets:"
	@echo "  make venv     - Create a virtual environment in .venv/"
	@echo "  make install  - Install the project (auto-detects venv)"
	@echo "  make test     - Run pytest tests"
	@echo "  make clean    - Remove build artifacts and venv"

venv:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	else \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "Virtual environment created at $(VENV_DIR)"; \
		echo "Activate it with: source $(VENV_DIR)/bin/activate"; \
	fi

install:
	@if [ -d "$(VENV_DIR)" ] && [ "$(IN_VENV)" = "0" ]; then \
		echo "Virtual environment detected but not activated."; \
		echo "Installing in venv..."; \
		$(VENV_PIP) install --upgrade pip; \
		$(VENV_PIP) install -e .[dev]; \
		$(VENV_PIP) install -e .[torch] $(PYTORCH_FLAGS); \
	elif [ "$(IN_VENV)" = "1" ]; then \
		echo "Installing in active virtual environment..."; \
		pip install --upgrade pip; \
		pip install -e .[dev]; \
		pip install -e .[torch] $(PYTORCH_FLAGS); \
	else \
		echo "No virtual environment detected."; \
		echo "Installing in system/user Python..."; \
		pip install --upgrade pip; \
		pip install -e .[dev]; \
		pip install -e .[torch] $(PYTORCH_FLAGS); \
	fi
	@echo "Installation complete!"

test:
	@if [ -d "$(VENV_DIR)" ] && [ "$(IN_VENV)" = "0" ]; then \
		echo "Running tests in venv..."; \
		$(VENV_PYTHON) -m pytest $(PYTEST_FLAGS); \
	elif [ "$(IN_VENV)" = "1" ]; then \
		echo "Running tests in active virtual environment..."; \
		pytest $(PYTEST_FLAGS); \
	else \
		echo "Running tests..."; \
		pytest $(PYTEST_FLAGS); \
	fi

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf $(VENV_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	@echo "Clean complete!"
