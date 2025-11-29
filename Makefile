.PHONY: help venv install install-cpu install-gpu test test-cov clean

PYTHON := python
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
PYTORCH_INDEX ?= cu118
PYTORCH_FLAGS := --index-url https://download.pytorch.org/whl/$(PYTORCH_INDEX)

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
	@echo "[INFO] Installing with PyTorch index $(PYTORCH_INDEX)"
	@if [ -d "$(VENV_DIR)" ] && [ "$(IN_VENV)" = "0" ]; then \
		echo "Virtual environment detected but not activated."; \
		echo "Installing in venv..."; \
		$(VENV_PIP) install --upgrade pip; \
		$(VENV_PIP) install -e .[dev]; \
		$(VENV_PIP) install --no-cache-dir $(PYTORCH_FLAGS) torch; \
	elif [ "$(IN_VENV)" = "1" ]; then \
		echo "Installing in active virtual environment..."; \
		pip install --upgrade pip; \
		pip install -e .[dev]; \
		pip install --no-cache-dir $(PYTORCH_FLAGS) torch; \
	else \
		echo "No virtual environment detected."; \
		echo "Installing in system/user Python..."; \
		pip install --upgrade pip; \
		pip install -e .[dev]; \
		pip install --no-cache-dir $(PYTORCH_FLAGS) torch; \
	fi
	@echo "Installation complete!"

install-cpu:
	@$(MAKE) install PYTORCH_INDEX=cpu

install-gpu:
	@$(MAKE) install PYTORCH_INDEX=cu118

test:
	@if [ -d "$(VENV_DIR)" ] && [ "$(IN_VENV)" = "0" ]; then \
		echo "Running tests in venv..."; \
		$(VENV_PYTHON) -m pytest -m "not gpu"; \
	elif [ "$(IN_VENV)" = "1" ]; then \
		echo "Running tests in active virtual environment..."; \
		pytest -m "not gpu"; \
	else \
		echo "Running tests..."; \
		pytest -m "not gpu"; \
	fi

TEST_FLAGS ?= -m "not gpu and not slow" --cov=SNIaSpectrumNN --cov-report=term-missing

test-cov:
	@if [ -d "$(VENV_DIR)" ] && [ "$(IN_VENV)" = "0" ]; then \
		echo "Running coverage tests in venv..."; \
		$(VENV_PYTHON) -m pytest $(TEST_FLAGS); \
	elif [ "$(IN_VENV)" = "1" ]; then \
		echo "Running coverage tests in active virtual environment..."; \
		pytest $(TEST_FLAGS); \
	else \
		echo "Running coverage tests..."; \
		pytest $(TEST_FLAGS); \
	fi
	@echo "Coverage test run complete"

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
