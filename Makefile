.PHONY: train ui test lint format clean

CONFIG ?= configs/base.yaml

train:
	python scripts/train.py --config $(CONFIG)

ui:
	streamlit run src/ui/app.py

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v --timeout=120

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

check: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf outputs/ logs/ .ruff_cache/
