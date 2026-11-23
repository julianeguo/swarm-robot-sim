.PHONY: test lint fmt demo clean

test:
	python3 -m pytest

lint:
	python3 -m ruff check .

fmt:
	python3 -m ruff format .

demo:
	python3 scripts/demo_random.py --config configs/default.yaml --render

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
