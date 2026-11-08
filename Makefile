.PHONY: test lint fmt demo clean

test:
	python3 -m pytest

lint:
	python3 -m ruff check .

fmt:
	python3 -m ruff format .

demo:
	python3 scripts/random_walk_demo.py --config configs/debug_tiny.yaml --steps 40 --render

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
