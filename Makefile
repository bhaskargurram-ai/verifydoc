.PHONY: install lint format typecheck test cov results demo clean

PY ?= python

install:
	pip install -e ".[dev,pdf,ui]"

lint:
	ruff check verifydoc tests
	black --check verifydoc tests

format:
	ruff check --fix verifydoc tests
	black verifydoc tests

typecheck:
	mypy verifydoc

test:
	pytest

cov:
	pytest --cov=verifydoc --cov-report=term-missing

# Regenerate every table/figure in paper/ from pinned configs + seeds.
# (cord/funsd download their source data once into data/, then cache)
results:
	$(PY) scripts/run_benchmark.py --config configs/demo.yaml --out results
	$(PY) scripts/run_benchmark.py --config configs/cord.yaml --out results/cord
	$(PY) scripts/run_benchmark.py --config configs/funsd.yaml --out results/funsd
	$(PY) scripts/grouped_conformal_experiment.py
	$(PY) scripts/grounded_conformal_real.py
	$(PY) scripts/annotator_agreement.py


demo:
	streamlit run ui/streamlit_app.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
