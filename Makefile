.PHONY: install lint format typecheck test cov results paper demo clean

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
	$(PY) scripts/run_benchmark.py --config configs/demo.yaml --out paper/generated
	$(PY) scripts/run_benchmark.py --config configs/cord.yaml --out paper/generated/cord
	$(PY) scripts/run_benchmark.py --config configs/funsd.yaml --out paper/generated/funsd
	$(PY) scripts/grouped_conformal_experiment.py
	$(PY) scripts/grounded_conformal_real.py
	$(PY) scripts/annotator_agreement.py
	$(PY) scripts/tables_to_latex.py paper/generated

# Compile the paper (needs a LaTeX toolchain); tables come from `make results`.
paper:
	$(PY) scripts/grouped_conformal_experiment.py
	$(PY) scripts/grounded_conformal_real.py
	$(PY) scripts/annotator_agreement.py
	$(PY) scripts/tables_to_latex.py paper/generated
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

demo:
	streamlit run ui/streamlit_app.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
