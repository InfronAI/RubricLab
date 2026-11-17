PYTHON ?= .venv/bin/python
CONFIG ?= configs/m1.json
DATA ?= data/m1
RUN ?= experiments/local

.PHONY: setup doctor prepare run resume test report demo
setup:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install -e '.[dev,demo]'
doctor:
	$(PYTHON) -m openrubrics_lab doctor
prepare:
	$(PYTHON) -m openrubrics_lab prepare --config $(CONFIG) --output $(DATA)
run:
	$(PYTHON) -m openrubrics_lab run --config $(CONFIG) --data $(DATA) --output $(RUN)
resume:
	$(PYTHON) -m openrubrics_lab run --config $(CONFIG) --data $(DATA) --output $(RUN) --resume
test:
	$(PYTHON) -m pytest -q
	.venv/bin/ruff check src tests
report:
	$(PYTHON) -m openrubrics_lab report $(RUN)
demo:
	$(PYTHON) -m openrubrics_lab demo --run $(RUN)
