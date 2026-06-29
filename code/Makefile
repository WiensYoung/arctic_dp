.PHONY: install test smoke paper-small ablation runtime data-sensitivity paper clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[paper]"

test:
	$(PYTHON) -m pytest -q

smoke:
	arctic-sci1 --config configs/sci1/sci1_smoke.yaml

paper-small:
	arctic-sci1 --config configs/sci1/sci1_paper_small.yaml

ablation:
	arctic-sci1 --config configs/sci1/sci1_ablation.yaml

runtime:
	arctic-sci1 --config configs/sci1/sci1_runtime.yaml

data-sensitivity:
	arctic-sci1 --config configs/sci1/sci1_data_sensitivity.yaml

paper:
	arctic-sci1 --config configs/sci1/sci1_paper_full.yaml

clean:
	rm -rf results/sci1_submission/smoke_* results/sci1_submission/paper_*
