.PHONY: install test smoke paper-small paper clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[paper]"

test:
	$(PYTHON) -m pytest -q

smoke:
	arctic-sci1 --profile smoke --seeds 1 --controllers pid precision ice_aware full no_cbf no_cvar no_observer no_fallback --no-traces

paper-small:
	arctic-sci1 --profile paper --seeds 5 --controllers pid smc precision ice_aware full no_cbf no_cvar no_observer no_fallback

paper:
	arctic-sci1 --config configs/sci1/sci1_submission.yaml

clean:
	rm -rf results/sci1_submission/smoke_* results/sci1_submission/paper_*
