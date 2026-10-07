.DEFAULT_GOAL := help
PYTHON ?= python3
EXP ?= block0_qproj
RUN_ID ?=
.PHONY: help run dry-run runs
help:
	@printf '%s\n' 'make run EXP=block0_qproj [RUN_ID=my-run]' \
		'make dry-run EXP=block0_qproj' 'make runs'
run:
	$(PYTHON) scripts/run_experiment.py $(EXP) $(if $(RUN_ID),--run-id $(RUN_ID),)
dry-run:
	$(PYTHON) scripts/run_experiment.py $(EXP) --dry-run $(if $(RUN_ID),--run-id $(RUN_ID),)
runs:
	$(PYTHON) scripts/list_runs.py
