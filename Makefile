.PHONY: test test-tutorial test-agent-benchmark agent-benchmark validate list-models list-couplers clean

# The default active suite (CHE-67). `testpaths = ["tests"]` is what keeps the
# on-demand tutorial suite and the archived generations out of this.
test:
	pytest -q

# The on-demand tutorial suite: ~33 min, 60 reproductions of upstream
# Optiland/Chromatix tutorials against the pinned installs. A dependency-pin
# regression gate, not a PR gate -- run it when a pin or docker/ changes, after a
# substantial solver-integration change, or as a weekly sweep.
# See tests_tutorial/README.md.
test-tutorial:
	pytest -q tests_tutorial

# The CHE-71 V1 agent benchmark, graded against its own reference and negative
# participants. Deterministic and seconds long -- it is the benchmark's gate on
# itself, not a run against an agent. See benchmarks_agent/README.md.
test-agent-benchmark:
	pytest -q benchmarks_agent

# One end-to-end run of the V1 suite. `PARTICIPANT` selects who is graded:
# `reference` (the default), `broken:<mode>`, or `command:<argv>` for a real
# agent. TRIALS defaults to the declared 3.
PARTICIPANT ?= reference
TRIALS ?= 3
agent-benchmark:
	python -m agent.benchmark_suite \
	    --suite v1 --trials $(TRIALS) --participant $(PARTICIPANT) \
	    --context-policy per-task --output outputs/che71_agent_v1

validate:
	python scripts/validate_package.py
	python -m cli validate examples/graphs/ray_to_wave.yaml

list-models:
	python -m cli list-models

list-couplers:
	python -m cli list-couplers

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
