.PHONY: test test-tutorial test-agent-benchmark agent-benchmark validate list-models list-couplers clean

# Every target runs through ./run.sh, which is the only supported entry point
# (AGENTS.md, "Execution Environment -- Container Only").
#
# These targets used to invoke bare `pytest` and `python`. That was not merely a
# policy violation: measured by CHE-85, `make test-agent-benchmark` aborted with
# a collection error in 0.11 s on the host, because the host has none of the
# dependencies. A target that cannot run is worse than one that runs the wrong
# way -- it reads as a supported command right up until someone tries it.
#
# `make` is therefore a host-side convenience over the container, not a
# container-internal alias. If you are already inside the container, run the
# underlying command directly; ./run.sh from within a container will not work.

# The default active suite (CHE-67). `testpaths = ["tests"]` is what keeps the
# on-demand tutorial suite and the archived generations out of this.
test:
	./run.sh pytest -q

# The on-demand tutorial suite: ~33 min, 60 reproductions of upstream
# Optiland/Chromatix tutorials against the pinned installs. A dependency-pin
# regression gate, not a PR gate -- run it when a pin or docker/ changes, after a
# substantial solver-integration change, or as a weekly sweep.
# See tests_tutorial/README.md.
test-tutorial:
	./run.sh pytest -q tests_tutorial

# The CHE-71 V1 agent benchmark, graded against its own reference and negative
# participants. Deterministic and seconds long -- it is the benchmark's gate on
# itself, not a run against an agent. See benchmarks/agents/README.md.
test-agent-benchmark:
	./run.sh pytest -q benchmarks/agents

# One end-to-end run of the V1 suite. `PARTICIPANT` selects who is graded:
# `reference` (the default), `broken:<mode>`, or `command:<argv>` for a real
# agent. TRIALS defaults to the declared 3.
PARTICIPANT ?= reference
TRIALS ?= 3
agent-benchmark:
	./run.sh python -m agent.benchmark_suite \
	    --suite v1 --trials $(TRIALS) --participant $(PARTICIPANT) \
	    --context-policy per-task --output outputs/che71_agent_v1

validate:
	./run.sh python scripts/validate_package.py
	./run.sh python -m cli validate examples/graphs/ray_to_wave.yaml

list-models:
	./run.sh python -m cli list-models

list-couplers:
	./run.sh python -m cli list-couplers

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
