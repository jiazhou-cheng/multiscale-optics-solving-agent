.PHONY: test test-slow test-serial test-gpu test-tutorial test-agent-benchmark agent-benchmark check-arch validate list-models list-couplers clean

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
# on-demand tutorial suite and the archived generations out of this. CHE-140 put
# `-m "not slow" -n N --dist loadfile` in pyproject's addopts, so this is ~60 s
# rather than the ~375 s it was. CHE-171 re-measured N and moved it from 12 to 8:
# the suite saturates at ~60 s from eight workers on, so the extra four bought
# nothing on a shared host. The measurement table is in that addopts comment.
test:
	./run.sh pytest -q

# The `slow` selection: expensive numerical characterization and convergence
# measurement, deselected from the default gate by CHE-140. ~39 tests, ~3.5 min.
#
# Not optional work. These are the coupler exit gates -- the gradient-bias
# characterization (CHE-28), the wave-to-ray Monte-Carlo convergence fits, and
# the B2 stochastic-transition benchmark with its five-control battery. Run this
# before merging any change to coupler numerics, sampling densities, estimator
# weights or a benchmark family, and say in the PR that you did.
#
# `-p no:cacheprovider` is not used here on purpose: a failure in this suite is
# worth `--lf`.
test-slow:
	./run.sh pytest -q -m "slow"

# The whole default tree with nothing deselected and no sharding. The arbiter
# when a failure is suspected to be a cross-test interaction or a worker
# artifact rather than a real defect -- if it reproduces here, it is real.
# ~6 min. Also the honest "everything" command.
test-serial:
	./run.sh pytest -q -m "" -n 0

# The opt-in GPU suite. Needs the separately-built `agent_solver_gpu` image and
# a device; see docs/testing/gpu_environment.md. Select the device with MOA_GPUS
# (AGENTS.md prefers 6 and 7): `MOA_GPUS=device=6 make test-gpu`.
#
# `-o addopts=` replaces the default `-m "not slow" -n 8 --dist loadfile`
# wholesale, and the sharding is the reason rather than the marker. There is one
# device: eight workers would each import jax and open their own CUDA context on
# it, and JAX preallocates a large fraction of device memory per process, so the
# second worker OOMs on a GPU the first one is holding. AGENTS.md's shared-server
# policy is explicit that this is not a throughput decision -- one workload per
# GPU. This is the same class of defect CHE-140 found in the swap guard, where a
# resource mechanism degraded silently under sharding.
#
# Overriding addopts rather than appending `-n 0` also keeps this runnable on the
# current `agent_solver_gpu` image, which was built before CHE-140 pinned
# pytest-xdist and therefore does not have the plugin to accept `-n` at all.
test-gpu:
	./run.sh --gpu pytest -q -o addopts="-ra" -m gpu

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

# The two architecture gates of the new tree (CHE-171 / R01.1), run directly for a
# readable report. Both also run in the default suite via `tests/unit/`, which is
# what makes them a gate rather than a command someone remembers; this target is
# for the report, and for seeing *why* a gate failed without reading pytest
# assertion output.
check-arch:
	./run.sh python scripts/check_dependencies.py
	./run.sh python scripts/class_budget.py

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
