.PHONY: test test-slow test-serial test-gpu check-arch clean

# Every target runs through ./run.sh, which is the only supported entry point
# (AGENTS.md, "Execution Environment").
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
#
# Six targets were deleted with the reference implementation, for exactly the
# reason above: `test-tutorial`, `test-agent-benchmark`, `agent-benchmark`,
# `validate`, `list-models` and `list-couplers` all invoked `tests_tutorial/`,
# `benchmarks/agents/`, `scripts/validate_package.py` or `python -m cli`, none of
# which exist any more.

# The default suite. Since the deletion this is the two architecture gates and
# nothing else: 32 tests in 0.06 s. `-n 8 --dist loadfile` came out of
# addopts with the suite that justified it; see the measurement in pyproject.toml.
test:
	./run.sh pytest -q

# The `slow` selection, which addopts deselects from the default gate. It selects
# nothing today -- every test that carried the marker was deleted -- and is kept
# because the marker is still the declared home for expensive numerical
# characterization, which R02 onward will write. Run it before merging a change to
# coupler numerics, sampling densities, estimator weights or a benchmark family,
# and say in the PR that you did.
test-slow:
	./run.sh pytest -q -m "slow"

# The whole tree with nothing deselected. The arbiter when a failure is suspected
# to be a cross-test interaction rather than a real defect. Also the honest
# "everything" command. `-n 0` is now the default rather than an override, so this
# differs from `make test` only by selecting `slow` as well.
test-serial:
	./run.sh pytest -q -m "" -n 0

# The opt-in GPU suite. Needs the separately-built `agent_solver_gpu` image and a
# device. Select the device with MOA_GPUS (AGENTS.md prefers 6 and 7):
# `MOA_GPUS=device=6 make test-gpu`. Like `test-slow` it selects nothing yet: the
# GPU tests were deleted with the old tree, and so was the conftest hook that
# skipped them when no device was attached. Whichever ticket writes the first new
# GPU test owes that gating logic again -- `pre-rewrite-2026-08-30` has the
# version to re-derive it from.
#
# `-o addopts=` replaces the default selection wholesale. It no longer has
# sharding to undo, but it still drops `-m "not slow"`, and keeping the override
# means this command does not silently change meaning the next time addopts does.
test-gpu:
	./run.sh --gpu pytest -q -o addopts="-ra" -m gpu

# The two architecture gates (CHE-171 / R01.1), run directly for a readable
# report. Both also run in the default suite via `tests/unit/`, which is what
# makes them a gate rather than a command someone remembers; this target is for
# the report, and for seeing *why* a gate failed without reading pytest
# assertion output.
check-arch:
	./run.sh python scripts/check_dependencies.py
	./run.sh python scripts/class_budget.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
