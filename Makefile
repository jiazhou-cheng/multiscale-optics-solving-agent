.PHONY: test validate list-models list-couplers clean

test:
	pytest -q

validate:
	python scripts/validate_package.py
	python -m multiscale_optics_agent.cli validate examples/graphs/ray_to_wave.yaml

list-models:
	python -m multiscale_optics_agent.cli list-models

list-couplers:
	python -m multiscale_optics_agent.cli list-couplers

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
