.PHONY: lint test ci ci-docker experiment figures dct

lint:
	uv run black --check .
	uv run isort --check-only .
	uv run flake8 .

test:
	uv run pytest -q

ci: lint test

ci-docker:
	docker run --rm -v "$(CURDIR)":/src -w /src python:3.12-slim \
		sh -c "pip install -q uv && export UV_PROJECT_ENVIRONMENT=/tmp/venv && \
		uv run black --check . && uv run isort --check-only . && \
		uv run flake8 . && uv run pytest -q"

experiment:
	uv run python experiment.py

figures:
	uv run python figures.py

dct:
	cd experiments/dct && uv run python experiment.py && uv run python figures.py
