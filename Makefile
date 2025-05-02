# Run everything: lint, format, static checks, tests
all: lint format check test

# Run full test suite with verbose output and generate HTML report
test:
	pytest -q tests/ --html=report.html --self-contained-html


# Run linter (Ruff) on all source and test code
lint:
	ruff check naviflow_collocated tests utils main.py launch_job.py

# Format code using Ruff's formatter
format:
	ruff format naviflow_collocated tests utils main.py launch_job.py

# Check only for lint issues without auto-fixing
check:
	ruff check naviflow_collocated tests utils main.py launch_job.py

# Clean up bytecode, cache files, and artifacts
clean:
	find . -type d -name '__pycache__' -exec rm -r {} + ;\
	find . -type f -name '*.pyc' -delete ;\
	rm -rf .pytest_cache .ruff_cache .coverage report.html

# Export current conda environment (without build info or machine-specific paths)
env:
	conda env export --no-builds | grep -v "prefix:" > environment.yaml

# Mark targets as phony (not real files)
.PHONY: all test test-nv lint format check clean env