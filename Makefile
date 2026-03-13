environment:
	@poetry config virtualenvs.in-project true
	@poetry install

clean:
	@rm -rf .venv poetry.lock
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

test:
	@.venv/bin/python pytest -v tests

clear_pycache:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

quality:
	@pre-commit run --all-files
