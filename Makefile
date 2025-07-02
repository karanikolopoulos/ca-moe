environment:
	@poetry config virtualenvs.in-project 1
# @poetry config virtualenvs.path "$$(conda info --base)/envs"
	@poetry install

clear_pycache:
	@find . | grep -E "(pycache|.pyc|.pyo$)" | xargs rm -rf

diagnose:
	@pre-commit run --all-files

count_loc:
	@find . -name '*.py' | xargs wc -l | sort -nr