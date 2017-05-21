init:
	pip install -r REQUIREMENTS.txt

test:
	nosetests tests/

.PHONY: init tests