VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

.PHONY: venv install run test

venv:
	python -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(PYTHON) -m streamlit run streamlit_app.py

test: install
	$(VENV)/bin/pytest -q tests/test_integration_weaviate.py
