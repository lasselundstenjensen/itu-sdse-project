#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = new_customers_classifier
PYTHON_INTERPRETER = python
PIP = pip

# Pointing to the specific location of mlruns based on your tree
MLFLOW_URI = $(PROJECT_NAME)/mlruns

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: help install clean data lint format test train tune deploy docs docs-serve mlflow-ui

## Display this help text
help:
	@echo "$(PROJECT_NAME) - Customer Classifier Makefile"
	@echo "------------------------------------------------"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

#################################################################################
# SETUP & DEPENDENCIES                                                          #
#################################################################################

## Install dependencies from pyproject.toml in editable mode
install:
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -e .[dev]
	
## Install pre-commit hooks (if you use them)
install-hooks:
	pre-commit install

#################################################################################
# DATA & DVC                                                                    #
#################################################################################

## Pull latest version of raw_data.csv from remote storage
data-pull:
	dvc pull raw_data.csv.dvc

## track changes to raw_data.csv
data-add:
	dvc add $(PROJECT_NAME)/raw_data.csv
	git add raw_data.csv.dvc .gitignore

#################################################################################
# CODE QUALITY & TESTING                                                        #
#################################################################################

## Format code (Black)
format:
	black $(PROJECT_NAME) tests

## Lint code (Flake8)
lint:
	flake8 $(PROJECT_NAME)

## Run tests (Pytest) on tests/ folder
test:
	pytest tests/

#################################################################################
# MODEL PIPELINE                                                                #
#################################################################################

## Run Preprocessing
preprocess:
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).preprocessing

## Run Model Selection / Hyperparameter Tuning
tune:
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).model_selection

## Run Main Training (Model Dev)
train:
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).model_dev

## Run Deployment Script
deploy:
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).deploy

#################################################################################
# MLFLOW & VISUALIZATION                                                        #
#################################################################################

## Launch MLflow UI (Pointing to the internal mlruns folder)
mlflow-ui:
	@echo "Starting MLflow UI pointing to $(MLFLOW_URI)..."
	mlflow ui --backend-store-uri $(MLFLOW_URI)

#################################################################################
# DOCUMENTATION (MkDocs)                                                        #
#################################################################################

## Serve documentation locally
docs-serve:
	mkdocs serve -f docs/mkdocs.yml

## Build documentation site
docs-build:
	mkdocs build -f docs/mkdocs.yml

#################################################################################
# CLEANUP                                                                       #
#################################################################################

## Delete compiled Python files and temporary build artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf site/