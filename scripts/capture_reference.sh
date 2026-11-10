#!/bin/bash

# Execute from project root (as per convention)

#---- activate virtual environment
source .venv/bin/activate

#---- run the two notebooks to obtain reference results
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/01_training.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/02_deployment.ipynb

#---- name the reference results directory after the current commit id
COMMIT_HASH=$(git rev-parse HEAD)
cd tests/fixtures
mkdir "$COMMIT_HASH"

#---- copy the run's artifacts to the reference results directory
cp -r ../../artifacts "$COMMIT_HASH/"
echo "copied reference results for commit id $COMMIT_ID"
