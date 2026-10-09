#!/usr/bin/env bash

set -e

echo "========================================"
echo "1. Checking Git status"
echo "========================================"
git status


echo
echo "========================================"
echo "2. Compiling key Python files"
echo "========================================"

python -m py_compile \
  src/pipeline/model_training_pipeline.py \
  src/pipeline/run_model_training_pipeline.py \
  src/models/train_random_forest.py \
  src/models/run_random_forest_training.py \
  src/models/train_xgboost.py \
  src/models/run_xgboost_training.py


echo
echo "========================================"
echo "3. Compiling Vertex AI pipeline"
echo "========================================"

python -m src.pipeline.model_training_pipeline


echo
echo "========================================"
echo "4. Checking for stale file references"
echo "========================================"

if grep -R -n \
  -e "requirements-training.txt" \
  -e "random_forest_dockerfile" \
  -e "xgboost_dockerfile" \
  -e "random_forest_training_pipeline.py" \
  -e "run_random_forest_pipeline.py" \
  . \
  --exclude-dir=.git \
  --exclude="validate_repo.sh"
then
  echo
  echo "WARNING: stale references were found."
else
  echo "No stale references found."
fi


echo
echo "========================================"
echo "5. Checking Docker files"
echo "========================================"

test -f docker/random_forest/Dockerfile
test -f docker/random_forest/requirements.txt

test -f docker/xgboost/Dockerfile
test -f docker/xgboost/requirements.txt

echo "Docker files found."


echo
echo "========================================"
echo "6. Checking Cloud Build configs"
echo "========================================"

test -f cloudbuild-random-forest.yaml
test -f cloudbuild-xgboost.yaml

echo "Cloud Build configs found."


echo
echo "========================================"
echo "Repository validation completed."
echo "========================================"
