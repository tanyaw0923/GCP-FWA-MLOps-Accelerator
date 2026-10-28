#!/usr/bin/env bash
set -euo pipefail

# Integration test: verify that each model container uses the exact direct
# dependency versions declared in its requirements.txt.
#
# Prerequisites:
#   - Docker
#   - gcloud authenticated to the project
#   - Artifact Registry Docker authentication:
#       gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
#
# This test intentionally checks only direct dependencies declared by this
# repository. Transitive dependencies may vary unless separately locked.

REGISTRY="${REGISTRY:-us-central1-docker.pkg.dev/fwa-mlops-accelerator-demo/fwa-mlops}"

declare -A IMAGE_REQUIREMENTS=(
  ["random-forest-training:v1.0.0"]="docker/random_forest/requirements.txt"
  ["random-forest-serving:v1.0.1"]="docker/random-forest-serving/requirements.txt"
  ["xgboost-training:v1.0.0"]="docker/xgboost/requirements.txt"
  ["xgboost-serving:v1.0.1"]="docker/xgboost-serving/requirements.txt"
)

normalize_name() {
  printf '%s' "$1"     | sed -E 's/\[[^]]+\]//'     | tr '[:upper:]_' '[:lower:]-'
}

failures=0

for image in "${!IMAGE_REQUIREMENTS[@]}"; do
  requirements_file="${IMAGE_REQUIREMENTS[$image]}"

  echo
  echo "===== $image ====="

  if [[ ! -f "$requirements_file" ]]; then
    echo "FAIL: requirements file not found: $requirements_file"
    failures=$((failures + 1))
    continue
  fi

  freeze_output="$(
    docker run --rm --entrypoint pip       "$REGISTRY/$image"       freeze
  )"

  while IFS= read -r requirement || [[ -n "$requirement" ]]; do
    requirement="${requirement%%#*}"
    requirement="$(printf '%s' "$requirement" | xargs)"

    [[ -z "$requirement" ]] && continue

    if [[ "$requirement" != *"=="* ]]; then
      echo "FAIL: unpinned direct dependency in $requirements_file: $requirement"
      failures=$((failures + 1))
      continue
    fi

    declared_name="${requirement%%==*}"
    declared_version="${requirement#*==}"
    normalized_declared_name="$(normalize_name "$declared_name")"

    installed_line="$(
      printf '%s\n' "$freeze_output"         | awk -F'==' -v target="$normalized_declared_name" '
            function norm(s) {
              gsub(/\[[^]]+\]/, "", s)
              gsub(/_/, "-", s)
              return tolower(s)
            }
            NF == 2 && norm($1) == target { print; exit }
          '
    )"

    if [[ -z "$installed_line" ]]; then
      echo "FAIL: $declared_name is declared but not installed in $image"
      failures=$((failures + 1))
      continue
    fi

    installed_version="${installed_line#*==}"

    if [[ "$installed_version" != "$declared_version" ]]; then
      echo "FAIL: $declared_name version mismatch in $image"
      echo "      requirements.txt: $declared_version"
      echo "      container:        $installed_version"
      failures=$((failures + 1))
    else
      echo "PASS: $declared_name==$declared_version"
    fi
  done < "$requirements_file"
done

echo
if [[ "$failures" -ne 0 ]]; then
  echo "Container dependency reproducibility test FAILED with $failures issue(s)."
  exit 1
fi

echo "Container dependency reproducibility test PASSED."
