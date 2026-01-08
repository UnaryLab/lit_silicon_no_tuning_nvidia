#!/usr/bin/env bash

set -euxo pipefail
DIR="${1:-.}"

# get unique steps
STEPS=$(find "${DIR}" -type f -name '*step*.json' | sed 's/.*step//g' | sed 's/\.json//g' | sort -nu)

# combine gpus together per step
MS=()
for step in $STEPS
do
  CUR="${DIR}/merged_step${step}.pkl"
  python -m chopper.profile.merge -t $(find "${DIR}" -type f -name "*step${step}.json") -o "${CUR}"
  MS+=($CUR)
done

# combine samples together into one file
python -m chopper.profile.merge -p "${MS[@]}" -o "${DIR}/ts.pkl"

# clean up intermediate files
for merged_step in "${MS[@]}"
do
  rm $merged_step
done
