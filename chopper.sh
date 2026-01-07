#!/usr/bin/env bash

set -euxo pipefail
DIR="${1:-.}"

# get unique samples
ls $DIR
STEPS=$(fd 'step\d+' -e json $DIR | rg -o 'step(\d+)' -r '$1' | sort -nu)

# combine gpus together per sample
MS=()
for step in $STEPS
do
  CUR="${DIR}/merged_step${step}.pkl"
  python -m chopper.profile.merge -t $(fd "step${step}.json" $DIR) -o "${CUR}" -nv
  MS+=($CUR)
done

# combine samples together
python -m chopper.profile.merge -p "${MS[@]}" -o "${DIR}/ts.pkl" -nv

# clean up intermediate files
for merged_step in "${MS[@]}"
do
  rm $merged_step
done
