#!/usr/bin/env bash
set -o pipefail
export AWS_PROFILE=lukewma-workshop
export AWS_DEFAULT_REGION=us-west-2

pass=0 fail=0
notebooks=($(find ~/evals-workshop -name '*.ipynb' ! -name 'Complete-*' ! -path '*/04-04-MultiModal-RAG/*' ! -path '*/.ipynb_checkpoints/*' | sort))

for nb in "${notebooks[@]}"; do
  dir=$(dirname "$nb")
  base=$(basename "$nb")
  out="$dir/Complete-$base"

  # Remove stale Complete output
  rm -f "$dir"/Complete-*.ipynb

  echo "=== Running: $nb ==="
  timeout 600 python3.11 -m papermill "$nb" "$out" --cwd "$dir" -k python3 2>&1
  rc=$?

  if [ $rc -eq 0 ]; then
    echo "PASS: $nb"
    ((pass++))
  else
    echo "FAIL: $nb (exit $rc)"
    ((fail++))
  fi
  echo
done

echo "===== SUMMARY ====="
echo "PASS: $pass"
echo "FAIL: $fail"
echo "TOTAL: $((pass + fail))"
