#!/bin/bash

REPO="VinzZ00/ToolScripts"

branches=(
  "datasetGeneration/datasetR"
  "datasetGeneration/datasetS"
  "datasetGeneration/datasetT"
  "datasetGeneration/datasetU"
  "datasetGeneration/datasetV"
  "datasetGeneration/datasetW"
  "datasetGeneration/datasetX"
  "datasetGeneration/datasetY"
  "datasetGeneration/datasetZ"
  "datasetGeneration/datasetSaya"
  "datasetGeneration/datasetAmbil"
  "datasetGeneration/datasetTunggu"
  "datasetGeneration/datasetJalan"
)

for branch in "${branches[@]}"; do
  echo "✅ Approving PR for $branch..."

  gh pr review \
    --repo "$REPO" \
    --approve \
    "$branch" 2>/dev/null

  if [ $? -eq 0 ]; then
    echo "✔ Approved"
  else
    echo "⚠ Could not approve (maybe no PR / already approved / not allowed)"
  fi

  echo ""
done

echo "🎉 Done approving all PRs!"