#!/bin/bash

branches=(
  "datasetGeneration/datasetSaya"
  "datasetGeneration/datasetAmbil"
  "datasetGeneration/datasetTunggu"
  "datasetGeneration/datasetJalan"
)

for branch in "${branches[@]}"; do
  echo "🚀 Processing $branch..."

  # Check if branch exists locally
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    git checkout "$branch"

    # Check if already exists on remote
    if git ls-remote --exit-code --heads origin "$branch" > /dev/null 2>&1; then
      echo "⏭ Already exists on remote: $branch"
    else
      echo "📤 Pushing $branch..."
      git push -u origin "$branch"
    fi
  else
    echo "❌ Local branch not found: $branch"
  fi

  echo ""
done

echo "✅ Done!"