#!/bin/bash

REPO="VinzZ00/ToolScripts"
BASE_BRANCH="development-10Mar"

# Generate datasetR → datasetZ
alphabet_branches=()
for letter in {R..Z}; do
  alphabet_branches+=("datasetGeneration/dataset$letter")
done

# Custom branches
custom_branches=(
  "datasetGeneration/datasetSaya"
  "datasetGeneration/datasetAmbil"
  "datasetGeneration/datasetTunggu"
  "datasetGeneration/datasetJalan"
)

# Combine
branches=("${alphabet_branches[@]}" "${custom_branches[@]}")

for branch in "${branches[@]}"; do
  echo "🚀 Processing $branch..."

  # Check local branch
  if git show-ref --verify --quiet "refs/heads/$branch"; then

    # Push if needed
    if ! git ls-remote --exit-code --heads origin "$branch" > /dev/null 2>&1; then
      echo "📤 Pushing $branch..."
      git push -u origin "$branch"
    else
      echo "⏭ Already on remote"
    fi

    # Check if PR already exists
    if gh pr list --repo "$REPO" --head "$branch" --json number --jq 'length > 0' | grep -q true; then
      echo "⏭ PR already exists for $branch"
      continue
    fi

    # Create PR
    echo "🔀 Creating PR → $BASE_BRANCH"

    gh pr create \
      --repo "$REPO" \
      --base "$BASE_BRANCH" \
      --head "$branch" \
      --title "Merge $branch into $BASE_BRANCH" \
      --body "Auto-generated PR for $branch"

  else
    echo "❌ Local branch not found: $branch"
  fi

  echo ""
done

echo "✅ All PRs processed!"