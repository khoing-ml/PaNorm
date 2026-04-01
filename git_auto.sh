#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./git_auto.sh [commit_message] [remote] [branch]
# Examples:
#   ./git_auto.sh
#   ./git_auto.sh "update benchmark logs"
#   ./git_auto.sh "quick fix" origin master

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
	echo "Error: current directory is not a git repository."
	exit 1
fi

commit_msg="${1:-auto: update $(date '+%Y-%m-%d %H:%M:%S')}"
remote="${2:-origin}"

if [[ $# -ge 3 ]]; then
	branch="$3"
else
	branch="$(git rev-parse --abbrev-ref HEAD)"
fi

if [[ "$branch" == "HEAD" || -z "$branch" ]]; then
	echo "Error: could not determine branch (detached HEAD)."
	exit 1
fi

echo "Staging changes..."
git add -A

if git diff --cached --quiet; then
	echo "No staged changes to commit."
	exit 0
fi

echo "Committing on branch '$branch'..."
git commit -m "$commit_msg"

if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
	echo "Pushing to upstream..."
	git push
else
	echo "No upstream configured. Pushing with -u to $remote/$branch..."
	git push -u "$remote" "$branch"
fi

echo "Done."
