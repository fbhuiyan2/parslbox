#!/bin/sh
# Sync canonical docs/ into each skill's docs/ subdir.
#
# Skills must be self-contained (portable when installed standalone), so each
# skill keeps real copies of the docs it references instead of pointing outside
# its dir. Canonical source of truth is the top-level docs/ -- edit there, then
# run this. Pure POSIX sh (git + coreutils only, no interpreter dependency) so
# it works from a pre-commit hook in any environment.
#
#   scripts/sync-skill-docs.sh          copy canonical -> skill dirs
#   scripts/sync-skill-docs.sh --check   exit 1 if any copy is stale
set -eu

repo_root=$(cd "$(dirname "$0")/.." && pwd)
docs="$repo_root/docs"

check=0
[ "${1:-}" = "--check" ] && check=1
rc=0

sync_one() {
    skill=$1
    shift
    for name in "$@"; do
        src="$docs/$name"
        dst="$repo_root/skills/$skill/docs/$name"
        if [ ! -f "$src" ]; then
            echo "error: canonical doc not found: docs/$name" >&2
            rc=1
            continue
        fi
        if [ "$check" -eq 1 ]; then
            if [ ! -f "$dst" ] || ! cmp -s "$src" "$dst"; then
                echo "stale: skills/$skill/docs/$name" >&2
                rc=1
            fi
        else
            mkdir -p "$(dirname "$dst")"
            cp -p "$src" "$dst"
            echo "synced skills/$skill/docs/$name"
        fi
    done
}

# skill dir name  ->  docs it references
sync_one parslbox-cli commands.md apps.md pbx-run-details.md
sync_one parslbox-api api-details.md apps.md pbx-run-details.md

if [ "$rc" -ne 0 ] && [ "$check" -eq 1 ]; then
    echo "Skill docs out of sync -- run scripts/sync-skill-docs.sh" >&2
fi
exit "$rc"
