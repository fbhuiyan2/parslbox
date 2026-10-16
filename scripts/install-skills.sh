#!/bin/sh
# Install parslbox's agent skills into whichever agent harnesses are present.
#
# Each skill directory under skills/ is self-contained (it bundles its own
# docs/), so a harness only needs to see it at a path that harness scans.
# Default is a symlink into this repo, so `git pull` updates the installed
# skill; --copy makes a frozen snapshot for machines where a link will not do.
#
#   scripts/install-skills.sh                 detected harnesses, symlink
#   scripts/install-skills.sh --harness claude,opencode
#   scripts/install-skills.sh --all           every known harness, creating dirs
#   scripts/install-skills.sh --copy          frozen copies instead of links
#   scripts/install-skills.sh --list          what is detected and installed
#   scripts/install-skills.sh --check         exit 1 if anything is missing/stale
#   scripts/install-skills.sh --uninstall     remove only what this script put there
#
# Pure POSIX sh, like sync-skill-docs.sh -- it has to run on a login node with
# nothing installed yet.
set -eu

repo_root=$(cd "$(dirname "$0")/.." && pwd)
marker=.installed-by-parslbox

# Lists below are newline-separated and iterated with IFS set to newline:
# a home directory with a space in it would otherwise field-split into bogus
# targets and the script would write to the wrong place reporting success.
NL='
'
default_ifs=$IFS

# harness key : directory it scans, relative to $HOME
harness_dirs() {
    cat <<'TABLE'
claude      .claude/skills
opencode    .config/opencode/skills
codex       .codex/skills
agents      .agents/skills
cursor      .cursor/skills
gemini      .gemini/skills
pi          .pi/agent/skills
copilot     .copilot/skills
amp         .config/agents/skills
kiro        .kiro/skills
kilo        .config/kilo/skills
factory     .factory/skills
trae        .trae/skills
kimi        .kimi/skills
devin       .config/devin/skills
TABLE
}

# Harnesses that scan more than one of the directories above. Installing into
# several of them makes that harness load the same skill once per location.
overlapping_harness='opencode'
overlapping_dirs='.config/opencode/skills .claude/skills .agents/skills'

skills() {
    for d in "$repo_root"/skills/*/; do
        if [ -f "$d/SKILL.md" ]; then basename "$d"; fi
    done
}

dir_for() {
    harness_dirs | while read -r key rel; do
        if [ "$key" = "$1" ]; then printf '%s\n' "$HOME/$rel"; fi
    done
}

mode=install
copy=0
selection=
force_all=0

while [ $# -gt 0 ]; do
    case "$1" in
        --harness) selection=$(printf '%s' "$2" | tr ',' ' '); shift 2 ;;
        --harness=*) selection=$(printf '%s' "${1#*=}" | tr ',' ' '); shift ;;
        --all) force_all=1; shift ;;
        --copy) copy=1; shift ;;
        --list) mode=list; shift ;;
        --check) mode=check; shift ;;
        --uninstall) mode=uninstall; shift ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

targets=
if [ -n "$selection" ]; then
    for key in $selection; do
        d=$(dir_for "$key")
        [ -n "$d" ] || { echo "unknown harness: $key" >&2; exit 2; }
        targets="$targets$key:$d$NL"
    done
elif [ "$force_all" -eq 1 ]; then
    targets=$(harness_dirs | while read -r key rel; do printf '%s:%s\n' "$key" "$HOME/$rel"; done)
else
    targets=$(harness_dirs | while read -r key rel; do
        if [ -d "$HOME/$rel" ]; then printf '%s:%s\n' "$key" "$HOME/$rel"; fi
    done)
fi

if [ -z "$targets" ] && [ "$mode" != list ]; then
    echo "No agent harness directories found under $HOME." >&2
    echo "Name one explicitly: --harness claude   (or --all to create them)" >&2
    exit 1
fi

installed_dirs=
rc=0

IFS=$NL
for entry in $targets; do
    IFS=$default_ifs
    key=${entry%%:*}
    dir=${entry#*:}

    if [ "$mode" = list ]; then
        if [ -d "$dir" ]; then state=present; else state="-"; fi
        printf '%-10s %-34s %s\n' "$key" "${dir#"$HOME"/}" "$state"
        for s in $(skills); do
            if [ -L "$dir/$s" ]; then
                printf '    %-18s -> %s\n' "$s" "$(readlink "$dir/$s")"
            elif [ -d "$dir/$s" ]; then
                printf '    %-18s %s\n' "$s" "$([ -f "$dir/$s/$marker" ] && echo "copy" || echo "not ours")"
            fi
        done
        continue
    fi

    for s in $(skills); do
        src="$repo_root/skills/$s"
        dst="$dir/$s"

        case "$mode" in
        uninstall)
            if [ -L "$dst" ] && [ "$(readlink "$dst")" = "$src" ]; then
                rm "$dst"; echo "removed $dst"
            elif [ -d "$dst" ] && [ -f "$dst/$marker" ]; then
                rm -rf "$dst"; echo "removed $dst"
            fi
            ;;
        check)
            if [ -L "$dst" ] && [ "$(readlink "$dst")" = "$src" ]; then
                :
            elif [ -d "$dst" ] && [ -f "$dst/$marker" ]; then
                :
            else
                echo "missing: $dst" >&2; rc=1
            fi
            ;;
        install)
            mkdir -p "$dir"
            if [ -L "$dst" ]; then
                if [ "$(readlink "$dst")" = "$src" ]; then
                    rm "$dst"
                else
                    echo "refusing to replace $dst (symlink to $(readlink "$dst"))" >&2
                    rc=1
                    continue
                fi
            elif [ -d "$dst" ]; then
                if [ -f "$dst/$marker" ]; then
                    rm -rf "$dst"
                else
                    echo "refusing to replace $dst (not installed by this script)" >&2
                    rc=1
                    continue
                fi
            fi
            if [ "$copy" -eq 1 ]; then
                cp -R "$src" "$dst"
                : > "$dst/$marker"
                echo "copied  $dst"
            else
                ln -s "$src" "$dst"
                echo "linked  $dst -> $src"
            fi
            installed_dirs="$installed_dirs${dir#"$HOME"/}$NL"
            ;;
        esac
    done
    IFS=$NL
done
IFS=$default_ifs

# Warn when one harness scans several of the directories we just wrote to.
if [ "$mode" = install ]; then
    hits=
    n=0
    for od in $overlapping_dirs; do
        IFS=$NL
        for id in $installed_dirs; do
            IFS=$default_ifs
            if [ "$od" = "$id" ]; then
                case " $hits " in *" $od "*) ;; *) hits="$hits $od"; n=$((n + 1)) ;; esac
            fi
            IFS=$NL
        done
        IFS=$default_ifs
    done
    if [ "$n" -gt 1 ]; then
        echo ""
        echo "WARNING: $overlapping_harness reads more than one directory you just installed into:"
        for h in $hits; do echo "    ~/$h"; done
        echo "  It scans all of them, so each skill is loaded once per location and skill"
        echo "  names must be unique across them. Other harnesses are unaffected."
        echo "  If you use $overlapping_harness, keep one location and drop the rest, e.g.:"
        echo "    scripts/install-skills.sh --uninstall --harness opencode"
    fi
fi

exit "$rc"
