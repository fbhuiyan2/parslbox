"""
Scheduler Directive Parsing and Merging for ParslBox

This module provides functions to parse PBS/SLURM scheduler directives
and merge them using a three-layer override chain:
    Template directives (base) → config sched_opts (per-system) → CLI/API sched_opts (per-run)
"""

import re
from typing import Optional, List


def extract_directive_key(line: str) -> Optional[str]:
    """
    Parse a #PBS or #SBATCH directive line and return its canonical key.

    Key extraction rules:
    - PBS `#PBS -l key=value` → key = resource name (e.g., "filesystems", "place", "walltime")
    - PBS `#PBS -X value` (single-letter flags) → key = flag (e.g., "-N", "-q", "-A")
    - SLURM `#SBATCH --flag=value` or `#SBATCH --flag value` → key = "--flag"
    - Non-directive lines → None

    Args:
        line: A single line from a scheduler script

    Returns:
        The canonical key string, or None for non-directive lines
    """
    stripped = line.strip()

    # PBS directives
    pbs_match = re.match(r'#PBS\s+(-\w)\s*(.*)', stripped)
    if pbs_match:
        flag = pbs_match.group(1)
        rest = pbs_match.group(2).strip()
        if flag == '-l' and '=' in rest:
            # Resource directive: #PBS -l key=value
            resource_key = rest.split('=', 1)[0].strip()
            return resource_key
        else:
            # Single-letter flag: #PBS -N, #PBS -q, etc.
            return flag

    # SLURM directives (long flags like --constraint, short flags like -C)
    sbatch_match = re.match(r'#SBATCH\s+(--[\w-]+|-\w)', stripped)
    if sbatch_match:
        return sbatch_match.group(1)

    return None


def merge_sched_opts(
    template_content: str,
    config_sched_opts: Optional[str],
    cli_sched_opts: Optional[List[str]],
) -> str:
    """
    Merge scheduler directives from three layers: template, config, and CLI/API.

    Override chain: template → config → CLI/API.
    Each layer replaces directives with matching keys from previous layers.
    New directives are inserted at the {sched_opts} placeholder location.

    Args:
        template_content: The already-rendered template script content
        config_sched_opts: Multi-line string of directives from config YAML (or None)
        cli_sched_opts: List of directive strings from CLI --sched-opts or API sched_opts param (or None)

    Returns:
        The final script content with all directives merged
    """
    lines = template_content.split('\n')

    # Build ordered list of (line_index, key) for template directive lines
    template_directives = []  # [(line_index, key)]
    sched_opts_line_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == '{sched_opts}':
            sched_opts_line_idx = i
            continue
        key = extract_directive_key(line)
        if key is not None:
            template_directives.append((i, key))

    # Collect override directives from config
    config_directives = []
    if config_sched_opts:
        for raw_line in config_sched_opts.strip().split('\n'):
            raw_line = raw_line.strip()
            if raw_line:
                config_directives.append(raw_line)

    # Collect override directives from CLI
    cli_directives = []
    if cli_sched_opts:
        for raw_line in cli_sched_opts:
            raw_line = raw_line.strip()
            if raw_line:
                cli_directives.append(raw_line)

    # Apply overrides: config then CLI
    # For each override layer, matching keys replace in-place, new keys are collected
    new_directives = []  # directives to insert at {sched_opts}

    # Track which template line indices have been overridden
    # and what the current effective key→line mapping is
    # We work with the actual lines array for in-place replacement

    def _apply_overrides(override_lines):
        """Apply a list of override directive lines."""
        for directive_line in override_lines:
            key = extract_directive_key(directive_line)
            if key is None:
                # Not a recognizable directive, add as-is to new directives
                new_directives.append(directive_line)
                continue

            # Check if this key exists in template directives
            replaced = False
            for idx, tpl_key in template_directives:
                if tpl_key == key:
                    # Detect indentation from the original template line
                    original = lines[idx]
                    indent = original[: len(original) - len(original.lstrip())]
                    lines[idx] = indent + directive_line.lstrip()
                    replaced = True
                    break

            if not replaced:
                # Check if this key was already added as a new directive
                found_in_new = False
                for j, existing in enumerate(new_directives):
                    existing_key = extract_directive_key(existing)
                    if existing_key == key:
                        new_directives[j] = directive_line
                        found_in_new = True
                        break
                if not found_in_new:
                    new_directives.append(directive_line)

    _apply_overrides(config_directives)
    _apply_overrides(cli_directives)

    # Replace {sched_opts} placeholder with new directives
    if sched_opts_line_idx is not None:
        placeholder_line = lines[sched_opts_line_idx]
        indent = placeholder_line[: len(placeholder_line) - len(placeholder_line.lstrip())]
        if new_directives:
            replacement = '\n'.join(indent + d.lstrip() for d in new_directives)
            lines[sched_opts_line_idx] = replacement
        else:
            # Remove the placeholder line entirely
            lines.pop(sched_opts_line_idx)

    return '\n'.join(lines)
