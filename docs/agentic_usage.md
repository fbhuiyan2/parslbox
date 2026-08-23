# Agentic usage

ParslBox is built to be driven by an AI agent as well as by hand. It ships two things you can add to any agentic harness:

- **An MCP server** — exposes job add/submit/query/cancel as callable tools.
- **Two skills** — `parslbox-cli` (the `pbx` shell commands) and `parslbox-api` (the Python `ParslBox` API). Each is a self-contained directory under [`skills/`](../skills) that teaches the agent how to use parslbox.

Install the MCP dependencies with `pip install ".[agentic]"`.

---

## MCP server

Start the server in either mode:

```bash
# HTTP mode (standalone server, default port 9795)
python -m parslbox.mcp.mcp_server

# HTTP mode on a different port (if 9795 is already in use)
python -m parslbox.mcp.mcp_server --port 8080

# stdio mode (for editor/agent integration)
python -m parslbox.mcp.mcp_server --stdio
```

Exposed tools: `add_jobs`, `submit_pbs_job`, `submit_slurm_job`, `cancel_pbs_job`, `cancel_slurm_job`, `remove_jobs`, `update_job`, `filter_jobs`, `list_jobs`, `get_job`, `get_jobs`.

For an HTTP client example, see [`examples/chemgraph_parslbox_example/`](../examples/chemgraph_parslbox_example/).

### Add the MCP server to Claude Code (user scope — available in every directory)

```bash
claude mcp add-json parslbox -s user '{"command":"/path/to/your/parslbox_env/bin/python","args":["-m","parslbox.mcp.mcp_server","--stdio"],"cwd":"/path/to/parslbox-repo-dir"}'
```

Replace both paths with your own:
- `command` → the `python` inside your parslbox environment (find it with `which python` after activating that environment)
- `cwd` → your local parslbox repo directory

Then restart Claude Code and run `/mcp` to confirm `parslbox` is connected.

**Project-scoped alternative:** the repo also ships a [`.mcp.json`](../.mcp.json) for automatic discovery — Claude Code launched from the repo directory will offer to connect. Edit its two `/path/to/...` placeholders first. Use this if you only want parslbox available when working inside the repo; use the `add-json` command above for it everywhere.

### Add the MCP server to OpenCode (user level)

Add the `parslbox` MCP server to your global OpenCode config so it's available in every project. Edit `~/.config/opencode/opencode.json` and add an `mcp` block (merge it alongside any existing keys):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "parslbox": {
      "type": "local",
      "command": [
        "/path/to/your/parslbox_env/bin/python",
        "-m",
        "parslbox.mcp.mcp_server",
        "--stdio"
      ],
      "cwd": "/path/to/parslbox-repo-dir",
      "enabled": true,
      "timeout": 120000
    }
  }
}
```

Replace both paths with your own:
- `command[0]` → the `python` inside your parslbox environment (find it with `which python` after activating that environment)
- `cwd` → your local parslbox repo directory
- `timeout` → how long (ms) OpenCode waits for the server; `120000` = 120s (see [Before you launch](#before-you-launch-env-vars-warm-up-and-timeouts))

Restart OpenCode, then run `opencode mcp list` to confirm `parslbox` is connected.

### Before you launch: env vars, warm-up, and timeouts

The harness starts the MCP server for you, so get a few things ready **before** you launch Claude Code or OpenCode.

1. **Set the environment variables parslbox needs** (e.g. `PBX_DB_PATH`, `PBX_CONFIG_PATH`) in the shell you launch the harness from — the MCP server inherits that environment. Both must be absolute paths:

   ```bash
   export PBX_DB_PATH=/path/to/job_database_pbx.db
   export PBX_CONFIG_PATH=/path/to/config.yaml
   ```

2. **Warm up parslbox** with one `pbx` command in that same shell:

   ```bash
   pbx ls
   ```

   On large shared filesystems (e.g. Aurora), the *first* `pbx` call can be slow — cold Python imports plus metadata lookups over the parallel filesystem. Warming the cache first lets the MCP server start quickly enough to beat the harness's connect timeout. Skip it and the harness may try to start the server, hit the timeout, and keep retrying.

3. **Raise the MCP timeout to ~120s** so a cold start isn't killed:

   - **Claude Code** — set before launching:
     ```bash
     export MCP_TIMEOUT=120000        # server startup (ms)
     export MCP_TOOL_TIMEOUT=120000   # long-running tool calls (ms)
     ```
   - **OpenCode** — add `"timeout": 120000` to the `parslbox` block in `opencode.json` (shown above).

Putting it together for a Claude Code session:

```bash
export PBX_DB_PATH=/path/to/job_database_pbx.db
export MCP_TIMEOUT=120000
export MCP_TOOL_TIMEOUT=120000
pbx ls        # warm up
claude        # (OpenCode: just run `opencode` — its timeout lives in opencode.json)
```

---

## Skills

The two skills live under [`skills/`](../skills) — `parslbox-cli` and `parslbox-api`. Each directory is self-contained (it bundles its own `docs/`), so you can install it either by copying or by symlinking.

### Where skills live

- **Claude Code** reads `~/.claude/skills/<name>/`
- **OpenCode** reads `~/.config/opencode/skills/`, **and also `~/.claude/skills/`** and `~/.agents/skills/`

Because `~/.claude/skills/` is read by **both** tools, it's the best single home if you use both (see the conflict note below).

### Install by copy (standalone snapshot)

```bash
cp -r skills/parslbox-cli skills/parslbox-api ~/.claude/skills/
```

A copy is frozen — re-copy after you pull repo updates.

### Install by symlink (tracks the repo)

```bash
ln -s /abs/path/to/parslbox/skills/parslbox-cli ~/.claude/skills/parslbox-cli
ln -s /abs/path/to/parslbox/skills/parslbox-api ~/.claude/skills/parslbox-api
```

A symlink always reflects the repo, so `git pull` updates the skill automatically. Use absolute paths.

> Both methods work because each skill bundles its own `docs/` — nothing points outside the skill directory.

### Using OpenCode and Claude Code together — avoid double-loading

If you have Claude Code installed, put the skills **only** in `~/.claude/skills/`. OpenCode reads that location too, so both tools discover them from one place. Do **not** also add them under `~/.config/opencode/skills/` — OpenCode would then load each skill twice, and skill names must be unique across locations.

(OpenCode-only, no Claude Code? Then `~/.config/opencode/skills/` is fine as the single home.)

---

## Driving parslbox from an agent — best practices

These are operational lessons for adding jobs at scale over MCP:

- **One compact call beats many.** Prefer a single `add_jobs` with a `paths` list (one shared `app`/`config`/resources) over one call per job. Fewer round-trips, smaller payloads, less to go wrong.
- **Bulk add with `all:<dir>` — use an absolute `<dir>`.** `all:<dir>` adds every subdirectory of `<dir>`. Over MCP, a bare `all` resolves against the **server's** working directory (usually not what you want), so always pass an absolute base dir.
- **MCP calls are synchronous — big batches can hit the client timeout.** Adding hundreds/thousands of jobs can take longer than the client's response window. The call may report a timeout **while the database writes still commit** (each insert commits individually). Don't assume failure.
- **On a timeout, verify then retry.** Check the DB with `list_jobs` / `filter_jobs` before re-issuing. Retries are safe: adds are idempotent via a `(path, in_file)` UNIQUE constraint — re-adding an existing job comes back as a failure, not a duplicate.
- **Add sequentially, not in parallel.** Bulk adds serialize on SQLite's single writer lock; firing several `add_jobs` calls at once just makes them contend and raises the chance of a timeout.
- **If you control the client timeout, raise it.** For OpenCode, add a `timeout` (ms) to the `mcp` block. Note OpenCode's `timeout` governs tool *fetching*, not necessarily each tool-call response — so pair it with the smaller-batch/sequential guidance above.
