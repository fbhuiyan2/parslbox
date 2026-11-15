# Running jobs on a remote machine

Write and organise your jobs on your own computer. Run them on an HPC machine,
without logging in to it.

You keep working in a normal directory on your laptop — making job folders,
adding them to pbx, checking what has finished. When you are ready, `pbx qsub`
sends everything over and submits it to the HPC scheduler for you. The results
of that submission come back into your local database, so `pbx ls` on your
laptop tells you what happened.

Command-by-command reference: [`commands.md`](commands.md#pbx-local).

---

## Part 1 — Understanding the pieces

### Globus, in plain terms

Globus is a service used by most HPC facilities. Two parts of it matter here,
and they are **completely separate things** that happen to share a name.

**Globus Compute** runs small pieces of code for you on the HPC machine. You
start a program called an *endpoint* on the HPC login node; it sits there
waiting, and when you send it something to do, it runs it as you, with your
permissions. This is what lets your laptop say "run `qsub` over there".

An endpoint is identified by a **UUID** — a long string like
`9a96ff32-7f6e-46d8-8d0b-463175ef8299`. You will be asked for it during setup.
You start the endpoint yourself; nobody creates it for you.

**Globus Transfer** copies files between machines. Before it can copy anything,
both machines have to be registered with Globus. A registered filesystem is
called a **collection**, and like an endpoint it has a UUID. Every copy needs
two: one for the machine the files come from, one for the machine they go to.

Your HPC almost certainly has a collection already — the facility set it up,
and its UUID will be in their documentation. Your own machine does not, unless
you install [Globus Connect
Personal](https://www.globus.org/globus-connect-personal), which registers it
and hands you its UUID.

### Which do you need?

Here is the useful part: **most of the time, you only need Compute.**

| What is moving | Goes over | What you must have |
|---|---|---|
| The job database (every push, pull, and submit) | **Compute** | a Compute endpoint |
| Your job folders and their input files | **Transfer** | collections on *both* machines |

The database is small — a few thousand jobs compress to well under a megabyte —
so it travels inside the Compute message itself. That means creating projects,
adding jobs, submitting, and checking progress **all work with just a Compute
endpoint**. No collections, no Globus Connect Personal, nothing installed on
your laptop.

You only need Transfer when the *input files themselves* have to move — when
you build job folders on your laptop and want them copied over. If you create
your job folders directly on the HPC machine instead (or they are already
there), you can skip Transfer entirely. That is covered in
[Part 4](#part-4--sending-your-input-files-too), and you can ignore it until
you need it.

### The picture

```
  YOUR LAPTOP                                   HPC MACHINE
  ~/work/proj1/                                 /lus/flare/.../proj1/
    .pbxlocal.yaml                                     run_a/
    job_database_pbx-local.db                          run_b/
    run_a/                                             job_database_pbx-local.db
    run_b/                                                   |
        |                                                    v
        |------- Compute: database + "run qsub" ------>  pbx qsub -> PBS
        |------- Transfer: job folders (optional) ---->
```

### The one idea behind it

**Your local database stores HPC paths, right from the first job you add.**

When you run `pbx add ./run_a` in a project whose remote root is
`/lus/flare/.../proj1`, pbx stores the path as `/lus/flare/.../proj1/run_a` —
not the path on your laptop.

This looks odd at first (`pbx ls` shows you paths that do not exist on your
machine) but it is what keeps everything else simple. The database already
describes the machine the jobs will run on, so nothing has to be converted,
rewritten, or handed back and forth. The same file works in both places.

---

## Part 2 — Setting up

### What you need first

**On your own machine:**

From your parslbox clone:

```bash
# on your own machine, in your parslbox clone
poetry install --extras "remote"      # or: pip install ".[remote]"
```

The `remote` extra is what adds the Globus libraries. It goes **here**, on the
machine you are working from — not on the HPC machine.

Then run `pbx config` once, here as well:

```bash
pbx config                            # on your own machine
```

Every `pbx` command refuses to start without a `config.yaml` on the machine it
is typed on — without one you get an interactive setup interview instead of the
command you asked for. What you put in it does not matter for a local project,
because the submission uses the *remote* machine's config; the file just has to
exist.

**On the HPC machine**, you need two things:

- **A running Globus Compute endpoint**, in a Python environment where
  `import parslbox` works.
- **A parslbox `config.yaml`** (`pbx config`). Only needs mentioning during
  setup if it is not in the default location.

parslbox gets used in two different situations on the HPC machine:

| Environment | When it is used | What it needs |
|---|---|---|
| The **endpoint's** — whichever one you start `globus-compute-endpoint` from | Whenever your laptop submits, pushes, or pulls | Just enough for `import parslbox` to work. It does **not** need the `remote` extra, and parsl does not even have to work here — the only parslbox code called is `submit_job`; everything else parslbox runs there is plain standard library. |
| The **allocation's** — whatever your config's `pbx_python_env_setup` activates | Later, once the scheduler starts your job | Everything your jobs genuinely need: parsl, MPI, your app's dependencies. |

**These can be the same environment.** parslbox, parsl and the Globus packages
coexist fine — install `globus-compute-endpoint` alongside your existing
parslbox and you are done.

> **The care is needed when adding the endpoint to an environment that already
> works.** `globus-compute-sdk` pins `dill`, `tblib` and `globus-sdk` to exact
> or narrow versions, so installing it will change whatever is already there.
> parsl imports `globus_sdk` whenever it is present, so a version combination
> it dislikes breaks `import parsl` — and that breaks your jobs inside the
> allocation, not the endpoint, which makes it an easy failure to misdiagnose.
>
> If the environment your jobs run in is working and you would rather not
> disturb it, put the endpoint in a fresh virtual environment instead. If you
> do install into it, check `python -c "import parsl"` afterwards.

> **Start the endpoint *after* installing parslbox into its environment.** The
> endpoint inherits the environment of the shell that launches it. If you
> install something afterwards, restart the endpoint so it picks it up.

To start one:

```bash
# on the HPC login node, in the environment where parslbox is installed
pip install globus-compute-endpoint
globus-compute-endpoint configure myproject     # browser login on first use
globus-compute-endpoint start myproject
globus-compute-endpoint list                    # read off its UUID
```

The endpoint is a process on the login node, and it has to be running whenever
you push, pull or submit from your laptop. It survives you logging out, but not
a login-node reboot — `globus-compute-endpoint list` tells you whether it is
still up, and `start` brings it back. Globus documents the rest at
<https://globus-compute.readthedocs.io>; your facility may have its own
instructions that take precedence.

### Step 1 — Create the project

Make a directory on your laptop and turn it into a local project:

```bash
# on your own machine
mkdir ~/work/proj1 && cd ~/work/proj1
pbx local init
```

It explains itself and then asks a few questions. Only the first two matter to
start with; press Enter to skip the rest.

```
╭─────────────────────────────── pbx local init ───────────────────────────────╮
│ /home/you/work/proj1 becomes the local project root.                         │
│                                                                              │
│ Jobs you add here are stored with the paths they will have on the remote     │
│ machine, so nothing is rewritten later. You keep working in this directory;  │
│ pbx qsub sends the work over.                                                │
╰──────────────────────────────────────────────────────────────────────────────╯
Remote project root (absolute path):
Globus Compute endpoint UUID on the remote (blank to add later):
Path to config.yaml on the remote (blank for its default):
Globus Transfer collection UUID for this machine (blank to skip):
Globus Transfer collection UUID for the remote (blank to skip):
```

| Question | What to answer |
|---|---|
| Remote project root | The absolute path where the work will live on the HPC machine, e.g. `/lus/flare/projects/PROJ/you/proj1`. It does not have to exist yet. |
| Compute endpoint UUID | From `globus-compute-endpoint list` on the HPC machine. |
| Remote `config.yaml` | Leave blank unless yours is somewhere unusual. |
| The two Transfer collections | **Leave both blank for now.** You only need these to copy job folders over — see [Part 4](#part-4--sending-your-input-files-too). |

> **Anything left blank can be filled in later by editing `.pbxlocal.yaml`
> directly.** It is a plain text file in the project root, and the keys are the
> same ones `init` asked about (`compute_endpoint`, `remote_config`,
> `transfer_local`, `transfer_remote`). Re-running `pbx local init` in an
> existing project will *not* update them — it reports that the project already
> exists and changes nothing — so editing the file is the way to add settings
> after the fact.

When it finishes:

```
✅ Created local project in /home/you/work/proj1
project   /home/you/work/proj1                  created 2026-09-12
remote    /lus/flare/projects/PROJ/you/proj1
database  job_database_pbx-local.db
endpoint  9a96ff32-7f6e-46d8-8d0b-463175ef8299

Point your shell at this project:
  export PBX_DB_PATH="/home/you/work/proj1/job_database_pbx-local.db"
```

Nothing has touched the network yet. Two files now exist: the database, and
`.pbxlocal.yaml` holding the settings you just entered.

### Step 2 — Run the export line

**Copy the `export` line and run it.** This is what tells pbx which project you
are working on:

```bash
export PBX_DB_PATH="/home/you/work/proj1/job_database_pbx-local.db"
```

`init` cannot do this for you — no program can change the environment of the
shell that launched it, which is why `python -m venv` makes you run `activate`
too. If you lose the line, `cd` back to the project and run `pbx local init`
again — it finds the existing project and prints the line. (`pbx local
status` cannot help here: it locates the project *through* `PBX_DB_PATH`.)

Two things to watch:

- **It names the file, not the folder.** Pointing it at the directory gets you
  a different filename (`job_database_pbx.db`) — a new, empty database. Every
  `pbx` command notices and stops before creating it:

  ```
  ❌ Error: /home/you/work/proj1 is a local project, but PBX_DB_PATH resolves to
      /home/you/work/proj1/job_database_pbx.db
  A local project's database is job_database_pbx-local.db. Inside a local
  project, PBX_DB_PATH must name the file:
      export PBX_DB_PATH="/home/you/work/proj1/job_database_pbx-local.db"
  ```

  The last line is the fix, ready to paste. Outside a local project a directory
  is still a perfectly good `PBX_DB_PATH` — the rule applies only where a
  `.pbxlocal.yaml` is present.
- **It lasts only as long as your shell.** In a new terminal, run it again. To
  come back to this project tomorrow, that export line is all you need — the
  project is not tied to your current directory.

### Step 3 — Log in to Globus (once)

The first command that contacts the endpoint will ask you to log in. Run it
**in a normal terminal**:

```bash
pbx local status
```

It prints a URL. Open it in a browser, approve, and paste the code back:

```
Please authenticate with Globus here:
-------------------------------------
https://auth.globus.org/v2/oauth2/authorize?client_id=...
-------------------------------------
Enter the resulting Authorization Code here:
```

> **This needs a real terminal.** Run it through an agent, an editor task, a
> script, or anything else without an interactive prompt and it fails
> immediately with `CommandLineLoginFlowEOFError: An EOF was read when an
> authorization code was expected`. That error means only "there was nowhere to
> type" — nothing is wrong with your setup.

This is a one-time thing. Globus saves the login in `~/.globus_compute/` and
everything afterwards — including non-interactive use — picks it up silently.

`pbx local status` is a good command to spend the login on, because it only
reads; it cannot damage anything if the setup is wrong.

For genuinely unattended use (cron, CI, an agent), set
`GLOBUS_COMPUTE_CLIENT_ID` and `GLOBUS_COMPUTE_CLIENT_SECRET` to a Globus
confidential client instead, which skips the browser.

### Step 4 — Check the connection

After logging in, the same command tells you whether everything is wired up:

```
project   /home/you/work/proj1                          created 2026-09-12
remote    /lus/flare/projects/PROJ/you/proj1
endpoint  9a96ff32-7f…                                  reachable
database  job_database_pbx-local.db                     0 jobs

last synced  never
local        empty, never synced
remote       no database there yet

endpoint: aurora-uan-0009, python 3.12.13, parslbox 1.0.1

export PBX_DB_PATH="/home/you/work/proj1/job_database_pbx-local.db"
```

The last line before the export is the one to read. It comes from the endpoint
itself and confirms the two things most likely to be wrong: that the endpoint
is running, and that parslbox is *installed* in its environment. (It reads the
package metadata rather than importing, so a parslbox that is present but
broken still reports a version.) If no version comes back, install parslbox
into the endpoint's environment and restart the endpoint before going further.

You can have this checked at creation time by adding `--verify` to the `pbx
local init` in Step 1. It has no effect on a project that already exists.

---

## Part 3 — Day-to-day use

### Step 5 — Add jobs

Exactly as you normally would:

```bash
mkdir run_a && cp input.lmp run_a/
pbx add run_a -a lammps-kk -c aurora-gpu -i input.lmp -t batch1
pbx ls
```

`pbx ls` shows the path on the HPC machine:

```
│ 1  │ lammps-kk │ Ready │ ... │ /lus/flare/.../proj1/run_a │
```

That is correct and expected — see [the one idea](#the-one-idea-behind-it)
above. The folder `run_a` is still sitting on your laptop; the database is just
describing where it is going.

**Paths follow two rules:**

| Where the path is | What happens |
|---|---|
| Inside your project folder | Must exist on your machine. Stored with the HPC prefix swapped in. |
| Anywhere else | Must be absolute. Stored exactly as written, and **not** checked — pbx assumes it already refers to the HPC machine. |

The second rule is how you point at things that only exist on the HPC machine,
like a shared dataset or an environment script:

```bash
pbx add run_a ... --envfile /lus/flare/shared/env.sh
```

That file does not exist on your laptop, and pbx will not complain.

### Step 6 — Submit

The ordinary command:

```bash
pbx qsub -c aurora-gpu -N run1 -q debug --select 1 -T 30 -A myproject -t batch1
```

In a local project this quietly does three things instead of one:

1. **Sends the database up** to the HPC machine.
2. **Submits there** — writes a `submit.sh` under
   `<remote root>/pbx_runs/<timestamp>/` and runs `qsub` on it.
3. **Brings the database back**, so your local copy reflects anything that
   finished in the meantime.

What you see:

```
⬆️  Pushed the database to /lus/flare/.../proj1/job_database_pbx-local.db (1 jobs)
🌎 Submitted on endpoint 9a96ff32…
📁 Remote run directory: /lus/flare/.../proj1/pbx_runs/234002_100926
📝 Remote submit script: /lus/flare/.../proj1/pbx_runs/234002_100926/submit.sh
🚀 Job submitted successfully! Job ID: 8819271.aurora-pbs-0001
📊 Monitor on the remote with: qstat 8819271.aurora-pbs-0001
💡 Run 'pbx local pull' to bring progress back here.
```

> **The scheduler job ID is printed, but is not yet in your database.** It gets
> written when the allocation actually starts running, which may be minutes or
> hours later. Run `pbx local pull` then and it will be there.

`pbx sbatch` works the same way on Slurm machines.

If you are already logged in to the HPC and want `pbx qsub` to behave normally
— submitting from the machine you are on — add `--no-local`.

### Step 7 — Check on it, and bring progress back

```bash
pbx local pull
pbx ls
```

```
⬇️  Pulled 12 KB (1 jobs)
```

Now `pbx ls` shows the real state:

```
│ 1  │ lammps-kk │ Done │ ... │ 8819271.aur... │ /lus/flare/.../proj1/run_a │
```

To look before you touch anything, `pbx local status` reports which side has
moved:

```
last synced  2026-09-12T15:04:11Z  (pull)
local        unchanged
remote       12 rows changed since

→ pull before you push
```

`status` never changes anything. It follows `PBX_DB_PATH` rather than your
current directory, so it works from any folder — which makes it the right thing
to run when you are not sure which project your shell is pointed at.

The matching command in the other direction is `pbx local push`, which sends
your database up without submitting anything. You need it when you have edited
job records — added jobs, flipped some back to `Restart` — and want the HPC
side to see them before the next submission. `pbx qsub` does it for you, so in
the ordinary flow you never type it. Add `--force` to overwrite a remote that
has moved, which discards whatever it recorded.

If you are offline, `pbx local status --local-only` skips the network entirely. Without it, an
unreachable endpoint is not an error: you get everything local, plus the last
known remote figures marked as stale.

> **Your jobs' output files stay on the HPC machine.** Pulling brings back the
> *database* — statuses, job IDs, timings — not trajectories or logs. Fetch
> those with `scp`, `rsync`, or the Globus web interface.

---

## Part 4 — Sending your input files too

Skip this section unless you build job folders on your laptop and need them
copied to the HPC machine. This is the only part that needs Globus Transfer,
and therefore the only part that needs collection UUIDs on both ends.

Setting it up is a one-time job with three parts: an app ID, your own machine
registered as a collection, and your HPC's collection UUID.

### Why an app ID is needed at all

When a program uses Globus, Globus asks which program it is. Every program
needs an ID to answer with.

Globus Compute has one built in — which is why nothing in Parts 1–3 asked you
for anything. The file-copying side has no built-in ID, so one has to be
supplied, and at the moment that means registering your own.

This ID is **not a secret**. No password is issued with it; it appears in plain
text in the login URL. It grants nothing on its own — every user still signs in
with their own Globus account and approves the app themselves. It is a label
saying which program is asking, nothing more.

Borrowing someone else's rarely works: an app registers the web addresses
Globus may return to after login, and a facility's app points at its own web
portal rather than the paste-a-code page a command-line tool needs. It is worth
asking your facility's support desk whether they provide an ID for user
scripting, but do not count on it.

### Step 1 — Register an app (once)

Go to <https://app.globus.org/settings/developers> and sign in.

Choose **"Register a thick client or script that will be installed and run by
users on their devices."** *Thick client* is old jargon for software installed
on your own computer, as opposed to something running in a browser. Its
description — "cannot manage a client secret" — is the property you want.

Globus also calls this kind of app a **Native App**, and that is the term
parslbox uses in its error messages. Thick client, native app, and "the app ID"
in this section are all the same thing.

Then:

| Field | Value |
|---|---|
| Project | Create one. It is only a folder for grouping; any name. |
| App Name | Anything, e.g. `parslbox-globus-transfer`. This is what you will see on the approval screen. |
| Redirects | `https://auth.globus.org/v2/web/auth-code` |

Leave the rest blank, and ignore client secrets — this kind of app does not get
one. Copy the **Client ID** it gives you and export it:

```bash
# on your own machine
export PBX_GLOBUS_CLIENT_ID="<client id>"
```

Add it to your shell profile so it persists.

### Step 2 — Register your own machine as a collection

Globus can only copy between machines it knows about. A registered filesystem
is called a **collection**, and every copy needs two: the machine files come
from, and the machine they go to.

Your HPC already has one. Your own machine does not, until you install
[Globus Connect Personal](https://www.globus.org/globus-connect-personal):

```bash
# on your own machine -- the one holding the job folders
curl -LO https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
tar xzf globusconnectpersonal-latest.tgz
cd globusconnectpersonal-*/
./globusconnectpersonal -setup      # browser login, then name the collection
./globusconnectpersonal -start &
```

`-setup` prints the new collection's UUID — that is your `transfer_local`. It
must be **running** (`-start`) for transfers to work; check with `-status`.

By default it shares your home directory, which is enough if your projects live
under it. Anything outside needs adding in the Globus Connect Personal
preferences.

### Step 3 — Find your HPC's collection UUID

Your facility's documentation usually names it. At ALCF, for example, the
collection covering `/lus/flare/projects` is `alcf#dtn_flare`. Otherwise search
for it at <https://app.globus.org/collections>.

To get the UUID: click the collection's name in the search results, then the
**Overview** tab. The UUID is a field there. (Opening the collection in the
file manager shows you its *files*, not its identity — that is a different
view.) That UUID is your `transfer_remote`.

**Expect two login prompts the first time**, and none afterwards. Most facility
collections need a second approval covering that one collection specifically,
on top of signing in to Globus. parslbox works out whether yours is one of them
and asks for it automatically. (In the Overview tab such a collection shows as
*Mapped Collection (GCS)*.)

**Where its root is.** A collection usually publishes part of a filesystem
rather than all of it, and inside the collection that part *is* the top. ALCF's
`alcf#dtn_flare` is rooted at `/lus/flare/projects`, so a job living at

```
/lus/flare/projects/ABC/you/proj1/run_a
└──── the collection ─┘└─── inside it ───┘
```

is addressed to Globus as `/ABC/you/proj1/run_a`. Everything else — the
database, the scheduler, the running job — still uses the full path, because on
the machine itself that is the real one. Only Transfer sees the short form.

**You do not have to work this out.** parslbox finds it on your first directory
push and writes it into `.pbxlocal.yaml` as `transfer_remote_root`. How it
knows is explained in [Part 6](#how-the-collection-root-is-found).

### Step 4 — Put the two UUIDs in the project

Your project already exists by now, and re-running `pbx local init` will not
update it — so edit `.pbxlocal.yaml` in the project root. Both keys are already
there with a value of `null`; replace the nulls:

```yaml
transfer_local:  414b6ad4-af94-11f1-b554-0ee7ef9370d9   # your machine
transfer_remote: f39a7a0f-5bfc-46ce-9615-ba9f8592814f   # the HPC
```

That is all you write. `transfer_remote_root` is already in the file with a
`null` value; the first `pbx local push --with-dirs` fills it in:

```
   collection is rooted at /lus/flare/projects — saved as transfer_remote_root,
   not looked up again
```

Globus Connect Personal serves the whole filesystem, so there is no
`transfer_local_root` and none is needed.

### Using it

```bash
pbx local push --with-dirs           # database + folders, no submission
pbx qsub ... --push-dirs             # folders, then submit
```

Both print a report before sending anything:

```
📁 Job directories — 12 of 400 job(s) in the database are Ready/Restart
       11 to send
        1 missing on this machine, skipped
          /home/you/work/proj1/run_007
```

**Read the first line, not the second.** `11 to send` looks like success on its
own. `12 of 400` is what tells you that 388 jobs were not included — usually
because they are already `Done` and you forgot to set them back to `Restart`.

Only folders for `Ready` and `Restart` jobs are sent, matching what `pbx qsub`
will actually run. Narrow further with `--apps` and `--tags`.

### Re-running after editing your scripts

The loop is:

```bash
pbx local pull                       # get the finished statuses
pbx update 1-20 --status Restart     # mark them to run again
pbx qsub ... --push-dirs             # send the edits, submit
```

The `pbx update` step is needed because `Done` jobs are not re-run and their
folders are not re-sent.

### `--sync-level`

When a file is already on the HPC machine, Globus has to decide whether to copy
it again. `--sync-level` chooses how it decides:

| Value | Compares | Speed |
|---|---|---|
| `exists` | Just the filename | instant |
| `size` | Name and byte count | instant |
| `mtime` | Name, size, modification time | instant |
| `checksum` *(default)* | Reads and hashes the whole file, both ends | slow on large files |

`checksum` is the default because it is the only one that cannot be fooled. The
cost is that it *reads every file on both machines* even when it ends up
copying nothing — so a re-push of large input files can take a long time and
look like a hang.

If your inputs are small (a few scripts and a structure file), leave it alone.
If they are large (restart files, wavefunctions), use `--sync-level mtime`.

Only files present on your laptop are examined. Output written on the HPC side
is never looked at, so a run producing 50 GB of results does not slow this down.

`--sync-level` is a `pbx local push` flag only. `pbx qsub --push-dirs` always
uses `checksum`, so with large inputs send them first and submit separately:

```bash
pbx local push --with-dirs --sync-level mtime     # on your own machine
pbx qsub -c aurora-gpu -N run1 -q debug --select 1 -T 30 -A YourProject
```

---

## Part 5 — When something is refused

### "The remote database has changed"

```
❌ The remote database has changed since the last sync (12 rows changed since).
Overwriting it would lose that work.
Run the opposite direction first, or pass --force to overwrite deliberately.
```

**What happened:** work finished on the HPC machine and your local copy does
not know about it yet. Pushing now would overwrite those results with your
older copy.

**What to do:** `pbx local pull`, then carry on.

The reverse message — "the local database has changed" — appears on `pbx local
pull` and means the opposite: you have edits that a pull would erase. Push
first.

There is no merging. One side is authoritative at a time, and pbx refuses
rather than guessing. `--force` overrides the refusal when you know the other
side's changes do not matter. Other parslbox documents call this check *the
guard*.

`pbx qsub` and `pbx sbatch` push before they submit, so they hit the same
guard — but they have no `--force` of their own, and say so:

```
❌ Error: The remote database has changed since the last sync (+12 jobs added since). Overwriting it would lose that work.

Options, safest first:
  pbx local status        show what changed on the remote
  pbx local pull          bring that work down, then submit again
  pbx local push --force  overwrite the remote with your copy, discarding it

A submission has no --force of its own. If an allocation is still running
there, pull: pushing over a database it is writing to discards the jobs it has
already finished.
```

Nothing was submitted — the refusal happens before the remote is contacted.

### A transfer that never finishes

A directory transfer that sits there making no progress is usually **not** a
hang, and usually not really a permissions problem either:

```
   1698d9ae… ACTIVE 0/2 files — PERMISSION_DENIED
```

Globus treats a path it cannot reach as something that might come back, so it
retries every couple of minutes for the whole 30-minute timeout instead of
failing. The usual cause is a wrong `transfer_remote_root`: parslbox hands
Globus a path the collection looks for *inside* itself and does not find.

parslbox works that value out for itself and refuses up front when it cannot,
so you should only see this if you set it by hand. Delete the
`transfer_remote_root` line from `.pbxlocal.yaml` and push again — the next
push re-derives it.

Interrupting with Ctrl-C is safe: the transfer was handed to Globus and runs
there, so stopping the command only stops the watching. Cancel the task itself
at <https://app.globus.org/activity>.

Other statuses worth recognising in that line: `PAUSED_BY_ADMIN` (the facility
has stopped transfers), `ENDPOINT_ERROR` (the collection is down), and
`CONNECTION_FAILED` on the source side, which usually means Globus Connect
Personal is not running — check with `globusconnectpersonal -status`.

### "Not this project's database"

The database sitting at your remote root belongs to a *different* project.
Usually this means two projects were pointed at the same remote directory.
Check `remote_root` in `.pbxlocal.yaml` before doing anything else.

### "Holds N job(s) and this project has never been synced"

Neither side has any record of the other, and the one about to be overwritten
is not empty. pbx cannot tell which is the real one. Decide, then use `--force`
in the direction you want.

### "Is a local project, but PBX_DB_PATH resolves to …"

`PBX_DB_PATH` names the project *directory* rather than the database file, so
it resolves to `job_database_pbx.db` — the ordinary name, not this project's.
Every `pbx` command stops here rather than creating that file and reporting an
empty project. The refusal ends with the corrected `export` line; paste it.

This is checked only where a `.pbxlocal.yaml` sits beside the database.
Everywhere else, pointing `PBX_DB_PATH` at a directory is normal and the
database is created there on first use.

---

## Part 6 — Background

You do not need any of this to use the feature.

### Why the database can ride inside a Compute message

Globus caps a Compute message at 10 MB in each direction — a hard service
limit, not something you can raise. A database is small and compresses well:
measured with full-length HPC paths, 100,000 jobs comes to about 1.4 MB on the
wire. pbx checks against an 8 MB ceiling before sending and refuses early
rather than letting Globus reject the task.

Moving an oversized database over Transfer instead is **not implemented**. The
error message suggests it, but you would have to do it by hand.

### Why change detection does not use the file timestamp

The obvious way to tell whether a database changed is its modification time.
That does not work here. SQLite runs in WAL mode, where a commit is appended to
a separate `-wal` file and the main `.db` file is left untouched until a
checkpoint. The HPC side could run a hundred jobs with the timestamp unmoved —
so the check would wave through exactly the case it exists to catch.

Instead pbx asks SQLite directly for `MAX(timestamp)` and `COUNT(*)` — the
pair is called the database's *fingerprint* elsewhere in these docs. The first
catches new and updated rows, the second catches deletions, and both are
correct regardless of WAL.

For the same reason, a pull does not copy the live file. It runs
`VACUUM INTO` first, which writes a clean, consistent snapshot — important
because pulling mid-run is exactly when the file is being written to.

### How the collection root is found

A Transfer collection will not tell you which part of the filesystem it
publishes. Asked directly, Globus reports the underlying path as `null` and a
default directory of `/{server_default}/`, and the `absolute_path` in a
directory listing is relative to the collection like everything else. There is
no field to read.

So parslbox works it out. The root has to be *some* prefix of your
`remote_root`, which leaves only a handful of possibilities — one per component
of the path. It tries each one, shallowest first:

| it tries listing | on `alcf#dtn_flare` |
|---|---|
| `/lus/flare/projects/ABC/you/proj1` | not found |
| `/flare/projects/ABC/you/proj1` | not found |
| `/projects/ABC/you/proj1` | not found |
| `/ABC/you/proj1` | **found** |

so the root is the part that was cut off: `/lus/flare/projects`.

Finding a directory is not enough on its own to trust it. Project names repeat
across a facility's filesystems, so `/ABC/you/proj1` may well exist on more
than one of them, and a collection for the wrong one would answer just as
happily and then quietly take your files there.

What settles it is the database. It goes up first, over the Compute endpoint —
which runs on the machine that runs your jobs, so there is no question which
filesystem it landed on. parslbox then accepts a directory only if the listing
shows that database at exactly the size it just wrote. A same-named directory
on a different filesystem does not have it, and is refused:

```
Could not find /lus/eagle/projects/ABC/you/proj1 through Transfer collection …
Paths tried:
  /lus/eagle/projects/ABC/you/proj1  (not found)
  …
  /ABC/you/proj1  (exists, but job_database_pbx-local.db is 12288 bytes, not 9999)
Either the collection is on a different filesystem than the Compute endpoint,
or it publishes a subtree that does not contain /lus/eagle/projects/ABC/you/proj1.
```

The answer is written to `.pbxlocal.yaml` and never looked up again. Setting
`transfer_remote_root` yourself skips the whole thing, including the check.

### Limits

- **Output files stay on the HPC machine.** Bringing them back is not
  implemented; copy them with the Globus web interface, or your usual tool.
- **You have to register your own Globus app ID** for file transfers
  ([Part 4](#part-4--sending-your-input-files-too)). The intention is to ship
  one with parslbox, the way Globus Compute already does, registered through
  ALCF or a similar home — that is not done yet.
- **There is no merge between the two sides, and there will not be.** One side
  is authoritative at a time.
- **One remote machine per project.**
- Job folders are sent 500 recursive items per Globus task. A single request
  naming several thousand directories fails with an opaque API error.
- **The endpoint's parslbox is not upgraded with yours.** `pbx local status`
  prints the endpoint's version; compare it against your own with
  `pip show parslbox` after upgrading either side.
