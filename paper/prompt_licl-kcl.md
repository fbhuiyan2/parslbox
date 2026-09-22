# LiCl-KCl Training-Data Workflow Prompt

```text
/mlip-training-data-generation Use the skill to generate a DFT-labeled training
dataset for binary molten-salt system. Run it end-to-end in one shot:
prep locally, register the full ParslBox DAG (MD -> prep_dft -> DFT -> aggregate) in
order, then submit with a single pbx qsub. Ask me only if something is genuinely
ambiguous or a required input is missing - otherwise use the settings below.

=== SYSTEM ===
- LiCl-KCl binary molten salt.
- Compositions (mol% LiCl): 0, 25, 50, 75, 100  (0 = pure KCl, 100 = pure LiCl).
- Liquid MD temperatures: 1050 K, 1150 K, 1250 K  (every composition x every T).
- Endpoint densities for box packing: LiCl 1.502 g/cc, KCl 1.527 g/cc
  (interpolate for intermediate compositions).
- Melting points (reference): LiCl 883 K, KCl 1044 K.
- Include BOTH crystals AND surfaces (slabs).
  - LiCl crystal: use Materials Project entry mp-22905.
  - KCl crystal: use Materials Project entry mp-23193.
  - Sample crystals + surfaces at 300 K and 400 K (both below both melting points).

=== FOUNDATION MODEL (drives the MD sampling) ===
- MACE, omat_pbe head, version mh-0.
- Checkpoint (.model) absolute path: [FOUNDATION_MODEL_PATH]

=== ENVIRONMENT ===
- Tools venv (parslbox, ase, pymatgen, mp-api, dscribe, skmatter, packmol, numpy, pyyaml):
    source [TOOLS_ENV_ACTIVATE]
  Verify the needed packages import before running each script; if the LAMMPS/MACE
  MD path needs symmetrix + mace-torch and they're missing, tell me before proceeding.
- MP_API key: [REDACTED MP API KEY]

=== HPC / ParslBox ===
- pbx system config: aurora-tile
- PBS project/allocation: [REDACTED ALLOCATION]
- Queue: capacity
- Per-job resources: MD jobs 1 GPU/job; VASP jobs 1 GPU/job (--nodes 1 --ngpus 1).
- Set PBX_DB_PATH=$(pwd) at the project root; submit with pbx qsub --rundir ./pbxrun.
- Don't pick the batch node count / walltime up front - after the DAG is registered,
  run pbx info to see how the jobs pack, then propose node count + walltime and
  confirm with me before the final qsub.

=== DFT (VASP) LABELING ===
- POTCAR library: [POTCAR_LIBRARY]
- Functional: PBE (incar_pbe).
- INCAR overrides: ENCUT = 700, EDIFF = 1e-7, KSPACING = 0.25.
- ISPIN = 1 (closed-shell main-group salt; drop MAGMOM).
- LWAVE = .FALSE., LCHARG = .FALSE. (many single-points, no restart benefit).
- Keep other incar_pbe template defaults. Show me the full assembled INCAR
  (build_vasp_inputs.py --preview, with the real ENCUT/KSPACING) for sign-off
  BEFORE registering the VASP jobs.

=== SAMPLING SIZES ===
- MD production length: 100 ps.
- Frames per MD trajectory for DFT labeling: 50.

Final output: training_data.xyz at the project root (per-frame info['tag'] and
info['functional']). Use dry-run / preview modes before each registration and
submission step.
```
