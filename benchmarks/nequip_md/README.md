# NequIP E0/E1/B0/B1 MD benchmark

> **Unified-runner note:** the shell benchmark/validation drivers in this
> directory are retained for audit of the original E0/E1/B0/B1 work. They use
> the historical `force_temp=True` plus `Stationary` initialization and must
> not be mixed with the common project results. Formal Cu/H2O and DynaMat runs
> use `run_md_test.py` / `run_md_matbench.py` and `nequip.md_route:run_md`, which
> consume the shared `MDConfig` velocity, thermostat, and warmup semantics.

This directory implements four ASE MD baselines with the same
structures, initial velocities, timestep, warmup, production steps, and output
format.

The production reference configuration matches the eSEN and MatRIS baselines:
ASE `NVTBerendsen`, timestep 1 fs, `taut=100 fs`, Maxwell-Boltzmann velocities
with `force_temp=True`, seed 42, and target temperatures 300 K and 800 K. The
benchmark also retains an explicit NVE mode for diagnostics.

| Mode | Model execution | Neighbor list | Offline compilation |
| --- | --- | --- | --- |
| E0 | eager, standard e3nn | matscipy/CPU | no |
| E1 | `torch.compile`, standard e3nn | matscipy/CPU | no; first call compiles |
| B0 | AOTInductor + OpenEquivariance | matscipy/CPU | yes |
| B1 | the same B0 artifact | alchemiops/GPU | no additional compilation |

All modes explicitly disable TF32. E1 also disables TorchInductor CUDA Graphs;
the B0/B1 artifact is compiled with `triton.cudagraphs=False`.

## Expected cluster paths

The scripts default to:

```text
repository:
  /share/home/fushibo/MD_opt/nequip
model package:
  nequip.net:mir-group/NequIP-OAM-L:0.1
downloaded package:
  /share/home/fushibo/MD_opt/nequip/benchmark_artifacts/models/
  NequIP-OAM-L-0.1.nequip.zip
compiled artifact:
  /share/home/fushibo/MD_opt/nequip/benchmark_artifacts/
  NequIP-OAM-L-0.1-torch211-cu126-sm90-ase-oeq-no-cg.nequip.pt2
```

Run every command from the repository checkout. The active environment must
contain NequIP, ASE, matscipy, OpenEquivariance, and the alchemiops package used
by NequIP's `alchemiops` neighbor-list backend.

B1 specifically requires NVIDIA's `nvalchemi-toolkit-ops` distribution (import
name `nvalchemiops`). Official releases require Python 3.11 or newer. The
unified project environment is Python 3.12, PyTorch 2.11.0, and the CUDA 12.6
PyTorch wheel. Install NequIP without allowing pip to replace that torch build:

```bash
conda activate md_opt
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=9.0

cd /public-data/fushibo/nequip
python -m pip install -e . --no-build-isolation \
  -c /public-data/fushibo/md-opt-constraints.txt
python -m pip install openequivariance nvalchemi-toolkit-ops \
  -c /public-data/fushibo/md-opt-constraints.txt

python benchmarks/nequip_md/check_unified_environment.py
python benchmarks/nequip_md/check_unified_environment.py --require-accelerators
```

The source APIs accept this combination: NequIP 0.19 requires Torch >=2.2,
OpenEquivariance requires Torch >=2.7, and NequIP supports the
`nvalchemiops.torch.neighbors` API used by toolkit-ops >=0.3. This is only a
static compatibility check. We have not run the server-side H100 AOTInductor
compile under Torch 2.11/CUDA 12.6 yet; successful compilation plus the E0/B0/B1
trajectory parity test is the acceptance criterion. Never reuse the old Torch
2.10/CUDA 12.8 `.pt2` artifact in this environment.

## Structure manifest

The portable single-GPU pipeline generates every benchmark structure under
`benchmark_structures/` and writes a repository-relative manifest there:

```text
Cu16, Cu32, Cu64, Cu192, Cu512, Cu1024
H2O16, H2O32, H2O60, H2O64, H2O192, H2O512, H2O1024
```

That is 13 systems. `CuN` contains N Cu atoms, while `H2ON` contains N water
molecules (3N atoms). Both structure generation and MD velocity initialization
use seed 42 by default.

Generate the structures without running a benchmark:

```bash
python benchmarks/nequip_md/generate_structures.py
```

To run a custom structure set instead, copy and edit the example manifest:

```bash
cd /share/home/fushibo/MD_opt/nequip
cp benchmarks/nequip_md/systems.example.tsv benchmarks/nequip_md/systems.tsv
```

The manifest is tab separated, with one label and one structure path per line.
Generated manifests use paths relative to the repository root.

## Complete baseline on physical GPU 0

With the benchmark environment active, an internet-connected machine can run
structure generation, official-model download, B0/B1 compilation, numerical
validation, five performance repeats, and result summarization with:

```bash
DOWNLOAD_MODEL=1 bash benchmarks/nequip_md/run_baseline_gpu0.sh
```

On an offline H100 compute server, first run the download command on a login
node, then run:

```bash
bash benchmarks/nequip_md/run_baseline_gpu0.sh
```

All inputs and outputs controlled by this wrapper use repository-relative
paths. Each run receives a timestamped result directory so results from
different runs cannot be mixed.

## Compile B0/B1 once

On a regular 8-GPU server, with the environment already active:

```bash
CUDA_VISIBLE_DEVICES=1 \
  bash benchmarks/nequip_md/compile_model.sh
```

`CUDA_VISIBLE_DEVICES` selects the physical server GPU. The resulting artifact
is shared by B0 and B1.

To compile the optional constant-fold candidate, use a different output:

```bash
CUDA_VISIBLE_DEVICES=1 WITH_CONSTANT_FOLD=1 \
  bash benchmarks/nequip_md/compile_model.sh
```

Do not mix the constant-fold and non-constant-fold artifacts within one
E0/E1/B0/B1 comparison.

Because cluster compute nodes are offline, download the official package once
on an internet-connected login node before submitting Slurm jobs:

```bash
python benchmarks/nequip_md/fetch_official_model.py \
  --source nequip.net:mir-group/NequIP-OAM-L:0.1 \
  --output benchmark_artifacts/models/NequIP-OAM-L-0.1.nequip.zip
```

Then submit compilation:

```bash
bash benchmarks/nequip_md/submit_compile_slurm.sh
```

The download command fetches `nequip.net:mir-group/NequIP-OAM-L:0.1`, verifies
the ZIP, and records its source plus SHA256 at the stable package path above.
Slurm submission and compute jobs use `--verify-only`, so they never access the
network and reject a missing, modified, or differently sourced package. E0/E1
and B0/B1 therefore use the exact same official package. AOTInductor writes to
a temporary `.nequip.pt2`; the script publishes the formal artifact only after
NequIP's eager-versus-AOTI numerical check succeeds. Reuse requires valid
artifact and source-model checksums.

All Slurm jobs use the fixed cluster setup:

```text
partition: h100
GPU:       --gres=gpu:1
CPU:       --cpus-per-task=32
CUDA:      module load cuda/12.6
Conda:     /public-data/fushibo/miniconda3, environment md_opt
logs:      /public-data/fushibo/nequip/log/
```

Every Slurm stage sources `setup_slurm_env.sh`. It purges modules inherited
from the submit shell, loads CUDA 12.6, derives `CUDA_HOME` from that module's
`nvcc`, and rejects a missing CUDA header before launching Python. The compile
dependency also builds the OpenEquivariance extension in a versioned shared
`TORCH_EXTENSIONS_DIR`; validation and benchmark arrays reuse it instead of
starting concurrent first-time JIT builds. TorchInductor, Triton, and CUDA
generated-code caches are job-local under `SLURM_TMPDIR` (or `/tmp`) so stale
code from an earlier failed job cannot affect numerical validation.

## Validate 1000-step trajectories before timing

E0 is the numerical reference. Every mode starts from identical positions,
momenta, cell, PBC, timestep, and thermostat settings, then independently runs
all 1000 steps. The absolute error is the difference in total model
potential energy in eV; it is not divided by the number of atoms.

Required checkpoints:

```text
step 1:  abs(error) < 1e-8 eV
step 50: abs(error) < 1e-6 eV
```

Step 100 and step 1000 errors are always reported but have no pass threshold.
Even when step 1 or step 50 fails, the mode still completes step 1000 and is
marked `failed` in validation and performance JSON.

Run all six trajectory checks on one GPU:

```bash
GPU_ID=0 bash benchmarks/nequip_md/validate_all_modes.sh
```

Initial force and stress differences are retained as diagnostics, but the pass
decision follows the two trajectory-energy thresholds above. A numerical
failure returns a successful process exit status so that performance testing
continues and records the failed status.

## Unified model-owned route

The permanent outer runners use `nequip.md_route:run_md`. `--model-path`
always names the official saved package, including for B0/B1. The AOTInductor
artifact is passed separately so its provenance cannot be confused with the
model weights:

```text
E0/eager: saved package + e3nn eager + matscipy, TF32 disabled
E1/compile: saved package + torch.compile + matscipy, TF32 and CUDA Graphs disabled
B0: saved-package provenance + AOTI/OpenEquivariance artifact + matscipy
B1: same B0 artifact + alchemiops GPU neighbor list
```

E1/B0/B1 are retained engineering comparison modes; they are not renamed to
project opt1/opt2/opt3 because those stage names have fixed meanings. Future
stages continue to live in `nequip.md_stages.opt1` through `opt4`.

Run the canonical E0 Cu/H2O baseline from the common project root:

```bash
cd /public-data/fushibo
CUDA_VISIBLE_DEVICES=0 python run_md_test.py \
  --model nequip \
  --model-backend nequip.md_route:run_md \
  --model-path /public-data/fushibo/checkpoints/nequip/NequIP-OAM-L-0.1.nequip.zip \
  --stage baseline \
  --backend E0 \
  --structure-path /public-data/fushibo/md_test_data \
  --temperature-k 300 800 \
  --integrator berendsen \
  --steps 1000 \
  --warmup-steps 3 \
  --timing-repeats 5 \
  --output /public-data/fushibo/results/nequip/e0-cu-h2o
```

After compiling, B0/B1 use a JSON route option (the same artifact for both):

```bash
COMPILED=/public-data/fushibo/nequip/benchmark_artifacts/NequIP-OAM-L-0.1-torch211-cu126-sm90-ase-oeq-no-cg.nequip.pt2

CUDA_VISIBLE_DEVICES=0 python run_md_test.py \
  --model nequip \
  --model-backend nequip.md_route:run_md \
  --model-path /public-data/fushibo/checkpoints/nequip/NequIP-OAM-L-0.1.nequip.zip \
  --stage baseline --backend B1 \
  --route-options "{\"compiled_model_path\":\"$COMPILED\"}" \
  --structure-path /public-data/fushibo/md_test_data/Cu16.cif \
  --temperature-k 300 --steps 10 --observation-step 1 10 --warmup-steps 3 \
  --output /public-data/fushibo/results/nequip/b1-smoke
```

Do the 10-step smoke for E0, E1, B0, and B1 first. Then run 1000 steps and
compare step 1/50/100/1000 energy and force outputs before treating B0/B1 as
usable engineering references.

## Project Opt1: GPU-resident eager MD

`stage=opt1 --backend gpu-resident` uses the same official saved package and
eager model semantics as E0, while moving the complete NVT state and neighbor
construction to CUDA.  The implementation is
`nequip.md_stages.opt1`; it uses the public
`NequIPTorchSimCalc.from_saved_model(...)` entry point with the AlchemiOps
neighbor list.  Positions, momenta, masses, forces, and both thermostat
implementations are FP64 CUDA tensors.  Model parameters retain the dtype in
the checkpoint.

This stage intentionally rejects `.pt2` inputs and accelerated equivariance
modules.  AOTInductor, OpenEquivariance, `torch.compile`, CUDA Graphs, TF32,
and model-specific fusion are not Opt1.  If the official `.nequip.zip` cannot
be restored by the installed NequIP/PyTorch combination, loading fails with
the original exception chained; it never silently substitutes B0/B1.

Install the editable repository and Opt1 runtime dependencies in `md_opt`:

```bash
cd /public-data/fushibo/nequip
python -m pip install -e . --no-deps
python -m pip install torch-sim-atomistic nvalchemi-toolkit-ops
```

Run an E0-versus-Opt1 10-step smoke on one GPU:

```bash
cd /public-data/fushibo
CUDA_VISIBLE_DEVICES=0 python run_md_test.py \
  --model nequip \
  --model-backend nequip.md_route:run_md \
  --model-path /public-data/fushibo/checkpoints/nequip/NequIP-OAM-L-0.1.nequip.zip \
  --stage opt1 --backend gpu-resident --baseline-backend E0 \
  --structure /public-data/fushibo/md_test_data/Cu16.cif \
  --temperature-k 300 --integrator berendsen \
  --steps 10 --warmup-steps 3 --observation-step 1 10 \
  --timing-repeats 1 --dtype float64 --device cuda:0 \
  --output /public-data/fushibo/outputs/opt1-smoke/nequip
```

The Opt1 acceptance tests cover one-step ASE 3.29 parity for Berendsen and
Nose-Hoover-chain integration, retained baseline routing, later-stage policy
exclusion, persistent TorchSim topology, and Matbench step-0 trajectory
fields. Run them with the shared project root on `PYTHONPATH`:

```bash
cd /public-data/fushibo/nequip
PYTHONPATH=/public-data/fushibo python -m pytest \
  tests/unit/test_md_opt1.py -q
```

## Matbench DynaMat comparison

The exact matching registry entry is `NequIP-OAM-L:0.1`; its leaderboard YAML
is `models/nequip/nequip-oam-l-0.1.yml`. The published calculation uses a
TorchScript `.nequip.pth` compiled from the registry model. Current NequIP
explicitly rejects TorchScript compilation on Torch >=2.10, while the published
YAML records Python 3.12, `torch<2.10`, and H200 hardware. Therefore the unified
Torch 2.11/CUDA 12.6 E0 run uses the same official weights but is not a bitwise
reproduction of the published execution artifact or runtime.

The current official ZIP also records that it was packaged with NequIP 0.14.0,
Torch 2.7.0+cu128, e3nn 0.5.6, and an OpenEquivariance development build. The
package format is intended for loading by newer NequIP, but that metadata does
not prove Torch 2.11 runtime compatibility; the E0 10-step smoke is mandatory.

Run the common public-metric protocol with no warmup:

```bash
cd /public-data/fushibo
CUDA_VISIBLE_DEVICES=0 python run_md_matbench.py \
  --model nequip \
  --model-backend nequip.md_route:run_md \
  --model-path /public-data/fushibo/checkpoints/nequip/NequIP-OAM-L-0.1.nequip.zip \
  --stage baseline \
  --backend E0 \
  --structure-path /public-data/fushibo/matbench-discovery-data/md/2026-06-29-dynamat-v1.0-reference-trajectories.h5 \
  --matbench-repo /public-data/fushibo/matbench-discovery \
  --leaderboard-model-yaml /public-data/fushibo/matbench-discovery/models/nequip/nequip-oam-l-0.1.yml \
  --integrator nose_hoover_chain \
  --steps 80000 --timestep-fs 0.25 --thermostat-time-fs 25 \
  --warmup-steps 0 --record-interval 10 \
  --output /public-data/fushibo/results/nequip/matbench-e0
```

Compare RDF, ADF, vDOS, and pressure metrics to the YAML. The public HDF5 does
not contain the private energy/force labels, so its published energy RMSE and
force RMSE can only be quoted, not recomputed. Chaotic trajectory divergence,
H100 versus H200, Torch version, and eager versus TorchScript are expected to
prevent exact equality even when the common protocol is correct.

## Regular 8xH100 server

The server script distributes system groups over the GPUs. All repeats and
E0/E1/B0/B1 modes for one system run sequentially as separate processes on the
same physical GPU. Their order rotates between repeats. With six systems, at
most six of the eight H100 GPUs are used concurrently.

```bash
NGPUS=8 REPEATS=5 STEPS=1000 WARMUP_STEPS=3 \
  bash benchmarks/nequip_md/run_8xh100.sh
```

By default the performance script requires
`benchmark_results/nequip_e0_e1_b0_b1/validation/<label>.json` and embeds the
corresponding validation status and checkpoint errors into every result.

Optional CPU pinning can prevent the three CPU-neighbor-list modes from
contending for the same cores:

```bash
export CPUSET_0=0-15
export CPUSET_1=16-31
# ... set CPUSET_2 through CPUSET_7 for the actual server topology
bash benchmarks/nequip_md/run_8xh100.sh
```

Do not copy the example core ranges without checking `lscpu -e` and the GPU/NUMA
topology first. If independent CPU core/NUMA allocation cannot be guaranteed,
use `NGPUS=1` for the strict publication run; `NGPUS=8` is primarily the faster
throughput mode.

## Slurm H100 compile, validation, performance, and summary pipeline

The pipeline submits all stages with `sbatch` and scheduler dependencies:

```bash
REPEATS=5 \
STEPS=1000 \
bash benchmarks/nequip_md/submit_pipeline_slurm.sh
```

Each invocation creates a timestamped result directory below
`benchmark_results/nequip_e0_e1_b0_b1/`, preventing results from different runs
or repeat counts from being mixed. Set `RUN_ID=my-run-name` to choose a stable
name.

The dependency chain is:

```text
compile AOTI+OpenEquivariance model
    --afterok-->
validation array (one job per system)
    --afterok-->
performance array (one job per system, all repeats in that job)
    --afterany-->
summary job (writes summary.json and summary.csv)
```

For the current manifest, validation has 13 array tasks and performance has 13
array tasks. Each performance task owns one structure, one H100, and runs all
repeats plus E0/E1/B0/B1 sequentially on that same GPU. Mode order rotates
between repeats.

Numerical threshold failures do not fail validation jobs because they are data,
so performance still runs and records `failed`. Runtime failures stop the
`afterok` validation-to-performance dependency. Summary uses `afterany`, so it
can still collect partial output after all performance tasks terminate.

To submit only the performance stage after validation files already exist:

```bash
REPEATS=5 \
STEPS=1000 \
bash benchmarks/nequip_md/submit_benchmarks_slurm.sh
```

The complete pipeline submits the summary automatically. For a separately
submitted performance array, submit summary using the returned job ID:

```bash
BENCHMARK_JOB_ID=<job-id> bash benchmarks/nequip_md/submit_summary_slurm.sh
```

Set `SLURM_ACCOUNT` only if the cluster requires an account.

## Timing semantics

The following operations are outside the measured MD region:

- model/package loading;
- E1's first `torch.compile` invocation;
- OpenEquivariance JIT initialization;
- three warmup MD steps;
- model hashing and JSON writing.

After warmup, positions, cell, PBC, and momenta are restored to the identical
initial snapshot. Timing uses one CUDA synchronization before and after the
whole production trajectory; there is no benchmark-added per-step sync.
Trajectory and ASE logging are disabled.

## Summarize results

```bash
python benchmarks/nequip_md/summarize_results.py \
  benchmark_results/nequip_e0_e1_b0_b1
```

This writes `summary.json` and `summary.csv`, reports median/mean/spread, and
calculates E1/E0, B0/E0, B1/B0, and B1/E0 speedups. It also warns if B0 and B1
did not use the same compiled artifact hash, or if any optimized mode failed or
lacks E0 trajectory validation.
