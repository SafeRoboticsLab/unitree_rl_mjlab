```bash
python3 scripts/play.py Unitree-Go2-Parkour --checkpoint-file=logs/rsl_rl/go2_parkour/2026-03-23_21-45-59/model_15000.pt
```

```bash
python3 scripts/play.py Unitree-Go2-Flat --checkpoint-file=logs/rsl_rl/go2_velocity/2026-03-18_23-30-35/model_1000.pt
```

## Log storage

Training logs (checkpoints, tensorboard, ONNX, YAML configs) live on scratch.
The project-local `logs/` directory is a symlink:

```
/projects/FISAC/nr1053/unitree_rl_mjlab/logs
  -> /scratch/gpfs/FISAC/nr1053/unitree_rl_mjlab/logs
```

Everything written by `scripts/train.py` (which uses `Path("logs") / "rsl_rl" / <experiment>`)
transparently lands on scratch. Paths like `logs/rsl_rl/go2_parkour/...` in the
commands above resolve through the symlink — no code change, no CLI flag.

To revert: `rm /projects/FISAC/nr1053/unitree_rl_mjlab/logs`
(removes only the symlink; scratch contents are untouched).

To recreate:
```bash
mkdir -p /scratch/gpfs/FISAC/nr1053/unitree_rl_mjlab/logs
ln -s /scratch/gpfs/FISAC/nr1053/unitree_rl_mjlab/logs \
      /projects/FISAC/nr1053/unitree_rl_mjlab/logs
```

## Workstation runner (non-Slurm)

`run_train.sh` is the non-Slurm equivalent of `run_train.slurm`. Target:
workstation with 2x RTX PRO 6000 + 64 CPU cores.

### Usage
- `./run_train.sh` — default task (`Unitree-Go2-Safety-Shield`),
  `num_envs=2048`, both GPUs.
- `./run_train.sh <Task> [num_envs]` — override task / env count.
- `GPUS=0 ./run_train.sh` — single-GPU (skips torchrunx).
- `GPUS=0,1 ./run_train.sh` — both GPUs (default; `train.py` dispatches via
  torchrunx, one process per GPU).
- `SCRATCH_DIR=/some/other/path ./run_train.sh` — override the artifact root
  (defaults to `/scratch/gpfs/FISAC/nr1053/unitree_rl_mjlab/`).

### How artifacts get to scratch
The script `cd`s into `$SCRATCH_DIR` before invoking `python`. Because every
output sink anchors on CWD, this single change captures them all:

| Sink                                                      | Lands at                                          |
|-----------------------------------------------------------|---------------------------------------------------|
| Console log (`tee`)                                       | `$SCRATCH_DIR/local_runs/<task>_<ts>.log`         |
| RL run dir (checkpoints, tensorboard, `params/`, `git/`, `videos/train/`) | `$SCRATCH_DIR/logs/rsl_rl/<experiment>/<ts>/`     |
| torchrunx per-rank logs (multi-GPU)                       | `…/<ts>/torchrunx/` (nested in RL run dir)        |
| Wandb offline runs                                        | `$SCRATCH_DIR/wandb/` (also pinned via `WANDB_DIR`)|
| `MUJOCO_LOG.TXT`                                          | `$SCRATCH_DIR/MUJOCO_LOG.TXT`                     |
| NaN guard dumps (only if `--enable-nan-guard`)            | `/tmp/mjlab/nan_dumps` — override with `--env.sim.nan-guard.output-dir=…` |

The mechanism: `train.py` builds `Path("logs") / "rsl_rl" / experiment_name`
relative to CWD; wandb defaults to CWD; MuJoCo writes `MUJOCO_LOG.TXT` to
CWD. So `cd $SCRATCH_DIR` + invoking `scripts/train.py` by **absolute path**
(via `$PROJECT_DIR`) redirects everything at once. This supersedes the
symlink approach above — it also covers wandb and MuJoCo, not just rsl_rl.

### CPU thread budget
With 2 GPUs, BLAS/OMP threads are capped at `64 / num_gpus / 4 = 8` per
worker process (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`). Conservative on purpose: PyTorch and MuJoCo each
spawn their own threadpools, and oversubscription on a 64-core box is
easier to hit than to detect. Bump up if `top` shows CPU underuse.

### Gotchas
- `set -u` (nounset) trips on `/etc/bashrc` (`BASHRCSOURCED: unbound
  variable`) and on conda's activate hooks. The script wraps
  `source ~/.bashrc; conda activate …` in `set +u` … `set -u`.
- For multi-GPU, `train.py` overrides `MUJOCO_EGL_DEVICE_ID` per-rank
  inside `run_train()`, so the script's `EGL_DEVICE_ID=0` only matters on
  the single-GPU path.
- The script passes `--gpu-ids all` for multi-GPU and `--gpu-ids 0` for
  single-GPU explicitly. With `--gpu-ids all`, `train.py` reads
  `CUDA_VISIBLE_DEVICES` (set by the script from `$GPUS`) and indexes into
  it; with `[0]` it picks the first visible device.