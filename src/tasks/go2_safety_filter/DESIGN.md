# Go2 Safety Filter — Design

A streamlined codebase that trains a **walking policy** (nominal locomotion on
rough terrain) and a **safety policy** (HJ-reachability value function + fallback
controller on parkour terrain with gap-crossing exposure), then combines them at
deploy time with a **hard-switch shield**:

```
a_t = π_walk(s_t)    if V_safe(s_t) >  τ      (nominal walking)
a_t = π_safe(s_t)    if V_safe(s_t) <= τ      (safety fallback)
```

Everything lives under [src/tasks/go2_safety_filter/](.).

---

## 1. Prior work in this repo (what we reuse verbatim)

| Component | Path | Why we reuse |
|---|---|---|
| `ParkourSafetyVecEnvWrapper` | [src/tasks/parkour/rl/safety_vecenv_wrapper.py](../parkour/rl/safety_vecenv_wrapper.py) | Computes `g(s) = min(base_height, tilt, non-foot contact)` margin, overrides env reward, emits `-0.1` on termination. |
| `SafetyPPO` | [src/tasks/parkour/rl/safety_ppo.py](../parkour/rl/safety_ppo.py) | Implements the Safety Bellman Backup `v_to_go = min(g_t, V(s_{t+1}))`, skips timeout bootstrapping. |
| `reset_robot_midair_over_gaps` | [src/tasks/parkour/mdp/safety_events.py](../parkour/mdp/safety_events.py) | Mid-air spawns above gaps with forward velocity — the key trick for teaching gap crossing. |
| Parkour MDP (rewards, obs, terms, curricula) | [src/tasks/parkour/mdp/](../parkour/mdp/) | Already tuned for quadruped parkour. |
| Parkour terrains | [src/tasks/parkour/terrains.py](../parkour/terrains.py) | Gap jump, crawl, rugged, mixed, flat. |
| `mjlab.terrains.config.ROUGH_TERRAINS_CFG` | mjlab package | Flat + stairs + slopes + waves + random rough (no gaps, no crawl). Exactly what the user asked for the walking policy. |
| `ParkourOnPolicyRunner` / `ParkourSafetyOnPolicyRunner` | [src/tasks/parkour/rl/](../parkour/rl/) | ONNX export + safety wrapper injection. |

The existing `Unitree-Go2-Parkour` and `Unitree-Go2-Parkour-Safety` tasks remain
untouched (user requirement #4).

---

## 2. Policy architectures and observation contracts

**Proprioception group (shared, identical structure in both tasks):**
```
base_ang_vel (3) + projected_gravity (3) + command (3) + phase (2)
+ base_height (1) + joint_pos (12) + joint_vel (12) + last_action (12)
+ height_scan (17×11 = 187)                       -> 235 per step
× history_length = 5                               -> 1175
```
Both the walking MLP and the safety CNN consume this group with the exact same
shape → the trained normalizers transfer 1:1 at deploy time.

**Depth group (safety only):**
```
depth_image: (B, 1, 64, 64), gaps→far remapped, clip 0.01..3.0 m
```

**Critic group (privileged, training only):** true `base_lin_vel`, foot height
/ contact / forces, forward distance + all proprio terms.

| Task | Actor input | Actor arch | Critic arch | Algo |
|---|---|---|---|---|
| `Unitree-Go2-Walking` | proprioception | MLPModel (512, 256, 128) | MLPModel (512, 256, 128) | PPO |
| `Unitree-Go2-Safety-Shield` | proprioception + depth | CNNModel (32/64/64, 5/3/3, stride 2) + MLP (256, 128) | MLPModel (512, 256, 128) | SafetyPPO |

Why they can share proprioception verbatim:
- We **keep** `height_scan` inside `proprioception` for the safety task (the
  upstream `parkour_safety_env_cfg` strips it for deployability-without-raycast;
  we override that because the shield runs in a parkour env where raycast is
  available). The safety critic still gets everything else as privileged info.
- Both tasks are built from the same `make_parkour_env_cfg()` factory, so
  observation ordering, noise, history length, and scales are byte-identical.

---

## 3. Walking task (`Unitree-Go2-Walking`)

Built from `make_parkour_env_cfg()` with these modifications:

- **Terrain:** swap `PARKOUR_TERRAINS_CFG` → `ROUGH_TERRAINS_CFG` (flat, stairs
  up/down, slopes, wave, random rough). **No gaps. No crawl barriers.**
- **Sensors:** drop `front_depth` camera (no depth branch needed in MLP).
- **Observations:** drop the `depth` group. Keep `proprioception` and `critic`.
- **Commands:** widen twist to full range: `lin_vel_x ∈ [-1.0, 2.0]`,
  `lin_vel_y ∈ [-1.0, 1.0]`, `ang_vel_z ∈ [-1.0, 1.0]`, 5% standing, heading
  enabled. This matches `mjlab.tasks.velocity` defaults.
- **Rewards:** drop `gap_crossing`, `gap_crossing_bonus`, `forward_progress`
  (parkour-specific). Keep all standard locomotion rewards. Keep
  `is_terminated: -50` (per user direction).
- **Events:** standard `reset_root_state_uniform` (no mid-air reset). Push-robot
  retained.
- **Terminations:** `time_out`, `bad_orientation (70°)`, `illegal_contact`. Drop
  `base_too_low` (no gaps to fall into).
- **Curriculum:** `terrain_levels_vel` from the velocity package (distance-based
  terrain level up/down).

---

## 4. Safety task (`Unitree-Go2-Safety-Shield`)

Built on top of `unitree_go2_parkour_safety_env_cfg` with these targeted tunes:

1. **Keep `height_scan` in proprioception.** We restore it (the parent config
   strips it). Reason: the shield runs in parkour sim where raycast is
   available, and identical proprio structure lets the walking policy plug into
   the same env with no obs bridging.
2. **Widen velocity command to `vx ∈ [0.3, 1.2]`.** The parent config's
   `[0, 0.5]` makes the gait-phase observation nearly stationary and biases
   π_safe toward walking styles that don't match π_walk. A wider command
   consistent with the walking policy's training distribution produces a
   smoother hand-off.
3. **Keep `is_terminated: -50` and `midair_fraction=0.5`.** Reward is overridden
   by `g(s)` in the wrapper regardless.

All other modifications from the parent config (mid-air resets over gaps,
stronger pushes, SafetyPPO hyperparameters) are inherited as-is.

---

## 5. Runtime shield

At play time both checkpoints are loaded into a single `ManagerBasedRlEnv`
configured as the safety env in play mode (parkour terrain + all sensors).

```python
class ShieldedPolicy:
    def __init__(self, walk_actor, safe_actor, safe_critic, threshold):
        ...

    def __call__(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            v_safe = self.safe_critic(obs).squeeze(-1)        # (num_envs,)
        unsafe = v_safe <= self.threshold                      # bool mask
        a_walk = self.walk_actor(obs)
        a_safe = self.safe_actor(obs)
        return torch.where(unsafe.unsqueeze(-1), a_safe, a_walk)
```

- **Threshold `τ`** defaults to `0.05` (safety margin at which we switch in).
  Exposed as a CLI flag.
- **Hand-off is hard** (per user spec) — no blending.
- Switch events and `V_safe` trajectories are logged for diagnostics.

The play env uses the safety env's play config (parkour terrain, infinite
episode, mid-air-reset fraction 0.25 for interesting visualization).

---

## 6. File layout

```
src/tasks/go2_safety_filter/
  __init__.py                  # task registration
  DESIGN.md                    # this file
  walking/
    __init__.py
    env_cfg.py                 # rough/flat terrain, no gaps/crawl
    rl_cfg.py                  # MLP PPO
  safety/
    __init__.py
    env_cfg.py                 # wraps parent parkour_safety; restores height_scan; widens vx
    rl_cfg.py                  # CNN SafetyPPO (identical to parent)
  shield/
    __init__.py
    shielded_policy.py         # hard-switch runtime wrapper
scripts/
  play_shielded.py             # load both .pt, drive parkour env with shield
```

## 7. Running it

```bash
# 1. Walking policy (quick, single MLP, rough terrain)
python scripts/train.py Unitree-Go2-Walking --env.scene.num-envs=4096

# 2. Safety policy (slower, CNN over depth, parkour terrain)
python scripts/train.py Unitree-Go2-Safety-Shield --env.scene.num-envs=256

# 3. Shielded play
python scripts/play_shielded.py \
    --walk-checkpoint=logs/rsl_rl/go2_walking_shield/<run>/model_<k>.pt \
    --safe-checkpoint=logs/rsl_rl/go2_safety_shield/<run>/model_<k>.pt \
    --threshold=0.05 --num-envs=4
```
