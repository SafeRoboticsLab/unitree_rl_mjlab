"""On-policy ISAACS runner: two-player adversarial reach-avoid on rsl_rl.

Owns both players of the zero-sum reach-avoid game:

* ctrl (max player)  — the task's ``ReachAvoidPPO`` over proprioception,
  warm-started from the single-agent checkpoint; unchanged algorithm.
* dstb (min player)  — ``DstbReachAvoidPPO`` (negated advantage) over the
  privileged ``critic`` observation group, emitting a 3-D force direction
  applied to ``base_link`` by :class:`AdversarialReachAvoidVecEnvWrapper`.

Phase machine (ISAACS alternation, on-policy):

    [dstb pretrain: ctrl frozen]  ->  [K dstb iters | M ctrl iters | ...]

During ctrl phases the disturbance comes from a fixed opponent mixture over
env slices (increment 1: ZERO / RANDOM / current frozen dstb; increment 2
replaces this with the league sampler).  The frozen player always acts
stochastically (ISAACS rollout semantics).
"""

from __future__ import annotations

import os
import time

import torch
from rsl_rl.env import VecEnv
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.storage.rollout_storage import RolloutStorage

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from src.tasks.parkour.rl.adversarial_vecenv_wrapper import (
  AdversarialReachAvoidVecEnvWrapper,
)
from src.tasks.parkour.rl.dstb_ppo import DstbReachAvoidPPO
from src.tasks.parkour.rl.league import (
  SLICE_CURRENT, SLICE_RANDOM, SLICE_ZERO, Leaderboard,
)
from src.tasks.parkour.rl.reach_avoid_runner import _install_log_std_clamp
from src.tasks.parkour.rl.runner import ParkourOnPolicyRunner

# Opponent codes for ctrl-phase env slices (increment 1 fixed mixture).
_OPP_ZERO = 0
_OPP_RANDOM = 1
_OPP_CURRENT = 2


def _clamp_log_std(optimizer, actor, lo: float, hi: float) -> None:
  """Same pattern as reach_avoid_runner's clamp, with custom bounds."""
  import math

  lo_v, hi_v = math.log(lo), math.log(hi)

  def _pre(opt, *a, **k):
    with torch.no_grad():
      for group in opt.param_groups:
        for p in group["params"]:
          if p.grad is not None and not torch.isfinite(p.grad).all():
            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

  def _post(opt, *a, **k):
    with torch.no_grad():
      if hasattr(actor, "log_std"):
        ls = actor.log_std.data
        torch.nan_to_num_(ls, nan=hi_v, posinf=hi_v, neginf=lo_v)
        ls.clamp_(min=lo_v, max=hi_v)

  optimizer.register_step_pre_hook(_pre)
  optimizer.register_step_post_hook(_post)
  _post(optimizer)


class Go2IsaacsOnPolicyRunner(ParkourOnPolicyRunner):
  """Two-player (ctrl vs dstb force) on-policy ISAACS runner."""

  def __init__(self, env: VecEnv, train_cfg: dict, log_dir=None, device="cpu"):
    icfg = train_cfg.get("isaacs", {})
    self._force_max = float(icfg.get("force_max", 50.0))
    self._pretrain_iters = int(icfg.get("dstb_pretrain_iters", 400))
    self._ctrl_per_cycle = int(icfg.get("ctrl_iters_per_cycle", 5))
    self._dstb_per_cycle = int(icfg.get("dstb_iters_per_cycle", 5))
    self._force_ramp_iters = int(icfg.get("force_scale_ramp_iters", 200))
    self._edge_clearance = float(icfg.get("rest_edge_clearance", 0.3))
    self._edge_ramp_iters = int(icfg.get("edge_ramp_ctrl_iters", 300))
    self._dstb_actor_cfg = dict(icfg.get("dstb_actor", {}))
    self._dstb_alg_cfg = dict(icfg.get("dstb_algorithm", {}))

    if isinstance(env, RslRlVecEnvWrapper) and not isinstance(
      env, AdversarialReachAvoidVecEnvWrapper
    ):
      env = AdversarialReachAvoidVecEnvWrapper(
        env.env,
        clip_actions=env.clip_actions,
        force_max=self._force_max,
        reach_mode="rest",
        v_rest=0.3,
        v_rest_norm=0.5,
        cross_bias_weight=0.3,
        cross_bias_scale=3.0,
        rest_edge_clearance=0.0,  # ramped in during ctrl phases
      )
    super().__init__(env, train_cfg, log_dir, device)
    _install_log_std_clamp(self)  # ctrl clamp [0.10, 0.40]

    # --- build the min player -------------------------------------------
    obs = self.env.get_observations().to(self.device)
    dstb_groups = {"actor": ["critic"], "critic": ["critic"]}
    self._dstb_actor_cfg.pop("class_name", None)
    actor_kwargs = dict(
      hidden_dims=self._dstb_actor_cfg.get("hidden_dims", (256, 256, 128)),
      activation=self._dstb_actor_cfg.get("activation", "elu"),
      obs_normalization=self._dstb_actor_cfg.get("obs_normalization", True),
      stochastic=True,
      init_noise_std=self._dstb_actor_cfg.get("init_noise_std", 0.5),
      noise_std_type=self._dstb_actor_cfg.get("noise_std_type", "log"),
    )
    dstb_actor = MLPModel(obs, dstb_groups, "actor", 3, **actor_kwargs).to(self.device)
    dstb_critic = MLPModel(
      obs, dstb_groups, "critic", 1,
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
    ).to(self.device)
    # The ctrl critic reads the same privileged group with the same arch:
    # initialize the min player's value estimates from the trained max critic.
    try:
      dstb_critic.load_state_dict(self.alg.critic.state_dict())
    except RuntimeError:
      print("[isaacs] dstb critic init from ctrl critic failed (arch mismatch); fresh init")

    dstb_obs = obs.select("critic")
    dstb_storage = RolloutStorage(
      "rl", self.env.num_envs, self.cfg["num_steps_per_env"], dstb_obs, [3],
      self.device,
    )
    alg_kwargs = dict(self._dstb_alg_cfg)
    alg_kwargs.pop("class_name", None)
    alg_kwargs.pop("rnd_cfg", None)
    alg_kwargs.pop("symmetry_cfg", None)
    self.dstb_alg = DstbReachAvoidPPO(
      dstb_actor, dstb_critic, dstb_storage, device=self.device, **alg_kwargs
    )
    _clamp_log_std(self.dstb_alg.optimizer, dstb_actor, 0.10, 0.60)

    # phase state
    self._iters_done = 0        # total isaacs iterations run
    self._ctrl_iters_done = 0   # ctrl-phase iterations (for edge ramp)
    # per-env force scale (post-pretrain curriculum: the adversary's strength
    # tracks what each env's ctrl can currently bear — anti-treadmill)
    self._env_force_scale = torch.full((self.env.num_envs,), 0.4,
                                       device=self.device)
    n = self.env.num_envs
    # League (increment 2): archive + softmax opponents over env slices.
    self._n_slices = int(icfg.get("num_slices", 8))
    lb_dir = os.path.join(log_dir or ".", "league")
    self._league = Leaderboard(
      save_top_k_ctrl=int(icfg.get("save_top_k_ctrl", 5)),
      save_top_k_dstb=int(icfg.get("save_top_k_dstb", 5)),
      softmax_rationality=float(icfg.get("softmax_rationality", 3.0)),
      model_dir=lb_dir,
    )
    # scratch nets for archived dstb opponents (state-dict loads per rollout)
    self._scratch_dstb = [
      MLPModel(obs, dstb_groups, "actor", 3, **actor_kwargs).to(self.device)
      for _ in range(self._league.kd)
    ]
    self._slice_opponents = [SLICE_ZERO, SLICE_RANDOM] + [SLICE_CURRENT] * (
      self._n_slices - 2
    )
    self._slice_of_env = (
      torch.arange(n, device=self.device) * self._n_slices // max(n, 1)
    ).clamp(max=self._n_slices - 1)
    # per-env reach-avoid episode flags (fill the board from training outcomes)
    self._ever_l = torch.zeros(n, dtype=torch.bool, device=self.device)
    self._ever_gneg = torch.zeros(n, dtype=torch.bool, device=self.device)

  # --- phase helpers ---------------------------------------------------------

  def _phase(self) -> str:
    if self._iters_done < self._pretrain_iters:
      return "dstb"
    k = (self._iters_done - self._pretrain_iters) % (
      self._dstb_per_cycle + self._ctrl_per_cycle
    )
    return "dstb" if k < self._dstb_per_cycle else "ctrl"

  def _force_scale_now(self) -> float:
    if self._iters_done >= self._force_ramp_iters:
      return 1.0
    return 0.4 + 0.6 * self._iters_done / max(self._force_ramp_iters, 1)

  def _resample_slice_opponents(self) -> None:
    """League: pick this rollout's opponent per env slice; load archived nets."""
    self._slice_opponents = self._league.sample_dstb_slices(self._n_slices)
    loaded = {}
    for si, opp in enumerate(self._slice_opponents):
      if opp >= 0 and opp not in loaded:
        scratch = self._scratch_dstb[len(loaded) % len(self._scratch_dstb)]
        self._league.load_actor(
          scratch, "dstb", self._league.dstb_steps[opp]
        )
        scratch.to(self.device)
        loaded[opp] = scratch
    self._loaded_scratch = loaded

  def _dstb_rollout_action(self, obs) -> torch.Tensor:
    """Ctrl-phase disturbance: per-slice league opponents (stochastic)."""
    n = self.env.num_envs
    act = torch.zeros(n, 3, device=self.device)
    cur_out = None
    for si, opp in enumerate(self._slice_opponents):
      mask = self._slice_of_env == si
      if not bool(mask.any()):
        continue
      if opp == SLICE_ZERO:
        continue
      if opp == SLICE_RANDOM:
        act[mask] = torch.randn(int(mask.sum()), 3, device=self.device)
      elif opp == SLICE_CURRENT:
        if cur_out is None:
          cur_out = self.dstb_alg.actor(
            obs.select("critic"), stochastic_output=True
          )
        act[mask] = cur_out[mask]
      else:
        a = self._loaded_scratch[opp](
          obs.select("critic"), stochastic_output=True
        )
        act[mask] = a[mask]
    return act

  # --- main loop -------------------------------------------------------------

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    self.dstb_alg.train_mode()
    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      phase = self._phase()
      if self._iters_done < self._pretrain_iters:
        self.env.set_force_scale(self._force_scale_now())
      else:
        self.env.set_force_scale(self._env_force_scale)
      # ramp the robustified-rest edge clearance during ctrl phases
      if phase == "ctrl":
        frac = min(1.0, self._ctrl_iters_done / max(self._edge_ramp_iters, 1))
        self.env.set_rest_edge_clearance(self._edge_clearance * frac)

      if phase == "ctrl":
        with torch.inference_mode():
          self._resample_slice_opponents()
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          if phase == "dstb":
            dstb_actions = self.dstb_alg.act(obs.select("critic"))
            ctrl_actions = self.alg.actor(obs, stochastic_output=True)
          else:
            ctrl_actions = self.alg.act(obs)
            dstb_actions = self._dstb_rollout_action(obs)
          self.env.set_dstb_action(dstb_actions)
          obs, rewards, dones, extras = self.env.step(
            ctrl_actions.to(self.env.device)
          )
          obs, rewards, dones = (
            obs.to(self.device), rewards.to(self.device), dones.to(self.device)
          )
          if phase == "dstb":
            self.dstb_alg.process_env_step(
              obs.select("critic"), rewards, dones, extras
            )
          else:
            self.alg.process_env_step(obs, rewards, dones, extras)
          self.logger.process_env_step(rewards, dones, extras, None)
          # reach-avoid episode flags for the league board
          ell = extras.get("target_margin")
          if ell is not None:
            self._ever_l |= ell.to(self.device) >= 0
          self._ever_gneg |= rewards < 0
          if self._iters_done >= self._pretrain_iters and bool(dones.any()):
            t_o = extras.get("time_outs", torch.zeros_like(dones)).bool()
            d_m = dones.bool()
            self._env_force_scale = torch.where(
              d_m & t_o, self._env_force_scale + 0.05, self._env_force_scale)
            self._env_force_scale = torch.where(
              d_m & ~t_o, self._env_force_scale - 0.05, self._env_force_scale)
            self._env_force_scale.clamp_(0.3, 1.0)
            # fill the league board from training outcomes
            succ = d_m & t_o & self._ever_l & ~self._ever_gneg
            if phase == "ctrl":
              for si, opp in enumerate(self._slice_opponents):
                m = d_m & (self._slice_of_env == si)
                if not bool(m.any()):
                  continue
                rate = float(succ[m].float().mean())
                col = (self._league.board.shape[1] - 1 if opp == SLICE_ZERO
                       else self._league.board.shape[1] - 2 if opp in
                       (SLICE_CURRENT, SLICE_RANDOM) else opp)
                self._league.ema_score(-1, col, rate)
            else:
              rate = float(succ[d_m].float().mean())
              self._league.ema_score(-1, self._league.board.shape[1] - 2, rate)
            self._ever_l = torch.where(d_m, torch.zeros_like(self._ever_l),
                                       self._ever_l)
            self._ever_gneg = torch.where(
              d_m, torch.zeros_like(self._ever_gneg), self._ever_gneg)

        collect_time = time.time() - start
        start = time.time()
        if phase == "dstb":
          self.dstb_alg.compute_returns(obs.select("critic"))
        else:
          self.alg.compute_returns(obs)

      if phase == "dstb":
        loss_dict = {f"dstb_{k}": v for k, v in self.dstb_alg.update().items()}
      else:
        loss_dict = self.alg.update()
        self._ctrl_iters_done += 1
      loss_dict["isaacs_phase_is_dstb"] = 1.0 if phase == "dstb" else 0.0
      loss_dict["isaacs_force_scale"] = float(self._env_force_scale.mean())
      learn_time = time.time() - start
      self._iters_done += 1
      self.current_learning_iteration = it

      cycle = self._dstb_per_cycle + self._ctrl_per_cycle
      if (self._iters_done > self._pretrain_iters
          and (self._iters_done - self._pretrain_iters) % cycle == 0):
        self._league.prune(self._iters_done, self.alg.actor,
                           self.dstb_alg.actor)
        loss_dict["league_archived_dstb"] = float(len(self._league.dstb_steps))

      self.logger.log(
        it=it, start_it=start_it, total_it=total_it,
        collect_time=collect_time, learn_time=learn_time,
        loss_dict=loss_dict, learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std, rnd_weight=None,
      )
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      self.save(os.path.join(
        self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
      ))
      self.logger.stop_logging_writer()

  # --- checkpointing ---------------------------------------------------------

  def save(self, path: str, infos=None):
    try:
      super().save(path, infos)  # ctrl dict + ONNX export + wandb upload
    except Exception as e:  # noqa: BLE001 — upload/export failures must not
      # prevent the dstb state from being appended to the checkpoint.
      print(f"[isaacs] base save export/upload failed ({e}); continuing")
    d = torch.load(path, weights_only=False)
    d["dstb_actor_state_dict"] = self.dstb_alg.actor.state_dict()
    d["dstb_critic_state_dict"] = self.dstb_alg.critic.state_dict()
    d["dstb_optimizer_state_dict"] = self.dstb_alg.optimizer.state_dict()
    d["isaacs_state"] = {
      "iters_done": self._iters_done,
      "ctrl_iters_done": self._ctrl_iters_done,
      "env_force_scale_mean": float(self._env_force_scale.mean()),
      "league_board": self._league.board.tolist(),
      "league_ctrl_steps": list(self._league.ctrl_steps),
      "league_dstb_steps": list(self._league.dstb_steps),
    }
    torch.save(d, path)

  def load(self, path: str, load_cfg=None, strict: bool = True,
           map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    d = torch.load(path, weights_only=False, map_location=map_location)
    if "dstb_actor_state_dict" in d:
      self.dstb_alg.actor.load_state_dict(d["dstb_actor_state_dict"], strict=strict)
      self.dstb_alg.critic.load_state_dict(d["dstb_critic_state_dict"], strict=strict)
      self.dstb_alg.optimizer.load_state_dict(d["dstb_optimizer_state_dict"])
      st = d.get("isaacs_state", {})
      self._iters_done = int(st.get("iters_done", 0))
      self._ctrl_iters_done = int(st.get("ctrl_iters_done", 0))
      print("[isaacs] loaded dstb player state")
    else:
      # Warm start from a single-agent checkpoint: dstb critic tracks the
      # freshly-loaded ctrl critic; dstb actor stays fresh.
      try:
        self.dstb_alg.critic.load_state_dict(self.alg.critic.state_dict())
      except RuntimeError:
        pass
      print("[isaacs] single-agent warm start (fresh dstb player)")
    return infos
