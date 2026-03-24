"""Parkour task runner with ONNX export."""

import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner


class ParkourOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)

    # Build metadata without relying on get_base_metadata (which assumes
    # an "actor" observation group). Our observation groups are
    # "proprioception", "depth", and "critic".
    env = self.env.unwrapped
    metadata = {
      "run_path": run_name,
      "task_type": "parkour",
      "observation_groups": list(env.observation_manager.active_terms.keys()),
      "command_names": list(env.command_manager.active_terms),
    }
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
