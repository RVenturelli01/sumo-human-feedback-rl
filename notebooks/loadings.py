import numpy as np
import torch
import torch.nn as nn

from human_feedback_rl.common.reward_nets import RewardNet, RewardEnsemble

_STATUS_DIM = 7  # [arrived, collided, off_road, timeout, running, teleported, removed_unknown]


class _Member(RewardNet):
    def __init__(self, obs_space, act_space, net: nn.Sequential, uses_status: bool):
        super().__init__(obs_space, act_space)
        self.net = net
        self._uses_status = uses_status

    def forward(self, state, action, next_status=None, done=None):
        if self._uses_status:
            x = torch.cat([state, action, next_status, done.unsqueeze(-1)], dim=1)
        else:
            x = torch.cat([state, action], dim=1)
        return self.net(x).squeeze(-1)


def _infer_arch(sd: dict, member_idx: int = 0):
    """Return (input_dim, hidden_dims, has_bias_last) by inspecting state dict keys."""
    prefix = f"members.{member_idx}.net."
    weight_keys = sorted(
        [k for k in sd if k.startswith(prefix) and k.endswith(".weight")],
        key=lambda k: int(k[len(prefix):].split(".")[0]),
    )
    if not weight_keys:
        raise ValueError(f"No weights found for member {member_idx} in checkpoint.")
    input_dim = sd[weight_keys[0]].shape[1]
    hidden_dims = [sd[k].shape[0] for k in weight_keys[:-1]]
    last_idx = weight_keys[-1][len(prefix):].split(".")[0]
    has_bias_last = f"{prefix}{last_idx}.bias" in sd
    return input_dim, hidden_dims, has_bias_last


def _build_net(input_dim: int, hidden_dims: list, has_bias_last: bool) -> nn.Sequential:
    layers, in_d = [], input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_d, h), nn.Tanh()]
        in_d = h
    layers.append(nn.Linear(in_d, 1, bias=has_bias_last))
    return nn.Sequential(*layers)


def load_reward_ensemble(path: str, obs_space, act_space, device: str = "cpu") -> RewardEnsemble:
    """Load a RewardEnsemble from a .pt checkpoint.

    Supports both raw state dicts and wrapped checkpoints of the form
    {"state_dict": ..., "n_members": ..., "obs_dim": ..., "act_dim": ...}.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        n_members = ckpt.get("n_members")
    else:
        sd = ckpt
        n_members = None

    # Saved from NormalizedRewardNet wrapper: keys are "net.members.*" → strip "net."
    if any(k.startswith("net.members.") for k in sd):
        sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}

    if n_members is None:
        indices = {int(k.split(".")[1]) for k in sd if k.startswith("members.")}
        n_members = max(indices) + 1

    obs_dim = int(np.prod(obs_space.shape))
    act_dim = int(np.prod(act_space.shape))

    input_dim, hidden_dims, has_bias_last = _infer_arch(sd)
    uses_status = (input_dim == obs_dim + act_dim + _STATUS_DIM + 1)

    members = [
        _Member(obs_space, act_space, _build_net(input_dim, hidden_dims, has_bias_last), uses_status)
        for _ in range(n_members)
    ]
    ensemble = RewardEnsemble(obs_space, act_space, members)
    ensemble.load_state_dict(sd)
    ensemble.eval()

    if device != "cpu":
        ensemble.to(device)

    return ensemble

