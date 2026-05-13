class _NullLogger:
    def log(self,  msg, *a, **kw): print(f"[rollout] {msg}")
    def warn(self, msg, *a, **kw): print(f"[rollout WARN] {msg}")
    def record(self, *a, **kw):    pass
    def dump(self,   *a, **kw):    pass
    def __getattr__(self, name):   return lambda *a, **kw: None


class _CompatibleMember(RewardNet):
    _STATUS_DIM = 7

    def __init__(self, obs_space, act_space, input_dim, hidden_dims, has_bias_last, obs_dim, act_dim):
        super().__init__(obs_space, act_space)
        self._uses_status = (input_dim == obs_dim + act_dim + self._STATUS_DIM + 1)
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        layers.append(nn.Linear(in_d, 1, bias=has_bias_last))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action, next_status=None, done=None):
        if self._uses_status and next_status is not None and done is not None:
            x = torch.cat([state, action, next_status, done.unsqueeze(-1)], dim=1)
        else:
            x = torch.cat([state, action], dim=1)
        return self.net(x).squeeze(-1)


def _parse_member_arch(sd, member_idx=0):
    prefix = f"members.{member_idx}.net."
    wkeys  = sorted(
        [k for k in sd if k.startswith(prefix) and k.endswith(".weight")],
        key=lambda k: int(k[len(prefix):].split(".")[0]),
    )
    input_dim   = sd[wkeys[0]].shape[1]
    hidden_dims = [sd[k].shape[0] for k in wkeys[:-1]]
    last_idx    = wkeys[-1][len(prefix):].split(".")[0]
    has_bias    = f"{prefix}{last_idx}.bias" in sd
    return input_dim, hidden_dims, has_bias


def load_reward_model(path, obs_space, act_space):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd, n = ckpt["state_dict"], ckpt["n_members"]
    obs_dim, act_dim = ckpt["obs_dim"], ckpt["act_dim"]
    input_dim, hidden_dims, has_bias_last = _parse_member_arch(sd)
    uses_status = (input_dim == obs_dim + act_dim + 8)
    print(
        f"Checkpoint: {n} membri | input_dim={input_dim} "
        f"({'obs+act+status+done' if uses_status else 'obs+act'}) | "
        f"hidden={hidden_dims} | bias_last={has_bias_last}"
    )
    members = [
        _CompatibleMember(obs_space, act_space, input_dim, hidden_dims, has_bias_last, obs_dim, act_dim)
        for _ in range(n)
    ]
    ensemble = RewardEnsemble(obs_space, act_space, members)
    ensemble.load_state_dict(sd)
    ensemble.eval()
    return ensemble
