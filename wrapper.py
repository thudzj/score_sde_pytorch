# coding=utf-8
from sde_lib import VESDE, VPSDE, subVPSDE
import torch

# define the score wrapper
class ScoreFnWrapper:
    def __init__(self, sde, score_fn, timesteps, continuous, score_mean=None):
        self.sde = sde
        self.score_fn = score_fn
        self.timesteps = timesteps
        self.continuous = continuous

        if score_mean is None:
            self.score_mean = 0
        else:
            self.score_mean = torch.load(score_mean).to(timesteps.device)
    
    def __call__(self, x, t):
        t_ = t.view(-1)[0].item()
        t_idx = torch.nonzero(self.timesteps == t_)[0].item()

        # get the timestep 's' (if 't' is the last step, just set 's' to 't')
        s_idx = t_idx + 1 if t_idx + 1 < self.timesteps.shape[0] else t_idx
        s = self.timesteps[s_idx]
        s = torch.ones(x.shape[0], device=x.device) * s

        # fetch alpha and sigma for 's' and 't' given various diffusion models
        if (self.continuous and isinstance(self.sde, (VPSDE, VESDE))) or isinstance(self.sde, subVPSDE):
            α_t, σ_t = self.sde.marginal_prob(torch.ones(x.shape[0], device=x.device), t)
            α_s, σ_s = self.sde.marginal_prob(torch.ones(x.shape[0], device=x.device), s)
        elif isinstance(self.sde, VPSDE):
            labels = t * (self.sde.N - 1)
            α_t, σ_t = self.sde.sqrt_alphas_cumprod.to(labels.device)[labels.long()], self.sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            labels = s * (self.sde.N - 1)
            α_s, σ_s = self.sde.sqrt_alphas_cumprod.to(labels.device)[labels.long()], self.sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
        elif isinstance(self.sde, VESDE):
            timestep = (t * (self.sde.N - 1) / self.sde.T).long()
            α_t, σ_t = torch.ones_like(t), self.sde.discrete_sigmas.to(t.device)[timestep]

            timestep = (s * (self.sde.N - 1) / self.sde.T).long()
            α_s, σ_s = torch.ones_like(s), self.sde.discrete_sigmas.to(s.device)[timestep]
        else:
            raise NotImplementedError(f"SDE class {self.sde.__class__.__name__} not yet supported.")

        σ2_t = σ_t ** 2
        σ2_s = σ_s ** 2

        α_ts =  α_t / α_s
        r_ts = 1 / α_ts
        σ2_ts = σ2_t - α_ts ** 2 * σ2_s

        # # check if there is a bug
        # print(t_idx, t.view(-1)[0], α_t.view(-1)[0], σ2_t.view(-1)[0])
        # print(s_idx, s.view(-1)[0], α_s.view(-1)[0], σ2_s.view(-1)[0])

        # Note that we use the 'score_fn' functional which is the estimation of the true scores,
        # so we replace all -ε_t(x_t; θ) / σ_t with score_fn(x, t) in this code
        score_t_one_step = self.score_fn(x, t) - self.score_mean[t_idx]
        μ_hat_st = (x + score_t_one_step * σ2_ts[:, None, None, None]) / α_ts[:, None, None, None]
        
        # σ2_hat_ts = σ2_s * σ2_ts / σ2_t # this is unused currently
        score_t_two_step = r_ts[:, None, None, None] * (self.score_fn(μ_hat_st, s) - self.score_mean[s_idx])

        β = 0.5
        score = score_t_one_step * β + score_t_two_step * (1 - β)
        return score