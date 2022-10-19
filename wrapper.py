# coding=utf-8
from sde_lib import VESDE, VPSDE, subVPSDE
import torch

# define the score wrapper
def score_fn_wrapper(sde, score_fn, continuous, calibration=False, score_mean=None, device='cuda'):
    if score_mean is None:
        score_mean = None
    else:
        score_mean = torch.load(score_mean).to(device)
    
    def new_score_fn(x, t, timesteps):
        t_ = t.view(-1)[0].item()
        t_idx = torch.nonzero(timesteps == t_)[0].item()

        # get the timestep 's' (if 't' is the last step, just set 's' to 't')
        s_idx = t_idx + 1 if t_idx + 1 < timesteps.shape[0] else t_idx
        s = timesteps[s_idx]
        s = torch.ones(x.shape[0], device=x.device) * s

        # fetch alpha and sigma for 's' and 't' given various diffusion models
        if (continuous and isinstance(sde, (VPSDE, VESDE))) or isinstance(sde, subVPSDE):
            α_t, σ_t = sde.marginal_prob(torch.ones(x.shape[0], device=x.device), t)
            α_s, σ_s = sde.marginal_prob(torch.ones(x.shape[0], device=x.device), s)
        elif isinstance(sde, VPSDE):
            labels = t * (sde.N - 1)
            α_t, σ_t = sde.sqrt_alphas_cumprod.to(labels.device)[labels.long()], sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            labels = s * (sde.N - 1)
            α_s, σ_s = sde.sqrt_alphas_cumprod.to(labels.device)[labels.long()], sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
        elif isinstance(sde, VESDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            α_t, σ_t = torch.ones_like(t), sde.discrete_sigmas.to(t.device)[timestep]

            timestep = (s * (sde.N - 1) / sde.T).long()
            α_s, σ_s = torch.ones_like(s), sde.discrete_sigmas.to(s.device)[timestep]
        else:
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

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
        score_t_one_step = score_fn(x, t) - (0 if score_mean is None else score_mean[t_idx])

        if calibration:
            μ_hat_st = (x + score_t_one_step * σ2_ts[:, None, None, None]) / α_ts[:, None, None, None]
            
            # σ2_hat_ts = σ2_s * σ2_ts / σ2_t # this is unused currently
            score_t_two_step = r_ts[:, None, None, None] * (score_fn(μ_hat_st, s) - (0 if score_mean is None else score_mean[s_idx]))

            β = 0.5
            score = score_t_one_step * β + score_t_two_step * (1 - β)
        else:
            score = score_t_one_step
        return score
    
    return new_score_fn