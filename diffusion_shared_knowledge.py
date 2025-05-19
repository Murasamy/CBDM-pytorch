# from dd_code.backdoor.benchmarks.pytorch-ddpm.main import self
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from torch_utils.ops import upfirdn2d


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def uniform_sampling(n, N, k):
    return np.stack([np.random.randint(int(N/n)*i, int(N/n)*(i+1), k) for i in range(n)])


def dist(X, Y):
    sx = torch.sum(X**2, dim=1, keepdim=True)
    sy = torch.sum(Y**2, dim=1, keepdim=True)
    return torch.sqrt(-2 * torch.mm(X, Y.T) + sx + sy.T)


def topk(y, all_y, K):
    dist_y = dist(y, all_y)
    return torch.topk(-dist_y, K, dim=1)[1]


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self,
                model, beta_1, beta_T, T, dataset,
                num_class, cfg, cb, tau, weight, finetune, 
                shared_knowledge=False,
                shared_portion = .1, 
                shared_resolution=8, 
                resample_filter=[1, 3, 3, 1]):
        super().__init__()

        self.model = model
        self.T = T
        self.dataset = dataset
        self.num_class = num_class
        self.cfg = cfg
        self.cb = cb
        self.tau = tau
        self.weight = weight
        self.finetune = finetune
        self.shared_knowledge = shared_knowledge
        self.shared_portion = shared_portion
        self.shared_resolution = shared_resolution
        

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        
    def downsample_uc(self, img):  # ToDo: improve efficiency
        assert self.shared_resolution > 0
        img_resolution = img.shape[-1]
        if self.shared_resolution >= img_resolution:
            return img
        while img_resolution != self.shared_resolution:
            img = upfirdn2d.downsample2d(img, self.resample_filter)
            img_resolution = img.shape[-1]
        return img

    def forward(self, x_0, y_0, augm=None):
        """
        Algorithm 1.
        """
        # original codes
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0) 

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        if self.cfg or self.cb:
            if torch.rand(1)[0] < 1/10:
                y_0 = None

        h, h_low = self.model(x_t, t, y=y_0, augm=augm)
        noise_downsample = self.downsample_uc(noise)
        
        loss = F.mse_loss(h, noise, reduction='none')
        loss_reg = loss_com = torch.tensor(0).to(x_t.device)
        if self.cb and y_0 is not None:
            y_bal = torch.Tensor(np.random.choice(
                                 self.num_class, size=len(x_0),
                                 p=self.weight.numpy() if not self.finetune else None,
                                 )).to(x_t.device).long()

            h_bal = self.model(x_t, t, y=y_bal, augm=augm)
            weight = t[:, None, None, None] / self.T * self.tau
            loss_reg = weight * F.mse_loss(h, h_bal.detach(), reduction='none')
            loss_com = weight * F.mse_loss(h.detach(), h_bal, reduction='none')
            return loss, loss_reg + 1/4 * loss_com

        if self.shared_knowledge:
            print('shared_knowledge')
            loss_low = self.shared_portion * F.mse_loss(h_low, noise_downsample, reduction='none')
            return loss, loss_low
        
        print('not shared_knowledge')
        return loss, loss_reg



       

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, num_class, img_size=32, var_type='fixedlarge'):
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.num_class =  num_class
        self.img_size = img_size
        self.var_type = var_type
        
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'alphas_bar', alphas_bar)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps): 
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, y=None, omega=0.0, method='free'):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped}[self.var_type]

        model_log_var = extract(model_log_var, t, x_t.shape)
        unc_eps = None
        augm = torch.zeros((x_t.shape[0], 9)).to(x_t.device)

        # Mean parameterization
        eps = self.model(x_t, t, y=y, augm=augm)
        if omega > 0 and (method == 'cfg'):
            unc_eps = self.model(x_t, t, y=None, augm=None)
            guide = eps - unc_eps
            eps = eps + omega * guide
        
        x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
        model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T, omega=0.0, method='cfg'):
        """
        Algorithm 2.
        """
        x_t = x_T.clone()
        y = None

        if method == 'uncond':
            y = None
        else:
            y = torch.randint(0, self.num_class, (len(x_t),)).to(x_t.device)

        with torch.no_grad():
            for time_step in tqdm(reversed(range(0, self.T)), total=self.T):
                t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y,
                                                     omega=omega, method=method)

                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                
                x_t = mean + torch.exp(0.5 * log_var) * noise

        return torch.clip(x_t, -1, 1), y

if __name__ == '__main__':
    from model.model_share_knowledge import UNet

    batch_size = 8
    net_model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1,
        cond="uncond", augm=False, num_class=100, shared_knowledge=True,
        shared_resolution=8,)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = torch.randint(100, (batch_size, ))
    # pred_noise, pred_noise_low = net_model(x, t, y, None,)

    trainer = GaussianDiffusionTrainer(
        net_model, 1e-4, 0.02, 1000, None,
        1000, cfg = 0.1, cb = False, 
        tau = 1., weight=None,
        finetune=False, 
        shared_knowledge=True,
        shared_portion=0.1, shared_resolution=8,
        resample_filter=[1, 3, 3, 1])
    
    loss_ddpm, loss_addition = trainer(x, y, None)
    print(loss_ddpm.shape, loss_addition.shape)

    # x = torch.randn(batch_size, 3, 32, 32)

    # def downsample_uc(img, res_uc):  # ToDo: improve efficiency
    #     assert res_uc > 0
    #     img_resolution = img.shape[-1]
    #     if res_uc >= img_resolution:
    #         return img
    #     while img_resolution != res_uc:
    #         filter_tensor = upfirdn2d.setup_filter([1, 3, 3, 1])
    #         img = upfirdn2d.downsample2d(img, filter_tensor)
    #         img_resolution = img.shape[-1]
    #     return img
    # print(downsample_uc(x, 8).shape)
