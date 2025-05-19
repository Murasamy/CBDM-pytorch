import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout,
                 cond, augm, num_class, freeze_down_latent_label = False):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.freeze_down_latent_label = freeze_down_latent_label
        
        if freeze_down_latent_label:
            print("Freeze down latent label")
    
        if cond:
            self.label_embedding = nn.Embedding(num_class, tdim)
        else:
            self.label_embedding = None

        if augm:
            self.augm_embedding = nn.Linear(9, tdim, bias=False)
        else:
            self.augm_embedding = None

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(out_ch, 3, 3, stride=1, padding=1)
        )

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def freeze_non_upsampling_layers(self):
        """
        Freeze all layers except the Upsampling layers.
        """
        # Freeze Downsampling layers
        for param in self.head.parameters():
            param.requires_grad = False
        for layer in self.downblocks:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze Middle layers
        for layer in self.middleblocks:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze Tail layers
        for param in self.tail.parameters():
            param.requires_grad = False

        # Ensure Upsampling layers remain trainable
        for layer in self.upblocks:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_all_layers(self):
        """
        Unfreeze all layers in the model.
        """
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, t, y=None, augm=None):

        if not self.freeze_down_latent_label:
            # Timestep embedding
            temb = self.time_embedding(t)

            # Label embedding for conditional generation 
            if y is not None and self.label_embedding is not None:
                assert y.shape[0] == x.shape[0]
                temb = temb + self.label_embedding(y)

            # Label embedding for conditional generation 
            if augm is not None and self.augm_embedding is not None:
                assert augm.shape[0] == x.shape[0]
                temb = temb + self.augm_embedding(augm)

            h = self.head(x)
            hs = [h]
            # Downsampling
            for layer in self.downblocks:
                h = layer(h, temb)
                hs.append(h)
            # Middle
            for layer in self.middleblocks:
                h = layer(h, temb)
            # Upsampling
            for layer in self.upblocks:
                if isinstance(layer, ResBlock):
                    h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, temb)

        elif self.freeze_down_latent_label:
            temb = self.time_embedding(t)
            temb_mixed = temb.clone()

            # Label embedding for conditional generation 
            if y is not None and self.label_embedding is not None:
                assert y.shape[0] == x.shape[0]
                temb_mixed = temb + self.label_embedding(y)

            # Label embedding for conditional generation 
            if augm is not None and self.augm_embedding is not None:
                assert augm.shape[0] == x.shape[0]
                temb_mixed = temb + self.augm_embedding(augm)

            h = self.head(x)
            hs = [h]

            for layer in self.downblocks:
                h = layer(h, temb)
                hs.append(h)
            # Middle
            for layer in self.middleblocks:
                h = layer(h, temb)
            # Upsampling
            for layer in self.upblocks:
                if isinstance(layer, ResBlock):
                    h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, temb_mixed)

        h = self.tail(h)
        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)

    class GaussianDiffusionTrainer(nn.Module):
        def __init__(self,
                    model, beta_1, beta_T, T, dataset,
                    num_class, cfg, cb, tau, weight, finetune, temperature_beta = False, temperature_beta_lamdba = None, edm2_truncate = False, edm2_truncate_portion =  0):
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
            self.temperature_beta = temperature_beta
            self.temperature_beta_lamdba = temperature_beta_lamdba

            if self.temperature_beta:
                # check if num_class equals to the length of weight
                betas_list = []
                sqrt_alphas_bar_label_list = []
                sqrt_one_minus_alphas_bar_list = []

                assert len(weight) == num_class
                for label in range(num_class):
                    betas = torch.linspace(beta_1, beta_T, T).double()
                    betas_new = self.temperature_beta_func(betas, label)
                    # self.register_buffer(f'betas_label_{label}', betas_new)
                    betas_list.append(betas_new)
                    alphas = 1. - betas_new
                    alphas_bar = torch.cumprod(alphas, dim=0)
                    sqrt_alphas_bar_label = torch.sqrt(alphas_bar)
                    sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
                    # self.register_buffer(
                        # f'sqrt_alphas_bar_label_{label}', torch.sqrt(alphas_bar))
                    # self.register_buffer(
                    #     f'sqrt_one_minus_alphas_bar_label_{label}', torch.sqrt(1. - alphas_bar))
                    sqrt_alphas_bar_label_list.append(sqrt_alphas_bar_label)
                    sqrt_one_minus_alphas_bar_list.append(sqrt_one_minus_alphas_bar)
                self.register_buffer('betas', torch.stack(betas_list)) # (num_class, T)
                self.register_buffer('sqrt_alphas_bar_label', torch.stack(sqrt_alphas_bar_label_list)) # (num_class, T)
                self.register_buffer('sqrt_one_minus_alphas_bar_label', torch.stack(sqrt_one_minus_alphas_bar_list)) # (num_class, T)
                    
            else:            
                self.register_buffer(
                    'betas', torch.linspace(beta_1, beta_T, T).double())
                alphas = 1. - self.betas
                alphas_bar = torch.cumprod(alphas, dim=0)

                self.register_buffer(
                    'sqrt_alphas_bar', torch.sqrt(alphas_bar))
                self.register_buffer(
                    'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
            
        def temperature_beta_func(self, beta, label):
            omega_c = 1 / self.weight[label]
            omega_c_max = 1 / self.weight.min()
            return beta * (1 - self.temperature_beta_lamdba * (omega_c / omega_c_max))


        def forward(self, x_0, y_0, augm=None):
            """
            Algorithm 1.
            x_0: (batch_size, 3, 32, 32)
            y_0: (batch_size, )
            """
            # original codes
            t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
            noise = torch.randn_like(x_0) 

            if self.temperature_beta:
                # modify version of extract(self.sqrt_alphas_bar, t, x_0.shape), get sqrt_alphas_bar_label with y_0 and t, reshape into x_0.shape
                # sqrt_alphas_bar_label.shape = (num_class, T), t.shape = (batch_size, )
                # print("self.sqrt_alphas_bar_label\n", self.sqrt_alphas_bar_label)
                # print("y_0\n", y_0)
                # print("t\n", t)
                sqrt_alphas_bar = self.sqrt_alphas_bar_label[y_0, t]  # Shape: (batch_size,)
                # print(sqrt_alphas_bar)
                sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar_label[y_0, t]  # Shape: (batch_size,)
                # print(*([1] * (len(x_0.shape) - 1)))
                sqrt_alphas_bar = sqrt_alphas_bar.view(-1, *([1] * (len(x_0.shape) - 1)))  # Shape: (batch_size, 1, 1, 1)
                sqrt_one_minus_alphas_bar = sqrt_one_minus_alphas_bar.view(-1, *([1] * (len(x_0.shape) - 1)))  # Shape: (batch_size, 1, 1, 1)
                # print(sqrt_alphas_bar.shape)

                x_t = (
                    sqrt_alphas_bar * x_0 +
                    sqrt_one_minus_alphas_bar * noise
                )


            else:
                x_t = (
                    extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                    extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

            if self.cfg or self.cb:
                if torch.rand(1)[0] < 1/10:
                    y_0 = None

            # return # debug

            h = self.model(x_t, t, y=y_0, augm=augm)
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

    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, dataset,
        FLAGS.num_class, FLAGS.cfg, FLAGS.cb, FLAGS.tau, weight, FLAGS.finetune, FLAGS.temperature_beta, FLAGS.temperature_beta_lambda).to(device)
    with trange(FLAGS.ckpt_step, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0, y_0 = next(datalooper)

            # when using ADA, the augmentation parameters will also be returned by the dataloader
            augm = None
            if type(x_0) == list:
                x_0, augm = x_0
                augm = augm.to(device)

            x_0 = x_0.to(device)
            y_0 = y_0.to(device)

            loss_ddpm, loss_reg = trainer(x_0, y_0, augm)
            loss_ddpm = loss_ddpm.mean()
            loss_reg = loss_reg.mean()
            loss = loss_ddpm + loss_reg if FLAGS.cb and loss_reg > 0 else loss_ddpm
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # logs
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_ddpm', loss_ddpm, step)
            writer.add_scalar('loss_reg', loss_reg, step)
            pbar.set_postfix(loss='%.5f' % loss)

            # sample
            if step != FLAGS.ckpt_step and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0, _  = ema_sampler(fixed_x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'fixed_x_T': fixed_x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(step)))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                # net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID = evaluate(ema_sampler, ema_model, False)
                metrics = {
                    'IS': ema_IS[0],
                    'IS_std': ema_IS[1],
                    'FID': ema_FID
                }
                print(step, metrics)
                pbar.write(
                    '%d/%d ' % (step, FLAGS.total_steps) +
                    ', '.join('%s:%.5f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + '\n')
    writer.close()