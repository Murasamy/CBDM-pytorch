import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

# from torch_utils import misc
# from torch_utils.ops import conv2d_resample
# from torch_utils.ops import fma


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

# def modulated_conv2d(
#     x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
#     weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
#     styles,                     # Modulation coefficients of shape [batch_size, in_channels].
#     noise           = None,     # Optional noise tensor to add to the output activations.
#     up              = 1,        # Integer upsampling factor.
#     down            = 1,        # Integer downsampling factor.
#     padding         = 0,        # Padding with respect to the upsampled image.
#     resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
#     demodulate      = True,     # Apply weight demodulation?
#     flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
#     fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
# ):
#     batch_size = x.shape[0]
#     out_channels, in_channels, kh, kw = weight.shape
#     misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
#     misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
#     misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

#     # Pre-normalize inputs to avoid FP16 overflow.
#     if x.dtype == torch.float16 and demodulate:
#         weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
#         styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

#     # Calculate per-sample weights and demodulation coefficients.
#     w = None
#     dcoefs = None
#     if demodulate or fused_modconv:
#         w = weight.unsqueeze(0) # [NOIkk]
#         w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
#     if demodulate:
#         dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
#     if demodulate and fused_modconv:
#         w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

#     # Execute by scaling the activations before and after the convolution.
#     if not fused_modconv:
#         x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
#         x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
#         if demodulate and noise is not None:
#             x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
#         elif demodulate:
#             x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
#         elif noise is not None:
#             x = x.add_(noise.to(x.dtype))
#         return x

#     # Execute as one fused op using grouped convolution.
#     with misc.suppress_tracer_warnings(): # this value will be treated as a constant
#         batch_size = int(batch_size)
#     misc.assert_shape(x, [batch_size, in_channels, None, None])
#     x = x.reshape(1, -1, *x.shape[2:])
#     w = w.reshape(-1, in_channels, kh, kw)
#     x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
#     x = x.reshape(batch_size, -1, *x.shape[2:])
#     if noise is not None:
#         x = x.add_(noise)
#     return x
    
# class FullyConnectedLayer(torch.nn.Module):
#     def __init__(self,
#         in_features,                # Number of input features.
#         out_features,               # Number of output features.
#         bias            = True,     # Apply additive bias before the activation function?
#         activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
#         lr_multiplier   = 1.0,        # Learning rate multiplier.
#         bias_init       = 0,        # Initial value for the additive bias.
#     ):
#         super().__init__()
#         self.activation = activation
#         self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
#         self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
#         self.weight_gain = lr_multiplier / np.sqrt(in_features)
#         self.bias_gain = lr_multiplier

#     def forward(self, x):
#         w = self.weight.to(x.dtype) * self.weight_gain
#         b = self.bias
#         if b is not None:
#             b = b.to(x.dtype)
#             if self.bias_gain != 1:
#                 b = b * self.bias_gain

#         if self.activation == 'linear' and b is not None:
#             x = torch.addmm(b.unsqueeze(0), x, w.t())
#         else:
#             x = x.matmul(w.t())
#             x = bias_act.bias_act(x, b, act=self.activation)
#         return x


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout,
                 cond, augm, num_class, shared_knowledge=False, shared_resolution=8,):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.shared_knowledge = shared_knowledge
        self.shared_resolution = shared_resolution
    
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

        # for i in range(len(self.upblocks)):
        #     print(self.upblocks[i])

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

    def forward(self, x, t, y=None, augm=None):
        # Timestep embedding
        temb = self.time_embedding(t)
        temb_label = temb.clone()

        # Label embedding for conditional generation 
        if y is not None and self.label_embedding is not None:
            assert y.shape[0] == x.shape[0]
            temb_label = temb + self.label_embedding(y)

        # Label embedding for conditional generation 
        if augm is not None and self.augm_embedding is not None:
            assert augm.shape[0] == x.shape[0]
            temb_label = temb + self.augm_embedding(augm)

        # Downsampling
        h = self.head(x)
        hs = [h]
        
        for layer in self.downblocks:
            h = layer(h, temb)
            print(h.shape)
            hs.append(h)
        
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # print(h.shape)
        # Upsampling

        if not self.shared_knowledge:
            for layer in self.upblocks:
                if isinstance(layer, ResBlock):
                    h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, temb)
            h = self.tail(h)
            assert len(hs) == 0
            return h

        if self.shared_knowledge:
            # Peiyang revised version
            counter = 0
            for counter in range(len(self.upblocks)):
                layer = self.upblocks[counter]
                if isinstance(layer, ResBlock):
                    h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, temb)
                print(h.shape)
                # check if the resolution equals to the shared resolution
                if h.shape[2] == self.shared_resolution and h.shape[3] == self.shared_resolution:
                    print("Shared resolution reached")

                    self.skip = nn.Sequential(
                        nn.GroupNorm(32, h.shape[1]),
                        Swish(),
                        nn.Conv2d(h.shape[1], 3, 3, stride=1, padding=1)
                    ).to(h.device)

                    h_low = self.skip(h)
                    print(h_low.shape)
                    counter += 1
                    break
                
                counter += 1

            for counter in range(counter, len(self.upblocks)):
                layer = self.upblocks[counter]
                if isinstance(layer, ResBlock):
                    h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, temb_label)
                print(h.shape)
            h = self.tail(h)

            assert len(hs) == 0

            return h, h_low


if __name__ == '__main__':
    batch_size = 8
    net_model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1,
        cond="uncond", augm=False, num_class=100, shared_knowledge=True,
        shared_resolution=8,)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = torch.randint(100, (batch_size, ))
    pred_noise, pred_noise_low = net_model(x, t, y, None,)

