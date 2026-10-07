import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TIMESTEP EMBEDDING
# ============================================================

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

# ============================================================
# MODEL UNET
# ============================================================

def groupnorm(ch: int, max_groups: int = 8):
    g = min(max_groups, ch)
    while ch % g != 0:
        g -= 1
        if g == 1:
            break
    return nn.GroupNorm(g, ch)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kT,t_dim=None):
        super().__init__()

        padT = kT // 2

        self.norm1 = groupnorm(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kT, 3, 3),
            padding=(padT, 1, 1),
            padding_mode="replicate",
        )
        if t_dim is not None:
            self.t_proj=nn.Linear(t_dim,out_channels)
        else:
            self.t_proj=None


        self.norm2 = groupnorm(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kT, 3, 3),
            padding=(padT, 1, 1),
            padding_mode="replicate",
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x,t_emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.t_proj is not None and t_emb is not None:
            h = h + self.t_proj(t_emb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DownsampleSpaceOnly(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.conv = nn.Conv3d(
            ch,
            ch,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            padding_mode="replicate",
        )

    def forward(self, x):
        return self.conv(x)


class UpsampleSpaceOnly(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.conv = nn.Conv3d(
            ch,
            ch,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=8, kT=3,t_dim=None):
        super().__init__()
        self.t_dim=t_dim
        
        if t_dim is not None:
            self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        else:
            self.time_mlp=None


        self.in_conv = nn.Conv3d(
            in_channels,
            base,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )

        self.enc1 = DoubleConv(base, base, kT,t_dim=t_dim)
        self.down1 = DownsampleSpaceOnly(base)

        self.enc2 = DoubleConv(base, base * 2, kT,t_dim=t_dim)
        self.down2 = DownsampleSpaceOnly(base * 2)

        self.enc3 = DoubleConv(base * 2, base * 4, kT,t_dim=t_dim)
        self.down3 = DownsampleSpaceOnly(base * 4)

        self.mid1 = DoubleConv(base * 4, base * 4, kT,t_dim=t_dim)
        self.mid2 = DoubleConv(base * 4, base * 4, kT,t_dim=t_dim)

        self.up3 = UpsampleSpaceOnly(base * 4)
        self.dec3 = DoubleConv(base * 8, base * 4, kT,t_dim=t_dim)

        self.up2 = UpsampleSpaceOnly(base * 4)
        self.dec2 = DoubleConv(base * 6, base * 2, kT,t_dim=t_dim)

        self.up1 = UpsampleSpaceOnly(base * 2)
        self.dec1 = DoubleConv(base * 3, base, kT,t_dim=t_dim)

        self.out_norm = groupnorm(base)
        self.out = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x_in,t=None):
        t_emb=None

        if self.time_mlp is not None and t is not None:
            t_emb=sinusoidal_timestep_embedding(t,self.t_dim)
            t_emb=self.time_mlp(t_emb)
            
        x = self.in_conv(x_in)

        e1 = self.enc1(x,t_emb)
        x = self.down1(e1)

        e2 = self.enc2(x,t_emb)
        x = self.down2(e2)

        e3 = self.enc3(x,t_emb)
        x = self.down3(e3)

        x = self.mid1(x,t_emb)
        x = self.mid2(x,t_emb)

        x = self.up3(x)
        x = self.dec3(torch.cat([x, e3], dim=1),t_emb)

        x = self.up2(x)
        x = self.dec2(torch.cat([x, e2], dim=1),t_emb)

        x = self.up1(x)
        x = self.dec1(torch.cat([x, e1], dim=1),t_emb)

        return self.out(F.silu(self.out_norm(x)))
