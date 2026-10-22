"""
Continuous Cosine Noise Schedule and Timestep Embeddings for FreqNiche-ST.
Strictly implements:
- T = 100 diffusion steps
- Cosine schedule for alpha_bar_t
- q_sample forward diffusion
- Sinusoidal timestep embedding generator
"""

import math
import torch
import torch.nn as nn


class CosineDiffusionSchedule(nn.Module):
    """
    Cosine noise schedule as proposed in Nichol & Dhariwal (2021).
    Provides smooth SNR decay without abrupt variance drops.
    """
    def __init__(self, timesteps: int = 100, s: float = 0.008):
        super().__init__()
        self.timesteps = timesteps
        self.s = s

        # Compute alpha_bar values for t in [0, T]
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        f_t = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = f_t / f_t[0]

        # Betas: beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        # Clip betas to prevent numerical explosion near t = T
        betas = torch.clip(betas, min=1e-4, max=0.02).float()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register buffers (float32)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples noisy latent x_t given x_0, timestep t, and Gaussian noise epsilon.

        Args:
            x_0: (B, M, k) clean target latent.
            t: (B,) integer timesteps in [1, timesteps].
            noise: (B, M, k) standard Gaussian noise.

        Returns:
            x_t: (B, M, k) noisy latent.
        """
        # t is 1-indexed (1 to T) -> convert to 0-indexed for buffer lookup
        t_idx = (t - 1).clamp(0, self.timesteps - 1)
        
        # (B, 1, 1)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t_idx].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t_idx].view(-1, 1, 1)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise


def get_sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Generates sinusoidal timestep embeddings for scalar timesteps.

    Args:
        timesteps: (B,) tensor of timesteps.
        dim: Embedding dimension (e.g. 128).

    Returns:
        (B, dim) timestep embeddings.
    """
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
