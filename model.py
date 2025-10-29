import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Define the beta schedule (linear in this case)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Pre-calculate the alphas and their cumulative products
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # Helper terms for the forward process q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Helper terms for the reverse process p(x_{t-1} | x_t, x_0)
        # beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        # Define alpha_bar_{-1} = 1 for t = 0 so that beta_tilde_0 = 0
        alphas_cumprod_prev = torch.cat([
            torch.ones(1, dtype=self.alphas_cumprod.dtype, device=self.alphas_cumprod.device),
            self.alphas_cumprod[:-1]
        ], dim=0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def _gather_to_shape(self, values, t, shape, device):
        """Gather per-timestep values and reshape for broadcasting.

        - values: 1D tensor of length T
        - t: LongTensor of shape (B,) with timestep indices
        - shape: target tensor shape to broadcast against (e.g., (B, N, 3))
        - device: device of the target tensors
        """
        t = t.to(device)
        vals = values.to(device).gather(0, t).float()
        batch = t.shape[0]
        return vals.view(batch, *([1] * (len(shape) - 1)))

    def add_noise(self, original_sample, noise, timesteps):
        """ Forward process: q(x_t | x_0) """
        device = original_sample.device
        sqrt_alphas_cumprod_t = self._gather_to_shape(self.sqrt_alphas_cumprod, timesteps, original_sample.shape, device)
        sqrt_one_minus_alphas_cumprod_t = self._gather_to_shape(self.sqrt_one_minus_alphas_cumprod, timesteps, original_sample.shape, device)
        
        noisy_sample = sqrt_alphas_cumprod_t * original_sample + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_sample

    def step(self, model_output, timestep, sample):
        """ Reverse process: p(x_{t-1} | x_t) """
        # This is a simplified DDIM-like step for clarity. A full DDPM step is slightly different.
        t = timestep
        
        # Get the alpha and beta values for the current timestep
        device = sample.device
        # Support both int and 0-dim tensors for t
        t_int = int(t) if isinstance(t, int) else int(t.item())
        alpha_t = self.alphas.to(device)[t_int]
        alpha_cumprod_t = self.alphas_cumprod.to(device)[t_int]
        beta_t = self.betas.to(device)[t_int]
        
        # Predict the original sample x_0
        pred_original_sample = (sample - torch.sqrt(1 - alpha_cumprod_t) * model_output) / torch.sqrt(alpha_cumprod_t)
        
        # Clip x_0 to prevent explosions
        pred_original_sample = torch.clamp(pred_original_sample, -1., 1.)
        
        # Compute the mean of the posterior q(x_{t-1} | x_t, x_0)
        alpha_cumprod_t_prev = self.alphas_cumprod.to(device)[t_int - 1] if t_int > 0 else torch.tensor(1.0, device=device)
        mean = torch.sqrt(alpha_cumprod_t_prev) * pred_original_sample + \
               torch.sqrt(1 - alpha_cumprod_t_prev - beta_t) * model_output
        
        # The final sample x_{t-1} is just the mean (DDIM step with eta=0)
        prev_sample = mean
        return {"prev_sample": prev_sample}

class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads,
            kdim=context_dim, vdim=context_dim, batch_first=True
        )
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, query, context):
        attn_output, _ = self.attention(query, context, context)
        return self.norm(query + attn_output)

class GraphMLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class GraphUNet(nn.Module):
    def __init__(self, model_dim=128, context_dim=128, num_heads=4):
        super().__init__()
        
        # 1. Timestep Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        
        # 2. Initial Projection
        self.initial_proj = nn.Linear(3, model_dim)
        
        # 3. Encoder Path (U-Net "Down")
        self.down_block1 = GraphMLPBlock(model_dim, model_dim)
        self.down_block2 = GraphMLPBlock(model_dim, model_dim * 2)
        
        # 4. Bottleneck with Cross-Attention
        self.bottleneck_attn = CrossAttentionBlock(
            query_dim=model_dim * 2, context_dim=context_dim, num_heads=num_heads
        )
        self.bottleneck_mlp = GraphMLPBlock(model_dim * 2, model_dim * 2)

        # 5. Decoder Path (U-Net "Up") with Skip Connections
        self.up_block1 = GraphMLPBlock(model_dim * 4, model_dim) # Cat(skip, up)
        self.up_block2 = GraphMLPBlock(model_dim * 2, model_dim)
        
        # 6. Final Projection
        self.final_proj = nn.Linear(model_dim, 3)

    def forward(self, noisy_cloth, timestep, body_context):
        # noisy_cloth: (B, N_cloth, 3)
        # timestep: (B,)
        # body_context: (B, N_body, context_dim)

        # Timestep embedding
        t_emb = self.time_mlp(timestep.float().unsqueeze(-1)) # (B, model_dim)
        t_emb = t_emb.unsqueeze(1) # (B, 1, model_dim) - for broadcasting

        # Initial projection
        x = self.initial_proj(noisy_cloth) # (B, N_cloth, model_dim)

        # --- Encoder ---
        skip1 = self.down_block1(x + t_emb) # (B, N_cloth, model_dim)
        x = self.down_block2(skip1)         # (B, N_cloth, model_dim*2)
        
        # --- Bottleneck ---
        x = self.bottleneck_attn(x, body_context)
        x = self.bottleneck_mlp(x)

        # --- Decoder with Skip Connections ---
        x = self.up_block1(torch.cat([x, skip1.repeat(1, 1, 2)], dim=-1)) # Match dims for skip
        x = self.up_block2(torch.cat([x, skip1], dim=-1))
        
        # Final projection to predict noise
        predicted_noise = self.final_proj(x)
        return predicted_noise
class ClothDiffusionModel(nn.Module):
    def __init__(self, num_body_vertices, num_cloth_vertices, body_feature_dim=128, model_dim=128, num_timesteps=1000):
        super().__init__()
        
        # Placeholder for the body encoder discussed previously
        # In a real scenario, this would be a complex SpatioTemporalModel
        self.body_encoder = nn.Sequential(
            nn.Linear(3, body_feature_dim),
            nn.ReLU()
        )
        
        self.denoising_unet = GraphUNet(model_dim=model_dim, context_dim=body_feature_dim)
        self.scheduler = DiffusionScheduler(num_timesteps=num_timesteps)

    def forward(self, clean_cloth, body_motion):
        """ Training step: returns the loss """
        # clean_cloth: (B, N_cloth, 3) - The ground truth
        # body_motion: (B, N_body, 3) - For simplicity, using a static pose as context
        
        B = clean_cloth.shape[0]
        device = clean_cloth.device
        
        # 1. Get body context (run encoder once)
        body_context = self.body_encoder(body_motion)
        
        # 2. Sample random noise and timesteps
        noise = torch.randn_like(clean_cloth)
        timesteps = torch.randint(0, self.scheduler.num_timesteps, (B,), device=device).long()
        
        # 3. Create noisy sample for the current step
        noisy_cloth = self.scheduler.add_noise(clean_cloth, noise, timesteps)
        
        # 4. Predict the noise using the U-Net
        predicted_noise = self.denoising_unet(noisy_cloth, timesteps, body_context)
        
        # 5. Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, body_motion, shape):
        """ Sampling/Inference step """
        # body_motion: (B, N_body, 3)
        # shape: (B, N_cloth, 3)
        
        B, N_cloth, _ = shape
        device = body_motion.device

        # 1. Get body context
        body_context = self.body_encoder(body_motion)
        
        # 2. Start with pure random noise
        sample = torch.randn(shape, device=device)
        
        # 3. Iteratively denoise
        for t in reversed(range(self.scheduler.num_timesteps)):
            timesteps = torch.full((B,), t, device=device, dtype=torch.long)
            predicted_noise = self.denoising_unet(sample, timesteps, body_context)
            sample = self.scheduler.step(predicted_noise, t, sample)["prev_sample"]
            
        return sample

# --- Example Usage ---
if __name__ == '__main__':
    B, N_cloth, N_body = 4, 512, 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy data
    gt_cloth_mesh = torch.randn(B, N_cloth, 3).to(device) # Ground truth cloth
    body_mesh = torch.randn(B, N_body, 3).to(device) # Body pose context

    # Initialize the model
    model = ClothDiffusionModel(
        num_body_vertices=N_body,
        num_cloth_vertices=N_cloth,
        body_feature_dim=128,
        model_dim=128
    ).to(device)

    # --- Test Training Step ---
    print("--- Testing Training Step ---")
    loss = model(gt_cloth_mesh, body_mesh)
    print(f"Calculated Loss: {loss.item()}")
    loss.backward() # Check if gradients flow
    print("Backward pass successful.")

    # --- Test Sampling Step ---
    print("\n--- Testing Sampling Step ---")
    generated_cloth = model.sample(body_mesh, shape=(B, N_cloth, 3))
    print(f"Generated cloth shape: {generated_cloth.shape}")
    assert generated_cloth.shape == (B, N_cloth, 3)
    print("Sampling successful.")
