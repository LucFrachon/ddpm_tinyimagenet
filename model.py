
from types import SimpleNamespace
import math
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from SmaAt_UNet.models.SmaAt_UNet import SmaAt_UNet
from torchmetrics.image.fid import FrechetInceptionDistance
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from image_utils import ValueScaler



""" Define the sinusoidal position embedding layer """

class PositionEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        assert embedding_dim % 2 == 0, f"Embedding dimension should be even; " \
                                       f"using the closest even number"
        self.embedding_dim = embedding_dim

    @torch.no_grad()
    def forward(self, positions):
        scales = (
            -math.log(10000) * torch.arange(self.embedding_dim / 2).to(positions)
            / (self.embedding_dim / 2)
        )
        scales = scales.exp()
        embeddings = positions @ scales[..., None].transpose(1, 0)
        final_embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        
        return final_embeddings


""" An assertion layer for debugging """
    
@torch.no_grad()  
def S(x, expected, include_bs = False):
    # Check that a tensor has the expected size, with or without its batch dimension.
    s1, s2 = x.size(), expected
    
    if not include_bs:
        s1 = s1[1:]
        msg_0 = "Batch-less s"
    else:
        msg_0 = "S"       

    msg = msg_0 + f"ize of tensor is {s1}, expected {expected}."
    assert s1 == s2, msg


""" Modified SmAtUNet backbone, with positional embeddings """
    
class SmAtUNetDiff(SmaAt_UNet):
    def __init__(self, time_emb_dim = 64, n_channels = 3, monitor_internals = False):
        super().__init__(n_channels=n_channels, n_classes=1, kernels_per_layer=2, bilinear=True, reduction_ratio=16)

        self.time_emb_dim = time_emb_dim
        self.pos_embed = nn.Sequential(
            PositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim * 2),
        )

        self.inc_t = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim * 2, 64 * 2))
        
        self.down1_t = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim * 2, 128 * 2))
        self.down2_t = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim * 2, 256 * 2))
        self.down3_t = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim * 2, 512 * 2))
        self.factor = 2 if self.bilinear else 1
        self.down4_t = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim * 2, (1024 // self.factor) * 2))

        self.up1_t = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim * 2, (512 // self.factor) * 2))
        self.up2_t = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim * 2, (256 // self.factor) * 2))
        self.up3_t = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim * 2, (128 // self.factor) * 2))
        self.up4_t = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim * 2, 64 * 2))
        self.out_t = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim * 2, 64 * 2))

        self.outc = nn.Conv2d(64, n_channels, 1)  # override default
        self.last_conv = nn.Conv2d(n_channels * 2, n_channels, 1)

        self.monitors = dict() if monitor_internals else None

    def forward(self, x, t):
        img_size = x.shape[1:]
        x_ini = x.clone()
        t = self.pos_embed(t)
        S(x, img_size, False), S(t, (self.time_emb_dim * 2, ), False)
        sci, shi = self.inc_t(t).view(-1, 64 * 2, 1, 1).chunk(2, dim=1)
        S(sci, (64, 1, 1), False), S(shi, (64, 1, 1), False)

        x1 = self.inc(x)
        S(x1, (64, *img_size[1:]), False)
        x = F.silu(x1 * (sci + 1) + shi)
        x1Att = self.cbam1(x1)
        
        sc1, sh1 = self.down1_t(t).view(-1, 128 * 2, 1, 1).chunk(2, dim=1)
        x2 = F.silu(self.down1(x1) * (sc1 + 1) + sh1)
        x2Att = self.cbam2(x2)
        
        sc2, sh2 = self.down2_t(t).view(-1, 256 * 2, 1, 1).chunk(2, dim=1)
        x3 = F.silu(self.down2(x2) * (sc2 + 1) + sh2)
        x3Att = self.cbam3(x3)
        
        sc3, sh3 = self.down3_t(t).view(-1, 512 * 2, 1, 1).chunk(2, dim=1)
        x4 = F.silu(self.down3(x3) * (sc3 + 1) + sh3)
        x4Att = self.cbam4(x4)
        
        sc4, sh4 = self.down4_t(t).view(-1, (1024 // self.factor) * 2, 1, 1).chunk(2, dim=1)
        x5 = F.silu(self.down4(x4) * (sc4 + 1) + sh4)
        x5Att = self.cbam5(x5)
                
        sc_1, sh_1 = self.up1_t(t).view(-1, (512 // self.factor) * 2, 1, 1).chunk(2, dim=1)
        x = self.up1(x5Att, x4Att)
        # S(x, (512 // self.factor)), S(sc_1, (512 // self.factor)), S(sh_1, (512 // self.factor))
        x = F.silu(x * (sc_1 + 1) + sh_1)
        
        sc_2, sh_2 = self.up2_t(t).view(-1, (256 // self.factor) * 2, 1, 1).chunk(2, dim=1)
        x = self.up2(x, x3Att)
        # S(x, (256 // self.factor)), S(sc_2, (256 // self.factor)), S(sh_2, (256 // self.factor))
        x = F.silu(x * (sc_2 + 1) + sh_2)
        
        x = self.up3(x, x2Att)
        sc_3, sh_3 = self.up3_t(t).view(-1, (128 // self.factor) * 2, 1, 1).chunk(2, dim=1)
        # S(x, (128 // self.factor)), S(sc_3, (128 // self.factor)), S(sh_3, (128 // self.factor))
        x = F.silu(x * (sc_3 + 1) + sh_3)
        
        x = self.up4(x, x1Att)
        sc_4, sh_4 = self.up4_t(t).view(-1, 64 * 2, 1, 1).chunk(2, dim=1)
        # S(x, (64, 64)), S(sc_4, 64), S(sh_4, 64)
        x = F.silu(x * (sc_4 + 1) + sh_4)

        sco, sho = self.out_t(t).view(-1, 64 * 2, 1, 1).chunk(2, dim=1)
        S(x, (64, *img_size[1:]), False)
        x = x * (sco + 1) + sho
        x = self.outc(x)
        S(x, img_size), S(x_ini, img_size)

        logits = self.last_conv(torch.cat([x, x_ini], dim=1))

        if self.monitors is not None:
            self.monitors['sci'], self.monitors['shi'] = sci, shi
            self.monitors['sc1'], self.monitors['sh1'] = sc1, sh1
            self.monitors['sc2'], self.monitors['sh2'] = sc2, sh2
            self.monitors['sc3'], self.monitors['sh3'] = sc3, sh3
            self.monitors['sc4'], self.monitors['sh4'] = sc4, sh4
            self.monitors['sc_1'], self.monitors['sh_1'] = sc_1, sh_1
            self.monitors['sc_2'], self.monitors['sh_2'] = sc_2, sh_2
            self.monitors['sc_3'], self.monitors['sh_3'] = sc_3, sh_3
            self.monitors['sc_4'], self.monitors['sh_4'] = sc_4, sh_4
            self.monitors['sco'], self.monitors['sho'] = sco, sho

        return logits    
    

""" Three different variance schedule functions """

@torch.no_grad()
def linear_schedule(min_ = 1e-4, max_ = 0.02, timesteps=200):
    return torch.linspace(min_, max_, timesteps)


@torch.no_grad()
def quadratic_schedule(min_ = 1e-4, max_ = 0.02, timesteps=200):
    return torch.linspace(min_ ** 0.5, max_ ** 0.5, timesteps) ** 2


@torch.no_grad()
def sigmoid_schedule(min_ = 1e-4, max_ = 0.02, timesteps=200):
    return torch.sigmoid(torch.linspace(-6, 6, timesteps)) * (max_ - min_) + min_


""" Helper functions to select the correct coefficients for the required timesteps """

@torch.no_grad()
def _at(vector, positions):
    positions = positions.to(dtype=torch.int64, device=vector.device).squeeze()
    return vector.gather(-1, positions)


@torch.no_grad()
def make_coeffs(schedule, beta_min, beta_max, timesteps):
    if schedule == 'linear':
        schedule = linear_schedule
    elif schedule == 'quadratic':
        schedule = quadratic_schedule
    elif schedule == 'sigmoid':
        schedule = sigmoid_schedule
    else:
        raise NotImplementedError
    
    A = SimpleNamespace()
    A.beta = schedule(min_=beta_min, max_=beta_max, timesteps=timesteps)
    A.alpha = 1 - A.beta
    A.alpha_bar = torch.cumprod(A.alpha, dim=0)
    A.alpha_bar_previous = F.pad(A.alpha_bar[:-1], (1, 0), value=1.0)
    A.sqrt_alpha = torch.sqrt(A.alpha)
    A.sqrt_alpha_bar = torch.sqrt(A.alpha_bar)
    A.sqrt_1_minus_alpha_bar = torch.sqrt(1 - A.alpha_bar)
    A.variance = A.beta * (1. - A.alpha_bar_previous) / (1. - A.alpha_bar)
    return A


""" Create noisier samples or denoised samples, and a denoising trajectory from random noise """
    
@torch.no_grad()
def noisify(x_0, t, coeffs, eps=None):
    if eps is None:
        eps = torch.randn_like(x_0)
    a1 = _at(coeffs.sqrt_alpha_bar, t).reshape(-1, *((1,) * (x_0.dim() - 1))).to(eps.device)
    a2 = _at(coeffs.sqrt_1_minus_alpha_bar, t).reshape(-1, *((1,) * (x_0.dim() - 1))).to(eps.device)
    return a1 * x_0 + a2 * eps


@torch.no_grad()
def sample_previous_timestep(x, t, index, model, coeffs):

    model.eval()

    if not x.device == t.device == model.device:
        x = x.to(model.device)
        t = t.to(model.device)
    
    b = _at(coeffs.beta, t).to(x.device)
    a1 = 1. / _at(coeffs.sqrt_alpha, t).to(x.device)
    a2 = _at(coeffs.sqrt_1_minus_alpha_bar, t).to(x.device)
    
    mu = (
        a1.reshape((x.size(0), *((1,) * (x.dim() - 1))))
        * (
            x - (b.reshape((x.size(0), *((1,) * (x.dim() - 1)))) * model(x, t)  # this is where the magic happens!
                / a2.reshape((x.size(0), *((1,) * (x.dim() - 1)))))
        )
    )
    if index == 0:
        return mu
    
    sigma2 = _at(coeffs.variance, t).to(device=x.device)
    eps = torch.randn_like(x)

    return mu + torch.sqrt(sigma2.reshape((-1, *((1,) * (x.dim() - 1))))) * eps


@torch.no_grad()
def sample_trajectory(
    timesteps,
    schedule,
    beta_min,
    beta_max,
    batch_size,
    img_size,
    channels,
    model,
    device = 'cuda',
    noise = None
):
    model.to(device)
    model.eval()
    
    coeffs = make_coeffs(schedule, beta_min, beta_max, timesteps)
    if noise is None:
        img = torch.randn((batch_size, channels, *img_size)).to(model.device)
    else: 
        img = torch.tensor(noise)
    
    if img.dim() < 4:
        img = img.unsqueeze(0)

    images = [img]
        
    for i in tqdm(reversed(range(0, timesteps)), total=timesteps):
        t = torch.full((batch_size, 1), i, dtype=torch.float32)
        img = sample_previous_timestep(img, t, i, model, coeffs)
        images.append(img.cpu())
        
    return images


""" Implement the Frechet Inception Distance metric to evaluate our model """

class FID:
    def __init__(self, datamodule: pl.LightningDataModule, clip_values = False, device = 'cuda'):
        self.fid_metric = FrechetInceptionDistance(reset_real_features = False, normalize=True).to(device)
        self.datamodule = datamodule
        self.device = device
        
        if clip_values:
            self.gen_image_scaler = ValueScaler(source_scale='-1_1', target_scale='0_1', to_numpy=False)
        else:
            self.gen_image_scaler = ValueScaler(source_scale='unbound', target_scale='0_1', to_numpy=False)
            
            
    def setup(self):
        # Compute the statistics for the real images
        print(f"FID update for real images: Using {len(self.datamodule.test_dataloader().dataset)} images.")
        for x, _ in self.datamodule.test_dataloader():
            x = x.to(self.device)
            x_scaled = ValueScaler(source_scale='-1_1', target_scale='0_1', to_numpy=False)(x)
            self.fid_metric.update(x_scaled, real=True)
            
    def compute(self, gen_images):
        gen_images = gen_images.to(self.device)
        
        if self.clip_values:
            x_scaled = gen_images.clamp(-1., 1.)
        x_scaled = self.gen_image_scaler(gen_images)
            
        self.fid_metric.update(x_scaled, real=False)
        return self.fid_metric.compute().cpu().item()

    
""" Encapsulate the model inside a LightningModule for easier use """

class UNetDiffModule(pl.LightningModule):
    def __init__(self, time_emb_dim = 32, lr = 3e-4, n_channels = 3, cosine_period = 100, schedule = 'quadratic'):
        super().__init__()
        self.save_hyperparameters()
        self.model = SmAtUNetDiff(time_emb_dim=time_emb_dim, n_channels=n_channels, monitor_internals=False)
        self.coeffs = make_coeffs(schedule, 1e-4, 0.02, 1000)
        self.monitors = self.model.monitors
                
    def forward(self, x, t):
        return self.model(x, t)
    
    def loss(self, eps, pred_eps):
        return F.mse_loss(eps, pred_eps)
    
    def training_step(self, batch, batch_idx):
        x, t = batch
        eps = torch.randn_like(x)
        x_noisy = noisify(x, t, self.coeffs, eps)
        pred_eps = self.forward(x_noisy, t)
        loss = self.loss(eps, pred_eps)
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
        
        if self.monitors is not None:
            for k, v in self.monitors.items():
                self.log(f"{k}", v.mean().detach(), on_step=True, on_epoch=False, prog_bar=False)

        return loss
    
    def validation_step(self, batch, batch_index):
        x, t = batch
        eps = torch.randn_like(x)
        x_noisy = noisify(x, t, self.coeffs, eps)
        pred_eps = self.forward(x_noisy, t)
        loss = self.loss(eps, pred_eps)
        self.log("val_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "LR", self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True
        )
        return loss
    
    def test_step(self, batch, batch_index):
        x, t = batch
        eps = torch.randn_like(x)
        x_noisy = noisify(x, t, self.coeffs, eps)
        pred_eps = self.forward(x_noisy, t)
        loss = self.loss(eps, pred_eps)
        self.log("test_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        return loss   
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.cosine_period, T_mult=1)
        return [optimizer], [scheduler]
