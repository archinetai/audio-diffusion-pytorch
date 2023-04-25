import torch
import torch.nn.functional as F
from audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler
from audio_encoders_pytorch import MelE1d, TanhBottleneck
from auraloss.freq import MultiResolutionSTFTLoss

autoencoder = DiffusionAE(
    encoder=MelE1d( # The encoder used, in this case a mel-spectrogram encoder
        in_channels=2,
        channels=512,
        multipliers=[1, 1],
        factors=[2],
        num_blocks=[12],
        out_channels=32,
        mel_channels=80,
        mel_sample_rate=48000,
        mel_normalize_log=True,
        bottleneck=TanhBottleneck(),
    ),
    inject_depth=6,
    net_t=UNetV0, # The model type used for diffusion upsampling
    in_channels=2, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
    loss_fn=MultiResolutionSTFTLoss(),  # The loss function used
)

# Train autoencoder with audio samples
audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
loss = autoencoder(audio)
loss.backward()

# Encode/decode audio
audio = torch.randn(1, 2, 2**18) # [batch, in_channels, length]
latent = autoencoder.encode(audio) # Encode
sample = autoencoder.decode(latent, num_steps=10) # Decode by sampling diffusion model conditioning on latent