import torch.optim as optim
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from .loss import DomainLoss

model_id = "runwayml/stable-diffusion-v1-5"


def get_pretrained(model_id):
    Tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    Encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    VAE = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    UNetLocked = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet")
    UNetTrained = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet")

    return Tokenizer, Encoder, VAE, UNetLocked, UNetTrained


def get_training_utils(model, vae, num_epochs=160, num_train_timesteps=1000, lr_base=4e-6):
    Optimizer = optim.Adam(model.parameters(), lr=lr_base)
    Scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        num_train_timesteps=num_train_timesteps
    )
    Loss = DomainLoss(vae.decode)
    lr_scheduler = optim.swa_utils.SWALR(
        Optimizer, anneal_strategy="linear", anneal_epochs=num_epochs, swa_lr=1.5e-6)

    return Optimizer, Scheduler, Loss, lr_scheduler
