import lightning as L
import torch
from model.model import get_pretrained, get_training_utils
from model.config import Config


class DomainStudio(L.LightningModule):
    def __init__(self, config: Config):
        super(DomainStudio, self).__init__()
        # Setup pretrained models and utilities
        self.tokenizer, self.encoder, self.vae, self.unet_locked, self.unet_trained = get_pretrained(
            config.model_id)

        # Lock weights of all models except UNetTrained
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet_locked.parameters():
            param.requires_grad = False

        self.optimizer, self.scheduler, self.loss, self.lr_scheduler = get_training_utils(
            self.unet_trained, self.vae, config.num_epochs, config.num_train_timesteps, config.lr_base)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        images = images.permute(0, 3, 1, 2)
        x_pr = torch.randn(images.shape).to(images.device)

        # create c_tar using clip
        labels_tr = batch['label_tr']
        tokens_tr = self.tokenizer(
            labels_tr, padding=True, return_tensors="pt")
        c_tar = self.encoder(**tokens_tr).last_hidden_state

        # create c_sou using clip
        labels_so = batch['label_so']
        tokens_so = self.tokenizer(
            labels_so, padding=True, return_tensors="pt")
        c_sou = self.encoder(**tokens_so).last_hidden_state

        # Sample noise to add to the images
        z = self.vae.encode(images).latent_dist.sample()
        z_pr = self.vae.encode(x_pr).latent_dist.sample()
        noise = torch.randn(z.shape).to(z.device)
        bs = z.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (bs,), device=z.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        z_t = self.scheduler.add_noise(z, noise, timesteps)

        # Get the random noise z_pr_t
        z_pr_t = self.scheduler.add_noise(z_pr, noise, timesteps)

        with torch.no_grad():
            # Predict the noise residual
            z_pr_sou = self.unet_locked(z_pr_t, timesteps, c_sou)["sample"]

        # Predict the noise residual
        z_ada = self.unet_trained(z_t, timesteps, c_tar)["sample"]
        z_pr_ada = self.unet_trained(z_pr_t, timesteps, c_sou)["sample"]
        loss = self.loss(z_ada, z, z_pr_sou, z_pr_ada, images)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        if self.lr_scheduler:
            return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.lr_scheduler, "monitor": "val_loss"}}
        else:
            return self.optimizer
