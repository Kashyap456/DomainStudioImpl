import torch
import lightning as L
from model.dataset import get_dataloader
from model.config import Config
from training_utils.model_lightning import DomainStudio
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    every_n_epochs=20, save_top_k=8, save_last=True, monitor="train_loss_epoch")

torch.cuda.empty_cache()
config = Config()
model = DomainStudio(config)
train_dataloader = get_dataloader("domain_set")
accelerator = "cuda" if torch.cuda.is_available() else "cpu"

trainer = L.Trainer(max_epochs=-1,
                    accelerator=accelerator, precision='16-mixed', callbacks=[checkpoint_callback])
trainer.fit(model, train_dataloader)
