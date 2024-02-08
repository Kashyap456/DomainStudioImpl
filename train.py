import torch
import lightning as L
from model.dataset import get_dataloader
from model.config import Config
from training_utils.model_lightning import DomainStudio
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(every_n_epochs=20, save_last=True)

torch.cuda.empty_cache()
config = Config()
model = DomainStudio(config)
train_dataloader = get_dataloader("domain_set")
accelerator = "cuda" if torch.cuda.is_available() else "cpu"

trainer = L.Trainer(max_epochs=config.num_epochs,
                    accelerator=accelerator, precision='16-mixed', accumulate_grad_batches=2)
trainer.fit(model, train_dataloader)
