import torch
import lightning as L
from model.dataset import get_dataloader
from model.config import Config
from training_utils.model_lightning import DomainStudio

config = Config()
model = DomainStudio(config)
train_dataloader = get_dataloader("domain_set")
accelerator = "cuda" if torch.cuda.is_available() else "cpu"

trainer = L.Trainer(max_epochs=config.num_epochs,
                    accelerator=accelerator)
trainer.fit(model, train_dataloader)
