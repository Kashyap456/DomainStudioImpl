import lightning as L
from model.dataset import get_dataloader
from model.config import Config
from lightning.model_lightning import DomainStudio

config = Config()
model = DomainStudio(config)
train_dataloader = get_dataloader("domain_set")

trainer = L.Trainer(max_epochs=config.num_epochs)
trainer.fit(model, train_dataloader)
