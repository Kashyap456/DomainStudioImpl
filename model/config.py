from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 4
    num_epochs: int = 160
    model_id: str = "runwayml/stable-diffusion-v1-5"
    num_train_timesteps: int = 1000
    lr_base: float = 4e-6
    lr_scheduler: bool = True
