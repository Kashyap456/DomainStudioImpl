import torch
from diffusers import StableDiffusionPipeline
from training_utils.model_lightning import DomainStudio
from model.config import Config


def load_custom_unet_pipeline(checkpoint_path, model_id="runwayml/stable-diffusion-v1-5"):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = DomainStudio(Config())
    model.load_state_dict(checkpoint['state_dict'])
    unet_trained = model.unet_trained
    # Load the pretrained Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)

    # Replace the UNet model in the pipeline with the one from the checkpoint
    pipeline.unet.load_state_dict(unet_trained.state_dict())

    return pipeline


# Example usage
checkpoint_path = "./checkpoints/nagai_mac.ckpt"
pipeline = load_custom_unet_pipeline(checkpoint_path).to("mps")
# pipeline = StableDiffusionPipeline.from_pretrained(
#    "runwayml/stable-diffusion-v1-5").to("mps")
print('here')
# Now you can use the pipeline for inference with a custom UNet model
generator = torch.Generator().manual_seed(0)
image = pipeline(prompt="A painting of a cloudy lit city at night from a plane window", generator=generator,
                 num_inference_steps=30).images[0]
print('here2')
image.show()
