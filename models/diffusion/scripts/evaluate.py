from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os

def evaluate(config, model, noise_scheduler, epoch):
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    images = pipeline(batch_size=config.eval_batch_size).images
    grid = make_image_grid(images, rows=4, cols=4)

    output_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(output_dir, exist_ok=True)
    grid.save(f"{output_dir}/{epoch:04d}.png")