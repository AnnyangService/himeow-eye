from accelerate import Accelerator
from tqdm.auto import tqdm
from scripts.evaluate import evaluate

import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

def train(config, model, train_dataloader, noise_scheduler, optimizer, lr_scheduler):
    accelerator = Accelerator()
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            images = batch["image"]
            noise = torch.randn(images.shape, device=images.device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # Evaluate and save model
        if epoch % config.save_image_epochs == 0:
            evaluate(config, accelerator.unwrap_model(model), noise_scheduler, epoch)