from scripts.config import TrainingConfig
from scripts.dataset import ImageDataset
from scripts.model import get_model
from scripts.train import train
from scripts.evaluate import evaluate

from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
import torch.optim as optim
from diffusers.optimization import get_cosine_schedule_with_warmup

if __name__ == "__main__":
    config = TrainingConfig()
    dataset = ImageDataset(image_dir="/home/minelab/바탕화면/ANN/himeow/filtered_by_breeds_datasets/brachy/abnormal", image_size=config.image_size)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    model = get_model(image_size=config.image_size)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 500, len(train_dataloader) * config.num_epochs)

    train(config, model, train_dataloader, noise_scheduler, optimizer, lr_scheduler)