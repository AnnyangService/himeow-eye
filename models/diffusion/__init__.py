from scripts.config import config
from scripts.dataset import train_dataloader
from scripts.model import get_model
from scripts.train import train_loop

from diffusers import DDPMScheduler
import torch.optim as optim
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import notebook_launcher

if __name__ == "__main__":
    
    #model 가져오기기
    model = get_model(image_size=config.image_size)
    #Scheduler 초기화
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps) #Scheduler 세팅팅
    #optimizer 초기화 #Standard AdamW Optimzier 사용    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate) #Standard AdamW Optimzier 사용
    
    #Cosine learning rate schedule  사용, 학습률(lr)을 훈련 과정에서 점진적으로 조정
    lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
    
    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)

    # train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    # gpt 절대 믿지마 