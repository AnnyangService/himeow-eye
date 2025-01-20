from dataloader import load_dataset
from transformers import SamProcessor
from sam_dataset import SAMDataset
from torch.utils.data import DataLoader
from transformers import SamModel 
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import os

# GPU 설정
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

# 모델 저장 경로 설정
save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models"
os.makedirs(save_dir, exist_ok=True)


# load dataset
dataset = load_dataset(
        image_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases",
        mask_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks/tiff_masks"
    )

# pytorch dataset
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
train_dataset = SAMDataset(dataset=dataset, processor=processor)

# pytorch dataloader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# load the model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# Freeze prompt_encoder 와 mask_decoder
for name, param in model.named_parameters():
    if name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
        param.requires_grad_(False)
    else:  # vision_encoder
        param.requires_grad_(True)

# train the model
optimizer = Adam(model.vision_encoder.parameters(), lr=1e-6, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


# 테스트용 에포크
num_epochs = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.DataParallel(model)
model.to(device)

# 최고 성능 모델 저장을 위한 변수
best_loss = float('inf')

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    mean_loss = mean(epoch_losses)
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean_loss}')
    
    # 최고 성능 모델 저장
    if mean_loss < best_loss:
        best_loss = mean_loss
        # module. 제거하고 vision encoder만 저장
        vision_encoder_dict = {name.replace('module.', ''): param 
                             for name, param in model.state_dict().items() 
                             if 'vision_encoder' in name}
        torch.save(vision_encoder_dict, os.path.join(save_dir, 'best_vision_encoder.pth'))

print("Training finished!")
