# from dataloader import load_dataset
# from transformers import SamProcessor
# from sam_dataset import SAMDataset
# from torch.utils.data import DataLoader
# from transformers import SamModel 
# from torch.optim import Adam
# import monai
# from tqdm import tqdm
# from statistics import mean
# import torch
# from torch.nn.functional import threshold, normalize
# import os
# import gc

# # GPU 설정
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# torch.cuda.empty_cache()
# gc.collect()


# # 모델 저장 경로 설정
# save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models"
# os.makedirs(save_dir, exist_ok=True)


# # load dataset
# dataset = load_dataset(
#         image_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases",
#         mask_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks/tiff_masks"
#     )

# # pytorch dataset
# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
# train_dataset = SAMDataset(dataset=dataset, processor=processor)

# # pytorch dataloader
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# # load the model
# model = SamModel.from_pretrained("facebook/sam-vit-base")

# # Freeze prompt_encoder 와 mask_decoder
# for name, param in model.named_parameters():
#     if name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
#         param.requires_grad_(False)
#     else:  # vision_encoder
#         param.requires_grad_(True)

# # train the model
# optimizer = Adam(model.vision_encoder.parameters(), lr=1e-6, weight_decay=0)
# seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


# num_epochs = 50

# # CUDA 설정
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = torch.nn.DataParallel(model)
# model.to(device)

# # 최고 성능 모델 저장을 위한 변수
# best_loss = float('inf')

# model.train()
# for epoch in range(num_epochs):
#     epoch_losses = []
#     for batch in tqdm(train_dataloader):
#       # forward pass
#       outputs = model(pixel_values=batch["pixel_values"].to(device),
#                       input_boxes=batch["input_boxes"].to(device),
#                       multimask_output=False)

#       # compute loss
#       predicted_masks = outputs.pred_masks.squeeze(1)
#       ground_truth_masks = batch["ground_truth_mask"].float().to(device)
#       loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

#       # backward pass (compute gradients of parameters w.r.t. loss)
#       optimizer.zero_grad()
#       loss.backward()

#       # optimize
#       optimizer.step()
#       epoch_losses.append(loss.item())

#     mean_loss = mean(epoch_losses)
#     print(f'EPOCH: {epoch}')
#     print(f'Mean loss: {mean_loss}')
    
#     # 최고 성능 모델 저장
#     if mean_loss < best_loss:
#         best_loss = mean_loss
#         # module. 제거하고 vision encoder만 저장
#         vision_encoder_dict = {name.replace('module.', ''): param 
#                              for name, param in model.state_dict().items() 
#                              if 'vision_encoder' in name}
#         torch.save(vision_encoder_dict, os.path.join(save_dir, 'best_vision_encoder.pth'))

# print("Training finished!")

##################################
##################################
##################################
##################################
##################################
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
import os
import gc
import json

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
gc.collect()

# 모델 저장 경로 설정
save_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models"
os.makedirs(save_dir, exist_ok=True)

def save_checkpoint(epoch, model, optimizer, loss, save_path):
  checkpoint = {
      'epoch': epoch,
      'vision_encoder_state_dict': {
          name.replace('module.', ''): param 
          for name, param in model.state_dict().items() 
          if 'vision_encoder' in name
      },
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
  }
  torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, checkpoint_path):
  checkpoint = torch.load(checkpoint_path)
  
  current_state_dict = model.state_dict()
  for name, param in checkpoint['vision_encoder_state_dict'].items():
      current_state_dict[f'module.{name}'] = param
  model.load_state_dict(current_state_dict)
  
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  
  return checkpoint['epoch'], checkpoint['loss']

# load dataset
dataset = load_dataset(
      image_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases",
      mask_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks/tiff_masks"
  )

# pytorch dataset
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
train_dataset = SAMDataset(dataset=dataset, processor=processor)

# pytorch dataloader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# load the model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# Freeze prompt_encoder와 mask_decoder
for name, param in model.named_parameters():
  if name.startswith("prompt_encoder") or name.startswith("mask_decoder"):
      param.requires_grad_(False)
  else:  # vision_encoder
      param.requires_grad_(True)

# train the model
optimizer = Adam(model.vision_encoder.parameters(), lr=1e-6, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.DataParallel(model)
model.to(device)

# 시작 epoch와 best_loss 초기화
start_epoch = 0
best_loss = float('inf')

# checkpoint 파일이 있다면 불러오기
checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
if checkpoint_files:
    # 가장 마지막 checkpoint 찾기
    last_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(save_dir, last_checkpoint)
    start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)
    print(f"Resuming from epoch {start_epoch} with best loss: {best_loss}")

# 학습 히스토리 저장용
history = {
  'losses': [],
  'learning_rates': []
}

# 학습 시작
model.train()
for epoch in range(start_epoch, num_epochs):
  print(f'\nCurrent Epoch: {epoch}/{num_epochs-1}')
  epoch_losses = []
  for batch in tqdm(train_dataloader):
      outputs = model(
          pixel_values=batch["pixel_values"].to(device),
          input_boxes=batch["input_boxes"].to(device),
          multimask_output=False
      )
      
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      epoch_losses.append(loss.item())

  mean_loss = mean(epoch_losses)
  print(f'EPOCH: {epoch}')
  print(f'Mean loss: {mean_loss}')
  
  # 히스토리 업데이트
  history['losses'].append(mean_loss)
  history['learning_rates'].append(optimizer.param_groups[0]['lr'])
  
  # 매 에폭마다 checkpoint 저장
  save_checkpoint(
      epoch + 1, 
      model,
      optimizer,
      mean_loss,
      os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
  )
  
  # 매 에폭마다 히스토리 저장
  with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
      json.dump(history, f)
  
  # 최고 성능 모델 따로 저장
  if mean_loss < best_loss:
      best_loss = mean_loss
      save_checkpoint(
          epoch + 1,
          model,
          optimizer,
          mean_loss,
          os.path.join(save_dir, 'best_checkpoint.pth')
      )

print("Training finished!")