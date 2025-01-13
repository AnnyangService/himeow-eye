import torch
from transformers import SamModel, SamProcessor, TrainingArguments, Trainer
from torch.utils.data import Dataset
from PIL import Image
import os

class EncoderFinetuning(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.image_files = [f for f in os.listdir(image_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        # SAM 프로세서를 사용하여 이미지 전처리
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"longest_edge": 1024},
            do_normalize=True
        )
        
        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "original_sizes": inputs.original_sizes.squeeze(0),
            "reshaped_input_sizes": inputs.reshaped_input_sizes.squeeze(0),
        }

def freeze_params(model, exclude_encoder=True):
    """모델의 파라미터를 freeze하고 vision encoder만 trainable하게 설정"""
    for name, param in model.named_parameters():
        if exclude_encoder and "vision_encoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def main():
    # GPU 설정
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델과 프로세서 초기화
    model_name = "facebook/sam-vit-huge"
    model = SamModel.from_pretrained(model_name)
    processor = SamProcessor.from_pretrained(model_name)
    
    # vision encoder만 학습하도록 설정
    freeze_params(model)
    
    # GPU로 모델 이동
    model = model.to(device)
    
    # 데이터셋 생성
    train_dataset = CatEyeDataset(
        image_dir="/path/to/your/images",
        processor=processor
    )
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./sam_encoder_finetuned",
        num_train_epochs=100,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        save_steps=20,  # 20 에포크마다 저장
        save_total_limit=3,  # 최근 3개의 체크포인트만 유지
        dataloader_num_workers=4,
        fp16=True,  # 메모리 효율을 위한 mixed precision training
        gradient_accumulation_steps=4,  # 메모리 효율을 위한 gradient accumulation
    )
    
    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # 학습 시작
    trainer.train()
    
    # 최종 모델 저장
    trainer.save_model("./sam_encoder_finetuned/final_model")

if __name__ == "__main__":
    main()