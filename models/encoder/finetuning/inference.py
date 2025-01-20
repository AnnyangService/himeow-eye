import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from dataloader import load_dataset
from sam_dataset import SAMDataset, get_bounding_box

def show_mask(mask, ax, random_color=False):
    """마스크를 시각화하는 함수"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def visualize_prediction(image, ground_truth, prediction, save_path=None):
    """예측 결과를 시각화하는 함수"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Ground Truth
    ax1.imshow(np.array(image))
    show_mask(ground_truth, ax1)
    ax1.set_title("Ground Truth")
    ax1.axis("off")
    
    # Prediction
    ax2.imshow(np.array(image))
    show_mask(prediction, ax2)
    ax2.set_title("Prediction")
    ax2.axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 데이터 로드
    dataset = load_dataset(
        image_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases",
        mask_dir="/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/other_diseases_gtmasks/tiff_masks"
    )
    
    # 모델과 프로세서 설정
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    # 학습된 가중치 로드
    saved_model_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/models/encoder/finetuning/custom_models/best_model.pth"
    model.load_state_dict(torch.load(saved_model_path))
    model.to(device)
    model.eval()
    
    # 테스트할 이미지 인덱스 선택
    idx = 10
    
    # 이미지와 마스크 로드
    image = dataset[idx]["image"]
    ground_truth_mask = np.array(dataset[idx]["label"])
    
    # 바운딩 박스 생성 및 모델 입력 준비
    prompt = get_bounding_box(ground_truth_mask)
    inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 예측
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    # 예측 결과 후처리
    pred_mask_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    pred_mask_prob = pred_mask_prob.cpu().numpy().squeeze()
    pred_mask = (pred_mask_prob > 0.5).astype(np.uint8)
    
    # 결과 시각화
    visualize_prediction(
        image=image,
        ground_truth=ground_truth_mask,
        prediction=pred_mask,
        save_path="prediction_result.png"
    )

if __name__ == "__main__":
    main()