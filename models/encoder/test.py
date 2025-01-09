from transformers import SamModel, SamProcessor
import torch
from PIL import Image
import torch.nn.functional as F

class ImageEncoder:
    def __init__(self):
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge")
        self.encoder = self.model.vision_encoder
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.encoder.eval()  # 평가 모드로 설정
        
    def encode_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.encoder(pixel_values=inputs.pixel_values)
        
        # 전체 패치의 평균을 이미지 임베딩으로 사용
        image_embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 1280]
        return image_embedding
    
    def compute_similarity(self, img1_path, img2_path):
        # 두 이미지의 임베딩 추출
        emb1 = self.encode_image(img1_path)
        emb2 = self.encode_image(img2_path)
        
        # 코사인 유사도 계산
        similarity = F.cosine_similarity(emb1, emb2)
        return similarity.item()

# 사용 예시
encoder = ImageEncoder()
similarity = encoder.compute_similarity("image1.jpg", "image2.jpg")
print(f"이미지 유사도: {similarity}")  # -1에서 1 사이의 값 (1에 가까울수록 유사)