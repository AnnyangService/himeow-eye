import torch
from transformers import SamModel, SamProcessor
from PIL import Image

class FeatureExtractor:
    def __init__(self, model_name="facebook/sam-vit-base", checkpoint_path=None, gpu_id=3):
        # GPU 설정
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # 모델 로드
        self.model = SamModel.from_pretrained(model_name)
        self.processor = SamProcessor.from_pretrained(model_name)
        
        if checkpoint_path:
            self.load_custom_checkpoint(checkpoint_path)
        
        self.model = self.model.to(self.device)
        self.vision_encoder = self.model.vision_encoder
        self.vision_encoder.eval()

    def load_custom_checkpoint(self, checkpoint_path):
        """파인튜닝된 sam vision encoder 불러오기"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        current_state_dict = self.model.state_dict()
        for name, param in checkpoint['vision_encoder_state_dict'].items():
            if 'vision_encoder' in name:
                current_state_dict[name] = param
        self.model.load_state_dict(current_state_dict)
        print(f"Loaded custom checkpoint from {checkpoint_path}")

    def extract_features(self, image_path):
        """이미지에서 특징 추출"""
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.vision_encoder(inputs.pixel_values)
                image_embeddings = features.last_hidden_state.contiguous()
                return image_embeddings.clone() 
                
        except Exception as e:
            print(f"Error in feature extraction for {image_path}: {str(e)}")
            return None