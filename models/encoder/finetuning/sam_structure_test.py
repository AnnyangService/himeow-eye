from transformers import SamModel

model = SamModel.from_pretrained("facebook/sam-vit-huge")

print("=== Encoder Related Parameters ===")
for name, _ in model.named_parameters():
    if 'encoder' in name.lower():  # encoder가 들어간 파라미터만
        print(name.split('.')[0])  # 첫 부분만 출력