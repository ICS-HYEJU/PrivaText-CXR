from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel

# 1. MedCLIP-ResNet50
model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
model.from_pretrained()

# 2. MedCLIP-ViT (Vision Transformer)
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()


import torchxrayvision as xrv

model = xrv.models.DenseNet(weights="densenet121-res224-all")

# 18°¡Áö ÁúÈ¯ ¸ñ·Ï Ãâ·Â
print(model.pathologies)
