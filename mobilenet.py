"""
Create image embeddings used in deep sort algorithm
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class MobileNetEmbedder:
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0)
        embedding = self.model(img_tensor).squeeze()
        embedding = embedding.float().cpu().numpy()
        embedding = embedding.mean(2).mean(1) 
        return embedding / np.linalg.norm(embedding)
