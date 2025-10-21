from PIL import Image
import torch
from torchvision import transforms
from model.EfficientNet import EfficientNet
import numpy as np

DEVICE = torch.device("cpu")

def preprocess(image: Image.Image):
    """Apply same normalization as training (without random augmentations)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image)


def get_embeddings(image: Image.Image, model):
    """Run the model on the image and return predicted labels."""
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embeddings = model(input_tensor) 

    return embeddings

def load_model():
    CONFIG = {
    'EMBEDDING_DIM': 224
}
    model = EfficientNet(embedding_dim=CONFIG['EMBEDDING_DIM'])
    checkpoint = torch.load("best_efficientnet_triplet.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def euclidean_distance(emb1, emb2):
    return torch.norm(emb1 - emb2, p=2, dim=1).item()

def find_closest_cat(test_emb, known_cats):
    closest_cat = None
    min_distance = float('inf')
    for cat in known_cats:
        name, emb = cat[0], cat[1]
        dist = euclidean_distance(test_emb, emb)
        if dist < min_distance:
            min_distance = dist
            closest_cat = name
    return closest_cat, min_distance
