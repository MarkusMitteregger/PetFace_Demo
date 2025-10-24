from PIL import Image
import torch
from torchvision import transforms
from model.EfficientNet import EfficientNet

class PetIdentifier:
    def __init__(self, model_path="best_efficientnet_triplet.pth", device="cpu", threshold=0.7514):
        self.device = torch.device(device)
        self.threshold = threshold
        self.model = self._load_model(model_path)
        self.known_pets = []  # Each: [name, embedding, image]

    def _load_model(self, model_path):
        model = EfficientNet(embedding_dim=128)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def get_embedding(self, image):
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor)
        return emb

    def add_pet(self, name, image):
        emb = self.get_embedding(image)
        self.known_pets.append([name, emb, image])

    def _euclidean_distance(self, emb1, emb2):
        return torch.norm(emb1 - emb2, p=2, dim=1).item()

    def identify(self, image):
        test_emb = self.get_embedding(image)
        closest, min_dist = None, float("inf")
        for name, emb, _ in self.known_pets:
            dist = self._euclidean_distance(test_emb, emb)
            if dist < min_dist:
                min_dist = dist
                closest = name
        if min_dist > self.threshold:
            closest = "Unknown"
        return closest, min_dist
