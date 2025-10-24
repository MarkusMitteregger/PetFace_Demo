from PIL import Image
import torch
from torchvision import transforms
from model.EfficientNet import EfficientNet

"""
pet_identifier.py

Contains PetIdentifier, a small utility class that wraps a pretrained
EfficientNet embedding model to register known pet images and identify
new images by nearest-neighbour search in embedding space.

Usage:
    pid = PetIdentifier(model_path="best_efficientnet_triplet.pth", device="cpu", threshold=0.5)
    pid.add_pet("Fido", Image.open("fido.jpg"))
    name, dist = pid.identify(Image.open("unknown.jpg"))
"""

class PetIdentifier:
    """
    PetIdentifier manages a list of known pets (name, embedding, image)
    and provides identification of an input image by comparing embeddings.

    Attributes:
        device (torch.device): device used for model and tensors.
        threshold (float): maximum distance to consider a match; otherwise "Unknown".
        model (torch.nn.Module): loaded EfficientNet embedding model.
        known_pets (list): list of [name, embedding, image] entries.
    """

    def __init__(self, model_path="best_efficientnet_triplet.pth", device="cpu", threshold=0.5):
        """
        Initialize the identifier.

        Args:
            model_path (str): path to a checkpoint containing "model_state_dict".
            device (str or torch.device): device specifier, e.g. "cpu" or "cuda".
            threshold (float): Euclidean distance threshold for recognizing a known pet.
        """
        self.device = torch.device(device)
        self.threshold = threshold
        # Load the embedding model (set to eval mode)
        self.model = self._load_model(model_path)
        # Each known pet stored as [name (str), embedding (torch.Tensor of shape [1, D]), image (PIL.Image)]
        self.known_pets = []

    def _load_model(self, model_path):
        """
        Load model weights into EfficientNet and prepare it for inference.

        Args:
            model_path (str): file path to checkpoint.

        Returns:
            torch.nn.Module: model loaded on the requested device in eval() mode.
        """
        model = EfficientNet(embedding_dim=128)
        checkpoint = torch.load(model_path, map_location=self.device)
        # Expect checkpoint to contain "model_state_dict"
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, image):
        """
        Preprocess a PIL image for the embedding model.

        Steps:
            - Resize to (224, 224)
            - Convert to tensor
            - Normalize with ImageNet mean/std

        Args:
            image (PIL.Image): input image to preprocess.

        Returns:
            torch.Tensor: preprocessed image tensor of shape [C, H, W].
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def get_embedding(self, image):
        """
        Compute the embedding for a PIL image.

        Args:
            image (PIL.Image): input image.

        Returns:
            torch.Tensor: embedding tensor of shape [1, embedding_dim] on self.device.
        """
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)  # add batch dim
        with torch.no_grad():
            emb = self.model(tensor)
        return emb

    def add_pet(self, name, image):
        """
        Register a known pet by computing and storing its embedding.

        Args:
            name (str): label/name for the pet.
            image (PIL.Image): image of the pet to register.
        """
        emb = self.get_embedding(image)
        # Store the original image for possible later use (display, re-compute, etc.)
        self.known_pets.append([name, emb, image])

    def _euclidean_distance(self, emb1, emb2):
        """
        Compute Euclidean distance between two embeddings.

        Both embeddings are expected to have shape [1, D]. The function
        subtracts and computes norm along the feature dimension, returning
        a Python float.

        Args:
            emb1 (torch.Tensor): tensor shape [1, D].
            emb2 (torch.Tensor): tensor shape [1, D].

        Returns:
            float: L2 distance between emb1 and emb2.
        """
        # result is a tensor of shape [1], .item() converts to float
        return torch.norm(emb1 - emb2, p=2, dim=1).item()

    def identify(self, image):
        """
        Identify the closest known pet to the provided image.

        Args:
            image (PIL.Image): image to identify.

        Returns:
            tuple: (name (str), distance (float)). If the smallest distance
                   exceeds self.threshold, name will be "Unknown".
        """
        test_emb = self.get_embedding(image)
        closest, min_dist = None, float("inf")
        # Linear search for nearest neighbour in embedding space
        for name, emb, _ in self.known_pets:
            dist = self._euclidean_distance(test_emb, emb)
            if dist < min_dist:
                min_dist = dist
                closest = name
        # If no known pet or distance too large, return Unknown
        if min_dist > self.threshold:
            closest = "Unknown"
        return closest, min_dist
