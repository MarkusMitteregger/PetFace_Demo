from PIL import Image
import torch
from torchvision import transforms
import pet_identifier as fn


if __name__ == "__main__":
    model = fn.load_model()
    # Example usage
    img_test = Image.open("Cats/Test_Tigercat.jpg")  # Replace with your image path
    cat_1 = Image.open("Cats/cat_3.png")  
    cat_2 = Image.open("Cats/cat_14.png")
    cat_3 = Image.open("Cats/cat_15.png")

    # Create dictionaries with embeddings of known cats
    list_1 = ["Fred", fn.get_embeddings(cat_1, model)]
    list_2 = ["George", fn.get_embeddings(cat_2, model)]
    list_3 = ["Ringo", fn.get_embeddings(cat_3, model)]
    known_cats = [list_1, list_2, list_3]

    # Get embedding for test image
    test_emb = fn.get_embeddings(img_test, model)

    closest_cat, distance = fn.find_closest_cat(test_emb, known_cats)
    print(f"Closest cat: {closest_cat}, Distance: {distance}")
    print("Done.")