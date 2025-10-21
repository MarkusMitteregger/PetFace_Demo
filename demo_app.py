import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import functions as fn


model = fn.load_model()
img_test = Image.open("Cats/Test_Tigercat.jpg")  # Replace with your image path
cat_1 = Image.open("Cats/cat_3.png")  
cat_2 = Image.open("Cats/cat_14.png")
cat_3 = Image.open("Cats/cat_15.png")

# Create dictionaries with embeddings of known cats
list_1 = ["Fred", fn.get_embeddings(cat_1, model), cat_1]
list_2 = ["George", fn.get_embeddings(cat_2, model), cat_2]
list_3 = ["Ringo", fn.get_embeddings(cat_3, model), cat_3]
known_cats = [list_1, list_2, list_3]
print("Model and known cats loaded.")


# -------------------------------
# 🎨 STREAMLIT APP UI
# -------------------------------
st.set_page_config(page_title="🐾 Pet Identifier", layout="wide")

st.title("🐶 Pet Identifier")

st.markdown("## Your registered Pets")

cols = st.columns(len(known_cats))
for idx, cat in enumerate(known_cats):
    with cols[idx]:
        st.image(cat[2], caption=cat[0], use_container_width=True)

st.markdown("---")
st.markdown(
    "Upload or take a photo of an animal and the model will identify your beloved pet."
)

uploaded_file = st.file_uploader(
    "Upload or take a photo",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    test_emb = fn.get_embeddings(image, model)


    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Cat to identify", use_container_width=True)

    with col2:
        st.subheader("🔍 Classification Results")
        with st.spinner("Running model..."):
            closest_cat, distance = fn.find_closest_cat(test_emb, known_cats)
        st.write(f"**Pet Name:** {closest_cat}")
        st.write(f"**Distance:** {distance:.4f}")
        st.success("✅ Pet identification complete!")

else:
    st.info("👆 Upload or take a photo to begin classification.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and PyTorch")

