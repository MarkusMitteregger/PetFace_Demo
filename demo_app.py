"""
Streamlit demo application for the PetIdentifier utility.

This app allows users to:
- Register pet images with a name (upload or camera).
- View the gallery of registered pets.
- Identify an unknown pet image by nearest-neighbour search in embedding space.

Session state stores a PetIdentifier instance so registrations persist
while the Streamlit session is active.
"""

import streamlit as st
from PIL import Image
from pet_identifier import PetIdentifier

st.set_page_config(page_title="🐾 Pet Identifier", layout="wide")
st.title("🐶 Pet Identifier")

# Initialize or reuse PetIdentifier in Streamlit session state
# This avoids reloading the model on every interaction.
if "identifier" not in st.session_state:
    st.session_state.identifier = PetIdentifier()

identifier = st.session_state.identifier

# ========================
# Section: Add New Pets
# ========================
st.header("Register your pets 🐾")

with st.expander("Add a new pet"):
    # Allow the user to choose upload vs camera input for registering a pet
    input_mode = st.radio(
        "Choose input method for your pet photo:",
        ["Upload image", "Use camera"],
        key="add_mode"
    )

    if input_mode == "Upload image":
        # File uploader returns a BytesIO-like object or None
        new_pet_image = st.file_uploader("Upload your pet photo", type=["jpg", "jpeg", "png"], key="add_upload")
    else:
        # camera_input returns an UploadedFile-like object when a photo is taken
        new_pet_image = st.camera_input("Take a photo of your pet", key="add_camera")

    new_pet_name = st.text_input("Enter pet name")

    # Add the pet only when a name and image are provided and button is pressed
    if st.button("Add pet") and new_pet_image and new_pet_name:
        # Convert to RGB PIL Image and shrink for better performance in the demo
        image = Image.open(new_pet_image).convert("RGB")
        image.thumbnail((512, 512))  # optional: reduce size to save memory / compute
        identifier.add_pet(new_pet_name, image)
        st.success(f"Added {new_pet_name}!")

# ========================
# Display Registered Pets
# ========================
if identifier.known_pets:
    st.subheader("Registered pets")
    # Create a column per registered pet for a compact gallery view
    cols = st.columns(len(identifier.known_pets))
    for i, (name, _, img) in enumerate(identifier.known_pets):
        with cols[i]:
            st.image(img, caption=name, width=150)
else:
    st.info("No pets registered yet. Add one above.")

st.markdown("---")

# ========================
# Section: Identify Pets
# ========================
st.header("Identify a pet 🐕🐈")

# Choose input method for identification (upload or camera)
input_mode = st.radio(
    "Choose input method for identification:",
    ["Upload image", "Use camera"],
    key="identify_mode"
)

if input_mode == "Upload image":
    upload = st.file_uploader("Upload or take a photo", type=["jpg", "jpeg", "png"], key="identify_upload")
else:
    upload = st.camera_input("Take a picture", key="identify_camera")

if upload is not None:
    # Prepare image and display side-by-side with results
    image = Image.open(upload).convert("RGB")
    image.thumbnail((512, 512))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Pet to identify", width=300)

    with col2:
        # Run identification in a spinner to show progress
        with st.spinner("Identifying..."):
            name, dist = identifier.identify(image)
        st.subheader("🔍 Results")
        st.write(f"**Pet:** {name}")
        st.write(f"**Distance:** {dist:.3f}")
else:
    st.info("📸 Upload or take a photo to identify your pet.")

st.caption("Made with ❤️ using Streamlit and PyTorch")