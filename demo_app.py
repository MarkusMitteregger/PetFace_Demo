import streamlit as st
from PIL import Image
from pet_identifier import PetIdentifier

st.set_page_config(page_title="🐾 Pet Identifier", layout="wide")
st.title("🐶 Pet Identifier")

# Initialize the PetIdentifier class in Streamlit session state
if "identifier" not in st.session_state:
    st.session_state.identifier = PetIdentifier()

identifier = st.session_state.identifier

# ========================
# Section: Add New Pets
# ========================
st.header("Register your pets 🐾")

with st.expander("Add a new pet"):
    input_mode = st.radio(
        "Choose input method for your pet photo:",
        ["Upload image", "Use camera"],
        key="add_mode"
    )

    if input_mode == "Upload image":
        new_pet_image = st.file_uploader("Upload your pet photo", type=["jpg", "jpeg", "png"], key="add_upload")
    else:
        new_pet_image = st.camera_input("Take a photo of your pet", key="add_camera")

    new_pet_name = st.text_input("Enter pet name")

    if st.button("Add pet") and new_pet_image and new_pet_name:
        image = Image.open(new_pet_image).convert("RGB")
        image.thumbnail((512, 512))  # optional resizing for performance
        identifier.add_pet(new_pet_name, image)
        st.success(f"Added {new_pet_name}!")

# ========================
# Display Registered Pets
# ========================
if identifier.known_pets:
    st.subheader("Registered pets")
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
    image = Image.open(upload).convert("RGB")
    image.thumbnail((512, 512))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Pet to identify", width=300)

    with col2:
        with st.spinner("Identifying..."):
            name, dist = identifier.identify(image)
        st.subheader("🔍 Results")
        st.write(f"**Pet:** {name}")
        st.write(f"**Distance:** {dist:.3f}")
else:
    st.info("📸 Upload or take a photo to identify your pet.")

st.caption("Made with ❤️ using Streamlit and PyTorch")