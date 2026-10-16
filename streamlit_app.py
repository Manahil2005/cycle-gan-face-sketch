import os
import streamlit as st
import torch
from PIL import Image
from models import load_generators, translate, is_sketch

st.set_page_config(
    page_title="Face ↔ Sketch | CycleGAN",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Face ↔ Sketch (CycleGAN)")
st.write(
    "Upload a real face or a sketch. The trained CycleGAN model "
    "will convert it into the other style."
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generators.pth")

@st.cache_resource
def get_model():
    device = torch.device("cpu")
    G_AB, G_BA, size = load_generators(MODEL_PATH, device)
    return G_AB, G_BA, size

try:
    with st.spinner("Loading your trained CycleGAN model..."):
        G_AB, G_BA, SIZE = get_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error("The model could not be loaded.")
    st.code(str(e))
    st.stop()

mode = st.selectbox(
    "Input image type",
    [
        "Auto-detect",
        "This is a real face/photo",
        "This is a sketch"
    ]
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Input image")
        st.image(image, use_container_width=True)

        if st.button("Convert image", type="primary"):
            with st.spinner("Converting image..."):
                if mode == "Auto-detect":
                    sketch = is_sketch(image)
                else:
                    sketch = mode == "This is a sketch"

                generator = G_BA if sketch else G_AB
                output = translate(generator, image, SIZE)
                output = output.resize((256, 256), Image.Resampling.BICUBIC)

            detected = "Sketch" if sketch else "Real face/photo"
            target = "Real face/photo" if sketch else "Sketch"

            st.write(f"Detected input: **{detected}**")
            st.subheader(f"Output: {target}")
            st.image(output, use_container_width=True)

            from io import BytesIO
            buffer = BytesIO()
            output.save(buffer, format="PNG")

            st.download_button(
                "Download converted image",
                data=buffer.getvalue(),
                file_name="cyclegan_output.png",
                mime="image/png"
            )

    except Exception as e:
        st.error("Could not process this image.")
        st.code(str(e))

st.caption("Powered by your trained CycleGAN generators.")
