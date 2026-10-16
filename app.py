```python
import os
from io import BytesIO

import streamlit as st
import torch
from PIL import Image

from models import load_generators, translate, is_sketch

st.set_page_config(
    page_title="Face ↔ Sketch | CycleGAN",
    page_icon="🎨",
    layout="centered",
)

st.title("🎨 Face ↔ Sketch (CycleGAN)")
st.write(
    "Convert a real face into a sketch, or a sketch into a real face."
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "generators.pth",
)


@st.cache_resource
def get_model():
    device = torch.device("cpu")
    G_AB, G_BA, size = load_generators(MODEL_PATH, device)
    return G_AB, G_BA, size


try:
    with st.spinner("Loading your trained CycleGAN model..."):
        G_AB, G_BA, SIZE = get_model()
except Exception as e:
    st.error("The model could not be loaded.")
    st.code(str(e))
    st.stop()

st.subheader("1. Choose an image source")

source = st.radio(
    "How would you like to provide your image?",
    ["📁 Upload an image", "📸 Take a photo with camera"],
)

image = None

if source == "📁 Upload an image":
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

else:
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

st.subheader("2. Select the input type")

mode = st.selectbox(
    "What kind of image did you provide?",
    [
        "Auto-detect",
        "This is a real face/photo",
        "This is a sketch",
    ],
)

if image is not None:
    st.subheader("Input image")
    st.image(image, caption="Your selected image", use_container_width=True)

    if st.button("🎨 Convert image", type="primary"):
        with st.spinner("Converting your image..."):
            if mode == "Auto-detect":
                sketch = is_sketch(image)
            else:
                sketch = mode == "This is a sketch"

            generator = G_BA if sketch else G_AB
            output = translate(generator, image, SIZE)
            output = output.resize(
                (256, 256),
                Image.Resampling.BICUBIC,
            )

        target = "Real face/photo" if sketch else "Sketch"

        st.subheader(f"Converted result: {target}")
        st.image(output, use_container_width=True)

        buffer = BytesIO()
        output.save(buffer, format="PNG")

        st.download_button(
            "⬇️ Download converted image",
            data=buffer.getvalue(),
            file_name="cyclegan_output.png",
            mime="image/png",
        )

st.caption("Powered by your trained CycleGAN generators.")
```
