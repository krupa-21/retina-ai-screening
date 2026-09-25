
import streamlit as st
import torch
import timm
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


MODEL_PATH = "best_retinopathy_model.pth"

IMG_SIZE = 224

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

DEVICE = torch.device("cpu")


val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=5
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def analyze_image(image, model):

    image = image.convert("RGB")

    input_tensor = val_transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

    confidence = probabilities[
        predicted_class
    ].item()

    target_layers = [
        model.conv_head
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    resized_image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    rgb_image = (
        np.array(resized_image)
        .astype(np.float32) / 255.0
    )

    heatmap = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    probability_data = {
        "Class": CLASS_NAMES,
        "Probability (%)": [
            round(
                float(probabilities[i].item() * 100),
                2
            )
            for i in range(5)
        ]
    }

    return (
        CLASS_NAMES[predicted_class],
        confidence * 100,
        pd.DataFrame(probability_data),
        heatmap
    )


st.set_page_config(
    page_title="Retina AI Screening",
    page_icon="🩺",
    layout="wide"
)


st.title("🩺 Retina AI Screening")

st.subheader(
    "Explainable AI for Diabetic Retinopathy Screening"
)

st.write(
    "Upload a retinal fundus image to receive an "
    "AI-assisted screening result and Grad-CAM explanation."
)


try:

    model = load_model()

except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()


uploaded_file = st.file_uploader(
    "Upload Retinal Fundus Image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Retinal Image",
        use_container_width=True
    )

    if st.button(
        "Analyze Image",
        type="primary"
    ):

        with st.spinner(
            "Analyzing retinal image..."
        ):

            prediction, confidence, probabilities, heatmap = (
                analyze_image(
                    image,
                    model
                )
            )

        st.success(
            f"Prediction: {prediction}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader(
            "Class Probabilities"
        )

        st.dataframe(
            probabilities,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Grad-CAM Explanation"
        )

        st.image(
            heatmap,
            caption="Regions influencing the model prediction",
            use_container_width=True
        )


st.divider()

st.caption(
    "This is an AI-assisted screening prototype "
    "and not a medical diagnosis. Results should be "
    "reviewed by a qualified healthcare professional."
)
