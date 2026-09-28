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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.title("Retina AI Screening")

st.write(
    "Explainable AI for diabetic retinopathy screening"
)

st.caption(
    "AI-assisted analysis of retinal fundus images "
    "with confidence analysis and Grad-CAM explanations."
)


st.divider()


st.subheader("Upload retinal image")

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type=["png", "jpg", "jpeg"]
)


try:

    model = load_model()

except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.divider()

    st.subheader("Screening analysis")

    image_column, analysis_column = st.columns(
        [1, 1],
        gap="large"
    )

    with image_column:

        st.write("**Retinal fundus image**")

        st.image(
            image,
            use_container_width=True
        )

    with analysis_column:

        st.write("**AI screening result**")

        analyze_button = st.button(
            "Analyze image",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            with st.spinner(
                "Analyzing retinal image..."
            ):

                (
                    prediction,
                    confidence,
                    probabilities,
                    heatmap
                ) = analyze_image(
                    image,
                    model
                )

            st.success(
                f"Predicted stage: {prediction}"
            )

            st.metric(
                label="Model confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                min(confidence / 100, 1.0)
            )

            st.write(
                "Confidence for the predicted class"
            )

    if analyze_button:

        st.divider()

        st.subheader("Prediction probabilities")

        probability_column, table_column = st.columns(
            [1.2, 0.8],
            gap="large"
        )

        with probability_column:

            chart_data = probabilities.copy()

            chart_data = chart_data.set_index(
                "Class"
            )

            st.bar_chart(
                chart_data["Probability (%)"]
            )

        with table_column:

            st.dataframe(
                probabilities,
                use_container_width=True,
                hide_index=True
            )


        st.divider()

        st.subheader("Explainable AI")

        st.write(
            "Grad-CAM highlights regions of the retinal "
            "image that contributed to the model prediction."
        )

        original_column, heatmap_column = st.columns(
            2,
            gap="large"
        )

        with original_column:

            st.write("**Original image**")

            st.image(
                image,
                use_container_width=True
            )

        with heatmap_column:

            st.write("**Grad-CAM explanation**")

            st.image(
                heatmap,
                use_container_width=True
            )

        st.info(
            "The Grad-CAM visualization explains model "
            "attention and should not be interpreted as "
            "a clinical diagnosis."
        )


st.divider()

st.subheader("About the system")

about_column, limitation_column = st.columns(
    2,
    gap="large"
)

with about_column:

    st.write("**Model**")

    st.write(
        "EfficientNet-B0 image classification model."
    )

    st.write("**Classification categories**")

    st.write(
        "No DR, Mild, Moderate, Severe, "
        "and Proliferative DR."
    )


with limitation_column:

    st.write("**Limitations**")

    st.write(
        "Predictions may be affected by image quality, "
        "dataset limitations, acquisition conditions, "
        "and differences between training and real-world images."
    )

    st.write("**Intended use**")

    st.write(
        "Research, education, and AI-assisted "
        "screening demonstration."
    )


st.divider()

st.warning(
    "Medical disclaimer: This is an AI-assisted screening "
    "prototype and not a medical diagnosis. Results should "
    "be reviewed by a qualified healthcare professional."
)