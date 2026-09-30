import streamlit as st
import torch
import timm
import cv2
import numpy as np
import pandas as pd

from PIL import Image, ImageFilter
from torchvision import transforms

from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
    KeepTogether
)

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

# Image validation thresholds

MIN_IMAGE_SIZE = 224

MIN_BRIGHTNESS = 25
MAX_BRIGHTNESS = 235

MIN_CONTRAST = 20

MIN_SHARPNESS = 5

MIN_FUNDUS_SCORE = 0.65

def calculate_image_quality(image):

    image = image.convert("RGB")

    image_array = np.array(
        image
    ).astype(np.float32)

    height, width = image_array.shape[:2]

    gray = (
        0.299 * image_array[:, :, 0]
        + 0.587 * image_array[:, :, 1]
        + 0.114 * image_array[:, :, 2]
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    edges = np.array(
        image.convert("L").filter(
            ImageFilter.FIND_EDGES
        )
    ).astype(np.float32)

    sharpness = float(
        np.var(edges)
    )

    checks = {
        "resolution": (
            width >= MIN_IMAGE_SIZE
            and height >= MIN_IMAGE_SIZE
        ),
        "brightness": (
            MIN_BRIGHTNESS
            <= brightness
            <= MAX_BRIGHTNESS
        ),
        "contrast": (
            contrast >= MIN_CONTRAST
        ),
        "sharpness": (
            sharpness >= MIN_SHARPNESS
        )
    }

    quality_score = sum(
        checks.values()
    ) / len(checks)

    return {
        "width": width,
        "height": height,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "checks": checks,
        "score": quality_score
    }


def calculate_fundus_suitability(image):

    image = image.convert("RGB")

    image_array = np.array(
        image
    )

    height, width = image_array.shape[:2]

    # Resize for consistent image processing.

    resized = cv2.resize(
        image_array,
        (512, 512)
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2GRAY
    )

    red = resized[:, :, 0].astype(np.float32)
    green = resized[:, :, 1].astype(np.float32)
    blue = resized[:, :, 2].astype(np.float32)

    # ---------------------------------------------------------
    # 1. RETINAL COLOR CHARACTERISTICS
    # ---------------------------------------------------------

    reddish_pixels = (
        (red > green * 1.05)
        & (red > blue * 1.08)
        & (red > 45)
    )

    reddish_ratio = float(
        np.mean(reddish_pixels)
    )

    # ---------------------------------------------------------
    # 2. CENTRAL RETINAL REGION
    # ---------------------------------------------------------

    center_y = 256
    center_x = 256

    yy, xx = np.ogrid[:512, :512]

    center_distance = np.sqrt(
        ((xx - center_x) / 220) ** 2
        + ((yy - center_y) / 220) ** 2
    )

    central_mask = center_distance <= 1.0

    center_region = resized[
        central_mask
    ]

    center_gray = gray[
        central_mask
    ]

    center_red = center_region[:, 0].astype(
        np.float32
    )

    center_green = center_region[:, 1].astype(
        np.float32
    )

    center_blue = center_region[:, 2].astype(
        np.float32
    )

    center_reddish_ratio = float(
        np.mean(
            (
                (center_red > center_green * 1.03)
                & (center_red > center_blue * 1.05)
            )
        )
    )

    center_brightness = float(
        np.mean(center_gray)
    )

    # ---------------------------------------------------------
    # 3. DARK OUTER BOUNDARY
    # ---------------------------------------------------------

    outer_mask = (
        center_distance >= 1.05
    ) & (
        center_distance <= 1.40
    )

    outer_region = gray[
        outer_mask
    ]

    outer_brightness = float(
        np.mean(outer_region)
    )

    # Fundus photographs often have a darker
    # boundary surrounding the retinal field.

    boundary_difference = (
        center_brightness
        - outer_brightness
    )

    dark_boundary = (
        outer_brightness < 100
        and boundary_difference > 8
    )

    # ---------------------------------------------------------
    # 4. CIRCULAR RETINAL FIELD
    # ---------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=120,
        param1=100,
        param2=45,
        minRadius=130,
        maxRadius=270
    )

    circular_field = False
    circle_score = 0.0

    if circles is not None:

        circles = np.round(
            circles[0]
        ).astype(int)

        best_circle = None
        best_score = 0.0

        for x, y, radius in circles:

            distance_from_center = np.sqrt(
                (x - 256) ** 2
                + (y - 256) ** 2
            )

            center_alignment = max(
                0.0,
                1.0
                - distance_from_center / 256.0
            )

            radius_score = 1.0 - min(
                abs(radius - 205) / 205,
                1.0
            )

            current_score = (
                0.55 * center_alignment
                + 0.45 * radius_score
            )

            if current_score > best_score:

                best_score = current_score
                best_circle = (
                    x,
                    y,
                    radius
                )

        if best_circle is not None:

            x, y, radius = best_circle

            circular_field = (
                best_score >= 0.55
                and 150 <= radius <= 260
            )

            circle_score = best_score

    # ---------------------------------------------------------
    # 5. RADIAL BRIGHTNESS STRUCTURE
    # ---------------------------------------------------------

    inner_mask = (
        center_distance <= 0.75
    )

    middle_mask = (
        (center_distance > 0.75)
        & (center_distance <= 1.0)
    )

    inner_brightness = float(
        np.mean(gray[inner_mask])
    )

    middle_brightness = float(
        np.mean(gray[middle_mask])
    )

    radial_difference = abs(
        inner_brightness
        - middle_brightness
    )

    radial_structure = (
        radial_difference >= 3
    )

    # ---------------------------------------------------------
    # 6. EDGE DISTRIBUTION
    # ---------------------------------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    reasonable_edge_density = (
        0.01 <= edge_density <= 0.25
    )

    # ---------------------------------------------------------
    # 7. ASPECT RATIO
    # ---------------------------------------------------------

    aspect_ratio = (
        width / height
    )

    reasonable_aspect_ratio = (
        0.65 <= aspect_ratio <= 1.60
    )

    # ---------------------------------------------------------
    # FUNDUS SCORE
    # ---------------------------------------------------------

    score = 0.0

    if circular_field:
        score += 0.30

    if dark_boundary:
        score += 0.20

    if center_reddish_ratio >= 0.20:
        score += 0.15

    if reddish_ratio >= 0.15:
        score += 0.10

    if radial_structure:
        score += 0.10

    if reasonable_edge_density:
        score += 0.05

    if reasonable_aspect_ratio:
        score += 0.05

    if 35 <= center_brightness <= 220:
        score += 0.05

    # ---------------------------------------------------------
    # STRICT FUNDUS GATE
    # ---------------------------------------------------------

    # A high score alone is not sufficient.
    #
    # At least one strong structural characteristic
    # must be present together with retinal characteristics.

    structural_evidence = (
        circular_field
        or dark_boundary
    )

    retinal_evidence = (
        center_reddish_ratio >= 0.20
        and reddish_ratio >= 0.15
    )

    suitable = (
        score >= MIN_FUNDUS_SCORE
        and structural_evidence
        and retinal_evidence
    )

    return {
        "score": score,
        "suitable": suitable,
        "circular_field": circular_field,
        "circle_score": circle_score,
        "dark_boundary": dark_boundary,
        "center_reddish_ratio": center_reddish_ratio,
        "reddish_ratio": reddish_ratio,
        "center_brightness": center_brightness,
        "outer_brightness": outer_brightness,
        "boundary_difference": boundary_difference,
        "radial_structure": radial_structure,
        "edge_density": edge_density,
        "aspect_ratio": aspect_ratio
    }



def validate_image(image):

    quality = calculate_image_quality(
        image
    )

    fundus = calculate_fundus_suitability(
        image
    )

    quality_passed = (
        quality["score"] >= 0.75
    )

    fundus_passed = (
        fundus["suitable"]
    )

    return {
        "quality": quality,
        "fundus": fundus,
        "quality_passed": quality_passed,
        "fundus_passed": fundus_passed,
        "passed": (
            quality_passed
            and fundus_passed
        )
    }


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

def create_pdf_report(
    image,
    prediction,
    confidence,
    probabilities,
    heatmap,
    validation
):

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=16
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        spaceBefore=10,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=6
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=10,
        textColor=colors.grey
    )

    story = []

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "RETINA AI SCREENING",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Assisted Diabetic Retinopathy Screening Report",
            subtitle_style
        )
    )

    screening_time = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    screening_id = datetime.now().strftime(
        "RAS-%Y%m%d-%H%M%S"
    )

    information_data = [
        [
            Paragraph("<b>Screening ID</b>", body_style),
            Paragraph(screening_id, body_style)
        ],
        [
            Paragraph("<b>Date & Time</b>", body_style),
            Paragraph(screening_time, body_style)
        ],
        [
            Paragraph("<b>Image Quality</b>", body_style),
            Paragraph(
                "Suitable"
                if validation["quality_passed"]
                else "Not suitable",
                body_style
            )
        ],
        [
            Paragraph("<b>Fundus Validation</b>", body_style),
            Paragraph(
                "Suitable"
                if validation["fundus_passed"]
                else "Not suitable",
                body_style
            )
        ]
    ]

    information_table = Table(
        information_data,
        colWidths=[
            45 * mm,
            125 * mm
        ]
    )

    information_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#F2F4F7")
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#D0D5DD")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        information_table
    )

    story.append(
        Spacer(1, 8)
    )

    # ---------------------------------------------------------
    # ORIGINAL IMAGE
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Retinal Fundus Image",
            heading_style
        )
    )

    original_buffer = BytesIO()

    image.save(
        original_buffer,
        format="PNG"
    )

    original_buffer.seek(0)

    original_report_image = ReportLabImage(
        original_buffer,
        width=150 * mm,
        height=105 * mm
    )

    original_report_image.hAlign = "CENTER"

    story.append(
        original_report_image
    )

    # ---------------------------------------------------------
    # AI RESULT
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "AI Screening Result",
            heading_style
        )
    )

    result_data = [
        [
            Paragraph(
                "<b>Predicted Stage</b>",
                body_style
            ),
            Paragraph(
                f"<b>{prediction}</b>",
                body_style
            )
        ],
        [
            Paragraph(
                "<b>Model Confidence</b>",
                body_style
            ),
            Paragraph(
                f"<b>{confidence:.2f}%</b>",
                body_style
            )
        ]
    ]

    result_table = Table(
        result_data,
        colWidths=[
            60 * mm,
            110 * mm
        ]
    )

    result_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#F2F4F7")
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#D0D5DD")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        result_table
    )

    # ---------------------------------------------------------
    # PROBABILITIES
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Prediction Probabilities",
            heading_style
        )
    )

    probability_rows = [
        [
            Paragraph("<b>Classification</b>", body_style),
            Paragraph("<b>Probability</b>", body_style)
        ]
    ]

    for _, row in probabilities.iterrows():

        probability_rows.append([
            Paragraph(
                str(row["Class"]),
                body_style
            ),
            Paragraph(
                f"{float(row['Probability (%)']):.2f}%",
                body_style
            )
        ])

    probability_table = Table(
        probability_rows,
        colWidths=[
            110 * mm,
            60 * mm
        ]
    )

    probability_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#F2F4F7")
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#D0D5DD")
            ),
            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "RIGHT"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            )
        ])
    )

    story.append(
        probability_table
    )

    # ---------------------------------------------------------
    # GRAD-CAM
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Explainable AI — Grad-CAM",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "The Grad-CAM visualization highlights regions "
            "that contributed most strongly to the model's "
            "prediction.",
            body_style
        )
    )

    heatmap_buffer = BytesIO()

    Image.fromarray(
        heatmap
    ).save(
        heatmap_buffer,
        format="PNG"
    )

    heatmap_buffer.seek(0)

    gradcam_report_image = ReportLabImage(
        heatmap_buffer,
        width=150 * mm,
        height=105 * mm
    )

    gradcam_report_image.hAlign = "CENTER"

    story.append(
        gradcam_report_image
    )

    # ---------------------------------------------------------
    # INTERPRETATION
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Screening Interpretation",
            heading_style
        )
    )

    if prediction == "No DR":

        interpretation = (
            "The model classified this retinal image in the "
            "No DR category. This result does not replace a "
            "professional eye examination."
        )

    else:

        interpretation = (
            "The model detected a diabetic retinopathy "
            "category in this retinal image. Professional "
            "evaluation by a qualified eye-care provider "
            "is recommended."
        )

    story.append(
        Paragraph(
            interpretation,
            body_style
        )
    )

    # ---------------------------------------------------------
    # LIMITATIONS
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Limitations",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "Predictions may be affected by image quality, "
            "dataset limitations, acquisition conditions, "
            "and differences between training and real-world "
            "images. The automated fundus validation is a "
            "preliminary image-based heuristic and is not a "
            "clinically validated fundus detector.",
            body_style
        )
    )

    # ---------------------------------------------------------
    # DISCLAIMER
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Medical Disclaimer",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "<b>This report is generated by an AI-assisted "
            "screening prototype and is not a medical diagnosis. "
            "Results should be reviewed by a qualified healthcare "
            "professional.</b>",
            body_style
        )
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "Generated by Retina AI Screening",
            small_style
        )
    )

    document.build(
        story
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()

st.set_page_config(
    page_title="Retina AI Screening",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)
if "screening_history" not in st.session_state:

    st.session_state.screening_history = []


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
                "Checking image suitability..."
            ):

                validation = validate_image(
                    image
                )

            st.write("**Image validation**")

            quality_column, fundus_column = st.columns(
                2,
                gap="large"
            )

            with quality_column:

                if validation["quality_passed"]:

                    st.success(
                        "Image quality: Suitable"
                    )

                else:

                    st.error(
                        "Image quality: Not suitable"
                    )

            with fundus_column:

                if validation["fundus_passed"]:

                    st.success(
                        "Fundus suitability: Suitable"
                    )

                else:

                    st.error(
                        "Fundus suitability: Not suitable"
                    )

            if not validation["passed"]:

                st.warning(
                    "This image did not pass the "
                    "automated suitability checks. "
                    "Please upload a clear retinal fundus "
                    "photograph for screening."
                )

                st.info(
                    "The validation system is a preliminary "
                    "technical check and does not guarantee "
                    "that an image is clinically suitable."
                )

                st.stop()

            st.success(
                "Image passed the preliminary suitability checks. "
                "Proceeding with AI screening."
            )

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
            screening_time = datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            )

            st.session_state.screening_history.insert(
                0,
                {
                    "time": screening_time,
                    "prediction": prediction,
                    "confidence": confidence
                }
            )
            # Improved prediction result

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

            # Severity scale

            st.write("**Predicted severity stage**")

            severity_display = [
                "No DR",
                "Mild",
                "Moderate",
                "Severe",
                "Proliferative DR"
            ]

            severity_text = "  →  ".join(
                [
                    f"**{stage}**"
                    if stage == prediction
                    else stage
                    for stage in severity_display
                ]
            )

            st.markdown(
                severity_text
            )

            # Screening interpretation

            st.write("**Screening interpretation**")

            if prediction == "No DR":

                st.info(
                    "The model classified this image in "
                    "the No DR category. This result does "
                    "not replace a professional eye examination."
                )

            else:

                st.info(
                    "The model detected a diabetic retinopathy "
                    "category in this retinal image. Professional "
                    "evaluation by a qualified eye-care provider "
                    "is recommended."
                )

    if analyze_button:

        st.divider()

        st.subheader("Prediction probabilities")

        st.write(
            "The chart shows the probability assigned by "
            "the model to each of the five classification categories."
        )

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

            display_probabilities = probabilities.copy()

            display_probabilities[
                "Class"
            ] = display_probabilities[
                "Class"
            ].apply(
                lambda x:
                f"{x}  ← Predicted"
                if x == prediction
                else x
            )

            st.dataframe(
                display_probabilities,
                use_container_width=True,
                hide_index=True
            )


        st.divider()

        st.subheader("Explainable AI")

        st.write(
            "Why did the model make this prediction?"
        )

        st.write(
            "Grad-CAM highlights regions of the retinal "
            "image that contributed most strongly to the "
            "model's prediction."
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
            "Grad-CAM represents model attention and should "
            "not be interpreted as a clinical diagnosis or "
            "as a precise map of disease."
        )
        st.divider()

        st.subheader("Screening report")

        st.write(
            "Generate a professional PDF report containing "
            "the screening result, prediction probabilities, "
            "original retinal image, and Grad-CAM explanation."
        )

        pdf_report = create_pdf_report(
            image=image,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            heatmap=heatmap,
            validation=validation
        )

        st.download_button(
            label="Download PDF screening report",
            data=pdf_report,
            file_name="retina_ai_screening_report.pdf",
            mime="application/pdf",
            use_container_width=True
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
st.divider()

st.subheader("Session screening history")

if len(st.session_state.screening_history) == 0:

    st.write(
        "No completed screenings in this session."
    )

else:

    history_rows = []

    for index, record in enumerate(
        st.session_state.screening_history,
        start=1
    ):

        history_rows.append(
            {
                "Screening": f"#{index}",
                "Date & Time": record["time"],
                "Predicted Stage": record["prediction"],
                "Confidence": (
                    f"{record['confidence']:.2f}%"
                )
            }
        )

    history_dataframe = pd.DataFrame(
        history_rows
    )

    st.dataframe(
        history_dataframe,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "Clear session history"
    ):

        st.session_state.screening_history = []

        st.rerun()