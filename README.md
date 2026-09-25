# Retina AI Screening

### Explainable AI for Diabetic Retinopathy Screening in Rural India

Retina AI Screening is an explainable deep-learning application designed to assist with the screening of diabetic retinopathy from retinal fundus images.

The system uses a fine-tuned **EfficientNet-B0** model to classify retinal images into five diabetic retinopathy severity levels and uses **Grad-CAM** to provide a visual explanation of the regions that influenced the model's prediction.

The application is built as a lightweight web interface using **Streamlit** and is deployed publicly through **Streamlit Community Cloud**.

**Live Demo:**
https://retina-ai-screening.streamlit.app/

---

## Overview

Diabetic retinopathy is a diabetes-related eye disease that can lead to vision loss when not detected and managed appropriately.

Retinal screening can help identify potential signs of diabetic retinopathy at an earlier stage. This project explores how deep learning and explainable AI can support such screening workflows, particularly in settings where access to specialized screening resources may be limited.

The system accepts a retinal fundus image and produces:

- Predicted diabetic retinopathy severity
- Prediction confidence
- Probability distribution across all five classes
- Grad-CAM visualization highlighting image regions that contributed to the prediction

The system is intended as an **AI-assisted screening prototype**, not as a replacement for professional medical diagnosis.

---

## Key Features

### Five-Class Classification

The model classifies retinal fundus images into:

| Class            | Description                        |
| ---------------- | ---------------------------------- |
| No DR            | No visible diabetic retinopathy    |
| Mild             | Mild diabetic retinopathy          |
| Moderate         | Moderate diabetic retinopathy      |
| Severe           | Severe diabetic retinopathy        |
| Proliferative DR | Proliferative diabetic retinopathy |

### Explainable AI

Grad-CAM is used to generate a visual heatmap showing the regions of the retinal image that contributed to the model's prediction.

This provides an additional layer of interpretability instead of presenting only a classification result.

### Confidence Analysis

The application displays the model's probability distribution across all five classes, allowing users to see how strongly the model distinguishes between possible categories.

### Web-Based Interface

The application provides a simple browser-based interface for uploading retinal fundus images and viewing the screening result.

### Public Deployment

The application is deployed using Streamlit Community Cloud and can be accessed through a web browser without requiring users to install Python, PyTorch, or the project locally.

---

## System Architecture

```text
Retinal Fundus Image
        │
        ▼
Image Preprocessing
        │
        ├── Resize to 224 × 224
        ├── Convert to Tensor
        └── ImageNet Normalization
        │
        ▼
EfficientNet-B0
        │
        ▼
Five-Class Classification
        │
        ├── No DR
        ├── Mild
        ├── Moderate
        ├── Severe
        └── Proliferative DR
        │
        ├───────────────────┐
        ▼                   ▼
Prediction             Grad-CAM
Confidence             Explanation
        │                   │
        └─────────┬─────────┘
                  ▼
          Streamlit Interface
```

---

## Model

The application uses **EfficientNet-B0** as its classification architecture.

```text
Architecture: EfficientNet-B0
Input Size: 224 × 224
Number of Classes: 5
Framework: PyTorch
Model Library: Torchvision / TIMM ecosystem
Inference Device: CPU
```

The trained model is loaded from:

```text
best_retinopathy_model.pth
```

A backup copy is also included:

```text
best_retinopathy_model_backup.pth
```

---

## Preprocessing

Input images are converted to RGB and resized to:

```text
224 × 224 pixels
```

ImageNet normalization is applied before inference:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## Explainability

The application uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize regions that contributed to the predicted class.

The Grad-CAM target layer is:

```text
model.conv_head
```

The resulting activation map is overlaid on the original retinal image to provide a visual explanation of the model's prediction.

Grad-CAM should be interpreted as an explanation of model behavior rather than as a medically validated indication of disease location.

---

## Technology Stack

| Component            | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| Deep Learning        | PyTorch                   |
| Model Architecture   | EfficientNet-B0           |
| Model Library        | TIMM                      |
| Image Processing     | Pillow                    |
| Computer Vision      | Torchvision               |
| Explainability       | Grad-CAM                  |
| Data Handling        | NumPy, Pandas             |
| Web Interface        | Streamlit                 |
| Version Control      | Git + GitHub              |
| Deployment           | Streamlit Community Cloud |

---

## Project Structure

```text
retina-ai-screening/
│
├── streamlit_app.py
├── requirements.txt
├── .gitignore
│
├── best_retinopathy_model.pth
└── best_retinopathy_model_backup.pth
```

---

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/krupa-21/retina-ai-screening.git
cd retina-ai-screening
```

### 2. Create a Virtual Environment

Windows:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the Application

```powershell
streamlit run streamlit_app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## Requirements

The project requires:

- Python 3.12
- PyTorch
- Torchvision
- TIMM
- Grad-CAM
- Streamlit
- NumPy
- Pandas
- Pillow

The complete dependency list is available in:

```text
requirements.txt
```

---

## Deployment

The application is deployed using Streamlit Community Cloud.

### Deployment Flow

```text
Local Development
       │
       ▼
      Git
       │
       ▼
    GitHub
       │
       ▼
Streamlit Community Cloud
       │
       ▼
   Public Web App
```

Any future changes can be made locally and pushed to the GitHub repository. Streamlit Community Cloud can then rebuild the application from the updated repository.

### Live Application

**https://retina-ai-screening.streamlit.app/**

---

## Intended Use

This project is intended as a research, educational, and demonstration prototype exploring the use of deep learning and explainable AI for diabetic retinopathy screening.

It is particularly designed to demonstrate how an AI-assisted screening workflow could be presented through an accessible web interface.

The model's predictions should not be used independently to make clinical decisions.

---

## Limitations

The current prototype has several limitations:

- Model performance depends on the quality and characteristics of the training dataset.
- Predictions may not generalize equally well across different cameras, populations, imaging conditions, or clinical environments.
- Grad-CAM visualizations indicate model attention and should not be interpreted as definitive medical evidence.
- The application has not been presented as a clinically validated diagnostic system.
- A qualified healthcare professional should review screening results before any medical decision is made.

---

## Medical Disclaimer

This application is an **AI-assisted screening prototype and not a medical diagnostic tool**.

The predictions, confidence values, and Grad-CAM visualizations are provided for research, educational, and demonstration purposes. They should not be used as a substitute for examination, diagnosis, or treatment by a qualified healthcare professional.

---

## Author

**Krupa Patel**

Computer Engineering
Government Polytechnic Ahmedabad

GitHub:
https://github.com/krupa-21

---

## License

This project is currently provided for educational and research purposes.
