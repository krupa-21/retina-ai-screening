# Retina AI Screening

### Explainable AI for Diabetic Retinopathy Screening

Retina AI Screening is an AI-assisted web application for screening diabetic retinopathy from retinal fundus images.

The system uses a fine-tuned **EfficientNet-B0** model to classify retinal images into five diabetic retinopathy severity categories and uses **Grad-CAM** to provide a visual explanation of the regions that influenced the model's prediction.

The application also performs preliminary image quality and fundus suitability checks before analysis, generates professional PDF screening reports, and maintains screening history during the current session.

**Live Application:**
https://retina-ai-screening.streamlit.app/

**GitHub Repository:**
https://github.com/krupa-21/retina-ai-screening

---

## Overview

Diabetic retinopathy is a diabetes-related eye disease that can lead to vision loss when not detected and managed appropriately.

This project explores how deep learning and explainable AI can support an accessible retinal screening workflow.

A user can upload a retinal fundus image and receive:

- Preliminary image quality assessment
- Fundus image suitability validation
- Predicted diabetic retinopathy stage
- Model confidence
- Probability distribution across all five classes
- Grad-CAM visualization
- Professional PDF screening report
- Session-based screening history

The application is an **AI-assisted screening prototype** and is not a replacement for professional medical diagnosis.

---

## Key Features

### Fundus Image Validation

Before model inference, the application performs preliminary checks for:

- Image resolution
- Brightness
- Contrast
- Sharpness
- Retinal color characteristics
- Circular retinal field structure
- Image composition and structural characteristics

Images that do not pass the validation stage are prevented from proceeding to model inference.

> The validation system is a heuristic screening layer and is not a clinically validated fundus detector.

### Five-Class Classification

The model classifies retinal fundus images into five categories:

| Class            | Description                                   |
| ---------------- | --------------------------------------------- |
| No DR            | No diabetic retinopathy detected by the model |
| Mild             | Mild diabetic retinopathy                     |
| Moderate         | Moderate diabetic retinopathy                 |
| Severe           | Severe diabetic retinopathy                   |
| Proliferative DR | Proliferative diabetic retinopathy            |

### Explainable AI

The application uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate a heatmap showing regions of the retinal image that contributed to the model's prediction.

Grad-CAM provides an interpretation of model behavior and should not be considered a medically validated map of disease.

### Confidence Analysis

The application displays:

- Predicted class
- Model confidence
- Probability distribution across all five classes

This allows users to inspect how the model distributes its prediction across the possible categories.

### Professional PDF Reports

A screening report can be generated containing:

- Screening date and identifier
- Image validation results
- Original retinal image
- Predicted stage
- Model confidence
- Class probabilities
- Grad-CAM explanation
- Screening interpretation
- Limitations
- Medical disclaimer

### Session Screening History

Completed screenings are recorded during the current browser session, allowing users to review previous predictions and confidence values.

No persistent patient database is used by the application.

### Web-Based Interface

The application provides a browser-based Streamlit interface without requiring users to install the project when using the deployed version.

### Public Deployment

The application is deployed through **Streamlit Community Cloud** and is accessible through a web browser.

---

## System Workflow

```text
                 Retinal Fundus Image
                          │
                          ▼
              Image Quality Validation
                          │
                          ▼
             Fundus Suitability Validation
                          │
                    ┌─────┴─────┐
                    │           │
                 Failed       Passed
                    │           │
                    ▼           ▼
              Stop Analysis   Preprocessing
                                │
                                ▼
                         EfficientNet-B0
                                │
                                ▼
                       Five-Class Prediction
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
             Prediction                 Grad-CAM
             Confidence                Explanation
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                       Streamlit Interface
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
              Screening       PDF Report    Session
               Result                       History
```

---

## Model

The application uses **EfficientNet-B0** for five-class retinal image classification.

```text
Architecture: EfficientNet-B0
Input Size: 224 × 224
Number of Classes: 5
Framework: PyTorch
Model Library: TIMM
Inference Device: CPU
```

The trained model is stored as:

```text
best_retinopathy_model.pth
```

---

## Preprocessing

Input images are:

1. Converted to RGB
2. Resized to `224 × 224`
3. Converted to tensors
4. Normalized using ImageNet statistics

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## Explainability

Grad-CAM is used to visualize image regions that contributed to the predicted class.

The target layer used for Grad-CAM is:

```text
model.conv_head
```

The resulting activation map is overlaid on the retinal image and presented alongside the prediction.

Grad-CAM represents model attention and should not be interpreted as definitive medical evidence or as a precise map of disease.

---

## Technology Stack

| Component               | Technology                |
| ----------------------- | ------------------------- |
| Programming Language    | Python                    |
| Deep Learning Framework | PyTorch                   |
| Model Architecture      | EfficientNet-B0           |
| Model Library           | TIMM                      |
| Image Processing        | Pillow                    |
| Computer Vision         | OpenCV                    |
| Image Transforms        | Torchvision               |
| Explainability          | Grad-CAM                  |
| Data Handling           | NumPy, Pandas             |
| Web Interface           | Streamlit                 |
| PDF Generation          | ReportLab                 |
| Version Control         | Git + GitHub              |
| Deployment              | Streamlit Community Cloud |

---

## Project Structure

```text
retina-ai-screening/
│
├── streamlit_app.py
├── requirements.txt
├── .gitignore
├── README.md
│
└── best_retinopathy_model.pth
```

---

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/krupa-21/retina-ai-screening.git
cd retina-ai-screening
```

### 2. Create a Virtual Environment

For Python 3.12:

```powershell
py -3.12 -m venv .venv
```

Activate the environment:

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

The application will be available at:

```text
http://localhost:8501
```

---

## Requirements

The application requires Python 3.12 and the dependencies listed in:

```text
requirements.txt
```

Major dependencies include:

- Streamlit
- PyTorch
- Torchvision
- TIMM
- Grad-CAM
- NumPy
- Pandas
- Pillow
- OpenCV
- ReportLab

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

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
 Public Web Application
```

### Live Application

https://retina-ai-screening.streamlit.app/

---

## Intended Use

This project is intended for:

- Research
- Education
- AI experimentation
- Demonstration of explainable medical AI
- Exploration of AI-assisted retinal screening workflows

The system should not be used independently to make clinical decisions.

---

## Limitations

The current prototype has several limitations:

- Model performance depends on the characteristics and limitations of the training dataset.
- Predictions may not generalize equally across different cameras, populations, image acquisition conditions, or clinical environments.
- The fundus suitability check is a heuristic validation layer and is not clinically validated.
- Grad-CAM visualizations describe model attention and should not be interpreted as definitive medical evidence.
- The application has not been presented as a clinically validated diagnostic system.
- Screening results should be reviewed by a qualified healthcare professional before any medical decision is made.

---

## Medical Disclaimer

**Retina AI Screening is an AI-assisted screening prototype and not a medical diagnostic tool.**

Predictions, confidence values, image validation results, and Grad-CAM visualizations are provided for research, educational, and demonstration purposes.

They should not be used as a substitute for examination, diagnosis, or treatment by a qualified healthcare professional.

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
