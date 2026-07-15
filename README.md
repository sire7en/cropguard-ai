# CropGuard AI — Tomato Disease Detection

CropGuard AI is a deep learning web application that detects tomato leaf diseases from uploaded or captured leaf images and provides confidence scores, severity assessments, and treatment recommendations.
###### Built with Streamlit and deployed on Streamlit Community Cloud.

---

## What it does
Upload or photograph a tomato leaf. CropGuard classifies it into one of 10 categories: 9 diseases and a healthy control and returns predictions information.

## Features
- Detects 9 tomato diseases and healthy leaves
- Confidence score for every prediction
- Severity assessment
- Treatment recommendations
- Downloadable PDF report
- Session history
- Responsive Streamlit interface

---

## Model
| Detail | Value |
|---|---|
| Architecture | ResNet50 (Transfer Learning) |
| Pre-trained on | ImageNet (1.2M images) |
| Training images | 14,535 |
| Validation images | 3,631 |
| Overall Accuracy | 94% |
| Macro F1 Score | 0.93 |
| Classes | 10 tomato disease categories |

The ResNet50 base is frozen. Only the custom classification head (531K parameters) was trained on the PlantVillage tomato subset.

---

## Disease Classes

| Class | Severity |
|---|---|
| Bacterial spot | High |
| Early blight | Medium |
| Late blight | High |
| Leaf Mold | Medium |
| Septoria leaf spot | Medium |
| Spider mites | Medium |
| Target Spot | Medium |
| Yellow Leaf Curl Virus | High |
| Mosaic virus | High |
| Healthy | None |

---

## Dataset

**PlantVillage Dataset**
Mendeley Data — DOI: [10.17632/tywbtsjrjv.1](https://data.mendeley.com/datasets/tywbtsjrjv/1)

18,166 tomato leaf images across 10 classes. A 14:1 Class imbalance was handled using class weighting during training.

---

## Tech Stack

- **Model:** TensorFlow · Keras · ResNet50 · Scikit-learn
- **App:** Streamlit · Plotly · Pandas · fpdf2
- **Dev:** Python 3.11 · Google Colab T4 GPU · Github · Git

---

## Run Locally

#### Clone the repository
- git clone https://github.com/sire7en/cropguard-ai.git
- cd cropguard

#### Create and activate environment
- conda create -n cropguard python=3.11
- conda activate cropguard

#### Install dependencies
- pip install -r requirements.txt

#### Run the app
- streamlit run app.py

---
## Live demo
#### Try it here: https://cropguard-api.streamlit.app/

---

## Known Limitations

- Trained on lab-condition PlantVillage images. Performance on real field photos may vary.
- Softmax classifier will return a prediction for any image including non-tomato inputs.
- A binary pre-classifier (Tomato vs Non-Tomato) is planned for the next version.
- Session history is in-memory only and resets on page refresh.

---

## Roadmap

- [ ] Binary pre-classifier to reject non-tomato inputs
- [ ] TensorFlow Lite conversion for offline edge deployment
- [ ] Expand to 5 Ghana Feed Programme crops (maize, pepper, cassava, cocoa)
- [ ] Real Ghanaian field-condition test set
- [ ] Persistent database backend for longitudinal disease tracking

---
## Disclaimer

CropGuard AI is intended as a decision-support tool and should not replace advice from qualified agricultural extension officers. Predictions should be verified before treatment decisions are made.