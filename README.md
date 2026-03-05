# CardioNet: Deep Learning for Cardiovascular Disease Detection from 12-Lead ECG Signals

**Hack4Health AI Challenge -- Lower Division Submission**

---

## Overview

Cardiovascular disease (CVD) is the leading cause of death worldwide, claiming approximately 17.9 million lives annually. Early detection through ECG analysis is effective but requires trained cardiologists who are scarce in underserved communities.

**CardioNet** is a 1D Residual Network with Squeeze-and-Excitation attention that classifies 12-lead ECG recordings into normal vs. cardiovascular abnormality categories. Trained on the PTB-XL dataset (21,799 clinical ECGs from PhysioNet), CardioNet achieves:

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9444** |
| Accuracy | **85.84%** |
| F1 Score | **0.8598** |

The model includes **1D Grad-CAM explainability**, producing temporal attention heatmaps on ECG waveforms that show clinicians exactly which portions of the cardiac cycle drive each prediction.

---

## Repository Structure

```
CardioNet/
|-- CardioNet_Hack4Health.ipynb    # Full reproducible notebook (Colab/Jupyter)
|-- CardioNet_Report.pdf           # 3-page project report
|-- CardioNet_Presentation.pptx    # Presentation deck (9 slides)
|-- requirements.txt               # Python dependencies
|-- README.md                      # This file
```

**Trained Model Weights:** [cardionet_final.pth (Google Drive)](https://drive.google.com/file/d/1J9G210-If_adQiDnb3B-ODv6Cxm4wQQ2/view?usp=sharing) (74.7 MB)

---

## Quick Start

### 1. Open in Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload `CardioNet_Hack4Health.ipynb` to Google Colab
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Run all cells (Runtime > Run all)

### 2. Local Setup

```bash
git clone https://github.com/brandonfwh/CardioNet.git
cd CardioNet
pip install -r requirements.txt
jupyter notebook CardioNet_Hack4Health.ipynb
```

---

## Dataset

We use the **PTB-XL** dataset from PhysioNet:

- 21,799 twelve-lead ECG recordings from 18,869 patients
- 10-second duration each, sampled at 100 Hz and 500 Hz
- Clinical diagnostic labels (SCP-ECG standard)
- 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP
- Pre-defined stratified folds for reproducible evaluation

The dataset is downloaded automatically in the notebook via the Kaggle API.

**Citation:**
> Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7(1), 154.

---

## Model Architecture

CardioNet is a **1D ResNet with Squeeze-and-Excitation (SE) channel attention**:

```
Input (12, 1000) -> Stem Conv (k=25, s=2) -> 4 Residual Stages (64/128/256/512)
    -> SE Attention -> Global Avg Pool -> FC Classifier -> Normal vs CVD
```

Key components:
- **1D Residual blocks** with skip connections for stable deep training
- **SE blocks** that learn to weight the importance of each feature channel
- **Data augmentation**: Gaussian noise, amplitude scaling, temporal shifting
- **Training**: AdamW + cosine annealing + class-weighted loss + early stopping

Total parameters: 18.7M

---

## Explainability

We implement **1D Grad-CAM** adapted for time-series ECG data. The method produces heatmaps overlaid on ECG waveforms, highlighting which temporal regions (ST-segment, QRS complex, T-wave) most influenced the model's prediction.

This enables clinicians to verify the model's reasoning against known electrophysiological patterns before acting on predictions.

---

## Technologies

- **Python 3.10+**
- **PyTorch** (model architecture and training)
- **wfdb** (ECG data loading)
- **SciPy** (signal preprocessing)
- **scikit-learn** (evaluation metrics)
- **Matplotlib / Seaborn** (visualization)
- **Google Colab** (GPU compute)

---

## AI Disclosure

Generative AI tools (Claude, Anthropic) were used to accelerate code development, debugging, and formatting. All research direction, hypothesis formation, model design decisions, and analytical conclusions are original work by the team.

---

## License

This project is submitted as part of the Hack4Health AI Challenge. The PTB-XL dataset is available under the PhysioNet Credentialed Health Data License.
