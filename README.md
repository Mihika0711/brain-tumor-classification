# 🧠 Brain Tumor Classification Using Deep Learning (VGG19)

An AI-powered deep learning model for binary classification of brain tumors using MRI scan images. This project uses **transfer learning** with the **VGG19** architecture, achieving ~94% accuracy. Designed and trained on **Google Colab**, the model demonstrates effective image classification for medical imaging applications.

---

## ✨ Key Features

- 🧬 **Transfer Learning with VGG19**
  - Fine-tuned VGG19 CNN on brain MRI images
  - Feature extraction from pre-trained ImageNet layers
  - Binary classification: Tumor vs No Tumor

- 🧠 **Medical Image Preprocessing**
  - Grayscale-to-RGB conversion
  - Image normalization and resizing (224x224)
  - Data augmentation using rotation, zoom, flip, etc.

- 📊 **Model Evaluation & Visualization**
  - Accuracy and loss plots
  - Confusion matrix and classification report
  - Optional model saving as `.h5` file

---

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib
- Google Colab

---

## 📁 Files Included

- `brain_tumor_classification.ipynb` – Main Colab notebook (code, training, evaluation)
- Model and dataset not included due to file size restrictions

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Mount your **Google Drive** containing the dataset
3. Install dependencies if needed
4. Run all cells in sequence to train, evaluate, and visualize the model

---

## 🧠 Model Architecture

The model is built using VGG19 (pre-trained on ImageNet) with custom layers:

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg19 (Functional)           (None, 7, 7, 512)         20024384
global_average_pooling2d     (None, 512)               0
dense (Dense)                (None, 128)               65664
dropout (Dropout)            (None, 128)               0
dense_1 (Dense)              (None, 1)                 129
=================================================================
Total params: 20,090,177
Trainable params: 65,793
Non-trainable params: 20,024,384
```

> 🔍 *Only the custom top layers are trainable. The VGG19 base is frozen to leverage high-level feature extraction.*

---

## 📊 Model Performance

- **Accuracy**: ~94%
- **Loss**: Low, with stable convergence
- Visualized using training/validation plots

---

## 📷 Sample Outputs

> *(Optional – Upload your own screenshots and link them here)*

```markdown
### Accuracy/Loss Graphs
![Training Curve](accuracy_graph.png)

### Sample Predictions
![Predicted MRI](sample_output.png)
```

---

## 🔗 Dataset Source

- MRI image dataset categorized into:
  - **yes/** – Brain tumor present
  - **no/** – No tumor
- Dataset accessed from Google Drive and used directly in Colab

---

## ⚠️ Disclaimer

This project is intended **for academic purposes only** and should **not be used for real-world clinical diagnosis** or decision-making.

---

## 🏷️ Version

- **Version**: 1.0.0  
- **Last Updated**: July 2025  
  
