# Face Mask Detection Using Deep Learning

A Face Mask Detection system built using **PyTorch**, **OpenCV**, and **Streamlit**. The model is trained from scratch using a custom Convolutional Neural Network (CNN) to classify whether a person is wearing a face mask or not.

---

## 📌 Project Overview

This project performs:

* Face Mask Classification using a custom CNN
* Face Detection using OpenCV Haar Cascades
* Image-based inference on uploaded images
* Interactive web application using Streamlit
* Model training, evaluation, and deployment

The system can classify:

* ✅ With Mask
* ⚠️ Without Mask

---

## 🏗️ Project Structure

```text
IWMI_Assessment/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│   └── best_model.pth
│
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│
├── src/
│   ├── BasicPreprocessing.py
│   ├── FaceMaskCNN.py
│   ├── FaceMaskTrainer.py
│   └── FaceMaskInference.py
│
├── requirements.txt
├── runtime.txt
└── README.md
```

> Note: `data/` contains the raw dataset organized by class (`with_mask/`, `without_mask/`). The 70/15/15 train/val/test split is performed in code (`BasicPreprocessing.py`) using a fixed random seed, not pre-split folders. and the `data/` file is not on the github repository

---

## 📊 Dataset

7553 images, nearly perfectly balanced:

| Class | Count | % |
|---|---|---|
| with_mask | 3725 | 49.32% |
| without_mask | 3828 | 50.68% |

Split 70/15/15 (seed=42): Train 5287 / Val 1132 / Test 1134
Batch size 32 → 166 / 36 / 36 batches.

Because the classes are nearly balanced, no class weighting or weighted loss was needed for `CrossEntropyLoss`.

---

## 🧠 Model Architecture

The model is a custom CNN built from scratch (no pretrained backbone).

### Feature Extractor

| Layer | Output Channels | Output Shape |
| --- | --- | --- |
| Conv2D + BatchNorm + ReLU | 32 | 32 × 128 × 128 |
| MaxPool2D | | 32 × 64 × 64 |
| Conv2D + BatchNorm + ReLU | 64 | 64 × 64 × 64 |
| MaxPool2D | | 64 × 32 × 32 |
| Conv2D + BatchNorm + ReLU | 128 | 128 × 32 × 32 |
| MaxPool2D | | 128 × 16 × 16 |

### Classifier

```text
Flatten
Linear(128 × 16 × 16 → 256)
ReLU
Dropout(0.5)
Linear(256 → 2)
```

### Input

```text
128 × 128 RGB image
```

### Output Classes

```text
with_mask
without_mask
```

### Key Design Decisions

* `padding=1` on all convolutions keeps spatial size constant within a block; only `MaxPool2D(2)` downsamples.
* `BatchNorm2D` placed after `Conv2D`, before `ReLU` — stabilizes training and allows a higher learning rate.
* `Dropout(0.5)` is applied only in the classifier, active during training and disabled by `model.eval()`.
* The model outputs raw logits (no softmax) — `CrossEntropyLoss` expects logits; softmax is applied at inference time.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd IWMI_Assessment
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model

Before running inference or the Streamlit application, train the model to generate the checkpoint file.

### Run Training

From the project root directory:

```bash
python src/FaceMaskTrainer.py
```

During training:

* `BasicPreprocessing` loads the dataset, builds transforms, splits into train/val/test, and creates DataLoaders
* Training loss and accuracy are displayed each epoch
* Validation loss/accuracy are monitored, and the learning rate is reduced via `ReduceLROnPlateau` if validation loss plateaus
* The best model (by validation accuracy) is saved automatically

### Training Outputs

```text
models/best_model.pth
results/training_curves.png
results/confusion_matrix.png
```

---

## 🌐 Running the Streamlit Application

After training has completed and `models/best_model.pth` exists:

### Start Streamlit

From the project root directory:

```bash
streamlit run app/streamlit_app.py
```

The application will launch in your browser.

### Features

#### Classify Whole Image

* Upload an image
* Predict mask status
* View confidence scores
* Visualize class probabilities as a bar chart

#### Detect Faces

* Detect faces using Haar Cascades
* Classify each detected face individually
* Display annotated image with predictions (green = with_mask, red = without_mask)

#### Sidebar

* Architecture summary
* Best validation accuracy (read from the saved checkpoint)

---

## 📊 Evaluation Metrics

The model is evaluated using:

* Training & validation loss/accuracy curves (`results/training_curves.png`)
* Confusion matrix (`results/confusion_matrix.png`)
* Per-class precision / recall / F1
* Test accuracy

### Results (10-epoch, CPU)

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 1 | 84.2% | 90.0% |
| 10 | 96.2% | 97.0% |

**Test Accuracy: 95.8% (1087/1134)**

Confusion matrix:

| | Predicted: with_mask | Predicted: without_mask |
|---|---|---|
| **Actual: with_mask** | 538 | 7 |
| **Actual: without_mask** | 40 | 549 |

---

## 🛠️ Technologies Used

* Python 3.11
* PyTorch
* TorchVision
* Streamlit
* Matplotlib
* scikit-learn

---

## 📸 Example Workflow

1. Train the CNN model (`python train.py --data_path data --epochs 10`)
2. Save the best checkpoint to `models/best_model.pth`
3. Launch the Streamlit application (`streamlit run app/streamlit_app.py`)
4. Upload an image
5. Receive prediction and confidence score
6. View detected faces and annotations

---

## 🔮 Future Improvements

* Real-time webcam detection
* Transfer Learning (ResNet, MobileNet)
* YOLO-based face detection
* Multi-face batch processing
* Docker deployment

---

## 👨‍💻 Author

Developed as part of a Deep Learning / Computer Vision internship assessment project.

**Technologies**: PyTorch, OpenCV, Streamlit