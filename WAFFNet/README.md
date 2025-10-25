# WAFFNet++

## Environment Setup

Activate your conda or virtual environment with **Python 3.12** (or compatible) and install dependencies:

```bash
pip install -r requirements.txt
```

---

## WAFFNet++ Model

The **WAFFNet++** model for Script Identification trained on the augmented version of **BSTD (12-language dataset)** can be downloaded from:

[Download Model Weights (Google Drive)](https://drive.google.com/file/d/1McEiKujTfxCBfwhAsnTG1P2uJw8OaDbj/view?usp=sharing)
[Download Training Data (Google Drvie)](https://drive.google.com/uc?id=1CxAveZr7HEp5lB4PSJt_yL9lTT9BcX1U)

---

## Supported Languages

The model is trained to classify the following 12 Indic scripts:

**Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu**

---

## Training

Specify dataset paths and hyperparameters in `config.py`.

To begin training:

```bash
python train.py
```

During training, checkpoints and logs will be saved automatically to the directories specified in the configuration file (`MODEL_SAVE_PATH`, `LOG_DIR`, etc.).

### Training uses:
- Focal Loss with `gamma=2.0`
- Cosine Annealing learning rate scheduler
- Early stopping based on validation loss
- Mixed-precision training (`torch.amp`)

---

## Evaluation

To evaluate a trained model on the test dataset:

```bash
python test.py
```

This will compute:
- Class-wise accuracy
- Confusion matrix
- Precision, recall, and F1-score

All results are saved inside the `results/` folder.

---

## Inference

For single image or folder-based inference:

```bash
python infer.py --data_path <path_to_image_or_folder> --weights_path weights/waffnet_best_cbam.pth
```

Or run the **FastAPI-based inference web app**:

```bash
uvicorn app.main:app --reload
```

Then open:

```
http://127.0.0.1:8000/
```

Upload an image and view the predicted script directly on the UI.

---

## Project Structure

```
WAFFNet/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── inference.py         # Image prediction logic
│   ├── model_loader.py      # Model loading utilities
│   └── templates/
│       └── upload.html      # Web UI template
├── models/
│   └── waffnet.py
├── utils/
│   ├── dataset.py
│   ├── transforms.py
│   ├── losses.py
│   └── evaluations.py
├── weights/
│   └── waffnet_best_cbam.pth
├── train.py
├── test.py
├── infer.py
├── config.py
└── requirements.txt
```

---

## Acknowledgements

This implementation is part of the **Script Identification** project and extends previous work on CNN-based and ViT-based architectures for multilingual script classification.
