# Detectai
AI Generated vs Real Image Detector
# DetectAI 🔍
### AI Generated vs Real Image Detection System

A deep learning based system that detects whether an image is **Real** or **AI Generated** using Transfer Learning with EfficientNet-B0.

---

## 🚀 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/Gouri0590/ai-image-detector)

---

## 📊 Results
| Metric | Score |
|--------|-------|
| Accuracy | **99%** |
| AUC-ROC | **0.9991** |
| F1-Score | **0.99** |

---

## 🛠️ Technologies Used
- Python 3
- PyTorch & TorchVision
- EfficientNet-B0 (via TIMM)
- Gradio
- Google Colab (T4 GPU)
- CIFAKE Dataset (Kaggle)
- Hugging Face Spaces

---

## 📂 Dataset
**CIFAKE** — 120,000 images
- 60,000 Real images (from CIFAR-10)
- 60,000 AI Generated images (Stable Diffusion)

---

## 🧠 How It Works
1. Input image is resized to 224×224
2. EfficientNet-B0 extracts visual features
3. Custom classifier head predicts Real or Fake
4. Confidence score is displayed

```
Input Image → Preprocessing → EfficientNet-B0 → Classifier → Real / Fake
```

---

## 📁 Files
| File | Description |
|------|-------------|
| `app.py` | Gradio web interface |
| `best_model.pth` | Trained model weights |
| `requirements.txt` | Required libraries |

---

## ⚙️ Run Locally
```bash
pip install -r requirements.txt
python app.py
```

---

