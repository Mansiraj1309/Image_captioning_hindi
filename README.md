# 🖼️ Cozy Image Captioning in Hindi (Glassmorphism Edition)

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-black.svg)

A seamless, beautiful web application powered by deep learning that generates rich **Hindi captions** and **Audio (Text-to-Speech)** for any uploaded image.

## ✨ Features
- **Modern Glassmorphism UI**: A stunning, premium frontend with animated fluid gradients, crafted for an elegant user experience.
- **Interactive Drag & Drop**: Easily drag your images directly onto the upload zone.
- **AI-Powered Hindi Captioning**: Automatically generates highly accurate Hindi sentences describing the image using an Encoder-Decoder neural network.
- **Audio Playback**: Automatically speaks the generated Hindi caption out loud using Google Text-to-Speech (gTTS).
- **Magical Loading States**: Animated UI feedback while the PyTorch model runs inference in the background.

## 📦 Required Model Files
Before running the app, you MUST download the model and vocabulary binaries and place them in the root directory:
- 🧠 [image_captioning_model.pth](https://drive.google.com/file/d/1txKnKwwB9beunMP3t1w3dQm85sDrvzoP/view?usp=sharing)
- 📚 [vocab.pkl](https://drive.google.com/file/d/1XssaSKX-tbnNpIHIrgOdHXLT2zcJQB5s/view?usp=sharing)

## 🛠️ Installation & Setup

Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mansiraj1309/Image_captioning_hindi.git
   cd Image_captioning_hindi
   ```
2. **Create a Virtual Environment**:
   ```bash
   conda create -n image-captioning python=3.12
   conda activate image-captioning
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application**:
   ```bash
   python app.py
   ```

## 🚀 Usage

1. Open your browser and navigate to **[http://localhost:5005](http://localhost:5005)**.
2. Drag and drop an image or click "Browse" to select one (Supports PNG, JPG, JPEG).
3. Click on **"कैप्शन प्राप्त करें" (Get Caption)**.
4. Wait a few magical seconds for the model to "weave the story".
5. View your generated Hindi caption alongside the generated audio player!

---
> *Example Output: For an image of a family gathering, the model might output and speak “परिवार एक साथ भोजन कर रहा है” (A family is eating together).*
