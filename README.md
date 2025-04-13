# Image_captioning_hindi
A Flask web app for generating Hindi captions using a deep learning model.
## Model Files
Download the following files and place them in the project root:
- [image_captioning_model.pth](https://drive.google.com/file/d/1txKnKwwB9beunMP3t1w3dQm85sDrvzoP/view?usp=sharing)
- [vocab.pkl](https://drive.google.com/file/d/1XssaSKX-tbnNpIHIrgOdHXLT2zcJQB5s/view?usp=sharing)
## Features
- Upload images (PNG, JPG, JPEG) through a user-friendly web interface.
- Generate Hindi captions automatically using a trained neural network.
- Supports real-time captioning with a responsive design.

## Prerequisites
- Python 3.12 or higher
- Conda (for environment management)
- Access to the model files hosted on Google Drive

## Installation
Follow these steps to set up and run the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mansiraj1309/Image_captioning_hindi.git
   cd Image_captioning_hindi
2. **Run**
-conda create -n image-captioning python=3.12
-conda activate image-captioning
3. **Run**
-pip install -r requirements.txt
4. **Download model files mentioned above**
5. **Run the application using** 
-python app.py

## Usage
Web Interface:
Access http://127.0.0.1:5001 after running app.py.
Click “Choose File” to select an image.
Click “Generate Caption” to view the Hindi caption.
Supported Formats: PNG, JPG, JPEG.
Example Output: For an image of a family gathering, the model might output “परिवार एक साथ भोजन कर रहा है” (A family is eating together).
