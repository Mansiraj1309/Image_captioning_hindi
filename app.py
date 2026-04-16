from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from PIL import Image
import torch
from torchvision import transforms
from model_utils import load_model, generate_caption_for_image
from gtts import gTTS

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_path = 'image_captioning_model.pth'


try:
    model, _, vocab, _ = load_model(model_path, device)
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Image transformation
inception_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        try:
            caption, _, _ = generate_caption_for_image(filepath, model, vocab, inception_transform, device)
            caption_text = ' '.join(caption)
            # Generate audio for the caption
            audio_filename = f"caption_{os.path.splitext(file.filename)[0]}.mp3"
            audio_filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
            tts = gTTS(text=caption_text, lang='hi')  # 'hi' for Hindi
            if caption is None:
                return jsonify({'error': 'Could not generate caption for the image'}), 500
            tts.save(audio_filepath)
            # Return URLs for image and audio
            image_url = f'/uploads/{file.filename}'
            audio_url = f'/uploads/{audio_filename}'
            return jsonify({
                'caption': caption_text,
                'image_path': image_url,
                'audio_path': audio_url
            })
        except Exception as e:
            return jsonify({'error': f'Error generating caption or audio: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005, use_reloader=False)