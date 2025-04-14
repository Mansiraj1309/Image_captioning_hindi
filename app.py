from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from PIL import Image
import torch
from torchvision import transforms
from model_utils import load_model, generate_caption_for_image
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# TextVocabulary class (unchanged)
class TextVocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {k: j for j, k in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in sentence.lower().split():
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = text.lower().split()

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'image_captioning_model.pth'
vocab_path = 'vocab.pkl'

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
            # Return the URL that the frontend can use to access the image
            image_url = f'/uploads/{file.filename}'
            return jsonify({'caption': caption_text, 'image_path': image_url})
        except Exception as e:
            return jsonify({'error': f'Error generating caption: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
