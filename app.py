from flask import Flask, request, render_template, jsonify
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

# Define TextVocabulary
class TextVocabulary:
    def __init__(self):
        self.itos = {0: "<PAD>", 1: "<start>", 2: "<end>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.min_freq = 1
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.token_counter = Counter()

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def numericalize(self, text):
        tokens_list = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokens_list
        ]

    def build_vocab(self, sentence_list):
        word_count = 4
        trainer = BpeTrainer(special_tokens=["<PAD>", "<start>", "<end>", "<UNK>"])
        self.tokenizer.train_from_iterator(sentence_list, trainer)
        for sentence in sentence_list:
            tokens = self.tokenizer.encode(sentence).tokens
            self.token_counter.update(tokens)
        for token, count in self.token_counter.items():
            if count >= self.min_freq and token not in self.stoi:
                self.stoi[token] = word_count
                self.itos[word_count] = token
                word_count += 1

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
            caption, _, _ = generate_caption_for_image(filepath, model, vocab, inception_transform, device)  # Pass device
            caption_text = ' '.join(caption)
            return jsonify({'caption': caption_text, 'image_path': filepath})
        except Exception as e:
            return jsonify({'error': f'Error generating caption: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Using port 5001 as per previous fix