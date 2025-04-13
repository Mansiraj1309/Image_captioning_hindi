import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pickle
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# Define TextVocabulary (from cseminor.py)
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

# Define InceptionFeatureExtractor (from cseminor.py)
class InceptionFeatureExtractor(nn.Module):
    def __init__(self, inception, transform_input=False):
        super(InceptionFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x

# Define Encoder (from cseminor.py)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)  # Updated line
        inception.eval()
        for param in inception.parameters():
            param.requires_grad_(False)
        self.feature_extractor = InceptionFeatureExtractor(inception)

    def forward(self, images):
        features = self.feature_extractor(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        return features


# Define Attention (from cseminor.py)
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        if encoder_out.dim() != 3:
            encoder_out = encoder_out.unsqueeze(0)
        if decoder_hidden.dim() != 2:
            decoder_hidden = decoder_hidden.unsqueeze(0)
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        alpha = self.softmax(self.full_att(self.relu(att1 + att2)).squeeze(2))
        weighted = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return alpha, weighted

# Define Decoder (from cseminor.py)

class Decoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, attention_dim, encoder_dim, decoder_dim, dropout=0.5):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def generate_caption_beam_search(self, features, vocab, beam_size=3, max_len=20):
        batch_size = features.size(0)
        encoder_dim = features.size(-1)
        features = features.view(batch_size, -1, encoder_dim)
        num_pixels = features.size(1)
        h, c = self.init_hidden_state(features)
        start_token = torch.tensor([vocab.stoi['<start>']]).to(features.device)
        beams = [([start_token.item()], 0.0, h, c)]
        complete_beams = []
        with torch.no_grad():
            for _ in range(max_len):
                new_beams = []
                for caption, score, h, c in beams:
                    if caption[-1] == vocab.stoi['<end>']:
                        complete_beams.append((caption, score))
                        continue
                    input_id = torch.tensor([caption[-1]]).to(features.device)
                    embeddings = self.embedding(input_id)
                    alpha, attention_weighted = self.attention(features, h)
                    gate = self.sigmoid(self.f_beta(h))
                    attention_weighted = gate * attention_weighted
                    lstm_input = torch.cat([embeddings, attention_weighted], dim=1)
                    h, c = self.lstm_cell(lstm_input, (h, c))
                    output = self.fc(self.drop(h))
                    log_probs = torch.log_softmax(output, dim=-1)
                    top_log_probs, top_indices = log_probs.topk(beam_size, dim=-1)
                    for log_prob, word_idx in zip(top_log_probs.squeeze(0), top_indices.squeeze(0)):
                        new_caption = caption + [word_idx.item()]
                        new_score = score + log_prob.item()
                        new_beams.append((new_caption, new_score, h.clone(), c.clone()))
                new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                beams = new_beams
                if not beams:
                    break
        if complete_beams:
            best_caption, best_score = max(complete_beams, key=lambda x: x[1])
        else:
            best_caption, best_score = beams[0][:2]
        caption_text = [vocab.itos[idx] for idx in best_caption]
        alphas = []
        return caption_text, alphas

# Define EncoderDecoder (from cseminor.py)
class EncoderDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, attention_dim, encoder_dim, decoder_dim, dropout=0.5):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(embed_dim, vocab_size, attention_dim, encoder_dim, decoder_dim, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

def load_model(file_path, device):
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    params = checkpoint['model_params']
    model = EncoderDecoder(
        embed_dim=params['embed_dim'],
        vocab_size=params['vocab_size'],
        attention_dim=params['attention_dim'],
        encoder_dim=params['encoder_dim'],
        decoder_dim=params['decoder_dim'],
        dropout=params['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    vocab = checkpoint['vocab']
    epoch = checkpoint['epoch']
    return model, optimizer, vocab, epoch
# ... (previous imports and classes remain unchanged)

def generate_caption_for_image(image_path, model, vocab, transform, device, use_beam=True, beam_size=3):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None, None, None
    image_tensor = transform(image).unsqueeze(0).to(device)  # Use the passed device
    model.eval()
    with torch.no_grad():
        features = model.encoder(image_tensor)
        if use_beam:
            caption, alphas = model.decoder.generate_caption_beam_search(features, vocab, beam_size=beam_size)
        else:
            caption, alphas = model.decoder.generate_caption(features, vocab)
        clean_caption = [w for w in caption if w not in ['<start>', '<end>', '<pad>']]
    return clean_caption, alphas, image

# ... (rest of the file remains unchanged)
