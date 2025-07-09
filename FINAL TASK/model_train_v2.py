# THIS CODE DOES NOT WORK AS INTENDED. DON'T RUN IT. USE THE model_train.py INSTEAD. THE ISSUE IS IT DOES NOT LOAD THE PRE-TRAINED MODEL PROPERLY.ALSO IT USES CPU NOT GPU. IT SHOULD IDEALLY CONTINUE TRAINING FROM A PRE-TRAINED MODEL AND USE GPU IF AVAILABLE. 



import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import json
import pickle
from collections import Counter
import math
import numpy as np
from collections import defaultdict
import shutil

# Set seaborn style for fancy visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Download NLTK 'punkt' data for BLEU scoring
try:
    nltk.download('punkt')
except:
    pass

# Enhanced device setup
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA Available - Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device
    else:
        print("⚠ CUDA not available - Using CPU")
        print("Install CUDA-enabled PyTorch for GPU acceleration")
        return torch.device('cpu')

DEVICE = setup_device()

# Paths
TRAIN_JAVA_PATH = './FINAL TASK/data/train.java-cs.txt.java'
TRAIN_CS_PATH = './FINAL TASK/data/train.java-cs.txt.cs'
TRAIN_PAIRS_PATH = './FINAL TASK/data/train_java_cs_pairs.txt'

# Handle model paths and copying
old_model_dir = './FINAL TASK/data/saved_models_2/'
new_model_dir = './FINAL TASK/data/saved_models_3/'

# Create new directory
os.makedirs(new_model_dir, exist_ok=True)

# Copy existing model and vocab if available
old_model_path = os.path.join(old_model_dir, 'training_model.pth')
new_model_path = os.path.join(new_model_dir, 'training_model.pth')
old_vocab_path = os.path.join(old_model_dir, 'vocab.pkl')
new_vocab_path = os.path.join(new_model_dir, 'vocab.pkl')

if os.path.exists(old_model_path) and not os.path.exists(new_model_path):
    shutil.copy2(old_model_path, new_model_path)
    print(f"✓ Copied trained model to new directory")

if os.path.exists(old_vocab_path) and not os.path.exists(new_vocab_path):
    shutil.copy2(old_vocab_path, new_vocab_path)
    print(f"✓ Copied vocabulary to new directory")

# Set final paths
MODEL_SAVE_PATH = new_model_path
VOCAB_SAVE_PATH = new_vocab_path

# Combine train files into a single paired file if not already present
if not os.path.exists(TRAIN_PAIRS_PATH):
    with open(TRAIN_JAVA_PATH, encoding='utf-8') as f_java, open(TRAIN_CS_PATH, encoding='utf-8') as f_cs, open(TRAIN_PAIRS_PATH, 'w', encoding='utf-8') as f_out:
        for java_line, cs_line in zip(f_java, f_cs):
            java_line = java_line.strip()
            cs_line = cs_line.strip()
            if java_line and cs_line:
                f_out.write(f"{java_line}|||{cs_line}\n")
    print(f"Combined train files into {TRAIN_PAIRS_PATH}")
else:
    print(f"Paired train file already exists: {TRAIN_PAIRS_PATH}")

# --- CUSTOM TOKENIZER CLASS ---
class CodeTokenizer:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.SOS_TOKEN = '<sos>'
        self.EOS_TOKEN = '<eos>'
        
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        
        # Initialize with special tokens
        for i, token in enumerate(self.special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token
    
    def tokenize(self, text):
        """Tokenize code text"""
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isalnum() or char == '_':
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if char.strip():
                    tokens.append(char)
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        for text in texts:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)
        
        most_common = self.word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        for word, count in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Most common tokens: {most_common[:10]}")
    
    def encode(self, text):
        """Encode text to token ids"""
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]) for token in tokens]
        return [self.word2idx[self.SOS_TOKEN]] + ids + [self.word2idx[self.EOS_TOKEN]]
    
    def decode(self, ids):
        """Decode token ids to text"""
        tokens = []
        for id in ids:
            if id in self.idx2word:
                token = self.idx2word[id]
                if token not in self.special_tokens:
                    tokens.append(token)
        return ' '.join(tokens)
    
    def save(self, path):
        """Save tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, path):
        """Load tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_counts = data['word_counts']
            self.vocab_size = data['vocab_size']
    
    @property
    def pad_id(self):
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def unk_id(self):
        return self.word2idx[self.UNK_TOKEN]
    
    @property
    def sos_id(self):
        return self.word2idx[self.SOS_TOKEN]
    
    @property
    def eos_id(self):
        return self.word2idx[self.EOS_TOKEN]
    
    def __len__(self):
        return len(self.word2idx)

# --- DATA LOADING AND PROCESSING ---
with open(TRAIN_PAIRS_PATH, encoding='utf-8') as f:
    all_pairs = [line.strip().split('|||') for line in f if '|||' in line]

print(f"Total pairs: {len(all_pairs)}")

MAX_SAMPLES = 10000
if len(all_pairs) > MAX_SAMPLES:
    all_pairs = all_pairs[:MAX_SAMPLES]
    print(f"Limited to {MAX_SAMPLES} samples for faster training")

# Build or load tokenizer
tokenizer = CodeTokenizer(vocab_size=8000)

if os.path.exists(VOCAB_SAVE_PATH):
    print(f"Loading existing vocabulary from {VOCAB_SAVE_PATH}")
    tokenizer.load(VOCAB_SAVE_PATH)
else:
    print("Building new vocabulary...")
    all_texts = []
    for src, tgt in all_pairs:
        all_texts.append(src)
        all_texts.append(tgt)
    
    tokenizer.build_vocab(all_texts)
    tokenizer.save(VOCAB_SAVE_PATH)
    print(f"Vocabulary saved to {VOCAB_SAVE_PATH}")

# Dataset loader for code translation
class CodeTranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.processed_pairs = []
        for src, tgt in pairs:
            src_ids = tokenizer.encode(src)
            tgt_ids = tokenizer.encode(tgt)
            
            if len(src_ids) <= max_length and len(tgt_ids) <= max_length:
                self.processed_pairs.append((src_ids, tgt_ids))
        
        print(f"Filtered dataset: {len(self.processed_pairs)} pairs (max_length={max_length})")
    
    def __len__(self):
        return len(self.processed_pairs)
    
    def __getitem__(self, idx):
        return self.processed_pairs[idx]

def collate_fn(batch, pad_id):
    src_batch, tgt_batch = zip(*batch)
    
    src_pad = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in src_batch], 
                                        batch_first=True, padding_value=pad_id)
    tgt_pad = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tgt_batch], 
                                        batch_first=True, padding_value=pad_id)
    
    return src_pad, tgt_pad

# Create dataset and dataloader
train_dataset = CodeTranslationDataset(all_pairs, tokenizer, max_length=256)

# Adjust batch size based on GPU memory
BATCH_SIZE = 32
if DEVICE.type == 'cuda' and torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
    BATCH_SIZE = 16
    print("Reduced batch size to 16 for GPU memory optimization")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id))

def to_device(data):
    if isinstance(data, (list, tuple)):
        return [to_device(x) for x in data]
    return data.to(DEVICE)

# --- ENHANCED MODEL DEFINITION ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CodeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def create_mask(self, src, tgt, pad_id):
        src_key_padding_mask = (src == pad_id)
        tgt_key_padding_mask = (tgt == pad_id)
        
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        tgt_mask = tgt_mask.to(tgt.device)
        
        return src_key_padding_mask, tgt_key_padding_mask, tgt_mask
    
    def forward(self, src, tgt, pad_id, return_attention=False):
        src_key_padding_mask, tgt_key_padding_mask, tgt_mask = self.create_mask(src, tgt, pad_id)
        
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.fc_out(output)
    
    def beam_search(self, src, beam_size=5, max_length=100, pad_id=0, sos_id=1, eos_id=2):
        """Beam search decoding"""
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        src_key_padding_mask = (src == pad_id)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        
        # Get encoder output
        encoder_output = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Initialize beam
        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        beam_sequences = torch.full((batch_size, beam_size, 1), sos_id, device=device)
        beam_finished = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)
        
        for step in range(max_length):
            if beam_finished.all():
                break
            
            # Prepare decoder input
            tgt_input = beam_sequences.view(-1, beam_sequences.size(-1))
            tgt_key_padding_mask = (tgt_input == pad_id)
            
            # Create causal mask
            tgt_len = tgt_input.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)
            
            # Decoder forward pass
            tgt_emb = self.embedding(tgt_input) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
            
            # Expand encoder output for beam
            expanded_encoder_output = encoder_output.unsqueeze(1).expand(-1, beam_size, -1, -1).contiguous().view(-1, encoder_output.size(1), encoder_output.size(2))
            expanded_src_mask = src_key_padding_mask.unsqueeze(1).expand(-1, beam_size, -1).contiguous().view(-1, src_key_padding_mask.size(1))
            
            decoder_output = self.transformer.decoder(
                tgt_emb, expanded_encoder_output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=expanded_src_mask
            )
            
            # Get next token probabilities
            logits = self.fc_out(decoder_output[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, -1)
            
            # Calculate scores
            vocab_size = log_probs.size(-1)
            expanded_scores = beam_scores.unsqueeze(-1).expand(-1, -1, vocab_size)
            scores = expanded_scores + log_probs
            scores = scores.view(batch_size, -1)
            
            # Get top k scores and indices
            top_scores, top_indices = torch.topk(scores, beam_size, dim=-1)
            
            # Update beam
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update sequences
            new_sequences = []
            new_scores = []
            new_finished = []
            
            for b in range(batch_size):
                batch_sequences = []
                batch_scores = []
                batch_finished = []
                
                for i in range(beam_size):
                    beam_idx = beam_indices[b, i]
                    token_idx = token_indices[b, i]
                    
                    old_seq = beam_sequences[b, beam_idx]
                    new_seq = torch.cat([old_seq, token_idx.unsqueeze(0)])
                    
                    batch_sequences.append(new_seq)
                    batch_scores.append(top_scores[b, i])
                    batch_finished.append(beam_finished[b, beam_idx] or token_idx == eos_id)
                
                new_sequences.append(torch.stack(batch_sequences))
                new_scores.append(torch.stack(batch_scores))
                new_finished.append(torch.stack(batch_finished))
            
            beam_sequences = torch.stack(new_sequences)
            beam_scores = torch.stack(new_scores)
            beam_finished = torch.stack(new_finished)
        
        # Return best sequences
        best_sequences = beam_sequences[:, 0, :]  # Take best beam for each batch
        return best_sequences

# --- ENHANCED EVALUATION FUNCTIONS ---

def compute_token_level_accuracy(pred_ids, tgt_sample, pad_id):
    """Token-level accuracy with padding masked"""
    correct = 0
    total = 0
    
    for pred, tgt in zip(pred_ids, tgt_sample):
        mask = (tgt != pad_id)
        correct += (pred[mask] == tgt[mask]).sum().item()
        total += mask.sum().item()
    
    return correct / total if total > 0 else 0

def compute_sequence_accuracy(pred_ids, tgt_sample, pad_id):
    """Sequence-level accuracy (exact match)"""
    correct = 0
    total = 0
    
    for pred, tgt in zip(pred_ids, tgt_sample):
        # Remove padding
        pred_seq = pred[tgt != pad_id]
        tgt_seq = tgt[tgt != pad_id]
        
        if torch.equal(pred_seq, tgt_seq):
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def compute_simple_edit_distance(seq1, seq2):
    """Simple Levenshtein distance implementation without external dependencies"""
    if len(seq1) == 0:
        return len(seq2)
    if len(seq2) == 0:
        return len(seq1)
    
    # Create matrix
    matrix = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    
    # Initialize first row and column
    for i in range(len(seq1) + 1):
        matrix[i][0] = i
    for j in range(len(seq2) + 1):
        matrix[0][j] = j
    
    # Fill matrix
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                cost = 1
            
            matrix[i][j] = min(
                matrix[i-1][j] + 1,      # deletion
                matrix[i][j-1] + 1,      # insertion
                matrix[i-1][j-1] + cost  # substitution
            )
    
    return matrix[len(seq1)][len(seq2)]

def compute_edit_distances(pred_ids, tgt_sample, pad_id):
    """Compute edit distances between predictions and targets using simple implementation"""
    distances = []
    
    for pred, tgt in zip(pred_ids, tgt_sample):
        pred_seq = pred[tgt != pad_id].tolist()
        tgt_seq = tgt[tgt != pad_id].tolist()
        
        distance = compute_simple_edit_distance(pred_seq, tgt_seq)
        distances.append(distance)
    
    return distances

def compute_syntax_validity(pred_ids, tokenizer, lang='java'):
    """Compute syntax validity percentage using simple heuristics"""
    valid_count = 0
    total = len(pred_ids)
    
    for pred in pred_ids:
        code_str = tokenizer.decode(pred.tolist())
        
        # Simple syntax checks (can be enhanced with actual parsers)
        if lang == 'java':
            # Check for balanced braces, semicolons, etc.
            brace_count = code_str.count('{') - code_str.count('}')
            paren_count = code_str.count('(') - code_str.count(')')
            bracket_count = code_str.count('[') - code_str.count(']')
            
            if brace_count == 0 and paren_count == 0 and bracket_count == 0:
                valid_count += 1
        else:  # C#
            # Similar checks for C#
            brace_count = code_str.count('{') - code_str.count('}')
            paren_count = code_str.count('(') - code_str.count(')')
            bracket_count = code_str.count('[') - code_str.count(']')
            
            if brace_count == 0 and paren_count == 0 and bracket_count == 0:
                valid_count += 1
    
    return valid_count / total if total > 0 else 0

def compute_bleu(pred_ids, tgt_sample, tokenizer):
    """Compute BLEU score"""
    smoothie = SmoothingFunction().method4
    scores = []
    
    for pred, tgt in zip(pred_ids, tgt_sample):
        pred_str = tokenizer.decode(pred.tolist())
        tgt_str = tokenizer.decode(tgt.tolist())
        
        if pred_str.strip() and tgt_str.strip():
            score = sentence_bleu([tgt_str.split()], pred_str.split(), 
                                smoothing_function=smoothie)
            scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0

def compute_semantic_match(pred_ids, tgt_sample, tokenizer):
    """Semantic match score (using BLEU as proxy)"""
    return compute_bleu(pred_ids, tgt_sample, tokenizer)

# --- VISUALIZATION FUNCTIONS ---

def plot_confusion_matrix(pred_ids, tgt_sample, pad_id, tokenizer, max_tokens=50):
    """Plot confusion matrix of token predictions"""
    y_true = []
    y_pred = []
    
    # Collect non-padding tokens
    for pred, tgt in zip(pred_ids, tgt_sample):
        for p, t in zip(pred.tolist(), tgt.tolist()):
            if t != pad_id:
                y_true.append(t)
                y_pred.append(p)
    
    # Limit to most frequent tokens for readability
    unique_tokens = list(set(y_true + y_pred))[:max_tokens]
    
    # Filter data
    filtered_true = [t for t in y_true if t in unique_tokens]
    filtered_pred = [p for p in y_pred if p in unique_tokens]
    
    if len(filtered_true) == 0 or len(filtered_pred) == 0:
        print("No data available for confusion matrix")
        return
    
    cm = confusion_matrix(filtered_true, filtered_pred, labels=unique_tokens)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=[tokenizer.idx2word.get(i, f'UNK_{i}') for i in unique_tokens],
                yticklabels=[tokenizer.idx2word.get(i, f'UNK_{i}') for i in unique_tokens])
    plt.title('Token Prediction Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Token', fontsize=14)
    plt.ylabel('True Token', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{new_model_dir}confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_edit_distance_histogram(distances):
    """Plot histogram of edit distances"""
    if len(distances) == 0:
        print("No edit distance data available")
        return
        
    plt.figure(figsize=(12, 8))
    sns.histplot(distances, bins=30, kde=True, color='purple', alpha=0.7)
    plt.title('Distribution of Edit Distances', fontsize=16, fontweight='bold')
    plt.xlabel('Edit Distance', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{new_model_dir}edit_distance_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_tsne_embeddings(model, tokenizer, sample_size=1000):
    """Plot t-SNE of token embeddings"""
    # Get embeddings
    embeddings = model.embedding.weight.data.cpu().numpy()
    
    # Sample embeddings for visualization
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = [tokenizer.idx2word.get(i, f'UNK_{i}') for i in indices]
    else:
        labels = [tokenizer.idx2word.get(i, f'UNK_{i}') for i in range(len(embeddings))]
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    reduced = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=range(len(reduced)), 
                         cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Token Embeddings', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{new_model_dir}tsne_embeddings.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_metrics(loss_history, bleu_history, accuracy_history, 
                         sequence_accuracy_history, edit_distance_history, 
                         syntax_validity_history, start_epoch=0):
    """Plot comprehensive training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss
    if loss_history:
        axes[0, 0].plot(range(start_epoch, start_epoch + len(loss_history)), loss_history, 
                        color='red', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # BLEU Score
    if bleu_history:
        axes[0, 1].plot(range(start_epoch, start_epoch + len(bleu_history)), bleu_history, 
                        color='blue', linewidth=2, marker='s', markersize=4)
        axes[0, 1].set_title('BLEU Score', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Evaluation Step')
        axes[0, 1].set_ylabel('BLEU Score')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Token Accuracy
    if accuracy_history:
        axes[0, 2].plot(range(start_epoch, start_epoch + len(accuracy_history)), accuracy_history, 
                        color='green', linewidth=2, marker='^', markersize=4)
        axes[0, 2].set_title('Token-Level Accuracy', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Evaluation Step')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Sequence Accuracy
    if sequence_accuracy_history:
        axes[1, 0].plot(range(start_epoch, start_epoch + len(sequence_accuracy_history)), 
                        sequence_accuracy_history, color='orange', linewidth=2, marker='d', markersize=4)
        axes[1, 0].set_title('Sequence Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Evaluation Step')
        axes[1, 0].set_ylabel('Sequence Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Edit Distance
    if edit_distance_history:
        axes[1, 1].plot(range(start_epoch, start_epoch + len(edit_distance_history)), 
                        edit_distance_history, color='purple', linewidth=2, marker='v', markersize=4)
        axes[1, 1].set_title('Average Edit Distance', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Evaluation Step')
        axes[1, 1].set_ylabel('Edit Distance')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Syntax Validity
    if syntax_validity_history:
        axes[1, 2].plot(range(start_epoch, start_epoch + len(syntax_validity_history)), 
                        syntax_validity_history, color='brown', linewidth=2, marker='p', markersize=4)
        axes[1, 2].set_title('Syntax Validity %', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Evaluation Step')
        axes[1, 2].set_ylabel('Validity %')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{new_model_dir}comprehensive_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_prediction_samples(model, tokenizer, train_loader, num_samples=10, use_beam_search=True):
    """Generate prediction samples for manual inspection"""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(train_loader):
            if len(samples) >= num_samples:
                break
            
            src, tgt = to_device(src), to_device(tgt)
            
            if use_beam_search:
                pred_ids = model.beam_search(src, beam_size=5, max_length=256, 
                                           pad_id=tokenizer.pad_id, 
                                           sos_id=tokenizer.sos_id, 
                                           eos_id=tokenizer.eos_id)
            else:
                # Greedy decoding
                tgt_input = tgt[:, :-1]
                output = model(src, tgt_input, tokenizer.pad_id)
                pred_ids = output.argmax(-1)
            
            # Convert to samples
            for i in range(min(src.size(0), num_samples - len(samples))):
                src_str = tokenizer.decode(src[i].tolist())
                tgt_str = tokenizer.decode(tgt[i].tolist())
                pred_str = tokenizer.decode(pred_ids[i].tolist())
                
                samples.append({
                    'source': src_str,
                    'target': tgt_str,
                    'prediction': pred_str
                })
    
    return samples

# Initialize model, loss, optimizer
model = CodeTransformer(len(tokenizer), d_model=512, nhead=8, 
                       num_encoder_layers=4, num_decoder_layers=4, dropout=0.1).to(DEVICE)

# Enhanced model loading with epoch tracking
START_EPOCH = 0
if os.path.exists(MODEL_SAVE_PATH):
    print(f"✓ Loading existing model from {MODEL_SAVE_PATH}")
    try:
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                START_EPOCH = checkpoint['epoch']
                print(f"✓ Resuming from epoch {START_EPOCH}")
        else:
            model.load_state_dict(checkpoint)
            START_EPOCH = 20  # Assume we're continuing from epoch 20
            print(f"✓ Model loaded, continuing from epoch {START_EPOCH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting from scratch...")
        START_EPOCH = 0
else:
    print("No existing model found. Starting from scratch.")

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training history
loss_history = []
bleu_history = []
accuracy_history = []
sequence_accuracy_history = []
edit_distance_history = []
syntax_validity_history = []
semantic_match_history = []

# Training parameters
ADDITIONAL_EPOCHS = 30
TOTAL_EPOCHS = START_EPOCH + ADDITIONAL_EPOCHS

print(f"Continuing training from epoch {START_EPOCH} for {ADDITIONAL_EPOCHS} more epochs...")

# Enhanced training loop
def train_enhanced(model, loader, criterion, optimizer, scheduler, start_epoch=0, total_epochs=50):
    model.train()
    
    for epoch in range(start_epoch, total_epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, (src, tgt) in enumerate(pbar):
            src, tgt = to_device(src), to_device(tgt)
            
            # Use teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt_input, tokenizer.pad_id)
            
            # Compute loss with padding masked
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{epoch_loss/num_batches:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Enhanced evaluation every 2 epochs
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                eval_loss = 0
                eval_batches = 0
                all_pred_ids = []
                all_tgt_ids = []
                
                # Evaluate on multiple batches
                for eval_batch_idx, (src_sample, tgt_sample) in enumerate(loader):
                    if eval_batch_idx >= 10:  # Limit evaluation batches
                        break
                    
                    src_sample, tgt_sample = to_device(src_sample), to_device(tgt_sample)
                    
                    # Use beam search for evaluation
                    pred_ids = model.beam_search(src_sample, beam_size=3, max_length=256, 
                                               pad_id=tokenizer.pad_id, 
                                               sos_id=tokenizer.sos_id, 
                                               eos_id=tokenizer.eos_id)
                    
                    # Also compute loss
                    tgt_input = tgt_sample[:, :-1]
                    tgt_output = tgt_sample[:, 1:]
                    output = model(src_sample, tgt_input, tokenizer.pad_id)
                    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                    
                    eval_loss += loss.item()
                    eval_batches += 1
                    
                    all_pred_ids.extend(pred_ids.cpu())
                    all_tgt_ids.extend(tgt_output.cpu())
                
                avg_eval_loss = eval_loss / eval_batches
                
                # Compute all metrics
                token_acc = compute_token_level_accuracy(all_pred_ids, all_tgt_ids, tokenizer.pad_id)
                seq_acc = compute_sequence_accuracy(all_pred_ids, all_tgt_ids, tokenizer.pad_id)
                bleu = compute_bleu(all_pred_ids, all_tgt_ids, tokenizer)
                edit_distances = compute_edit_distances(all_pred_ids, all_tgt_ids, tokenizer.pad_id)
                avg_edit_distance = np.mean(edit_distances) if edit_distances else 0
                syntax_validity = compute_syntax_validity(all_pred_ids, tokenizer, lang='java')
                semantic_match = compute_semantic_match(all_pred_ids, all_tgt_ids, tokenizer)
                
                # Store metrics
                accuracy_history.append(token_acc)
                sequence_accuracy_history.append(seq_acc)
                bleu_history.append(bleu)
                edit_distance_history.append(avg_edit_distance)
                syntax_validity_history.append(syntax_validity)
                semantic_match_history.append(semantic_match)
                
                print(f"Evaluation Metrics:")
                print(f"  Loss: {avg_eval_loss:.4f}")
                print(f"  Token Accuracy: {token_acc:.4f}")
                print(f"  Sequence Accuracy: {seq_acc:.4f}")
                print(f"  BLEU Score: {bleu:.4f}")
                print(f"  Avg Edit Distance: {avg_edit_distance:.2f}")
                print(f"  Syntax Validity: {syntax_validity:.4f}")
                print(f"  Semantic Match: {semantic_match:.4f}")
                
                # Show sample predictions
                print("\nSample Beam Search Predictions:")
                for i in range(min(2, len(all_pred_ids))):
                    if i < src_sample.size(0):
                        src_str = tokenizer.decode(src_sample[i].tolist())
                        tgt_str = tokenizer.decode(tgt_sample[i].tolist())
                        pred_str = tokenizer.decode(all_pred_ids[i].tolist())
                        
                        print(f"SRC:  {src_str[:100]}...")
                        print(f"TGT:  {tgt_str[:100]}...")
                        print(f"PRED: {pred_str[:100]}...")
                        print("---")
            
            model.train()
        
        # Save model checkpoint and create visualizations
        if (epoch + 1) % 5 == 0:
            # Save with epoch information
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
            print(f"Model checkpoint saved to {MODEL_SAVE_PATH}")
            
            # Create comprehensive training plots
            plot_training_metrics(loss_history, bleu_history, accuracy_history, 
                                sequence_accuracy_history, edit_distance_history, 
                                syntax_validity_history, start_epoch=start_epoch)
        
        # Create detailed visualizations every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                try:
                    # Get sample batch for visualizations
                    sample_src, sample_tgt = next(iter(loader))
                    sample_src, sample_tgt = to_device(sample_src), to_device(sample_tgt)
                    
                    # Beam search predictions
                    beam_pred_ids = model.beam_search(sample_src, beam_size=3, max_length=256, 
                                                    pad_id=tokenizer.pad_id, 
                                                    sos_id=tokenizer.sos_id, 
                                                    eos_id=tokenizer.eos_id)
                    
                    # Create visualizations
                    print("Creating visualizations...")
                    
                    # Confusion matrix
                    plot_confusion_matrix(beam_pred_ids, sample_tgt[:, 1:], tokenizer.pad_id, tokenizer)
                    
                    # Edit distance histogram
                    edit_distances = compute_edit_distances(beam_pred_ids, sample_tgt[:, 1:], tokenizer.pad_id)
                    plot_edit_distance_histogram(edit_distances)
                    
                    # t-SNE embeddings
                    plot_tsne_embeddings(model, tokenizer)
                    
                except Exception as e:
                    print(f"Error creating visualizations: {e}")
            
            model.train()

# Start enhanced training
print("Starting enhanced training with all metrics and visualizations...")
train_enhanced(model, train_loader, criterion, optimizer, scheduler, 
               start_epoch=START_EPOCH, total_epochs=TOTAL_EPOCHS)

# Save final model
final_checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': TOTAL_EPOCHS,
    'loss_history': loss_history,
    'bleu_history': bleu_history,
    'accuracy_history': accuracy_history
}
torch.save(final_checkpoint, MODEL_SAVE_PATH)
print(f"\nEnhanced training finished! Final model saved to {MODEL_SAVE_PATH}")

# Final comprehensive evaluation
print("\n" + "="*80)
print("FINAL COMPREHENSIVE EVALUATION")
print("="*80)

model.eval()
with torch.no_grad():
    # Generate prediction samples for manual inspection
    print("Generating prediction samples for manual inspection...")
    prediction_samples = generate_prediction_samples(model, tokenizer, train_loader, 
                                                   num_samples=20, use_beam_search=True)
    
    # Save prediction samples
    with open(f'{new_model_dir}prediction_samples.json', 'w') as f:
        json.dump(prediction_samples, f, indent=2)
    
    # Display some samples
    print("\nSample Predictions (Beam Search):")
    for i, sample in enumerate(prediction_samples[:5]):
        print(f"\nSample {i+1}:")
        print(f"SOURCE: {sample['source'][:150]}...")
        print(f"TARGET: {sample['target'][:150]}...")
        print(f"PREDICTION: {sample['prediction'][:150]}...")
        print("-" * 50)
    
    # Final metrics computation
    print("\nComputing final metrics on larger evaluation set...")
    final_pred_ids = []
    final_tgt_ids = []
    
    for eval_batch_idx, (src_sample, tgt_sample) in enumerate(train_loader):
        if eval_batch_idx >= 20:  # Larger evaluation set
            break
        
        src_sample, tgt_sample = to_device(src_sample), to_device(tgt_sample)
        
        # Beam search predictions
        pred_ids = model.beam_search(src_sample, beam_size=5, max_length=256, 
                                   pad_id=tokenizer.pad_id, 
                                   sos_id=tokenizer.sos_id, 
                                   eos_id=tokenizer.eos_id)
        
        final_pred_ids.extend(pred_ids.cpu())
        final_tgt_ids.extend(tgt_sample[:, 1:].cpu())
    
    # Compute final metrics
    final_token_acc = compute_token_level_accuracy(final_pred_ids, final_tgt_ids, tokenizer.pad_id)
    final_seq_acc = compute_sequence_accuracy(final_pred_ids, final_tgt_ids, tokenizer.pad_id)
    final_bleu = compute_bleu(final_pred_ids, final_tgt_ids, tokenizer)
    final_edit_distances = compute_edit_distances(final_pred_ids, final_tgt_ids, tokenizer.pad_id)
    final_avg_edit_distance = np.mean(final_edit_distances) if final_edit_distances else 0
    final_syntax_validity = compute_syntax_validity(final_pred_ids, tokenizer, lang='java')
    final_semantic_match = compute_semantic_match(final_pred_ids, final_tgt_ids, tokenizer)
    
    print("\nFINAL EVALUATION RESULTS:")
    print(f"Token-Level Accuracy (w/ padding masked): {final_token_acc:.4f}")
    print(f"Sequence Accuracy: {final_seq_acc:.4f}")
    print(f"BLEU Score: {final_bleu:.4f}")
    print(f"Average Edit Distance: {final_avg_edit_distance:.2f}")
    print(f"Syntax Validity %: {final_syntax_validity:.4f}")
    print(f"Semantic Match Score: {final_semantic_match:.4f}")
    
    # Create final visualizations
    print("\nCreating final visualizations...")
    try:
        # Final confusion matrix
        plot_confusion_matrix(final_pred_ids, final_tgt_ids, tokenizer.pad_id, tokenizer)
        
        # Final edit distance histogram
        plot_edit_distance_histogram(final_edit_distances)
        
        # Final t-SNE plot
        plot_tsne_embeddings(model, tokenizer)
        
        # Final comprehensive metrics plot
        plot_training_metrics(loss_history, bleu_history, accuracy_history, 
                             sequence_accuracy_history, edit_distance_history, 
                             syntax_validity_history, start_epoch=START_EPOCH)
    except Exception as e:
        print(f"Error creating final visualizations: {e}")

print("\n" + "="*80)
print("TRAINING AND EVALUATION COMPLETE!")
print("="*80)
print(f"Model saved to: {os.path.abspath(MODEL_SAVE_PATH)}")
print(f"Prediction samples saved to: {os.path.abspath(f'{new_model_dir}prediction_samples.json')}")
print(f"All visualizations saved to: {os.path.abspath(new_model_dir)}")
