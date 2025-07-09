import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from sklearn.model_selection import train_test_split
import json
import pickle
from collections import Counter
import math

# Download NLTK 'punkt' data for BLEU scoring
try:
    nltk.download('punkt')
except:
    pass

# Paths
TRAIN_JAVA_PATH = './FINAL TASK/data/train.java-cs.txt.java'
TRAIN_CS_PATH = './FINAL TASK/data/train.java-cs.txt.cs'
TRAIN_PAIRS_PATH = './FINAL TASK/data/train_java_cs_pairs.txt'
MODEL_SAVE_PATH = './FINAL TASK/data/saved_models2/training_model.pth'
VOCAB_SAVE_PATH = './FINAL TASK/data/saved_models2/vocab.pkl'

# Ensure the directory for saved models exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

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
        # Basic tokenization for code - split on whitespace and common delimiters
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isalnum() or char == '_':
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if char.strip():  # Don't add whitespace as tokens
                    tokens.append(char)
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        # Count all tokens
        for text in texts:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)
        
        # Get most common tokens (excluding special tokens)
        most_common = self.word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        # Add to vocabulary
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
# Read all pairs for training
with open(TRAIN_PAIRS_PATH, encoding='utf-8') as f:
    all_pairs = [line.strip().split('|||') for line in f if '|||' in line]

print(f"Total pairs: {len(all_pairs)}")

# Limit dataset size for faster training/debugging
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

# Show sample pairs
for i in range(min(3, len(all_pairs))):
    print(f"Sample {i+1}:")
    print(f"SRC: {all_pairs[i][0][:100]}...")
    print(f"TGT: {all_pairs[i][1][:100]}...")
    print(f"SRC encoded: {tokenizer.encode(all_pairs[i][0])[:20]}...")
    print(f"TGT encoded: {tokenizer.encode(all_pairs[i][1])[:20]}...")
    print("---")

# Dataset loader for code translation
class CodeTranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-process and filter pairs
        self.processed_pairs = []
        for src, tgt in pairs:
            src_ids = tokenizer.encode(src)
            tgt_ids = tokenizer.encode(tgt)
            
            # Filter out sequences that are too long
            if len(src_ids) <= max_length and len(tgt_ids) <= max_length:
                self.processed_pairs.append((src_ids, tgt_ids))
        
        print(f"Filtered dataset: {len(self.processed_pairs)} pairs (max_length={max_length})")
    
    def __len__(self):
        return len(self.processed_pairs)
    
    def __getitem__(self, idx):
        return self.processed_pairs[idx]

def collate_fn(batch, pad_id):
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_pad = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in src_batch], 
                                        batch_first=True, padding_value=pad_id)
    tgt_pad = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tgt_batch], 
                                        batch_first=True, padding_value=pad_id)
    
    return src_pad, tgt_pad

# Create dataset and dataloader
train_dataset = CodeTranslationDataset(all_pairs, tokenizer, max_length=256)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id))

# Set device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(DEVICE)})")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU")

def to_device(data):
    if isinstance(data, (list, tuple)):
        return [to_device(x) for x in data]
    return data.to(DEVICE)

# --- IMPROVED MODEL DEFINITION ---
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
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def create_mask(self, src, tgt, pad_id):
        # Create padding masks
        src_key_padding_mask = (src == pad_id)
        tgt_key_padding_mask = (tgt == pad_id)
        
        # Create causal mask for target
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        tgt_mask = tgt_mask.to(tgt.device)
        
        return src_key_padding_mask, tgt_key_padding_mask, tgt_mask
    
    def forward(self, src, tgt, pad_id):
        # Create masks
        src_key_padding_mask, tgt_key_padding_mask, tgt_mask = self.create_mask(src, tgt, pad_id)
        
        # Embeddings and positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.fc_out(output)

# Initialize model, loss, optimizer
model = CodeTransformer(len(tokenizer), d_model=512, nhead=8, 
                       num_encoder_layers=4, num_decoder_layers=4, dropout=0.1).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

EPOCHS = 20
loss_history = []
bleu_history = []
accuracy_history = []

# Helper functions for evaluation
def compute_accuracy(pred_ids, tgt_sample, pad_id):
    correct = 0
    total = 0
    
    for pred, tgt in zip(pred_ids, tgt_sample):
        # Remove padding
        pred_seq = pred[tgt != pad_id]
        tgt_seq = tgt[tgt != pad_id]
        
        min_len = min(len(pred_seq), len(tgt_seq))
        if min_len > 0:
            correct += (pred_seq[:min_len] == tgt_seq[:min_len]).sum().item()
            total += min_len
    
    return correct / total if total > 0 else 0

def compute_bleu(pred_ids, tgt_sample, tokenizer):
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

# Training loop
def train(model, loader, criterion, optimizer, scheduler, epochs=EPOCHS):
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (src, tgt) in enumerate(pbar):
            src, tgt = to_device(src), to_device(tgt)
            
            # Use teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt_input, tokenizer.pad_id)
            
            # Compute loss
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
        
        # Evaluation
        if (epoch + 1) % 2 == 0:  # Evaluate every 2 epochs
            model.eval()
            with torch.no_grad():
                eval_loss = 0
                eval_batches = 0
                
                # Evaluate on a few batches
                for eval_batch_idx, (src_sample, tgt_sample) in enumerate(loader):
                    if eval_batch_idx >= 5:  # Limit evaluation batches
                        break
                    
                    src_sample, tgt_sample = to_device(src_sample), to_device(tgt_sample)
                    
                    tgt_input = tgt_sample[:, :-1]
                    tgt_output = tgt_sample[:, 1:]
                    
                    output = model(src_sample, tgt_input, tokenizer.pad_id)
                    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                    
                    eval_loss += loss.item()
                    eval_batches += 1
                
                avg_eval_loss = eval_loss / eval_batches
                pred_ids = output.argmax(-1)
                
                bleu = compute_bleu(pred_ids, tgt_output, tokenizer)
                acc = compute_accuracy(pred_ids, tgt_output, tokenizer.pad_id)
                
                bleu_history.append(bleu)
                accuracy_history.append(acc)
                
                print(f"Evaluation - Loss: {avg_eval_loss:.4f}, BLEU: {bleu:.4f}, Accuracy: {acc:.4f}")
                
                # Show sample predictions
                print("\nSample predictions:")
                for i in range(min(2, src_sample.size(0))):
                    src_str = tokenizer.decode(src_sample[i].tolist())
                    tgt_str = tokenizer.decode(tgt_sample[i].tolist())
                    pred_str = tokenizer.decode(pred_ids[i].tolist())
                    
                    print(f"SRC:  {src_str[:100]}...")
                    print(f"TGT:  {tgt_str[:100]}...")
                    print(f"PRED: {pred_str[:100]}...")
                    print("---")
            
            model.train()
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model checkpoint saved to {MODEL_SAVE_PATH}")
            
            # Save training progress plot
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(loss_history, label='Train Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True)
            
            if bleu_history:
                plt.subplot(1, 3, 2)
                plt.plot(range(1, len(bleu_history) + 1), bleu_history, label='BLEU', color='blue')
                plt.xlabel('Evaluation Step')
                plt.ylabel('BLEU Score')
                plt.title('BLEU Score')
                plt.legend()
                plt.grid(True)
            
            if accuracy_history:
                plt.subplot(1, 3, 3)
                plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, label='Accuracy', color='green')
                plt.xlabel('Evaluation Step')
                plt.ylabel('Accuracy')
                plt.title('Token Accuracy')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'./FINAL TASK/data/saved_models2/training_progress_epoch_{epoch+1}.png')
            plt.close()

# Start training
print("Starting training...")
train(model, train_loader, criterion, optimizer, scheduler, epochs=EPOCHS)

# Save final model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nTraining finished! Final model saved to {MODEL_SAVE_PATH}")
print("Absolute path:", os.path.abspath(MODEL_SAVE_PATH))

# Final evaluation
print("\nFinal evaluation...")
model.eval()
with torch.no_grad():
    total_loss = 0
    total_batches = 0
    
    for src_sample, tgt_sample in train_loader:
        src_sample, tgt_sample = to_device(src_sample), to_device(tgt_sample)
        
        tgt_input = tgt_sample[:, :-1]
        tgt_output = tgt_sample[:, 1:]
        
        output = model(src_sample, tgt_input, tokenizer.pad_id)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        
        total_loss += loss.item()
        total_batches += 1
        
        if total_batches >= 10:  # Limit evaluation
            break
    
    avg_loss = total_loss / total_batches
    pred_ids = output.argmax(-1)
    
    final_bleu = compute_bleu(pred_ids, tgt_output, tokenizer)
    final_acc = compute_accuracy(pred_ids, tgt_output, tokenizer.pad_id)
    
    print(f"Final evaluation - Loss: {avg_loss:.4f}, BLEU: {final_bleu:.4f}, Accuracy: {final_acc:.4f}")