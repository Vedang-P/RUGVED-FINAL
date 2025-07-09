import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

java_path = './FINAL TASK/data/test.java-cs.txt.java'
cs_path = './FINAL TASK/data/test.java-cs.txt.cs'
out_path = './FINAL TASK/data/java_cs_pairs.txt'

with open(java_path, encoding='utf-8') as f_java, open(cs_path, encoding='utf-8') as f_cs, open(out_path, 'w', encoding='utf-8') as f_out:
    for java_line, cs_line in zip(f_java, f_cs):
        java_line = java_line.strip()
        cs_line = cs_line.strip()
        if java_line and cs_line:
            f_out.write(f"{java_line}|||{cs_line}\n")

# Data Preprocessing and Exploration for test.java-cs.txt.cs
# This script will analyze the dataset to understand its structure for Java <-> C# translation.

DATA_PATH = './FINAL TASK/data/java_cs_pairs.txt'

# Read a sample of the file
def read_sample_lines(path, n=10):
    lines = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line.strip())
    return lines

# Analyze the file
sample_lines = read_sample_lines(DATA_PATH, 20)
print('Sample lines from dataset:')
for i, line in enumerate(sample_lines):
    print(f'{i+1}: {line}')

# Check for delimiter usage and line structure
def analyze_lines(lines):
    delimiter = '|||'  # expected delimiter for src/tgt pairs
    has_delimiter = [delimiter in line for line in lines]
    print(f'Lines with delimiter "{delimiter}": {sum(has_delimiter)}/{len(lines)}')
    if not any(has_delimiter):
        print('No delimiter found. Assuming each line is a single code snippet (likely not paired).')
    else:
        print('Delimiter found. Each line is a Java/C# pair.')

analyze_lines(sample_lines)

# Check total lines and file size
num_lines = sum(1 for _ in open(DATA_PATH, encoding='utf-8'))
file_size = os.path.getsize(DATA_PATH)
print(f'Total lines: {num_lines}, File size: {file_size/1024:.2f} KB')

# If no delimiter, treat each line as a single code snippet (not paired)
# If delimiter exists, treat as paired data for translation

# Dataset loader for code translation
delimiter = '|||'
class CodeTranslationDataset(Dataset):
    def __init__(self, path, delimiter=delimiter):
        self.pairs = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                if delimiter in line:
                    src, tgt = line.strip().split(delimiter)
                    self.pairs.append((src, tgt))
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

def tokenize(text):
    return list(text)

class Vocab:
    def __init__(self, texts):
        chars = set()
        for text in texts:
            chars.update(tokenize(text))
        self.chars = sorted(list(chars)) + ['<pad>', '<sos>', '<eos>']
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
    def encode(self, text):
        return [self.stoi['<sos>']] + [self.stoi[c] for c in tokenize(text)] + [self.stoi['<eos>']]
    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids if i in self.itos and self.itos[i] not in ['<sos>', '<eos>', '<pad>']])
    def __len__(self):
        return len(self.chars)

class CodeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        out = self.transformer(src_emb.permute(1,0,2), tgt_emb.permute(1,0,2))
        return self.fc(out.permute(1,0,2))

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_enc = [src_vocab.encode(s) for s in src_batch]
    tgt_enc = [tgt_vocab.encode(t) for t in tgt_batch]
    src_pad = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in src_enc], batch_first=True, padding_value=src_vocab.stoi['<pad>'])
    tgt_pad = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tgt_enc], batch_first=True, padding_value=tgt_vocab.stoi['<pad>'])
    return src_pad, tgt_pad

# Load dataset
dataset = CodeTranslationDataset(DATA_PATH)
src_texts = [src for src, tgt in dataset]
tgt_texts = [tgt for src, tgt in dataset]
src_vocab = Vocab(src_texts)
tgt_vocab = Vocab(tgt_texts)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Set device to GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Move model to device
def to_device(data):
    if isinstance(data, (list, tuple)):
        return [to_device(x) for x in data]
    return data.to(DEVICE)

model = CodeTransformer(len(src_vocab)).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.stoi['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop with progress bar and metrics
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for src, tgt in pbar:
            src, tgt = to_device(src), to_device(tgt)
            optimizer.zero_grad()
            output = model(src, tgt[:,:-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'batch_loss': loss.item()})
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Optionally, show a few predictions for sanity check
        model.eval()
        with torch.no_grad():
            src_sample, tgt_sample = next(iter(loader))
            src_sample, tgt_sample = to_device(src_sample), to_device(tgt_sample)
            pred = model(src_sample, tgt_sample[:,:-1])
            pred_ids = pred.argmax(-1)
            print("Sample predictions:")
            for i in range(min(3, src_sample.size(0))):
                src_str = src_vocab.decode(src_sample[i].tolist())
                tgt_str = tgt_vocab.decode(tgt_sample[i].tolist())
                pred_str = tgt_vocab.decode(pred_ids[i].tolist())
                print(f"SRC: {src_str}\nTGT: {tgt_str}\nPRED: {pred_str}\n---")
        model.train()

# Inference function
def translate(model, src_code, src_vocab, tgt_vocab, max_len=256):
    model.eval()
    src = torch.tensor([src_vocab.encode(src_code)], dtype=torch.long).to(DEVICE)
    tgt = torch.tensor([[tgt_vocab.stoi['<sos>']]], dtype=torch.long).to(DEVICE)
    for _ in range(max_len):
        out = model(src, tgt)
        next_token = out[0, -1].argmax(-1).item()
        tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
        if next_token == tgt_vocab.stoi['<eos>']:
            break
    return tgt_vocab.decode(tgt[0].tolist())

# Example usage:
train(model, loader, criterion, optimizer, epochs=5)
# print(translate(model, "public void foo() { ... }", src_vocab, tgt_vocab))
# Save model
torch.save(model.state_dict(), "code_transformer.pth")
print("Model saved to code_transformer.pth")

# Save the current trained model without retraining
MODEL_SAVE_PATH = "model_trained_on_test_dataset_5_epochs.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

import os
print("Saving to:", os.path.abspath(MODEL_SAVE_PATH))