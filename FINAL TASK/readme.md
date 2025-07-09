# 🧠 Final Task: Transformer from Scratch + Attention is All You Need

This task involves building the **Transformer architecture from scratch**, based on the original paper _"Attention is All You Need"_, and applying it (optionally) to a code translation dataset. The goal is to understand, implement, and explain the inner workings of this foundational deep learning model.

---

## 📌 Objectives

1. 📄 **Read & Understand the Paper:**  
   [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

2. ⚙️ **Implement a Transformer from Scratch:**  
   - No PyTorch `nn.Transformer`, no shortcuts — build from the ground up.
   - Cover:
     - Multi-Head Self-Attention
     - Positional Encoding
     - Layer Normalization
     - Masking (causal & padding)
     - Position-wise Feedforward Layers
     - Encoder-Decoder architecture

3. 🧪 **(Optional)** Train your model on the following dataset:  
   [Hugging Face CodeXGLUE: Code-to-Code Translation (Java ↔ C#)](https://huggingface.co/datasets/google/code_x_glue_cc_code_to_code_trans)

---

## 🗂️ Files Included

| File Name                       | Description                                     |
|--------------------------------|-------------------------------------------------|
| `transformer_from_scratch.py`  | Core implementation of the Transformer model    |
| `utils.py`                     | Positional encoding, masking, helper functions  |
| `train.py`                     | Training loop with optional dataset integration |
| `paper_notes.md`               | Summary and breakdown of the Transformer paper  |
| `code_translation_dataset/`    | Optional data directory for CodeXGLUE           |
| `README.md`                    | This file                                       |

---

## 📚 Key Concepts Implemented

| Component              | Description                                           |
|------------------------|-------------------------------------------------------|
| Multi-Head Attention   | Allows the model to focus on different representations|
| Scaled Dot-Product     | Efficient attention score calculation                 |
| Positional Encoding    | Injects order information into embeddings             |
| Feedforward Layers     | Non-linearity and depth per Transformer block         |
| Layer Normalization    | Improves training stability                           |
| Masking (Lookahead)    | Prevents the model from seeing future tokens          |
| Encoder-Decoder        | Complete Transformer pipeline                         |

---

## 📊 Dataset (Optional)

**CodeXGLUE: Code-to-Code Translation**  
- Tasks: Translate Java ↔ C# code snippets  
- Fields: `src`, `tgt`  
- Format: JSON or JSONL  

You can load using:
```python
from datasets import load_dataset
dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans", "java-cs")
