# 🤖 Text Sentiment Analysis using Transformer

## 📌 Description

This project implements a **Transformer-based Sentiment Analysis model** using **TensorFlow** and **Keras**. The model classifies input text as either **Positive** or **Negative** by learning contextual relationships between words through the Transformer architecture.

The project demonstrates the complete deep learning workflow, including dataset preparation, text preprocessing, tokenization, positional encoding, self-attention, Transformer encoder implementation, model training, evaluation, and prediction.

---

## 🎯 Objectives

* Understand the Transformer architecture
* Learn Natural Language Processing (NLP) fundamentals
* Implement self-attention mechanisms
* Perform sentiment classification
* Gain practical experience with TensorFlow and Keras

---

## 🚀 Features

* Custom sentiment dataset
* Text tokenization and padding
* Word embedding generation
* Positional encoding
* Multi-Head Self-Attention
* Custom Transformer Encoder implementation
* Binary sentiment classification
* Model training and evaluation
* Prediction on unseen sentences

---

## 🏗️ Model Architecture

```text
Input Sentences
       │
       ▼
Text Vectorization
       │
       ▼
Token Embedding
       │
       ▼
Positional Encoding
       │
       ▼
Transformer Encoder
       │
       ▼
Global Average Pooling
       │
       ▼
Dense Layer (ReLU)
       │
       ▼
Dropout
       │
       ▼
Sigmoid Output Layer
```

---

## 🛠️ Technologies Used

* Python 3.x
* TensorFlow 2.x
* Keras
* NumPy

---

## 📂 Project Structure

```text
Text_Sentiment_Analysis/
│── TransformerSentimentAnalysis.py
│── README.md
```

---

## ⚙️ Workflow

1. Create sentiment dataset
2. Tokenize and preprocess text
3. Generate token embeddings
4. Apply positional encoding
5. Compute self-attention
6. Build Transformer encoder
7. Train the model
8. Evaluate performance
9. Predict sentiment for new sentences

---

## ▶️ Installation

Install the required packages:

```bash
pip install tensorflow numpy
```

---

## ▶️ Run the Project

```bash
python TransformerSentimentAnalysis.py
```

---

## 📊 Sample Prediction

```text
Sentence : this session is good
Prediction : Positive

Sentence : course was difficult
Prediction : Negative
```

---

## 📚 Concepts Covered

* Natural Language Processing (NLP)
* Text Vectorization
* Token Embeddings
* Positional Encoding
* Multi-Head Attention
* Transformer Encoder
* Binary Classification
* Model Evaluation

---

## 🎓 Learning Outcomes

After completing this project, you will understand:

* Transformer architecture
* Self-attention mechanism
* Tokenization techniques
* Word embeddings
* Positional encoding
* Deep learning model training
* Text sentiment classification

---

## 👤 Author

**Kartik Ganesh Jare**

---
## ⭐ Why This Repository Matters

This project demonstrates practical implementation of a modern NLP architecture using Transformers. It showcases skills in deep learning, TensorFlow, custom neural network development, and sentiment analysis, making it valuable for AI/ML portfolios, academic learning, and technical interviews.
