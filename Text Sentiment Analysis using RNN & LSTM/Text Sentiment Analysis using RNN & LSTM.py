# ============================================================================
# File Name   : TransformerSentimentAnalysis.py
# Description : Implementation of a Transformer-based Sentiment Analysis model
#               using TensorFlow and Keras. The program demonstrates the
#               complete Transformer pipeline including:
#                   1. Dataset Creation
#                   2. Text Tokenization
#                   3. Word Embedding
#                   4. Positional Encoding
#                   5. Self-Attention
#                   6. Transformer Encoder
#                   7. Model Building
#                   8. Model Training
#                   9. Model Evaluation
#                  10. Prediction on New Sentences
#
# Author      : Kartik Ganesh Jare
# Date        :  3 May 2026
# Framework   : TensorFlow 2.x, Keras
# ============================================================================

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def Marvellous_Header(step, title):
    """
    Function Name : Marvellous_Header()

    Description   : Displays a formatted heading for each execution step.

    Parameters    :
        step  (int)  : Step number
        title (str)  : Step title

    Returns       :
        None

    Author        : Kartik Ganesh Jare
    Date          : 3 May 2026
    """
    print("\n" + "=" * 70)
    print(f"Step {step}: {title}")
    print("=" * 70)


# ------------------------------------------------------------
# Step 1: Dataset
# Purpose :
#     Create a small sentiment dataset consisting of positive and
#     negative training sentences.
# ------------------------------------------------------------


Marvellous_Header(1, "Dataset Creation")
print("Explanation: Creating positive and negative training sentences.")

sentences = [
    "this course is very good",
    "i like this session",
    "transformer concept is excellent",
    "teaching is very clear",
    "this class is helpful",
    "i enjoyed the lecture",
    "this lecture is excellent",
    "session was very useful",
    "this topic is interesting",
    "i understood the concept",
    "teacher explained very well",
    "this course is amazing",

    "this course is boring",
    "i did not understand",
    "session was confusing",
    "teaching is not clear",
    "this class is difficult",
    "i did not like the lecture",
    "this topic is hard",
    "lecture was not useful",
    "i am confused",
    "teacher did not explain well",
    "this session is bad",
    "course was difficult"
]

labels = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

for s, l in zip(sentences, labels):
    print(s, "=>", "Positive" if l == 1 else "Negative")


# ------------------------------------------------------------
# Step 2: Tokenization and Padding
# Purpose:
#     Convert text into integer tokens and pad every sentence
#     to a fixed length required by the Transformer model.
# ------------------------------------------------------------

Marvellous_Header(2, "Tokenization and Padding")
print("Explanation: Each word is converted into a number. Padding makes every sentence length equal.")

sequence_length = 8
vocab_size = 1000

vectorizer = layers.TextVectorization(
    max_tokens=vocab_size,
    output_sequence_length=sequence_length
)

vectorizer.adapt(sentences)

x_data = vectorizer(sentences)

print("\nVocabulary:")
for index, word in enumerate(vectorizer.get_vocabulary()):
    print(index, ":", word)

print("\nSample Tokenized Output:")
for sentence, tokens in zip(sentences[:5], x_data.numpy()[:5]):
    print(sentence)
    print(tokens)
    print("-" * 50)


# ------------------------------------------------------------
# Step 3: Embedding and Positional Encoding
#
# Purpose:
#     Convert token IDs into dense vectors and add positional
#     information because Transformer does not process words
#     sequentially like RNN/LSTM.
# ------------------------------------------------------------

Marvellous_Header(3, "Embedding and Positional Encoding")
print("Token numbers are converted into vectors. Position information is added because Transformer does not read words sequentially like RNN.")

embed_dim = 16

sample_embedding_layer = layers.Embedding(vocab_size, embed_dim)
sample_position_layer = layers.Embedding(sequence_length, embed_dim)

sample_tokens = x_data[:1]
positions = tf.range(start=0, limit=sequence_length, delta=1)

token_embedding_output = sample_embedding_layer(sample_tokens)
position_embedding_output = sample_position_layer(positions)

print("\nSample Input Tokens:")
print(sample_tokens.numpy())

print("\nToken Embedding Shape:")
print(token_embedding_output.shape)

print("\nSample Token Embedding of First Word:")
print(token_embedding_output[0][0].numpy())

print("\nPosition Embedding Shape:")
print(position_embedding_output.shape)

print("\nSample Position Embedding of Position 0:")
print(position_embedding_output[0].numpy())


# ------------------------------------------------------------
# Step 4: Self-Attention
#
# Purpose:
#     Allow every word in the sentence to attend to every other
#     word and learn contextual relationships.
# ------------------------------------------------------------

Marvellous_Header(4, "Self-Attention")
print("Each word compares itself with every other word in the same sentence.")

sample_combined_embedding = token_embedding_output + position_embedding_output

attention_layer = layers.MultiHeadAttention(
    num_heads=2,
    key_dim=embed_dim
)

attention_output, attention_scores = attention_layer(
    sample_combined_embedding,
    sample_combined_embedding,
    return_attention_scores=True
)

print("\nAttention Output Shape:")
print(attention_output.shape)

print("\nAttention Scores Shape:")
print(attention_scores.shape)

print("\nSample Attention Score Matrix of Head 1:")
print(attention_scores[0][0].numpy())


# ------------------------------------------------------------
# Step 5: Custom Embedding Layer for Actual Model
# ------------------------------------------------------------

class TokenAndPositionEmbedding(layers.Layer):
    """
    Class Name    : TokenAndPositionEmbedding

    Description   :
        Combines Token Embedding and Positional Embedding.

        Each token is first converted into a dense vector.
        Positional information is then added so that the
        Transformer knows the order of words.

    Author        : Kartik Ganesh Jare
    Date          : 3 May 2026
    """
    def __init__(self, max_len, vocab_size, embed_dim):
        """
        Constructor

        Description :
            Initializes embedding layers.

        Parameters :
            max_len     : Maximum sequence length
            vocab_size  : Vocabulary size
            embed_dim   : Embedding dimension
        """
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, embed_dim)
        self.position_embedding = layers.Embedding(max_len, embed_dim)

    def call(self, x):
        """
        Function Name : call()

        Description :
            Generates token embedding and positional embedding
            then combines them.

        Parameters :
            x : Input token sequence

        Returns :
            Combined embedding tensor.
        """
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)

        token_embed = self.token_embedding(x)
        position_embed = self.position_embedding(positions)

        return token_embed + position_embed


# ------------------------------------------------------------
# Step 6: Transformer Encoder Block
# ------------------------------------------------------------

class TransformerEncoder(layers.Layer):
    """
    Class Name    : TransformerEncoder

    Description   :
        Implements a Transformer Encoder Block consisting of

            • Multi-Head Self Attention
            • Feed Forward Network
            • Residual Connections
            • Layer Normalization
            • Dropout

    Author        : Kartik Ganesh Jare
    Date          : 3 May 2026
    
    """
    def __init__(self, embed_dim, num_heads, ff_dim):
        """
        Constructor

        Description :
            Initializes all layers required for
            Transformer Encoder.

        Parameters :
            embed_dim : Embedding dimension
            num_heads : Number of Attention Heads
            ff_dim    : Feed Forward dimension
        """
        
        super().__init__()

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, x, training=False):
        """
        Function Name : call()

        Description :
            Performs

                1. Self Attention
                2. Residual Connection
                3. Layer Normalization
                4. Feed Forward Network
                5. Residual Connection
                6. Layer Normalization

        Parameters :
            x        : Input Tensor
            training : Training flag

        Returns :
            Encoded Tensor
        """

        attention_output = self.attention(x, x)

        attention_output = self.dropout1(attention_output, training=training)

        out1 = self.norm1(x + attention_output)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(ffn_output, training=training)

        return self.norm2(out1 + ffn_output)


# ------------------------------------------------------------
# Step 7: Model 
#
# Architecture
#
# Input Layer
#        ↓
# Token + Position Embedding
#        ↓
# Transformer Encoder
#        ↓
# Global Average Pooling
#        ↓
# Dense Layer
#        ↓
# Dropout
#        ↓
# Sigmoid Output Layer

# ------------------------------------------------------------

Marvellous_Header(5, "Model Building")
print("Combining Embedding, Positional Encoding, Transformer Encoder and Classification Layer.")

num_heads = 2
ff_dim = 32

inputs = layers.Input(shape=(sequence_length,), dtype=tf.int64)

x = TokenAndPositionEmbedding(
    sequence_length,
    vocab_size,
    embed_dim
)(inputs)

x = TransformerEncoder(
    embed_dim,
    num_heads,
    ff_dim
)(x)

x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(32, activation="relu")(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ------------------------------------------------------------
# Step 8: Model Training
#
# Purpose:
#     Train the Transformer model using Binary Cross Entropy
#     and Adam Optimizer.
# ------------------------------------------------------------

Marvellous_Header(6, "Model Training")
print("Model learns positive and negative sentence patterns from training data.")

history = model.fit(
    x_data,
    labels,
    epochs=80,
    batch_size=2,
    verbose=1
)


# ------------------------------------------------------------
# Step 9: Training Evaluation
# Purpose:
#     Evaluate training accuracy and loss to determine how
#     well the model has learned the sentiment patterns.
# ------------------------------------------------------------

Marvellous_Header(7, "Training Evaluation")
print("Explanation: Checking how much model learned from the training data.")

loss, accuracy = model.evaluate(x_data, labels, verbose=0)

print("Training Loss     :", loss)
print("Training Accuracy :", accuracy)


# ------------------------------------------------------------
# Step 10: Prediction
#
# Purpose:
#     Predict whether unseen sentences are Positive or Negative
#     using the trained Transformer model.
# ------------------------------------------------------------

Marvellous_Header(8, "Prediction")
print("Model predicts whether new sentences are positive or negative.")

test_sentences = [
    "this session is good",
    "i did not like this course",
    "teaching is excellent",
    "this course is confusing",
    "this class is very helpful",
    "course was difficult",
    "teacher explained very well",
    "i am confused"
]

test_data = vectorizer(test_sentences)

print("\nTest Sentence Tokens:")
for sentence, tokens in zip(test_sentences, test_data.numpy()):
    print(sentence)
    print(tokens)
    print("-" * 50)

predictions = model.predict(test_data)

print("\nFinal Prediction Output:")

for sentence, prediction in zip(test_sentences, predictions):
    score = prediction[0]
    result = "Positive" if score >= 0.5 else "Negative"

    print("-" * 60)
    print("Sentence :", sentence)
    print("Score    :", score)
    print("Result   :", result)