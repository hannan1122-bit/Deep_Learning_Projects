import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np
import re

# ------------- LOADING DATA FROM FILE ---------------
with open("text.txt", "r", encoding="utf-8") as file:  # use encoding for safety
    lines = file.readlines()

# ------- PREPROCESSING -----------
clean_lines = []
for line in lines:
    if ":" in line:
        _, sentence = line.split(":", 1)  # ✅ safer split (limit to 1 split)
        clean_lines.append(sentence.strip())
    else:
        clean_lines.append(line.strip())

content = " ".join(clean_lines)
content = content.lower()
content = re.sub(r'[^a-zA-Z0-9\s]', '', content)

# ---------- TOKENIZATION ----------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([content])
sequences = tokenizer.texts_to_sequences([content])[0]

# ---------- GENERATING SEQUENCES ----------
seq_length = 15
input_sequences = []

for i in range(seq_length, len(sequences)):
    n_gram_sequence = sequences[i - seq_length:i + 1]
    input_sequences.append(n_gram_sequence)

input_sequences = pad_sequences(input_sequences, maxlen=seq_length + 1)

# ---------- PREPARING TRAINING DATA ----------
input_sequences = np.array(input_sequences)
X = input_sequences[:, :-1]     # inputs
Y = input_sequences[:, -1]      # targets

vocab_size = len(tokenizer.word_index) + 1
Y = tf.keras.utils.to_categorical(Y, num_classes=vocab_size)

# ---------- MODEL CREATION ----------
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=50, input_length=seq_length),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # ✅ FIXED: "optiizer" → "optimizer"
model.fit(X, Y, epochs=50, batch_size=128, verbose=1)  # ✅ FIXED: "batrch_size" → "batch_size"

loss, accuracy = model.evaluate(X, Y, verbose=1)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# ---------- PREDICTION FUNCTION ----------
def predict_next_word(seed_text, max_seq_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""
