import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb

# ---------------------  TRAINING PART   -------------------------

# taking words from vocablory
num_words=10000
# maximum length of a sentence
max_words=195  

# data spliting
(X_train,Y_train),(X_test,Y_test)=imdb.load_data(num_words=num_words)
X_train=pad_sequences(X_train,maxlen=max_words)
X_test=pad_sequences(X_test,maxlen=max_words)

# model training section

model=Sequential(
    [
        Embedding(input_dim=num_words,output_dim=42,input_length=max_words)
        ,SimpleRNN(units=64,activation='tanh',return_sequences=True)
        ,LSTM(units=32, activation='tanh')
        ,Dense(units=1, activation='sigmoid')
    ]
)
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=5,batch_size=64,validation_split=0.2)

loss, accuracy=model.evaluate(X_test, Y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")


#---------------------------------- MODEL TESTING ON A USER SENTENCE ---------------------------

# getting words from vocablory
word_index = imdb.get_word_index()
# WE JUST NEED FIRST 10000 VOC.. AND FIRST 3 ARE ALREADY PRESERVED 
word_index = {k: (v + 3) for k, v in imdb.get_word_index().items() if v < num_words - 3}

def preprocess_input_text(text, word_index, maxlen=195):
    import numpy as np
    # Lowercase and split words
    words = text.lower().split()
    # Convert to word indexes (only if word exists in IMDb vocab)
    sequence = [word_index.get(word, 2) for word in words]  # 2 is for unknown words
    # Pad the sequence
    padded = pad_sequences([sequence], maxlen=maxlen)
    return padded

user_input = input("Enter The sentence: ")
processed_input = preprocess_input_text(user_input, word_index, maxlen=max_words)

prediction = model.predict(processed_input)[0][0]

if prediction > 0.5:
    print("🌟 Positive Review! (%.2f)" % prediction)
else:
    print("😞 Negative Review! (%.2f)" % prediction)
