from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
import json
import os 

app = Flask(__name__)

# 学習データの準備
data = [
      {"input": "", "output": ""},
]

inputs = [item["input"] for item in data]
outputs = [item["output"] for item in data]

# 文字をインデックスに変換
char_to_index = {char: idx for idx, char in enumerate(sorted(set("".join(inputs + outputs))))}
index_to_char = {idx: char for char, idx in char_to_index.items()}

def text_to_sequence(text):
    return [char_to_index[char] for char in text]

def sequence_to_text(sequence):
    return "".join([index_to_char[idx] for idx in sequence])

# モデルの構築
max_len = max(len(seq) for seq in inputs)
X = tf.keras.preprocessing.sequence.pad_sequences([text_to_sequence(seq) for seq in inputs], maxlen=max_len, padding='post')
y = tf.keras.preprocessing.sequence.pad_sequences([text_to_sequence(seq) for seq in outputs], maxlen=max_len, padding='post')

y = tf.keras.utils.to_categorical(y, num_classes=len(char_to_index))

model = Sequential([
    Embedding(input_dim=len(char_to_index), output_dim=64, input_length=max_len),
    LSTM(128, return_sequences=True),
    Dense(len(char_to_index), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 学習
model.fit(X, y, epochs=500, verbose=0)

# 推論関数
def generate_response(prompt):
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([text_to_sequence(prompt)], maxlen=max_len, padding='post')
    pred = model.predict(input_seq)[0]
    response = sequence_to_text(np.argmax(pred, axis=-1))
    return response

# Flaskルート
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prompt', methods=['POST'])
def prompt():
    data = request.json
    user_input = data.get('input', '')
    response = generate_response(user_input)
    return jsonify({"input": user_input, "output": response})

if __name__ == '__main__':
    app.run(debug=True)

