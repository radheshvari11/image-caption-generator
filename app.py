import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Load VGG16 model
cnn_model = VGG16()

# Create LSTM model
model = Sequential()

model.add(Embedding(input_dim=5000, output_dim=256))
model.add(LSTM(256))
model.add(Dense(5000, activation='softmax'))

print("Image Caption Generator Model Ready")
