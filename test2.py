"""
Train a model to predict masked numbers in a matrix
1. Generate a 256x256 matrix with numpy
2. Randomly mask 50% of the matrix entries
3. Train a neural network to predict masked entries of the matrix
4. Train for 10000 epochs, loss fn is mean squared error
5. Have the model predict the masked entries
6. Save the model
"""
import numpy as np
import tensorflow as tf


# 1.Generate a 256x256 matrix with numpy
def generate_matrix():
    matrix = np.random.rand(256, 256)
    return matrix


# 2. Randomly mask 50% of the matrix entries
def mask_matrix(matrix):
    mask = np.random.rand(256, 256)
    #mask random indices

    mask = mask < 0.5
    masked_matrix = matrix * mask
    return masked_matrix


# 3. Train a neural network to predict masked entries
def train_model(masked_matrix, filled_matrix):
    # 4. Train for 10000 epochs, loss fn is mean squared error
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu')])
    model.compile(optimizer='adam', loss="mse")

    model.fit(masked_matrix, filled_matrix, epochs=10000, verbose=1)
    return model

# 5. Have the model predict the masked entries
def predict_model(model, masked_matrix):
    predictions = model.predict(masked_matrix)
    return predictions


# 6. Save the model
def save_model(model):
    model.save('model.h5')
    return


# 7. Load the model
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model


# 8. Use the model to predict masked entries
def use_model(model, masked_matrix):
    predictions = model.predict(masked_matrix)
    return predictions


matrix = generate_matrix()
print(matrix)
masked_matrix = mask_matrix(matrix)
print(masked_matrix)
model = train_model(masked_matrix, matrix)
predictions = predict_model(model, masked_matrix)
save_model(model)
model = load_model()
print("Predictions")
print(predictions[0])
print(matrix[0])
print(np.sum(np.abs(predictions[0] - matrix[0])))
print(np.sum(np.abs(predictions - matrix))/256*256)
