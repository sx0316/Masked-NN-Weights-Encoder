"""
Write a masked language model for matrix completion
1. Embed a matrix and randomly mask 50% of the weights
2. Input the embedded matrix into a neural network
3. Have the model predict the masked weights
4. Train for 10000 epochs, loss fn is mean squared error
5. Save the model
"""
import numpy as np
import tensorflow as tf
import tensorflow_data as tfdata
import tensorflow_data.text as tftext
import tensorflow_data.text.tokenizer as tftext_tokenizer
import tensorflow_data.text.tokenizer_utils as tftext_tokenizer_utils
import tensorflow_data.text.tokenizer_utils.all_tokenizers as tftext_tokenizer_utils_all_tokenizers
import tensorflow_data.text.tokenizer_utils.all_tokenizers.tokenizer_base as tftext_tokenizer_utils_all_tokenizers_tokenizer_base
import tensorflow_data.text.tokenizer_utils.all_tokenizers.tokenizer_fast_bert as tftext_tokenizer_utils_all_tokenizers_tokenizer_fast_bert
import tensorflow_data.text.tokenizer_utils.all_tokenizers.tokenizer_fast_bert_en as tftext_tokenizer_utils_all_tokenizers_tokenizer_fast_bert_en
import tensorflow_data.text.tokenizer_utils.all_tokenizers.tokenizer_fast_bert_en_cased as tftext_tokenizer_utils_all_tokenizers_tokenizer_fast_bert_en_cased
import tensorflow_data.text.tokenizer_utils.all_tokenizers.tokenizer_fast_bert_en_uncased as tftext_tokenizer_utils_all_tokenizers_tokenizer_fast_bert_en_uncased
import tensorflow_data.text.tokenizer_utils.all_tokenizers.tokenizer_fast_bert_en_uncased_v2 as tftext_tokenizer_utils_all_tokenizers_tokenizer_fast_bert_en_uncased_v2
import tensorflow_data.text.tokenizer_utils.all_tokenizers.tokenizer_fast_bert_ro_cased as tftext_tokenizer_utils_all_tokenizers_tokenizer_fast_bert_ro_cased
import tensorflow_data.text.tokenizer_utils.all_tokenizers.tokenizer_fast_

# Generate a 256*256
# Embed the matrix and randomly mask 50% of the weights
# Input the embedded matrix into a neural network
# Have the model predict the masked weights
# Train for 10000 epochs, loss fn is mean squared error
# Save the model



# Mask the matrix

mask_indices = np.random.choice(matrix.shape[1], int(matrix.shape[1] * 0.5), replace=False)
masked_matrix = matrix.copy()
masked_matrix[:, mask_indices] = 0

# Embed the matrix
embedding_dim = 100
embedding_layer = tf.keras.layers.Embedding(matrix.shape[1], embedding_dim)
embedding_layer.build((None,))
embedding_layer.set_weights([matrix])
embedding_layer.trainable = False

# Input the embedded matrix into a neural network
input_layer = tf.keras.layers.Input(shape=(matrix.shape[1],))
x = embedding_layer(input_layer)
x = tf.keras.layers.Dense(100, activation='relu')(x)
x = tf.keras.layers.Dense(100, activation='relu')(x)
output_layer = tf.keras.layers.Dense(matrix.shape[1], activation='sigmoid')(x)
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse')
model.fit(masked_matrix, matrix, epochs=10000)

# Save the model
model.save('model.h5')

# Load the model
model = tf.keras.models.load_model('model.h5')

# Predict the masked weights
predicted_matrix = model.predict(masked_matrix)

# Evaluate the model
print(np.mean(np.square(predicted_matrix - matrix)))
# Plot the predicted and actual weights
import matplotlib.pyplot as plt
plt.matshow(predicted_matrix, cmap='RdBu')
plt.matshow(matrix, cmap='RdBu', alpha=0.5)
plt.show()
plt.savefig('predicted_matrix.png')
plt.close()
plt.matshow(masked_matrix, cmap='RdBu')
plt.show()
plt.savefig('masked_matrix.png')
plt.close()
plt.matshow(matrix - predicted_matrix, cmap='RdBu')
plt.show()
plt.savefig('difference_matrix.png')
plt.close()

#