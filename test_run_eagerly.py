import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile with run_eagerly=True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    run_eagerly=True
)

print(f"Model run_eagerly: {model.run_eagerly}")

# Create dummy data
X = np.random.random((10, 5))
y = np.random.random((10, 1))

# Test fit method
try:
    print("\nFitting model...")
    history = model.fit(X, y, epochs=2, batch_size=5, verbose=1)
    print("Fit successful!")
except Exception as e:
    print(f"Error during fit: {e}")
    
# Test prediction
try:
    print("\nMaking predictions...")
    preds = model.predict(X)
    print(f"Predictions shape: {preds.shape}")
    print("Predict successful!")
except Exception as e:
    print(f"Error during predict: {e}")

print("\nTest completed") 