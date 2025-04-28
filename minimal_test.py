import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

# Create a simple model directly with TensorFlow
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(5,)))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation=None))

# Compile with try/except for run_eagerly
try:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        run_eagerly=True
    )
    print("Compiled with run_eagerly=True")
except TypeError:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    print("Compiled without run_eagerly")

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