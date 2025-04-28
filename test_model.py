import tensorflow as tf
import numpy as np
import sys
sys.path.append('Non linear signature optimal stopping')
from Deep_signatures_optimal_stopping import LongstaffSchwartzModel

# Test LongstaffSchwartzModel
print("Creating model...")
model = LongstaffSchwartzModel(feature_dim=5, layers_number=2, nodes=8)

print("Compiling model...")
model.compile(learning_rate=0.001, loss='mse')
print(f"Model run_eagerly: {model.model.run_eagerly}")

# Create dummy data
print("Creating dummy data...")
X = np.random.random((10, 5))
y = np.random.random((10,))

# Test fit method
print("Fitting model...")
try:
    history = model.fit(X, y, epochs=2, batch_size=5, verbose=1)
    print("Fit successful!")
except Exception as e:
    print(f"Error during fit: {e}")
    
# Test prediction
try:
    print("Making predictions...")
    preds = model.predict(X)
    print(f"Predictions shape: {preds.shape}")
    print("Predict successful!")
except Exception as e:
    print(f"Error during predict: {e}")

print("Test completed") 