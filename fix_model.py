"""
ISL Model Compatibility Fix
Creates a TensorFlow-compatible ISL model
"""

import numpy as np
import os
import sys
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input

# --------------------------------------------------
# Check model file
# --------------------------------------------------
if not os.path.exists("isl_model.h5"):
    print("ERROR: isl_model.h5 not found")
    sys.exit(1)

# --------------------------------------------------
# Load labels
# --------------------------------------------------
if os.path.exists("isl_labels.npy"):
    labels = np.load("isl_labels.npy", allow_pickle=True)
    num_classes = len(labels)
else:
    num_classes = 36  # A-Z + 0-9

# --------------------------------------------------
# Load old model safely
# --------------------------------------------------
try:
    old_model = load_model("isl_model.h5", compile=False, safe_mode=False)
except:
    old_model = load_model("isl_model.h5", compile=False)

old_weights = old_model.get_weights()

# --------------------------------------------------
# Build new compatible model
# --------------------------------------------------
new_model = Sequential([
    Input(shape=(63,)),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

new_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# Transfer weights
# --------------------------------------------------
new_model.set_weights(old_weights)

# --------------------------------------------------
# Save fixed model
# --------------------------------------------------
new_model.save("isl_model_fixed.h5")

# --------------------------------------------------
# Verify model
# --------------------------------------------------
test_input = np.random.rand(1, 63).astype("float32")
_ = new_model.predict(test_input, verbose=0)

print("SUCCESS: isl_model_fixed.h5 created and verified")
