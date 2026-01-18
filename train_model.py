import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# =========================
# Load data
# =========================
X = np.load("isl_landmarks.npy")   # shape: (samples, 63)
y = np.load("isl_labels.npy")      # shape: (samples,)

# Encode labels
unique_labels = np.unique(y)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = np.array([label_map[label] for label in y])

y_cat = to_categorical(y_encoded)

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# =========================
# Build model
# =========================
model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(unique_labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# Train model
# =========================
model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =========================
# Save model & labels
# =========================
model.save("isl_model.h5")
np.save("isl_labels.npy", unique_labels)

print(" Training complete")
print(" isl_model.h5 saved successfully")
