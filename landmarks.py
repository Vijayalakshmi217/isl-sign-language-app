import os
import cv2
import numpy as np
import mediapipe as mp

# ----------------------------------
# CONFIG
# ----------------------------------
DATASET_DIR = "dataset"
IMG_SIZE = 256

X = []
y = []

total_images = 0
used_images = 0

# ----------------------------------
# MEDIAPIPE SETUP (FAST)
# ----------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6
)

print("Extracting landmarks from images...")

# ----------------------------------
# DATASET LOOP
# ----------------------------------
for label in sorted(os.listdir(DATASET_DIR)):
    label_path = os.path.join(DATASET_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        total_images += 1

        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            continue

        hand = result.multi_hand_landmarks[0]

        features = []
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])

        if len(features) == 63:
            X.append(features)
            y.append(label)
            used_images += 1

        if used_images % 100 == 0:
            print("Processed samples:", used_images)

hands.close()

# ----------------------------------
# SAVE FILES
# ----------------------------------
X = np.array(X)
y = np.array(y)

np.save("isl_landmarks.npy", X)
np.save("isl_labels.npy", y)

print("Done.")
print("Total images scanned:", total_images)
print("Samples used:", X.shape[0])
print("Feature shape:", X.shape)
