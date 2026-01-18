"""
ISL Static Image Prediction
Displays ONLY hand gestures (no text, no boxes)
"""

import cv2
import numpy as np
import os
import sys
import mediapipe as mp
from tensorflow import keras


class ISLImagePredictor:
    def __init__(self, model_path='isl_model.h5', labels_path='isl_labels.npy'):
        self.model = keras.models.load_model(model_path, compile=False)
        self.labels = np.load(labels_path, allow_pickle=True).tolist()

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def predict(self, features):
        pred = self.model.predict(features.reshape(1, -1), verbose=0)
        idx = np.argmax(pred[0])
        return self.labels[idx], pred[0][idx]

    def predict_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print("Invalid image")
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            print("No hand detected")
            cv2.imshow("ISL Prediction", image)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            return

        hand = result.multi_hand_landmarks[0]

        # Draw ONLY hand landmarks
        self.mp_draw.draw_landmarks(
            image,
            hand,
            self.mp_hands.HAND_CONNECTIONS
        )

        # Extract features (prediction happens silently)
        features = []
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])
        features = np.array(features)

        if len(features) == 63:
            pred, conf = self.predict(features)
            print(f"Predicted: {pred} ({conf*100:.2f}%)")

        cv2.imshow("ISL Prediction", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    if not os.path.exists("isl_model.h5"):
        print("Model file not found")
        return

    predictor = ISLImagePredictor()

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip()

    predictor.predict_image(image_path)


if __name__ == "__main__":
    main()
