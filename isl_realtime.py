"""
ISL REAL-TIME HAND GESTURE PREDICTION
Stable • Large Display • Speech Enabled
"""

import cv2
import numpy as np
import os
import winsound

# ================= MEDIA PIPE =================
try:
    import mediapipe as mp
except ImportError:
    print("ERROR: pip install mediapipe")
    exit()

# ================= CLASS =================
class ISLPredictor:
    def __init__(self, model_path="isl_model_fixed.h5", labels_path="isl_labels.npy"):

        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = None

        # Speech control
        self.last_spoken = None
        self.speak_cooldown = 0

        # Prediction buffer (stability)
        self.prediction_buffer = []
        self.buffer_size = 15
        self.min_stable = 10

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

    # ================= SPEECH =================
    def speak(self, text):
        try:
            cmd = (
                'PowerShell -Command '
                '"Add-Type -AssemblyName System.Speech; '
                '$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$speak.Speak(\'{text}\')"'
            )
            os.system(cmd)
        except:
            winsound.Beep(1000, 200)

    # ================= LOAD MODEL =================
    def load_model(self):
        from tensorflow import keras

        self.model = keras.models.load_model(self.model_path, compile=False)
        print("Model loaded")

        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path, allow_pickle=True).tolist()
        else:
            self.labels = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        print("Labels loaded:", self.labels)
        return True

    # ================= PREDICT =================
    def predict(self, features):
        features = features.reshape(1, -1)
        pred = self.model.predict(features, verbose=0)
        idx = np.argmax(pred[0])
        conf = pred[0][idx]
        return self.labels[idx], conf

    # ================= STABLE BUFFER =================
    def get_stable_prediction(self, pred):
        self.prediction_buffer.append(pred)
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)

        if self.prediction_buffer.count(pred) >= self.min_stable:
            return pred, True

        return pred, False

    # ================= MAIN LOOP =================
    def run(self):
        cap = cv2.VideoCapture(0)

        print("\nISL REAL-TIME STARTED")
        print("Press Q to Quit\n")

        current_prediction = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = self.hands.process(rgb)

            if self.speak_cooldown > 0:
                self.speak_cooldown -= 1

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]

                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS
                )

                features = []
                for lm in hand.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                if len(features) == 63:
                    pred, conf = self.predict(np.array(features))

                    if conf > 0.70:
                        stable_pred, is_stable = self.get_stable_prediction(pred)

                        if is_stable:
                            # Speak once per stable change
                            if stable_pred != current_prediction or self.speak_cooldown == 0:
                                current_prediction = stable_pred
                                self.speak(stable_pred)
                                self.speak_cooldown = 60

                            # ===== BIG GREEN DISPLAY =====
                            overlay_height = 200
                            cv2.rectangle(frame, (0, 0), (w, overlay_height), (0, 180, 0), -1)

                            text = f"PREDICTION : {stable_pred}"
                            (tw, th), _ = cv2.getTextSize(
                                text,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                3.5,
                                8
                            )

                            tx = (w - tw) // 2
                            ty = (overlay_height + th) // 2

                            cv2.putText(
                                frame,
                                text,
                                (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                3.5,
                                (255, 255, 255),
                                8,
                                cv2.LINE_AA
                            )

            else:
                self.prediction_buffer.clear()
                current_prediction = None

                cv2.putText(
                    frame,
                    "SHOW HAND GESTURE",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    3
                )

            cv2.imshow("ISL Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


# ================= MAIN =================
def main():
    predictor = ISLPredictor()
    predictor.load_model()
    predictor.run()


if __name__ == "__main__":
    main()
