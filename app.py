"""
ISL Sign Language Translator - Professional Web Application
Real-time hand gesture recognition with speech output and W&B tracking
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import wandb
from datetime import datetime
import os
from gtts import gTTS
import tempfile
import base64

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="ISL Sign Language Translator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #00A67E;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #00A67E 0%, #00875A 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-box {
        background: #F0F2F6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #00A67E;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        background-color: #00A67E;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00875A;
    }
</style>
""", unsafe_allow_html=True)

# ==================== W&B INITIALIZATION ====================
@st.cache_resource
def initialize_wandb():
    """Initialize Weights & Biases tracking"""
    try:
        # Use your W&B API key
        wandb.login(key='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')
        
        run = wandb.init(
            project="isl-sign-language-translator",
            config={
                "model_type": "keras",
                "framework": "tensorflow",
                "labels": "1-9, A-Z (35 classes)",
                "confidence_threshold": 0.70,
                "buffer_size": 10,
                "min_stable_predictions": 6
            },
            reinit=True
        )
        return run
    except Exception as e:
        st.sidebar.error(f"W&B initialization failed: {e}")
        return None

wandb_run = initialize_wandb()

# ==================== ISL APPLICATION CLASS ====================
class ISLTranslator:
    """Main ISL Sign Language Translator Application"""
    
    def __init__(self):
        self.model = None
        self.labels = None
        
        # MediaPipe hand detection
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            model_complexity=1
        )
        
        # Tracking metrics
        self.total_predictions = 0
        self.confidence_scores = []
        self.prediction_history = []
        
        # Speech control
        self.last_spoken = None
        self.speak_cooldown = 0
    
    @st.cache_resource
    def load_model(_self):
        """Load the trained ISL model and labels"""
        try:
            # Load model
            model = keras.models.load_model('isl_model.h5', compile=False)
            
            # Load labels
            if os.path.exists('isl_labels.npy'):
                labels = np.load('isl_labels.npy', allow_pickle=True).tolist()
            else:
                # Default: 1-9 then A-Z
                labels = [str(i) for i in range(1, 10)] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
            return model, labels
        except Exception as e:
            st.error(f" Error loading model: {str(e)}")
            return None, None
    
    def text_to_speech(self, text):
        """Convert text to speech and play audio"""
        try:
            # Generate speech using Google Text-to-Speech
            tts = gTTS(text=str(text), lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts.save(fp.name)
                temp_file = fp.name
            
            # Read audio file
            with open(temp_file, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Convert to base64 for HTML audio player
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            # Auto-play audio
            audio_html = f"""
                <audio autoplay style="display:none">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            
            # Clean up temporary file
            os.unlink(temp_file)
            
        except Exception as e:
            st.sidebar.warning(f" Speech unavailable: {str(e)}")
    
    def predict_gesture(self, features):
        """Predict sign language gesture from hand features"""
        try:
            # Reshape features for prediction
            features = features.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features, verbose=0)
            idx = np.argmax(prediction[0])
            confidence = float(prediction[0][idx])
            label = str(self.labels[idx])
            
            # Update metrics
            self.total_predictions += 1
            self.confidence_scores.append(confidence)
            self.prediction_history.append({
                'label': label,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # Log to Weights & Biases
            if wandb_run:
                wandb.log({
                    "prediction": label,
                    "confidence": confidence,
                    "total_predictions": self.total_predictions,
                    "timestamp": datetime.now().isoformat()
                })
            
            return label, confidence
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, 0.0
    
    def process_frame(self, frame):
        """Process video frame for hand detection and gesture recognition"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand skeleton
            self.mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Extract hand features (21 landmarks × 3 coordinates = 63 features)
            features = []
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            features = np.array(features)
            
            # Validate features
            if len(features) == 63:
                prediction, confidence = self.predict_gesture(features)
                return frame, prediction, confidence
        
        return frame, None, 0.0

# ==================== MAIN APPLICATION ====================
def main():
    """Main application entry point"""
    
    # Header
    st.markdown('<div class="main-header">ISL Sign Language Translator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Indian Sign Language Recognition with AI-Powered Translation</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize application
    app = ISLTranslator()
    app.model, app.labels = app.load_model()
    
    # Check if model loaded successfully
    if app.model is None or app.labels is None:
        st.error(" Failed to load model. Please ensure 'isl_model.h5' and 'isl_labels.npy' are in the project directory.")
        st.stop()
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header(" Dashboard")
        
        # Statistics
        st.metric("Total Predictions", app.total_predictions)
        
        if app.confidence_scores:
            avg_confidence = np.mean(app.confidence_scores)
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        st.markdown("---")
        
        # W&B Dashboard Link
        if wandb_run:
            st.success(" W&B Connected")
            st.markdown(f"### [View Live Dashboard]({wandb_run.get_url()})")
        else:
            st.warning(" W&B Not Connected")
        
        st.markdown("---")
        
        # Supported Signs
        st.header(" Supported Signs")
        st.markdown("**Numbers:** 1, 2, 3, 4, 5, 6, 7, 8, 9")
        st.markdown("**Letters:** A to Z (26 letters)")
        st.info(f"**Total Classes:** {len(app.labels)}")
        
        st.markdown("---")
        
        # Model Information
        st.header("Model Info")
        st.json({
            "Framework": "TensorFlow/Keras",
            "Hand Tracking": "MediaPipe",
            "Confidence Threshold": "70%",
            "Buffer Size": "10 frames",
            "Min Stable": "6 frames"
        })
        
        st.markdown("---")
        
        # Instructions
        st.header(" How to Use")
        st.markdown("""
        1. Click **'Start Camera'**
        2. Show hand gesture clearly
        3. Keep hand in camera view
        4. Hold gesture steady for 2 seconds
        5. Listen to spoken prediction
        6. View metrics in W&B dashboard
        """)
        
        st.markdown("---")
        
        # Tips for Better Results
        st.header("Pro Tips")
        st.success("""
        **Ensure good lighting**  
        **Use clear gestures**  
        **Keep hand centered**  
        **Hold gesture steady**  
        **Avoid background clutter**  
        """)
    
    # ==================== MAIN CONTENT ====================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Camera Feed")
        
        # Camera control
        camera_active = st.checkbox("Start Camera", value=False)
        
        # Video feed placeholder
        video_placeholder = st.empty()
        
        # Prediction display placeholders
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        if camera_active:
            # Open camera
            camera = cv2.VideoCapture(0)
            
            if not camera.isOpened():
                st.error("Cannot access camera. Please check camera permissions.")
            else:
                st.success("Camera Active - Show your hand gesture!")
                
                # Prediction stabilization buffer
                prediction_buffer = []
                buffer_size = 10
                min_stable = 6
                current_prediction = None
                
                # Main camera loop
                while camera_active:
                    ret, frame = camera.read()
                    if not ret:
                        st.error("Failed to read from camera")
                        break
                    
                    # Flip frame horizontally (mirror effect)
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    processed_frame, prediction, confidence = app.process_frame(frame)
                    
                    # Update speech cooldown
                    if app.speak_cooldown > 0:
                        app.speak_cooldown -= 1
                    
                    # Stable prediction logic
                    if prediction and confidence > 0.70:
                        # Add to buffer
                        prediction_buffer.append(prediction)
                        if len(prediction_buffer) > buffer_size:
                            prediction_buffer.pop(0)
                        
                        # Check for stable prediction
                        if len(prediction_buffer) >= min_stable:
                            most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                            count = prediction_buffer.count(most_common)
                            
                            if count >= min_stable:
                                # Speak when prediction changes or cooldown expires
                                if most_common != current_prediction or app.speak_cooldown == 0:
                                    current_prediction = most_common
                                    app.text_to_speech(most_common)
                                    app.speak_cooldown = 30
                                
                                # Draw prediction on frame
                                h, w = processed_frame.shape[:2]
                                
                                # Green header bar
                                cv2.rectangle(processed_frame, (0, 0), (w, 120), (0, 120, 0), -1)
                                
                                # Large centered text
                                text_size = cv2.getTextSize(most_common, cv2.FONT_HERSHEY_DUPLEX, 3.5, 6)[0]
                                text_x = (w - text_size[0]) // 2
                                
                                cv2.putText(
                                    processed_frame,
                                    most_common,
                                    (text_x, 85),
                                    cv2.FONT_HERSHEY_DUPLEX,
                                    3.5,
                                    (255, 255, 255),
                                    6
                                )
                                
                                # Display in Streamlit
                                prediction_placeholder.markdown(
                                    f'<div class="prediction-box"> {most_common} </div>',
                                    unsafe_allow_html=True
                                )
                                confidence_placeholder.markdown(
                                    f'<div class="confidence-box"><strong>Confidence:</strong> {confidence:.1%}</div>',
                                    unsafe_allow_html=True
                                )
                    else:
                        # Clear buffer when no hand detected
                        if not prediction:
                            prediction_buffer.clear()
                            current_prediction = None
                    
                    # Display frame
                    video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                
                camera.release()
        
        else:
            st.info(" Click **'Start Camera'** above to begin real-time detection")
    
    with col2:
        st.subheader("Live Statistics")
        
        # Recent predictions
        if app.prediction_history:
            st.markdown("### Recent Predictions")
            recent = app.prediction_history[-5:][::-1]  # Last 5, reversed
            
            for pred in recent:
                st.markdown(f"""
                <div class="info-card">
                    <strong>{pred['label']}</strong><br>
                    Confidence: {pred['confidence']:.1%}<br>
                    <small>{pred['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Confidence distribution
        if len(app.confidence_scores) > 0:
            st.markdown("### Confidence Scores")
            st.line_chart(app.confidence_scores[-20:])  # Last 20 scores
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>ISL Sign Language Translator</strong></p>
        <p>Built with  using Streamlit • MediaPipe • TensorFlow • Weights & Biases</p>
        <p> 2026 - Real-time AI-Powered Sign Language Recognition</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()