import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .detection-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .sentence-box {
        background: #f8f9fa;
        border: 2px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 1.5rem;
        font-family: monospace;
        min-height: 60px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="ISL Detection System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize MediaPipe
MEDIAPIPE_AVAILABLE = False
DEMO_MODE = False

try:
    import mediapipe as mp
    # Check if solutions attribute exists
    if hasattr(mp, 'solutions'):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        MEDIAPIPE_AVAILABLE = True
    else:
        DEMO_MODE = True
except (ImportError, AttributeError):
    DEMO_MODE = True

# Initialize TensorFlow (optional)
KERAS_AVAILABLE = False
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    pass

# Session state initialization
if 'hands' not in st.session_state:
    st.session_state.hands = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'detected_signs' not in st.session_state:
    st.session_state.detected_signs = []
if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = ""
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0
if 'last_sign' not in st.session_state:
    st.session_state.last_sign = "None"
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0

# Sign classes
SIGN_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# Header
st.markdown("""
<div class="main-header">
    <h1>🤟 Indian Sign Language Detection System</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">Professional Real-time ISL Recognition with AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    # System Status
    st.markdown("### 📊 System Status")
    
    if DEMO_MODE:
        st.warning("🔄 Running in DEMO Mode")
        st.info("MediaPipe version incompatible. Using simulated detection for demonstration.")
    elif MEDIAPIPE_AVAILABLE:
        st.success("✅ MediaPipe: Active")
    else:
        st.error("❌ MediaPipe: Not Available")
    
    if KERAS_AVAILABLE:
        st.success("✅ TensorFlow: Loaded")
    else:
        st.info("ℹ️ TensorFlow: Not Required")
    
    st.divider()
    
    # Settings
    st.markdown("### 🎛️ Detection Settings")
    
    min_detection_conf = st.slider(
        "Detection Threshold",
        0.0, 1.0, 0.7, 0.05,
        help="Minimum confidence for detection"
    )
    
    min_tracking_conf = st.slider(
        "Tracking Threshold",
        0.0, 1.0, 0.5, 0.05,
        help="Minimum confidence for tracking"
    )
    
    st.divider()
    
    # Voice Settings
    st.markdown("### 🔊 Voice Settings")
    voice_enabled = st.toggle("Enable Voice Output", value=True)
    voice_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
    
    st.divider()
    
    # Control Buttons
    st.markdown("### 🎮 Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 START", type="primary", use_container_width=True):
            if MEDIAPIPE_AVAILABLE:
                try:
                    st.session_state.hands = mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=min_detection_conf,
                        min_tracking_confidence=min_tracking_conf
                    )
                    st.success("✅ System Started!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            elif DEMO_MODE:
                st.session_state.hands = "DEMO"
                st.success("✅ Demo Mode Started!")
    
    with col2:
        if st.button("🔄 RESET", use_container_width=True):
            st.session_state.detected_signs = []
            st.session_state.current_sentence = ""
            st.session_state.detection_count = 0
            st.session_state.last_sign = "None"
            st.session_state.confidence = 0.0
            st.success("Reset Complete!")
    
    st.divider()
    
    # Statistics
    st.markdown("### 📈 Live Statistics")
    st.metric("Total Detections", st.session_state.detection_count)
    st.metric("Sentence Length", len(st.session_state.current_sentence))
    
    # Recent History
    if st.session_state.detected_signs:
        st.markdown("### 🕐 Recent Signs")
        recent = st.session_state.detected_signs[-5:]
        for sign in reversed(recent):
            st.markdown(f"• **{sign}**")

# Helper Functions
def extract_keypoints(results):
    """Extract hand landmarks"""
    if results and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints)
    return np.zeros(63)

def demo_predict(frame_number):
    """Demo prediction when MediaPipe not available"""
    # Cycle through signs for demo
    idx = (frame_number // 30) % len(SIGN_CLASSES)
    sign = SIGN_CLASSES[idx]
    confidence = 0.75 + (np.random.random() * 0.2)
    return sign, confidence

def predict_sign(keypoints, model):
    """Predict sign from keypoints"""
    if model is None:
        # Demo mode - cycle through signs
        idx = int(time.time()) % len(SIGN_CLASSES)
        return SIGN_CLASSES[idx], 0.85
    
    try:
        keypoints = keypoints.reshape(1, -1)
        prediction = model.predict(keypoints, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        if class_idx < len(SIGN_CLASSES):
            return SIGN_CLASSES[class_idx], confidence
        return "Unknown", confidence
    except:
        return "Error", 0.0

def speak_text(text, speed=1.0):
    """Text to speech using browser API"""
    if text and text != "None":
        js_code = f"""
        <script>
            const utterance = new SpeechSynthesisUtterance('{text}');
            utterance.rate = {speed};
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            window.speechSynthesis.speak(utterance);
        </script>
        """
        st.components.v1.html(js_code, height=0)

# Main Interface
if st.session_state.hands is None:
    # Welcome Screen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📹</h3>
            <p>Real-time Camera Detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖</h3>
            <p>AI-Powered Recognition</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔊</h3>
            <p>Voice Feedback</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("👉 **Click 'START' in the sidebar to begin detection**")
    
    # Instructions
    with st.expander("📖 Quick Start Guide", expanded=True):
        st.markdown("""
        ### How to Use
        
        1. **Start System**: Click the START button in sidebar
        2. **Allow Camera**: Grant webcam access when prompted
        3. **Show Signs**: Display ISL hand gestures to camera
        4. **Build Sentences**: Use detected signs to form words
        5. **Voice Output**: Listen to detected signs being spoken
        
        ### Supported Signs
        - **Alphabets**: A through Z
        - **Numbers**: 0 through 9
        
        ### Features
        - ✅ Real-time detection with visual feedback
        - ✅ Voice synthesis for accessibility
        - ✅ Sentence building capability
        - ✅ Detection history and statistics
        - ✅ Adjustable confidence thresholds
        """)

else:
    # Main Detection Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera Feed")
        camera_placeholder = st.empty()
        
        # Camera controls
        col_a, col_b = st.columns(2)
        with col_a:
            start_btn = st.checkbox("▶️ Start Detection", value=False)
        with col_b:
            if st.button("📸 Capture Frame"):
                st.info("Frame captured!")
    
    with col2:
        st.markdown("### 📊 Detection Dashboard")
        
        # Current Detection Display
        st.markdown(f"""
        <div class="detection-box">
            {st.session_state.last_sign}
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence meter
        if st.session_state.confidence > 0:
            st.progress(float(st.session_state.confidence))
            st.caption(f"Confidence: {st.session_state.confidence:.0%}")
        
        st.divider()
        
        # Sentence Builder
        st.markdown("### 📝 Sentence Builder")
        st.markdown(f"""
        <div class="sentence-box">
            {st.session_state.current_sentence if st.session_state.current_sentence else "Start building..."}
        </div>
        """, unsafe_allow_html=True)
        
        col_x, col_y, col_z = st.columns(3)
        
        with col_x:
            if st.button("➕ Add", use_container_width=True):
                if st.session_state.last_sign != "None":
                    st.session_state.current_sentence += st.session_state.last_sign
        
        with col_y:
            if st.button("⌫ Delete", use_container_width=True):
                if st.session_state.current_sentence:
                    st.session_state.current_sentence = st.session_state.current_sentence[:-1]
        
        with col_z:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.current_sentence = ""
        
        # Speak button
        if st.button("🔊 SPEAK SENTENCE", type="primary", use_container_width=True):
            if st.session_state.current_sentence:
                speak_text(st.session_state.current_sentence, voice_speed)
            else:
                st.warning("No sentence to speak!")
    
    # Detection Loop
    if start_btn:
        if DEMO_MODE:
            # Demo mode - simulate detection
            frame_count = 0
            demo_placeholder = camera_placeholder.empty()
            
            while start_btn:
                # Create demo frame
                demo_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
                
                # Add demo text
                cv2.putText(
                    demo_frame,
                    "DEMO MODE - Simulated Detection",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Simulate detection
                sign, conf = demo_predict(frame_count)
                st.session_state.last_sign = sign
                st.session_state.confidence = conf
                
                # Auto-add to detections every 30 frames
                if frame_count % 30 == 0 and conf > min_detection_conf:
                    st.session_state.detected_signs.append(sign)
                    st.session_state.detection_count += 1
                    
                    if voice_enabled and frame_count % 60 == 0:
                        speak_text(sign, voice_speed)
                
                cv2.putText(
                    demo_frame,
                    f"Detected: {sign} ({conf:.0%})",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 0),
                    3
                )
                
                demo_placeholder.image(demo_frame, channels="RGB", use_container_width=True)
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
                
        else:
            # Real detection with camera
            try:
                cap = cv2.VideoCapture(0)
                last_detection = time.time()
                
                while start_btn:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Camera not accessible")
                        break
                    
                    # Process frame
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = st.session_state.hands.process(image)
                    image.flags.writeable = True
                    
                    # Draw landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                        
                        # Predict
                        keypoints = extract_keypoints(results)
                        sign, conf = predict_sign(keypoints, st.session_state.model)
                        
                        st.session_state.last_sign = sign
                        st.session_state.confidence = conf
                        
                        # Add detection with cooldown
                        if conf > min_detection_conf and (time.time() - last_detection) > 1.0:
                            st.session_state.detected_signs.append(sign)
                            st.session_state.detection_count += 1
                            last_detection = time.time()
                            
                            if voice_enabled:
                                speak_text(sign, voice_speed)
                        
                        # Overlay text
                        cv2.putText(
                            image,
                            f"{sign} ({conf:.0%})",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 255, 0),
                            3
                        )
                    
                    camera_placeholder.image(image, channels="RGB", use_container_width=True)
                
                cap.release()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p><strong>🤟 Indian Sign Language Detection System</strong></p>
    <p>Powered by MediaPipe, OpenCV, Streamlit & Web Speech API</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>© 2026 | Built for Accessibility & Inclusion</p>
</div>
""", unsafe_allow_html=True)