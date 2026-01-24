import streamlit as st
import cv2
import numpy as np
import sys
import os

# Page configuration
st.set_page_config(
    page_title="ISL Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# Initialize MediaPipe with explicit error handling
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands_module
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    
    mp_hands = mp_hands_module
    MEDIAPIPE_AVAILABLE = True
    st.success("✅ MediaPipe loaded successfully!")
except Exception as e:
    st.error(f"❌ MediaPipe initialization error: {str(e)}")
    st.info(f"Python version: {sys.version}")
    st.info(f"MediaPipe module location: {mp.__file__ if 'mp' in locals() else 'Not found'}")

# Make TensorFlow optional
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    keras = None

# Initialize session state
if 'hands' not in st.session_state:
    st.session_state.hands = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Title
st.title("🤟 Indian Sign Language Detection System")
st.markdown("Real-time hand gesture recognition using MediaPipe and Deep Learning")

# Stop if MediaPipe not available
if not MEDIAPIPE_AVAILABLE:
    st.error("⚠️ Cannot proceed without MediaPipe. Check deployment logs.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Show system status
    st.subheader("System Status")
    st.success("✅ MediaPipe available")
    
    if KERAS_AVAILABLE:
        st.success("✅ TensorFlow available")
    else:
        st.info("ℹ️ TensorFlow not installed - hand detection only")
    
    # Detection settings
    st.subheader("Detection Configuration")
    min_detection_confidence = st.slider(
        "Detection Confidence",
        0.0, 1.0, 0.5, 0.05
    )
    
    min_tracking_confidence = st.slider(
        "Tracking Confidence",
        0.0, 1.0, 0.5, 0.05
    )
    
    # Initialize detector
    if st.button("Initialize Hand Detector"):
        try:
            st.session_state.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            st.success("✅ Hand detector initialized!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Model loading (optional - only if TensorFlow available)
    if KERAS_AVAILABLE:
        st.subheader("Model Configuration")
        model_path = st.text_input("Model Path (optional)", value="model.h5")
        
        if st.button("Load Model"):
            if os.path.exists(model_path):
                try:
                    st.session_state.model = keras.models.load_model(model_path)
                    st.success("✅ Model loaded!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
            else:
                st.warning("⚠️ Model file not found")

# Sign classes
SIGN_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

def extract_keypoints(results):
    """Extract hand landmarks as keypoints"""
    if results.multi_hand_landmarks:
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints)
    return np.zeros(63)

def predict_sign(keypoints, model):
    """Predict sign language gesture"""
    if not KERAS_AVAILABLE:
        return "TensorFlow not installed", 0.0
    
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        keypoints = keypoints.reshape(1, -1)
        prediction = model.predict(keypoints, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        if class_idx < len(SIGN_CLASSES):
            return SIGN_CLASSES[class_idx], confidence
        return "Unknown", confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Main tabs
tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "📸 Image Upload", "ℹ️ About"])

with tab1:
    st.header("Live Camera Detection")
    st.info("📌 Note: Webcam access may not work on deployed apps. Use 'Image Upload' tab instead.")
    
    if st.session_state.hands is None:
        st.warning("⚠️ Please initialize the hand detector from the sidebar first!")

with tab2:
    st.header("Upload Image for Detection")
    
    if st.session_state.hands is None:
        st.warning("⚠️ Please initialize the hand detector from the sidebar first!")
    else:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image_rgb, use_container_width=True)
                
                with col2:
                    st.subheader("Detection Result")
                    
                    results = st.session_state.hands.process(image_rgb)
                    annotated = image_rgb.copy()
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                annotated,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                        
                        st.image(annotated, use_container_width=True)
                        
                        if KERAS_AVAILABLE and st.session_state.model:
                            keypoints = extract_keypoints(results)
                            sign, conf = predict_sign(keypoints, st.session_state.model)
                            st.success(f"Detected: **{sign}**")
                            st.info(f"Confidence: **{conf:.2%}**")
                        else:
                            st.success("✅ Hand detected!")
                    else:
                        st.warning("No hands detected in the image")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Indian Sign Language Detection System
    
    This application uses computer vision to detect and track hands in images.
    
    ### 🔧 Technologies Used
    - **MediaPipe**: Hand landmark detection
    - **Streamlit**: Interactive web interface
    - **OpenCV**: Image processing
    - **TensorFlow** (Optional): Deep learning for gesture classification
    
    ### 📖 How to Use
    1. **Initialize Detector**: Click "Initialize Hand Detector" in the sidebar
    2. **Upload Image**: Go to "Image Upload" tab and upload a hand gesture image
    3. **View Results**: See the detected hand landmarks and predictions
    
    ### 💡 Features
    - Detects 0-9 digits and A-Z letters
    - Adjustable confidence thresholds
    - Visual feedback with hand landmarks
    """)
    
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MediaPipe", "✅ Available")
        st.metric("OpenCV", "✅ Installed")
    with col2:
        if KERAS_AVAILABLE:
            st.metric("TensorFlow", "✅ Installed")
        else:
            st.metric("TensorFlow", "❌ Not Installed")
        
        if st.session_state.hands:
            st.metric("Hand Detector", "✅ Initialized")
        else:
            st.metric("Hand Detector", "❌ Not Initialized")

st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🤟 Indian Sign Language Detection System</p>
        <p>Powered by MediaPipe & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)