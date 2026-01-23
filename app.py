import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import wandb
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ISL Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'hands' not in st.session_state:
    st.session_state.hands = None

# Title and description
st.title("🤟 Indian Sign Language Detection System")
st.markdown("Real-time hand gesture recognition using MediaPipe and Deep Learning")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model loading
    st.subheader("Model Configuration")
    model_path = st.text_input("Model Path", value="model.h5")
    
    if st.button("Load Model"):
        try:
            st.session_state.model = keras.models.load_model(model_path)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    # MediaPipe settings
    st.subheader("Detection Settings")
    min_detection_confidence = st.slider(
        "Min Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    min_tracking_confidence = st.slider(
        "Min Tracking Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Initialize hands detector
    if st.button("Initialize Detector"):
        st.session_state.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        st.success("✅ Hand detector initialized!")

# Sign language classes (customize based on your model)
SIGN_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

def extract_keypoints(results):
    """Extract hand landmarks as keypoints"""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            return np.array(keypoints)
    return np.zeros(63)  # 21 landmarks * 3 coordinates

def preprocess_keypoints(keypoints, target_shape=(63,)):
    """Preprocess keypoints for model input"""
    if len(keypoints) < target_shape[0]:
        keypoints = np.pad(keypoints, (0, target_shape[0] - len(keypoints)))
    elif len(keypoints) > target_shape[0]:
        keypoints = keypoints[:target_shape[0]]
    return keypoints.reshape(1, -1)

def predict_sign(keypoints, model):
    """Predict sign language gesture"""
    if model is None:
        return None, 0.0
    
    try:
        processed = preprocess_keypoints(keypoints)
        prediction = model.predict(processed, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        if class_idx < len(SIGN_CLASSES):
            return SIGN_CLASSES[class_idx], confidence
        return "Unknown", confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0.0

# Main application tabs
tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "📸 Image Upload", "ℹ️ About"])

with tab1:
    st.header("Live Camera Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        run_detection = st.checkbox("Start Detection")
        frame_placeholder = st.empty()
        
    with col2:
        st.subheader("Detection Results")
        result_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
    if run_detection:
        if st.session_state.hands is None:
            st.warning("⚠️ Please initialize the hand detector from the sidebar first!")
        elif st.session_state.model is None:
            st.warning("⚠️ Please load the model from the sidebar first!")
        else:
            cap = cv2.VideoCapture(0)
            
            stop_button = st.button("Stop Detection")
            
            while run_detection and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                # Convert BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Process with MediaPipe
                results = st.session_state.hands.process(image)
                
                # Draw landmarks
                image.flags.writeable = True
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Extract and predict
                    keypoints = extract_keypoints(results)
                    sign, confidence = predict_sign(keypoints, st.session_state.model)
                    
                    if sign:
                        # Display prediction on image
                        cv2.putText(
                            image,
                            f"{sign}: {confidence:.2f}",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 255, 0),
                            3
                        )
                        
                        # Update results
                        result_placeholder.metric("Detected Sign", sign)
                        confidence_placeholder.progress(float(confidence))
                
                # Display frame
                frame_placeholder.image(image, channels="RGB", use_container_width=True)
            
            cap.release()

with tab2:
    st.header("Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_container_width=True)
        
        with col2:
            st.subheader("Detection Result")
            
            if st.session_state.hands and st.session_state.model:
                # Process image
                results = st.session_state.hands.process(image_rgb)
                
                # Draw landmarks
                annotated_image = image_rgb.copy()
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Predict
                    keypoints = extract_keypoints(results)
                    sign, confidence = predict_sign(keypoints, st.session_state.model)
                    
                    st.image(annotated_image, use_container_width=True)
                    st.success(f"Detected Sign: **{sign}**")
                    st.info(f"Confidence: **{confidence:.2%}**")
                else:
                    st.warning("No hands detected in the image")
            else:
                st.warning("Please initialize detector and load model first")

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application detects Indian Sign Language (ISL) gestures in real-time using:
    - **MediaPipe** for hand landmark detection
    - **TensorFlow/Keras** for gesture classification
    - **Streamlit** for the web interface
    
    ### 🔧 How to Use
    1. **Load Model**: Enter your model path in the sidebar and click "Load Model"
    2. **Initialize Detector**: Configure detection settings and click "Initialize Detector"
    3. **Start Detection**: Use the "Live Detection" tab for real-time recognition
    4. **Upload Images**: Use the "Image Upload" tab for static image analysis
    
    ### 📊 Features
    - Real-time hand tracking with MediaPipe
    - Support for multiple hand detection
    - Adjustable confidence thresholds
    - Image upload capability
    - Visual feedback with landmarks
    
    ### 🛠️ Technical Stack
    - Python 3.11
    - Streamlit
    - MediaPipe
    - TensorFlow/Keras
    - OpenCV
    - NumPy
    
    ### 📝 Notes
    - Ensure good lighting for better detection
    - Keep hands clearly visible to the camera
    - Model accuracy depends on training data quality
    """)
    
    st.info("💡 Tip: Adjust the confidence thresholds in the sidebar for better results")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ for Indian Sign Language Recognition</p>
        <p>Powered by MediaPipe, TensorFlow, and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)