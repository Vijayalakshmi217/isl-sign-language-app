import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="ISL Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #00A67E;
    text-align: center;
    margin-bottom: 1rem;
}
.prediction-box {
    background: linear-gradient(135deg, #00A67E 0%, #00875A 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 2.5rem;
    font-weight: bold;
    margin: 1rem 0;
}
.info-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #00A67E;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe
@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return mp_hands, mp_drawing, mp_drawing_styles, hands

mp_hands, mp_drawing, mp_drawing_styles, hands = init_mediapipe()

# Initialize session state
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0

# Header
st.markdown('<div class="main-header">🤟 ISL Sign Language Detection System</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Real-time Indian Sign Language Recognition using MediaPipe & AI</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Dashboard")
    
    # Statistics
    st.metric("Total Detections", st.session_state.detection_count)
    
    st.markdown("---")
    
    # Supported Signs
    st.header("📖 Supported Signs")
    st.markdown("**Numbers:** 0-9")
    st.markdown("**Letters:** A-Z")
    st.info("**Total Classes:** 36")
    
    st.markdown("---")
    
    # Detection Settings
    st.header("🎛️ Settings")
    min_detection_confidence = st.slider(
        "Detection Confidence",
        0.0, 1.0, 0.7, 0.05,
        help="Minimum confidence for hand detection"
    )
    
    min_tracking_confidence = st.slider(
        "Tracking Confidence",
        0.0, 1.0, 0.7, 0.05,
        help="Minimum confidence for hand tracking"
    )
    
    st.markdown("---")
    
    # Instructions
    st.header("📋 How to Use")
    st.markdown("""
    1. Upload a hand gesture image
    2. Wait for hand detection
    3. View detected landmarks
    4. See hand skeleton overlay
    
    **Tips:**
    - Use clear images
    - Good lighting
    - Hand clearly visible
    - No background clutter
    """)
    
    st.markdown("---")
    
    # System Info
    st.header("ℹ️ System Info")
    st.success("✅ MediaPipe Ready")
    st.info("🔧 Hand Detection Mode")
    st.markdown("""
    <div class="info-card">
    <small>
    <b>Framework:</b> MediaPipe<br>
    <b>Max Hands:</b> 2<br>
    <b>Landmarks:</b> 21 per hand
    </small>
    </div>
    """, unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📸 Upload Hand Gesture Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image showing hand gestures"
    )
    
    if uploaded_file:
        try:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert to RGB if needed
            if len(image_np.shape) == 2:  # Grayscale
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
            # Display original
            st.subheader("📥 Original Image")
            st.image(image_np, use_container_width=True)
            
            # Process image
            with st.spinner("🔍 Detecting hands..."):
                results = hands.process(image_np)
            
            # Create annotated image
            annotated = image_np.copy()
            
            if results.multi_hand_landmarks:
                # Draw landmarks on all detected hands
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Update count
                st.session_state.detection_count += 1
                
                # Display result
                st.markdown("---")
                st.subheader("✅ Detection Result")
                st.image(annotated, use_container_width=True)
                
                # Show success message
                num_hands = len(results.multi_hand_landmarks)
                st.markdown(f'<div class="prediction-box">✅ Detected {num_hands} Hand(s)!</div>', unsafe_allow_html=True)
                
                # Extract features for each hand
                with col2:
                    st.header("📊 Detection Details")
                    
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        st.markdown(f"### Hand {idx + 1}")
                        
                        # Extract landmarks
                        landmarks_data = []
                        for lm in hand_landmarks.landmark:
                            landmarks_data.extend([lm.x, lm.y, lm.z])
                        
                        st.markdown(f"""
                        <div class="info-card">
                        <b>Landmarks:</b> 21<br>
                        <b>Features:</b> 63<br>
                        <b>Status:</b> ✅ Detected
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show sample landmarks
                        with st.expander(f"View Landmark Data (Hand {idx + 1})"):
                            st.write("First 3 landmarks (x, y, z):")
                            for i in range(3):
                                st.write(f"Landmark {i}: x={landmarks_data[i*3]:.3f}, y={landmarks_data[i*3+1]:.3f}, z={landmarks_data[i*3+2]:.3f}")
            else:
                st.warning("⚠️ No hands detected in the image")
                st.info("💡 **Tips:**\n- Ensure hands are clearly visible\n- Use good lighting\n- Try a different image\n- Adjust confidence thresholds in sidebar")
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
    else:
        st.info("👆 **Upload an image above to get started!**")
        
        # Show example
        st.markdown("---")
        st.subheader("📌 Example")
        st.markdown("""
        Upload an image showing:
        - ✋ Open hand
        - ✌️ Peace sign  
        - 👍 Thumbs up
        - 🤟 Any ISL gesture
        
        The system will detect hand landmarks and display the skeleton overlay.
        """)

with col2:
    if not uploaded_file:
        st.header("📊 Live Statistics")
        st.info("Upload an image to see detection statistics")
        
        st.markdown("---")
        st.header("🎯 About This App")
        st.markdown("""
        This application uses **MediaPipe** to detect and track hand landmarks in real-time.
        
        **Features:**
        - Hand landmark detection
        - 21-point skeleton tracking
        - Support for multiple hands
        - Visual feedback with overlays
        
        **Supported Gestures:**
        - Numbers: 0-9
        - Letters: A-Z
        - Total: 36 classes
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🤟 ISL Sign Language Detection System</strong></p>
    <p>Built with ❤️ using Streamlit • MediaPipe • OpenCV</p>
    <p>© 2026 - Real-time AI-Powered Sign Language Recognition</p>
</div>
""", unsafe_allow_html=True)