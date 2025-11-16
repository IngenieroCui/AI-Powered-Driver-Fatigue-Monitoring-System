"""
Configuration settings for the Driver Fatigue Detection System.
Contains all thresholds, camera settings, and landmark indices.
"""

import os

# ===== CAMERA SETTINGS =====
CAMERA_INDEX = 0  # Default camera (0 for built-in webcam)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# ===== DETECTION THRESHOLDS =====
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for drowsiness
MAR_THRESHOLD = 0.5   # Mouth Aspect Ratio threshold for yawning
HEAD_TILT_THRESHOLD = 20  # Head tilt angle threshold (degrees)

# ===== FRAME ANALYSIS SETTINGS =====
CONSECUTIVE_FRAMES_DROWSY = 20  # Frames to confirm drowsiness
CONSECUTIVE_FRAMES_YAWN = 10    # Frames to confirm yawning
MOVING_AVERAGE_WINDOW = 5       # Window size for smoothing metrics

# ===== MEDIAPIPE FACE MESH SETTINGS =====
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
REFINE_LANDMARKS = True

# ===== FACIAL LANDMARK INDICES =====
# MediaPipe Face Mesh landmark indices for specific facial features

# Right Eye landmarks (6 points for EAR calculation)
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Left Eye landmarks (6 points for EAR calculation)
LEFT_EYE = [33, 158, 160, 144, 153, 145]

# Mouth landmarks (8 points for MAR calculation)
MOUTH = [61, 84, 17, 314, 405, 320, 307, 375]

# Head pose estimation landmarks
HEAD_POSE_POINTS = {
    'nose_tip': 1,
    'chin': 175,
    'left_eye_corner': 33,
    'right_eye_corner': 263,
    'left_mouth_corner': 61,
    'right_mouth_corner': 291
}

# ===== ALERT SETTINGS =====
ENABLE_AUDIO_ALERTS = True
ENABLE_VISUAL_ALERTS = True
ALERT_SOUND_PATH = "assets/alert_sound.wav"  # Path to alert sound file

# Alert colors (BGR format for OpenCV)
COLOR_NORMAL = (0, 255, 0)      # Green
COLOR_WARNING = (0, 255, 255)   # Yellow
COLOR_DANGER = (0, 0, 255)      # Red
COLOR_TEXT = (255, 255, 255)    # White

# ===== LOGGING SETTINGS =====
ENABLE_LOGGING = True
LOG_DIRECTORY = "data/logs"
LOG_FILENAME = "fatigue_detection_log.csv"
LOG_INTERVAL = 1  # Log every N seconds

# ===== FILE PATHS =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# ===== DISPLAY SETTINGS =====
WINDOW_NAME = "Driver Fatigue Monitoring System"
FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_PADDING = 30

# ===== DEBUG SETTINGS =====
DEBUG_MODE = False  # Set to True for detailed console output
SHOW_LANDMARKS = False  # Set to True to visualize facial landmarks
SAVE_DEBUG_IMAGES = False  # Set to True to save frames for analysis

# ===== PERFORMANCE SETTINGS =====
SKIP_FRAMES = 0  # Skip N frames between detections (0 = process every frame)
RESIZE_FRAME = False  # Resize frame for faster processing
RESIZE_WIDTH = 320
RESIZE_HEIGHT = 240

# ===== CALIBRATION SETTINGS =====
CALIBRATION_MODE = False  # Set to True for initial user calibration
CALIBRATION_FRAMES = 100   # Number of frames for calibration
AUTO_ADJUST_THRESHOLD = False  # Automatically adjust thresholds based on user

# ===== ALERT MESSAGES =====
MESSAGES = {
    'normal': 'NORMAL',
    'warning': 'WARNING',
    'drowsy': 'DROWSY DETECTED!',
    'yawning': 'YAWNING DETECTED!',
    'head_tilt': 'HEAD TILT DETECTED!',
    'eyes_closed': 'EYES CLOSED!',
    'multiple_alerts': 'MULTIPLE FATIGUE SIGNS!'
}

# ===== SYSTEM REQUIREMENTS =====
MIN_PYTHON_VERSION = (3, 8)
REQUIRED_PACKAGES = [
    'opencv-python>=4.5.0',
    'mediapipe>=0.8.0',
    'numpy>=1.19.0',
    'pandas>=1.3.0'
]

def get_log_file_path():
    """Get the full path to the log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    return os.path.join(LOGS_DIR, LOG_FILENAME)

def get_model_file_path(filename):
    """Get the full path to a model file."""
    return os.path.join(MODELS_DIR, filename)

def validate_configuration():
    """Validate configuration settings and create necessary directories."""
    # Create directories if they don't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Validate thresholds
    assert 0 < EAR_THRESHOLD < 1, "EAR_THRESHOLD must be between 0 and 1"
    assert 0 < MAR_THRESHOLD < 2, "MAR_THRESHOLD must be between 0 and 2"
    assert 0 < HEAD_TILT_THRESHOLD < 90, "HEAD_TILT_THRESHOLD must be between 0 and 90"
    
    return True
