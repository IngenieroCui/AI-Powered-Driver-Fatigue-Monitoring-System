"""
Configuration settings for the Driver Fatigue Detection System.
Contains all thresholds, camera settings, and landmark indices.
"""

import os

# ===== CAMERA SETTINGS =====
CAMERA_INDEX = 0  # Default camera (0 for built-in webcam)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 60  # Increased for better temporal precision

# === CAMERA QUALITY PROFILES ===
# Configuración optimizada para diferentes tipos de cámara
CAMERA_PROFILES = {
    '720p_30fps': {
        'resolution': (1280, 720),
        'fps': 30,
        'quality_factor': 1.0,
        'detection_confidence': 0.5,
        'tracking_confidence': 0.5
    },
    '1080p_30fps': {
        'resolution': (1920, 1080), 
        'fps': 30,
        'quality_factor': 1.2,
        'detection_confidence': 0.6,
        'tracking_confidence': 0.6
    },
    '1080p_60fps': {
        'resolution': (1920, 1080),
        'fps': 60,
        'quality_factor': 1.3,
        'detection_confidence': 0.6,
        'tracking_confidence': 0.6
    },
    '4k_60fps': {
        'resolution': (3840, 2160),
        'fps': 60,
        'quality_factor': 1.5,
        'detection_confidence': 0.7,
        'tracking_confidence': 0.7
    }
}

# Perfil de cámara activo (cambiar según tu hardware)
CURRENT_CAMERA_PROFILE = '1080p_30fps'
ACTIVE_PROFILE = CAMERA_PROFILES[CURRENT_CAMERA_PROFILE]

# ===== DETECTION THRESHOLDS (ADAPTIVE) =====
# Base thresholds
BASE_EAR_THRESHOLD = 0.26
BASE_MAR_THRESHOLD = 0.6

# Thresholds ajustados por calidad de cámara
EAR_THRESHOLD = BASE_EAR_THRESHOLD  # EAR se mantiene constante, es independiente de resolución
MAR_THRESHOLD = BASE_MAR_THRESHOLD / ACTIVE_PROFILE['quality_factor']  # MAR mejora con mayor resolución

def set_camera_profile(profile_name):
    """
    Cambia dinámicamente el perfil de cámara y ajusta los parámetros.
    
    Args:
        profile_name: Nombre del perfil ('720p_30fps', '1080p_30fps', '1080p_60fps', '4k_60fps')
    
    Returns:
        dict: Configuración del perfil aplicado
    """
    global CURRENT_CAMERA_PROFILE, ACTIVE_PROFILE, EAR_THRESHOLD, MAR_THRESHOLD
    
    if profile_name in CAMERA_PROFILES:
        CURRENT_CAMERA_PROFILE = profile_name
        ACTIVE_PROFILE = CAMERA_PROFILES[profile_name]
        
        # Reajustar thresholds
        MAR_THRESHOLD = BASE_MAR_THRESHOLD / ACTIVE_PROFILE['quality_factor']
        
        print(f"📹 Perfil de cámara cambiado a: {profile_name}")
        print(f"   Resolución: {ACTIVE_PROFILE['resolution']}")
        print(f"   FPS: {ACTIVE_PROFILE['fps']}")
        print(f"   MAR Threshold ajustado: {MAR_THRESHOLD:.3f}")
        
        return ACTIVE_PROFILE
    else:
        print(f"❌ Perfil '{profile_name}' no encontrado. Perfiles disponibles:")
        for name in CAMERA_PROFILES.keys():
            print(f"   - {name}")
        return None

def get_recommended_profile():
    """
    Recomienda un perfil de cámara basado en capacidades del sistema.
    
    Returns:
        str: Nombre del perfil recomendado
    """
    import psutil
    import platform
    
    # Detectar capacidades básicas del sistema
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Heurística simple para recomendación
    if cpu_count >= 8 and memory_gb >= 16:
        return '1080p_60fps'  # Sistema potente
    elif cpu_count >= 4 and memory_gb >= 8:
        return '1080p_30fps'  # Sistema estándar (RECOMENDADO)
    else:
        return '720p_30fps'   # Sistema básico
    
def auto_configure_camera():
    """
    Configura automáticamente el mejor perfil para el hardware detectado.
    
    Returns:
        dict: Configuración aplicada
    """
    recommended = get_recommended_profile()
    print(f"🔍 Sistema detectado - Recomendación: {recommended}")
    return set_camera_profile(recommended)
HEAD_TILT_THRESHOLD = 25  # Head tilt angle threshold (degrees, increased to reduce false positives)

# ===== FRAME ANALYSIS SETTINGS =====
CONSECUTIVE_FRAMES_DROWSY = 15  # Frames to confirm drowsiness (reduced for faster response)
CONSECUTIVE_FRAMES_YAWN = 8     # Frames to confirm yawning (reduced, now using dynamic detection)
CONSECUTIVE_FRAMES_MICROSLEEP = 45  # Frames para microsueño (más estricto)
MOVING_AVERAGE_WINDOW = 7       # Window size for smoothing metrics (increased for better stability)

# ===== MEDIAPIPE FACE MESH SETTINGS =====
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
REFINE_LANDMARKS = True

# ===== FACIAL LANDMARK INDICES =====
# MediaPipe Face Mesh landmark indices for specific facial features

# Right Eye landmarks (6 points for EAR calculation)
# Order: [outer, upper1, upper2, inner, lower1, lower2]
RIGHT_EYE = [263, 386, 385, 362, 380, 374]  # (386,374) y (385,380) son pares verticales

# Left Eye landmarks (6 points for EAR calculation)
# Order: [outer, upper1, upper2, inner, lower1, lower2]
LEFT_EYE = [33, 159, 158, 133, 153, 145]    # (159,145) y (158,153) son pares verticales

# Mouth landmarks para MAR (4 puntos)
# Orden esperada en utils.calculate_MAR: width entre [0] y [2], height entre [1] y [3]
# Esquina-izq=61, esquina-der=291, labio-sup=13, labio-inf=14
MOUTH = [61, 13, 291, 14]

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
    'microsleep': 'MICROSLEEP DETECTED!',
    'eyes_closed': 'EYES CLOSED!',
    'head_tilt': 'HEAD TILT DETECTED!',
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
