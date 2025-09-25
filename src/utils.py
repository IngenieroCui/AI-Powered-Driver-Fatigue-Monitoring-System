"""
Utility functions for driver fatigue detection system.
Contains mathematical calculations for EAR, MAR, and other metrics.
"""

import numpy as np
import math

def _to_px(landmark, img_w, img_h):
    return np.array([landmark.x * img_w, landmark.y * img_h], dtype=np.float32)

def _dist(a, b):
    return float(np.linalg.norm(a - b))

def calculate_EAR(landmarks, eye_indices, image_shape):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    eye_indices order: [outer(p1), upper1(p2), upper2(p3), inner(p4), lower1(p5), lower2(p6)]
    Distancias en píxeles para evitar distorsión por normalización.
    """
    try:
        if len(eye_indices) != 6:
            return 0.0
        img_h, img_w = image_shape[:2]
        pts = [_to_px(landmarks[i], img_w, img_h) for i in eye_indices]

        v1 = _dist(pts[1], pts[5])  # p2 - p6
        v2 = _dist(pts[2], pts[4])  # p3 - p5
        h  = _dist(pts[0], pts[3])  # p1 - p4

        if h <= 1e-6:
            return 0.0
        ear = (v1 + v2) / (2.0 * h)
        # clamp razonable
        return float(max(0.0, min(1.0, ear)))
    except Exception as e:
        print(f"EAR calculation error: {e}")
        return 0.0


def calculate_MAR(landmarks, mouth_indices, image_shape, distance_factor=1.0):
    """
    MAR = vertical_distance / horizontal_distance
    mouth_indices: 4 puntos [left, top, right, bottom]
    Distancias en píxeles para mayor precisión.
    """
    try:
        if len(mouth_indices) < 4:
            return 0.0
            
        img_h, img_w = image_shape[:2]
        pts = [_to_px(landmarks[i], img_w, img_h) for i in mouth_indices[:4]]
        
        # Altura de la boca
        mouth_height = _dist(pts[1], pts[3])
        # Ancho de la boca
        mouth_width = _dist(pts[0], pts[2])
        
        if mouth_width <= 1e-6:
            return 0.0
            
        mar = mouth_height / mouth_width
        # Compensar por distancia
        mar = mar * distance_factor
        return float(max(0.0, min(2.0, mar)))
        
    except Exception as e:
        print(f"MAR calculation error: {e}")
        return 0.0


def calculate_head_pose(landmarks, face_indices):
    """
    Calculate head pose estimation for tilt detection.
    
    Args:
        landmarks: Facial landmarks from mediapipe
        face_indices: List of indices for face boundary points
        
    Returns:
        dict: Dictionary containing pitch, yaw, roll angles
    """
    # Key facial points for head pose estimation
    nose_tip = np.array([landmarks[1].x, landmarks[1].y])
    chin = np.array([landmarks[175].x, landmarks[175].y])
    left_eye_corner = np.array([landmarks[33].x, landmarks[33].y])
    right_eye_corner = np.array([landmarks[263].x, landmarks[263].y])
    
    # Calculate head tilt (roll angle)
    eye_vector = right_eye_corner - left_eye_corner
    roll_angle = math.atan2(eye_vector[1], eye_vector[0]) * 180 / math.pi
    
    # Calculate vertical alignment (pitch estimation)
    face_height = np.linalg.norm(nose_tip - chin)
    eye_center = (left_eye_corner + right_eye_corner) / 2
    nose_eye_distance = np.linalg.norm(nose_tip - eye_center)
    
    # Simple pitch approximation
    pitch_ratio = nose_eye_distance / face_height if face_height > 0 else 0
    pitch_angle = (pitch_ratio - 0.3) * 180  # Normalized approximation
    
    # Horizontal alignment (yaw estimation)
    nose_center_x = nose_tip[0]
    face_center_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
    yaw_offset = nose_center_x - face_center_x
    yaw_angle = yaw_offset * 180  # Simplified yaw estimation
    
    return {
        'roll': roll_angle,
        'pitch': pitch_angle,
        'yaw': yaw_angle
    }


def moving_average(values, window_size=5):
    """
    Calculate moving average for smoothing metric values.
    
    Args:
        values: List of numeric values
        window_size: Size of the moving window
        
    Returns:
        float: Moving average value
    """
    if len(values) < window_size:
        return sum(values) / len(values) if values else 0
    
    return sum(values[-window_size:]) / window_size


def detect_blink_pattern(ear_history, threshold=0.25, consecutive_frames=3):
    """
    Detect blink patterns from EAR history.
    
    Args:
        ear_history: List of recent EAR values
        threshold: EAR threshold for closed eyes
        consecutive_frames: Number of consecutive frames below threshold
        
    Returns:
        bool: True if blink pattern detected
    """
    if len(ear_history) < consecutive_frames:
        return False
    
    recent_values = ear_history[-consecutive_frames:]
    return all(ear < threshold for ear in recent_values)


def detect_yawn_pattern(mar_history, baseline=0.15, peak_threshold=0.5, min_duration=15, distance_scale=1.0):
    """
    Detect yawn patterns based on MAR dynamics rather than static threshold.
    
    Args:
        mar_history: List of recent MAR values
        baseline: Baseline MAR for closed mouth
        peak_threshold: Minimum MAR value to consider as potential yawn
        min_duration: Minimum duration of elevated MAR to confirm yawn
        distance_scale: Scale factor to compensate for distance to camera
        
    Returns:
        dict: Dictionary with yawn detection results
    """
    if len(mar_history) < min_duration:
        return {'is_yawning': False, 'yawn_progress': 0}
    
    # More conservative adjustments based on distance
    adjusted_peak_threshold = peak_threshold * (1.0 + (distance_scale - 1.0) * 0.8)
    adjusted_baseline = baseline * (1.0 + (distance_scale - 1.0) * 0.4)
    
    # Clamp thresholds to reasonable ranges
    adjusted_peak_threshold = max(0.4, min(1.2, adjusted_peak_threshold))
    adjusted_baseline = max(0.1, min(0.3, adjusted_baseline))
    
    current_mar = mar_history[-1]
    recent_values = mar_history[-min_duration:]
    
    # Check for sustained elevated MAR (more strict)
    elevated_frames = sum(1 for mar in recent_values if mar > adjusted_peak_threshold)
    
    # Look for the characteristic yawn pattern with stricter criteria
    if len(mar_history) >= 25:  # Need more history for reliable detection
        window = mar_history[-25:]
        
        # Find peak and analyze pattern
        max_mar = max(window)
        max_index = window.index(max_mar)
        
        # Check for proper yawn pattern: gradual rise -> peak -> gradual decline
        has_gradual_rise = False
        has_sustained_peak = False
        
        # Verify gradual rise before peak
        if max_index >= 8:  # Need enough frames before peak
            rise_section = window[:max_index]
            if len(rise_section) >= 5:
                # Check if there's a consistent rise
                rise_trend = sum(1 for i in range(1, len(rise_section)) 
                               if rise_section[i] > rise_section[i-1])
                has_gradual_rise = rise_trend >= len(rise_section) * 0.6
        
        # Check for sustained peak
        peak_frames = sum(1 for mar in recent_values[-min_duration//2:] 
                         if mar > adjusted_peak_threshold * 0.8)
        has_sustained_peak = peak_frames >= min_duration // 4
        
        # Final yawn detection with all criteria
        is_yawning = (has_gradual_rise and 
                     has_sustained_peak and 
                     max_mar > adjusted_peak_threshold and
                     elevated_frames >= min_duration // 3 and
                     max_mar > adjusted_baseline + 0.2)
        
        yawn_progress = min(1.0, elevated_frames / (min_duration // 2))
        
        return {
            'is_yawning': is_yawning,
            'yawn_progress': yawn_progress,
            'peak_mar': max_mar,
            'elevated_frames': elevated_frames,
            'adjusted_threshold': adjusted_peak_threshold,
            'distance_scale': distance_scale,
            'has_rise': has_gradual_rise,
            'has_sustained': has_sustained_peak
        }
    
    return {'is_yawning': False, 'yawn_progress': 0, 'distance_scale': distance_scale}


def detect_speech_vs_yawn(mar_history, fps=30):
    """
    Distingue entre habla y bostezo analizando patrones temporales específicos.
    
    HABLA características:
    - Frecuencia alta (2-8 Hz): movimientos rápidos de boca
    - MAR oscilante: sube y baja constantemente
    - Duración corta de picos: <0.5 segundos
    - Variación alta: desviación estándar elevada
    
    BOSTEZO características:
    - Frecuencia baja (<1 Hz): movimiento lento
    - MAR sostenido: mantiene valor alto 1-3 segundos
    - Patrón gradual: subida lenta -> mantenimiento -> bajada lenta
    - Variación baja: cambios suaves
    
    Args:
        mar_history: Lista de valores MAR recientes (al menos 60 frames)
        fps: Frames por segundo de la cámara
        
    Returns:
        dict: Resultado del análisis con probabilidades
    """
    if len(mar_history) < 30:
        return {'is_speech': False, 'is_yawn': False, 'confidence': 0}
    
    # Usar ventana de análisis de 2 segundos
    analysis_window = min(60, fps * 2)
    recent_mar = mar_history[-analysis_window:]
    
    # === ANÁLISIS DE FRECUENCIA ===
    # Contar cambios de dirección (picos y valles)
    direction_changes = 0
    for i in range(2, len(recent_mar)):
        # Detectar cambio de tendencia
        trend_prev = recent_mar[i-1] - recent_mar[i-2]
        trend_curr = recent_mar[i] - recent_mar[i-1]
        if (trend_prev > 0 > trend_curr) or (trend_prev < 0 < trend_curr):
            direction_changes += 1
    
    # Frecuencia de cambios por segundo
    frequency = direction_changes / (len(recent_mar) / fps)
    
    # === ANÁLISIS DE DURACIÓN DE PICOS ===
    threshold_high = np.mean(recent_mar) + np.std(recent_mar) * 0.5
    consecutive_high = 0
    max_consecutive_high = 0
    high_episodes = []
    
    for mar in recent_mar:
        if mar > threshold_high:
            consecutive_high += 1
        else:
            if consecutive_high > 0:
                high_episodes.append(consecutive_high)
                max_consecutive_high = max(max_consecutive_high, consecutive_high)
            consecutive_high = 0
    
    # Duración promedio de episodios altos (en segundos)
    avg_high_duration = np.mean(high_episodes) / fps if high_episodes else 0
    max_high_duration = max_consecutive_high / fps
    
    # === ANÁLISIS DE VARIABILIDAD ===
    mar_std = np.std(recent_mar)
    mar_range = np.max(recent_mar) - np.min(recent_mar)
    
    # === ANÁLISIS DE GRADIENTE (suavidad del cambio) ===
    gradients = np.diff(recent_mar)
    avg_gradient = np.mean(np.abs(gradients))
    max_gradient = np.max(np.abs(gradients))
    
    # === CRITERIOS DE DECISIÓN ===
    
    # Indicadores de HABLA:
    speech_indicators = {
        'high_frequency': frequency > 3.0,           # >3 cambios/seg
        'short_peaks': avg_high_duration < 0.4,     # Picos <0.4seg  
        'high_variability': mar_std > 0.08,         # Alta variación
        'rapid_changes': max_gradient > 0.15        # Cambios rápidos
    }
    
    # Indicadores de BOSTEZO:
    yawn_indicators = {
        'low_frequency': frequency < 1.5,           # <1.5 cambios/seg
        'long_peaks': max_high_duration > 1.0,     # Pico >1seg
        'sustained_high': max_consecutive_high > fps * 0.8,  # >0.8seg alto
        'gradual_changes': avg_gradient < 0.05     # Cambios graduales
    }
    
    # Calcular scores
    speech_score = sum(speech_indicators.values()) / len(speech_indicators)
    yawn_score = sum(yawn_indicators.values()) / len(yawn_indicators)
    
    # Decisión final con umbral de confianza
    confidence_threshold = 0.5
    
    is_speech = speech_score > confidence_threshold and speech_score > yawn_score
    is_yawn = yawn_score > confidence_threshold and yawn_score > speech_score
    
    confidence = max(speech_score, yawn_score) if (is_speech or is_yawn) else 0
    
    return {
        'is_speech': is_speech,
        'is_yawn': is_yawn,
        'confidence': confidence,
        'speech_score': speech_score,
        'yawn_score': yawn_score,
        'frequency': frequency,
        'avg_high_duration': avg_high_duration,
        'max_high_duration': max_high_duration,
        'variability': mar_std,
        'indicators': {
            'speech': speech_indicators,
            'yawn': yawn_indicators
        }
    }


def detect_microsleep_pattern(ear_history, current_ear, threshold=0.26, min_duration=30, max_blink_duration=8):
    """
    Detect microsleep episodes by analyzing sustained low EAR values.
    Uses both current EAR and history for accurate detection.
    
    Args:
        ear_history: List of recent EAR values (smoothed)
        current_ear: Current frame EAR value (immediate)
        threshold: EAR threshold for closed eyes
        min_duration: Minimum frames for microsleep detection
        max_blink_duration: Maximum frames for normal blink
        
    Returns:
        dict: Dictionary with microsleep detection results
    """
    if len(ear_history) < min_duration:
        return {'is_microsleep': False, 'duration': 0, 'severity': 0, 'current_ear': current_ear}
    
    # Usar EAR inmediato para conteo de frames consecutivos
    consecutive_low = 0
    test_history = ear_history + [current_ear]  # Incluir frame actual
    
    for ear in reversed(test_history):
        if ear < threshold:
            consecutive_low += 1
        else:
            break
    
    # Analizar ventana reciente para patrones
    recent_window = ear_history[-min_duration:]
    low_ear_frames = sum(1 for ear in recent_window if ear < threshold)
    
    # Incluir valor actual en análisis
    if current_ear < threshold:
        low_ear_frames += 1
    
    # Calcular porcentaje de tiempo con EAR bajo
    total_frames = len(recent_window) + 1  # +1 por frame actual
    low_ear_percentage = low_ear_frames / total_frames
    
    # Detectar microsueño: EAR bajo sostenido más allá de parpadeo normal
    is_microsleep = (consecutive_low > max_blink_duration and 
                    consecutive_low >= min_duration // 3) or \
                   (low_ear_percentage > 0.7 and consecutive_low > max_blink_duration)
    
    # Calcular severidad basada en duración y consistencia
    severity = min(1.0, (consecutive_low - max_blink_duration) / min_duration)
    
    return {
        'is_microsleep': is_microsleep,
        'duration': consecutive_low,
        'severity': severity,
        'low_ear_percentage': low_ear_percentage,
        'current_ear': current_ear,
        'threshold_used': threshold
    }


def detect_sudden_head_movement(current_pose, previous_pose, threshold=15):
    """
    Detect sudden head movements that might cause false positives.
    
    Args:
        current_pose: Current head pose dictionary
        previous_pose: Previous head pose dictionary  
        threshold: Movement threshold in degrees per frame
        
    Returns:
        bool: True if sudden movement detected
    """
    if not previous_pose or not current_pose:
        return False
    
    # Calculate movement deltas
    roll_delta = abs(current_pose['roll'] - previous_pose['roll'])
    pitch_delta = abs(current_pose['pitch'] - previous_pose['pitch'])
    yaw_delta = abs(current_pose['yaw'] - previous_pose['yaw'])
    
    # Check if any axis shows sudden movement
    sudden_movement = (roll_delta > threshold or 
                      pitch_delta > threshold or 
                      yaw_delta > threshold)
    
    return sudden_movement


def calculate_face_distance_scale(landmarks):
    """
    Calculate relative face distance based on facial feature sizes.
    Used to normalize MAR calculations for different distances to camera.
    
    Args:
        landmarks: Facial landmarks from mediapipe
        
    Returns:
        float: Distance scale factor (1.0 = normal, >1.0 = closer, <1.0 = farther)
    """
    try:
        # Use eye distance as reference for face size
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        eye_distance = np.linalg.norm(right_eye_outer - left_eye_outer)
        
        # Use nose to chin distance as another reference
        nose_tip = np.array([landmarks[1].x, landmarks[1].y])
        chin = np.array([landmarks[175].x, landmarks[175].y])
        face_height = np.linalg.norm(nose_tip - chin)
        
        # Calculate combined scale - larger values indicate closer to camera
        # Normalize based on typical values (adjust based on testing)
        eye_scale = eye_distance / 0.15  # Typical eye distance in normalized coords
        height_scale = face_height / 0.25  # Typical face height portion
        
        # Combine both measurements with weights
        distance_scale = (eye_scale * 0.6 + height_scale * 0.4)
        
        # Clamp to reasonable range
        distance_scale = max(0.3, min(2.0, distance_scale))
        
        return distance_scale
    
    except Exception as e:
        return 1.0  # Default scale if calculation fails


def format_metrics_display(metrics):
    """
    Format metrics for display on screen.
    
    Args:
        metrics: Dictionary containing various metrics
        
    Returns:
        list: List of formatted strings for display
    """
    display_lines = []
    
    if 'ear' in metrics:
        display_lines.append(f"EAR: {metrics['ear']:.3f}")
    
    if 'mar' in metrics:
        display_lines.append(f"MAR: {metrics['mar']:.3f}")
    
    if 'head_pose' in metrics:
        pose = metrics['head_pose']
        display_lines.append(f"Head Tilt: {pose['roll']:.1f}°")
    
    if 'status' in metrics:
        display_lines.append(f"Status: {metrics['status']}")
    
    return display_lines
