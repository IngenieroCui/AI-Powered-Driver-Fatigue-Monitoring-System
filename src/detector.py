"""
Driver Fatigue Detection System - Main Detection Logic
Handles face detection, landmark extraction, and fatigue analysis.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime

from .config import *
from .utils import (
    calculate_EAR, 
    calculate_MAR, 
    calculate_head_pose,
    calculate_face_distance_scale,
    moving_average,
    detect_blink_pattern,
    detect_yawn_pattern,
    detect_microsleep_pattern,
    detect_sudden_head_movement,
    detect_speech_vs_yawn,
    format_metrics_display
)


class FatigueDetector:
    """Main class for driver fatigue detection using computer vision."""
    
    def __init__(self):
        """Initialize the fatigue detector with MediaPipe and tracking variables."""
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Face mesh model
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=REFINE_LANDMARKS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        # Tracking variables
        self.ear_history = []
        self.mar_history = []
        self.drowsy_frame_count = 0
        self.yawn_frame_count = 0
        self.last_log_time = 0
        
        # Variables for dynamic detection
        self.mar_baseline = 0.2  # Baseline MAR for closed mouth
        self.mar_peak_detected = False
        self.yawn_start_time = None
        self.previous_mar = 0
        
        # Head movement tracking for filtering false positives
        self.previous_head_pose = {'roll': 0, 'pitch': 0, 'yaw': 0}
        self.head_movement_threshold = 15  # degrees per frame for sudden movement
        self.head_movement_frames = 0
        
        # Alert states
        self.is_drowsy = False
        self.is_yawning = False
        self.is_head_tilted = False
        
        # Initialize logging
        if ENABLE_LOGGING:
            self._init_logging()
    
    def _init_logging(self):
        """Initialize CSV logging for fatigue events."""
        log_file = get_log_file_path()
        
        # Create log file with headers if it doesn't exist
        try:
            with open(log_file, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'ear_right', 'ear_left', 'ear_avg', 
                    'mar', 'head_roll', 'head_pitch', 'head_yaw',
                    'drowsy_alert', 'yawn_alert', 'tilt_alert', 'status'
                ])
        except FileExistsError:
            # File already exists, which is fine
            pass
    
    def detect_fatigue(self, frame):
        """
        Main fatigue detection method.
        
        Args:
            frame: Input video frame from camera
            
        Returns:
            tuple: (processed_frame, metrics_dict)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize metrics
        metrics = {
            'ear_right': 0,
            'ear_left': 0,
            'ear_avg': 0,
            'mar': 0,
            'head_pose': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'status': 'NO_FACE',
            'alerts': []
        }
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Calculate metrics
                metrics = self._calculate_all_metrics(landmarks, frame.shape)
                
                # Detect fatigue patterns
                self._analyze_fatigue_patterns(metrics)
                
                # Update frame with visual feedback
                frame = self._draw_results(frame, metrics)
                
                # Log data if enabled
                if ENABLE_LOGGING:
                    self._log_metrics(metrics)
        
        else:
            # No face detected
            cv2.putText(frame, "No face detected", 
                       (50, 50), FONT_FACE, FONT_SCALE, COLOR_WARNING, FONT_THICKNESS)
        
        return frame, metrics
    
    def _calculate_all_metrics(self, landmarks, image_shape):
        """Calculate all fatigue detection metrics."""
        # Calculate face distance scale for normalization
        distance_scale = calculate_face_distance_scale(landmarks)
        
        # Calculate EAR for both eyes using pixel coordinates
        ear_right = calculate_EAR(landmarks, RIGHT_EYE, image_shape)
        ear_left = calculate_EAR(landmarks, LEFT_EYE, image_shape)
        ear_avg = (ear_right + ear_left) / 2.0
        
        # Calculate MAR using pixel coordinates
        mar = calculate_MAR(landmarks, MOUTH, image_shape, distance_scale)
        
        # Calculate head pose
        head_pose = calculate_head_pose(landmarks, HEAD_POSE_POINTS)
        
        # Update history for smoothing
        self.ear_history.append(ear_avg)
        self.mar_history.append(mar)
        
        # Keep only recent history
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)
        if len(self.mar_history) > 30:
            self.mar_history.pop(0)
        
        # Calculate smoothed values
        ear_smoothed = moving_average(self.ear_history, MOVING_AVERAGE_WINDOW)
        mar_smoothed = moving_average(self.mar_history, MOVING_AVERAGE_WINDOW)
        
        return {
            'ear_right': ear_right,
            'ear_left': ear_left,
            'ear_avg': ear_avg,
            'ear_smoothed': ear_smoothed,
            'mar': mar,
            'mar_smoothed': mar_smoothed,
            'head_pose': head_pose,
            'distance_scale': distance_scale,
            'status': 'ANALYZING',
            'alerts': []
        }
    
    def _analyze_fatigue_patterns(self, metrics):
        """Analyze metrics to detect fatigue patterns."""
        alerts = []
        
        # Usar EAR INMEDIATO para contadores (no suavizado)
        current_ear = metrics['ear_avg']  # EAR inmediato del frame actual
        
        if current_ear < EAR_THRESHOLD:
            self.drowsy_frame_count += 1
        else:
            self.drowsy_frame_count = max(0, self.drowsy_frame_count - 2)
        
        # Check for sudden head movement
        sudden_movement = detect_sudden_head_movement(
            metrics['head_pose'], self.previous_head_pose, self.head_movement_threshold
        )
        if sudden_movement:
            self.head_movement_frames = min(self.head_movement_frames + 1, 10)
        else:
            self.head_movement_frames = max(self.head_movement_frames - 1, 0)

        # HEAD TILT - Detección independiente (restaurada a funcionalidad anterior)
        head_roll = abs(metrics['head_pose']['roll'])
        if head_roll > HEAD_TILT_THRESHOLD:
            self.is_head_tilted = True
            alerts.append('HEAD_TILT')
        else:
            self.is_head_tilted = False

        # DETECCIÓN DE OJOS CERRADOS - Lógica corregida usando EAR inmediato
        if current_ear < EAR_THRESHOLD:
            if self.drowsy_frame_count >= CONSECUTIVE_FRAMES_MICROSLEEP:
                self.is_drowsy = True
                alerts.append('MICROSLEEP')
            elif self.drowsy_frame_count >= CONSECUTIVE_FRAMES_DROWSY:
                self.is_drowsy = True
                alerts.append('DROWSY') 
            elif self.drowsy_frame_count >= 3:  # Respuesta rápida
                alerts.append('EYES_CLOSED')
        else:
            self.is_drowsy = False
        
        # DETECCIÓN DE BOSTEZO MEJORADA - Filtro de habla vs bostezo
        if self.head_movement_frames <= 3 and len(self.mar_history) >= 30:
            # Analizar si es habla o bostezo
            speech_analysis = detect_speech_vs_yawn(self.mar_history, fps=ACTIVE_PROFILE['fps'])
            
            # Solo detectar bostezo si NO es habla y cumple criterios
            if not speech_analysis['is_speech'] and speech_analysis['is_yawn']:
                if speech_analysis['confidence'] > 0.6:  # Alta confianza
                    self.is_yawning = True
                    alerts.append('YAWNING')
                else:
                    self.is_yawning = False
            elif speech_analysis['is_speech']:
                # Es habla, no bostezo - registro para debug
                self.is_yawning = False
                # Opcional: agregar indicador de habla detectada
                metrics['speech_detected'] = True
            else:
                # Fallback a detección simple solo para casos extremos
                if metrics['mar'] > 0.8:  # Threshold muy alto para evitar false positives
                    self.is_yawning = True
                    alerts.append('YAWNING')
                else:
                    self.is_yawning = False
        else:
            self.is_yawning = False

        self.previous_head_pose = metrics['head_pose'].copy()
        metrics['alerts'] = alerts
        metrics['head_movement'] = self.head_movement_frames > 0
        
        # Estado basado en alertas
        if alerts:
            if len(alerts) > 1:
                metrics['status'] = 'CRITICAL'
            else:
                metrics['status'] = 'WARNING'
        else:
            metrics['status'] = 'NORMAL'
    
    def _draw_results(self, frame, metrics):
        """Draw detection results on the frame."""
        h, w, _ = frame.shape
        
        # Choose color based on status
        if metrics['status'] == 'CRITICAL':
            color = COLOR_DANGER
        elif metrics['status'] == 'WARNING':
            color = COLOR_WARNING
        else:
            color = COLOR_NORMAL
        
        # Draw metrics text
        y_offset = TEXT_PADDING
        
        # EAR information con color según el estado
        ear_status = "CERRADOS" if metrics['ear_avg'] < EAR_THRESHOLD else "ABIERTOS"
        ear_color = COLOR_DANGER if metrics['ear_avg'] < EAR_THRESHOLD else COLOR_NORMAL
        cv2.putText(frame, f"EAR: {metrics['ear_avg']:.3f} - OJOS {ear_status}", 
                   (30, y_offset), FONT_FACE, FONT_SCALE, ear_color, FONT_THICKNESS)
        y_offset += 30
        
        # MAR information simplificada
        mar_status = "ABIERTA" if metrics['mar'] > 0.5 else "NORMAL"
        mar_color = COLOR_WARNING if metrics['mar'] > 0.5 else COLOR_NORMAL
        cv2.putText(frame, f"MAR: {metrics['mar']:.3f} - BOCA {mar_status}", 
                   (30, y_offset), FONT_FACE, FONT_SCALE, mar_color, FONT_THICKNESS)
        y_offset += 30
        
        # Contador de frames para debug
        if self.drowsy_frame_count > 0:
            cv2.putText(frame, f"Frames ojos cerrados: {self.drowsy_frame_count}", 
                       (30, y_offset), FONT_FACE, FONT_SCALE, COLOR_WARNING, FONT_THICKNESS)
            y_offset += 30
        
        # Head movement indicator
        if metrics.get('head_movement', False):
            cv2.putText(frame, "HEAD MOVING - Yawn detection paused", 
                       (30, y_offset), FONT_FACE, FONT_SCALE, COLOR_WARNING, FONT_THICKNESS)
            y_offset += 30
        
        # Microsleep information if detected
        if 'microsleep_duration' in metrics:
            cv2.putText(frame, f"Microsleep: {metrics['microsleep_duration']} frames", 
                       (30, y_offset), FONT_FACE, FONT_SCALE, COLOR_DANGER, FONT_THICKNESS)
            y_offset += 30
        
        # Head pose information
        head_pose = metrics['head_pose']
        cv2.putText(frame, f"Head Tilt: {head_pose['roll']:.1f}°", 
                   (30, y_offset), FONT_FACE, FONT_SCALE, color, FONT_THICKNESS)
        y_offset += 30
        
        # Status
        cv2.putText(frame, f"Status: {metrics['status']}", 
                   (30, y_offset), FONT_FACE, 1.0, color, FONT_THICKNESS)
        
        # Alertas - Mostrar todas si hay múltiples
        if metrics['alerts']:
            alert_text = " | ".join(metrics['alerts']) if len(metrics['alerts']) > 1 else metrics['alerts'][0]
            text_size = cv2.getTextSize(alert_text, FONT_FACE, 1.8, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            # Fondo rojo sólido
            cv2.rectangle(frame, 
                         (text_x - 40, text_y - 60),
                         (text_x + text_size[0] + 40, text_y + 40),
                         COLOR_DANGER, -1)
            
            # Texto blanco grande
            cv2.putText(frame, alert_text, (text_x, text_y), 
                       FONT_FACE, 1.8, COLOR_TEXT, 4)
        
        return frame
    
    def _log_metrics(self, metrics):
        """Log metrics to CSV file."""
        current_time = time.time()
        
        # Log at specified intervals
        if current_time - self.last_log_time >= LOG_INTERVAL:
            log_file = get_log_file_path()
            
            with open(log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{metrics['ear_right']:.4f}",
                    f"{metrics['ear_left']:.4f}",
                    f"{metrics['ear_avg']:.4f}",
                    f"{metrics['mar']:.4f}",
                    f"{metrics['head_pose']['roll']:.2f}",
                    f"{metrics['head_pose']['pitch']:.2f}",
                    f"{metrics['head_pose']['yaw']:.2f}",
                    self.is_drowsy,
                    self.is_yawning,
                    self.is_head_tilted,
                    metrics['status']
                ])
            
            self.last_log_time = current_time
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
    
    def get_current_state(self):
        """Get current detection state."""
        return {
            'is_drowsy': self.is_drowsy,
            'is_yawning': self.is_yawning,
            'is_head_tilted': self.is_head_tilted,
            'drowsy_frame_count': self.drowsy_frame_count,
            'yawn_frame_count': self.yawn_frame_count
        }
