"""Head pose estimation utilities.

This module provides a simplified 2D head pose estimation extracted from the
previous `utils.py` module so it can be used independently by the classic
fatigue detector.
"""

import math
import numpy as np


def calculate_head_pose(landmarks, face_indices):
    """Calculate head pose estimation for tilt detection.

    Args:
        landmarks: Facial landmarks from mediapipe.
        face_indices: Mapping or collection of face points (kept for
            backward-compatibility; the current implementation uses fixed
            indices consistent with the original code).

    Returns:
        dict: Dictionary containing pitch, yaw, roll angles.
    """
    # Key facial points for head pose estimation (same indices as before)
    nose_tip = np.array([landmarks[1].x, landmarks[1].y])
    chin = np.array([landmarks[175].x, landmarks[175].y])
    left_eye_corner = np.array([landmarks[33].x, landmarks[33].y])
    right_eye_corner = np.array([landmarks[263].x, landmarks[263].y])

    eye_vector = right_eye_corner - left_eye_corner
    roll_angle = math.atan2(eye_vector[1], eye_vector[0]) * 180 / math.pi

    face_height = np.linalg.norm(nose_tip - chin)
    eye_center = (left_eye_corner + right_eye_corner) / 2
    nose_eye_distance = np.linalg.norm(nose_tip - eye_center)

    pitch_ratio = nose_eye_distance / face_height if face_height > 0 else 0
    pitch_angle = (pitch_ratio - 0.3) * 180

    nose_center_x = nose_tip[0]
    face_center_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
    yaw_offset = nose_center_x - face_center_x
    yaw_angle = yaw_offset * 180

    return {
        "roll": roll_angle,
        "pitch": pitch_angle,
        "yaw": yaw_angle,
    }
