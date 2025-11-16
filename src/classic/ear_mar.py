"""EAR and MAR calculation utilities.

This module provides Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
calculations extracted from the previous `utils.py` module so they can be
used independently by the classic fatigue detector.
"""

import numpy as np


def calculate_EAR(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR) for fatigue detection.

    Args:
        landmarks: Facial landmarks from mediapipe.
        eye_indices: List of 6 indices representing eye landmarks.

    Returns:
        float: Eye Aspect Ratio value.
    """
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])

    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def calculate_MAR(landmarks, mouth_indices):
    """Calculate Mouth Aspect Ratio (MAR) for yawn detection.

    Args:
        landmarks: Facial landmarks from mediapipe.
        mouth_indices: List of indices representing mouth landmarks.

    Returns:
        float: Mouth Aspect Ratio value.
    """
    mouth_points = []
    for idx in mouth_indices:
        point = np.array([landmarks[idx].x, landmarks[idx].y])
        mouth_points.append(point)

    vertical1 = np.linalg.norm(mouth_points[1] - mouth_points[7])
    vertical2 = np.linalg.norm(mouth_points[2] - mouth_points[6])
    vertical3 = np.linalg.norm(mouth_points[3] - mouth_points[5])
    horizontal = np.linalg.norm(mouth_points[0] - mouth_points[4])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2 + vertical3) / (3.0 * horizontal)
