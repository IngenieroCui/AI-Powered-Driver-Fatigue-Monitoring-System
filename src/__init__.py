"""
Driver Fatigue Detection System
A computer vision-based system for detecting driver drowsiness and fatigue.
"""

__version__ = "1.0.0"
__author__ = "Juan David Reyes Cure, Juan Esteban Monroy Castellanos, Camilo Novoa Montenegro"
__email__ = "fatigue.detection@university.edu"

from .detector import FatigueDetector
from .config import *
from .utils import *

__all__ = [
    'FatigueDetector',
    'calculate_EAR',
    'calculate_MAR',
    'calculate_head_pose',
    'moving_average',
    'detect_blink_pattern',
    'format_metrics_display'
]
