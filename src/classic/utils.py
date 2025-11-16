"""Utility helpers for driver fatigue detection (classic pipeline).

This module focuses on generic helpers such as smoothing, blink pattern
detection and formatting for on-screen display. The raw EAR/MAR and
head-pose calculations now live in:

- ``src.classic.ear_mar``
- ``src.classic.head_pose``
"""


def moving_average(values, window_size=5):
    """Calculate moving average for smoothing metric values.

    Args:
        values: List of numeric values.
        window_size: Size of the moving window.

    Returns:
        float: Moving average value.
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
