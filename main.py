"""
Driver Fatigue Detection System - Main Application
Entry point for the driver fatigue monitoring system.
"""

import cv2
import argparse
import sys
import os
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.classic.detector import FatigueDetector
from src.classic.config import *


class FatigueMonitoringApp:
    """Main application class for the fatigue monitoring system."""
    
    def __init__(self, camera_index=CAMERA_INDEX):
        """
        Initialize the monitoring application.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.detector = None
        self.cap = None
        
        # Validate configuration
        validate_configuration()
    
    def initialize_camera(self):
        """Initialize the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        print(f"Camera {self.camera_index} initialized successfully")
        print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return True
    
    def initialize_detector(self):
        """Initialize the fatigue detector."""
        try:
            self.detector = FatigueDetector()
            print("Fatigue detector initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing detector: {e}")
            return False
    
    def run(self):
        """Main execution loop."""
        print("Starting Driver Fatigue Detection System...")
        print("Press 'q' or ESC to quit")
        print("Press 's' to save current frame")
        print("-" * 50)
        
        # Initialize components
        if not self.initialize_camera():
            return False
        
        if not self.initialize_detector():
            return False
        
        frame_count = 0
        
        try:
            while True:
                # Read frame from camera
                success, frame = self.cap.read()
                if not success:
                    print("Warning: Failed to read frame from camera")
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame for fatigue detection
                processed_frame, metrics = self.detector.detect_fatigue(frame)
                
                # Display frame
                cv2.imshow(WINDOW_NAME, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Save frame
                    self.save_frame(processed_frame, frame_count)
                elif key == ord('d'):  # Toggle debug info
                    self.print_debug_info(metrics)
                
                frame_count += 1
                
                # Skip frames if configured
                if SKIP_FRAMES > 0:
                    for _ in range(SKIP_FRAMES):
                        self.cap.read()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"Error during execution: {e}")
        
        finally:
            self.cleanup()
        
        return True
    
    def save_frame(self, frame, frame_count):
        """Save the current frame to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}_{frame_count}.jpg"
        filepath = os.path.join(DATA_DIR, "samples", filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, frame)
        print(f"Frame saved: {filename}")
    
    def print_debug_info(self, metrics):
        """Print debug information."""
        print("\n" + "="*50)
        print("DEBUG INFO")
        print("="*50)
        print(f"EAR (Avg): {metrics.get('ear_avg', 0):.4f}")
        print(f"MAR: {metrics.get('mar', 0):.4f}")
        
        head_pose = metrics.get('head_pose', {})
        print(f"Head Roll: {head_pose.get('roll', 0):.2f}°")
        print(f"Head Pitch: {head_pose.get('pitch', 0):.2f}°")
        print(f"Head Yaw: {head_pose.get('yaw', 0):.2f}°")
        
        print(f"Status: {metrics.get('status', 'UNKNOWN')}")
        print(f"Alerts: {metrics.get('alerts', [])}")
        
        if self.detector:
            state = self.detector.get_current_state()
            print(f"Drowsy Frame Count: {state['drowsy_frame_count']}")
            print(f"Yawn Frame Count: {state['yawn_frame_count']}")
        
        print("="*50 + "\n")
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up resources...")
        
        if self.cap:
            self.cap.release()
        
        if self.detector:
            self.detector.cleanup()
        
        cv2.destroyAllWindows()
        print("Cleanup completed")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Driver Fatigue Detection System")
    
    parser.add_argument('--camera', '-c', type=int, default=CAMERA_INDEX,
                       help=f'Camera index (default: {CAMERA_INDEX})')
    
    parser.add_argument('--ear-threshold', '-e', type=float, default=EAR_THRESHOLD,
                       help=f'EAR threshold for drowsiness (default: {EAR_THRESHOLD})')
    
    parser.add_argument('--mar-threshold', '-m', type=float, default=MAR_THRESHOLD,
                       help=f'MAR threshold for yawning (default: {MAR_THRESHOLD})')
    
    parser.add_argument('--consecutive-frames', '-f', type=int, default=CONSECUTIVE_FRAMES_DROWSY,
                       help=f'Consecutive frames for drowsy alert (default: {CONSECUTIVE_FRAMES_DROWSY})')
    
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug mode')
    
    parser.add_argument('--no-logging', action='store_true',
                       help='Disable logging')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Update configuration based on arguments
    if args.ear_threshold != EAR_THRESHOLD:
        globals()['EAR_THRESHOLD'] = args.ear_threshold
    
    if args.mar_threshold != MAR_THRESHOLD:
        globals()['MAR_THRESHOLD'] = args.mar_threshold
    
    if args.consecutive_frames != CONSECUTIVE_FRAMES_DROWSY:
        globals()['CONSECUTIVE_FRAMES_DROWSY'] = args.consecutive_frames
    
    if args.debug:
        globals()['DEBUG_MODE'] = True
    
    if args.no_logging:
        globals()['ENABLE_LOGGING'] = False
    
    # Print system information
    print("="*60)
    print("DRIVER FATIGUE DETECTION SYSTEM")
    print("="*60)
    print(f"Version: 1.0")
    print(f"Camera Index: {args.camera}")
    print(f"EAR Threshold: {args.ear_threshold}")
    print(f"MAR Threshold: {args.mar_threshold}")
    print(f"Consecutive Frames: {args.consecutive_frames}")
    print(f"Debug Mode: {args.debug}")
    print(f"Logging Enabled: {not args.no_logging}")
    print("="*60)
    
    # Create and run the application
    app = FatigueMonitoringApp(camera_index=args.camera)
    success = app.run()
    
    if success:
        print("Application completed successfully")
    else:
        print("Application encountered errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
