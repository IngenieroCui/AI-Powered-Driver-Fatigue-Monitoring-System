"""
Test script to verify the Driver Fatigue Detection System installation.
Run this script to check if all dependencies are properly installed.
"""

import sys
import importlib
import traceback

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe', 
        'numpy': 'numpy',
        'pandas': 'pandas',
        'pygame': 'pygame',
        'matplotlib': 'matplotlib'
    }
    
    print("Testing package imports...")
    print("-" * 50)
    
    all_good = True
    
    for package, pip_name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✓ {package} ({pip_name}) - OK")
        except ImportError as e:
            print(f"✗ {package} ({pip_name}) - FAILED")
            print(f"  Error: {e}")
            print(f"  Install with: pip install {pip_name}")
            all_good = False
        except Exception as e:
            print(f"✗ {package} ({pip_name}) - ERROR")
            print(f"  Unexpected error: {e}")
            all_good = False
    
    return all_good

def test_camera():
    """Test if camera can be accessed."""
    print("\nTesting camera access...")
    print("-" * 30)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✓ Camera 0 - Available")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"✓ Camera resolution: {width}x{height}")
            else:
                print("⚠ Camera opened but cannot read frames")
            
            cap.release()
            return True
        else:
            print("✗ Camera 0 - Not accessible")
            print("  Make sure your camera is connected and not being used by another application")
            return False
    
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe face mesh functionality."""
    print("\nTesting MediaPipe face mesh...")
    print("-" * 35)
    
    try:
        import mediapipe as mp
        
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            print("✓ MediaPipe face mesh initialized successfully")
            return True
    
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        traceback.print_exc()
        return False

def test_project_structure():
    """Test if project structure is correct."""
    print("\nTesting project structure...")
    print("-" * 32)
    
    import os
    
    required_dirs = [
        'src',
        'data',
        'data/logs',
        'data/samples',
        'models'
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.MD',
        'src/__init__.py',
        'src/detector.py',
        'src/config.py',
        'src/utils.py'
    ]
    
    all_good = True
    
    # Check directories
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ Directory: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"✓ File: {file_path}")
        else:
            print(f"✗ File missing: {file_path}")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("="*60)
    print("DRIVER FATIGUE DETECTION SYSTEM - INSTALLATION TEST")
    print("="*60)
    
    tests = [
        ("Package imports", test_imports),
        ("Camera access", test_camera), 
        ("MediaPipe functionality", test_mediapipe),
        ("Project structure", test_project_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} - CRASHED")
            print(f"  Error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        icon = "✓" if result else "✗"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The system is ready to use.")
        print("Run: python main.py")
    else:
        print(f"\n⚠ {len(results) - passed} test(s) failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
