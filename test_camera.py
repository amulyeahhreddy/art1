import cv2

def test_cameras():
    """Test available camera indices and display their properties"""
    print("Testing available cameras...")
    
    for idx in range(3):  # Test indices 0, 1, 2
        cap = cv2.VideoCapture(idx)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"Camera {idx}:")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps}")
                print(f"  Frame shape: {frame.shape}")
                print(f"  Status: Working ✓")
                
                # Try to display a test frame
                cv2.imshow(f"Camera {idx} Test", frame)
                print(f"  Test window opened. Press any key to continue...")
                cv2.waitKey(2000)  # Wait 2 seconds
                cv2.destroyWindow(f"Camera {idx} Test")
                
            else:
                print(f"Camera {idx}: Opened but cannot read frames")
            
            cap.release()
        else:
            print(f"Camera {idx}: Not available")
    
    print("\nCamera test complete.")

if __name__ == "__main__":
    test_cameras()
