import cv2
import time
from hand_tracker import HandTracker
from mudra_recognizer import MudraRecognizer
from visual_effects import VisualEffects


def main():
    # Initialize webcam with fallback logic
    cap = None
    camera_indices = [0, 1, 2]  # Try different camera indices
    
    for idx in camera_indices:
        print(f"Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if ret:
                print(f"Successfully opened camera {idx}")
                break
            else:
                cap.release()
                cap = None
        else:
            if cap:
                cap.release()
                cap = None
    
    if cap is None:
        print("Error: Could not open any webcam. Please check:")
        print("1. Camera is connected and not in use by another application")
        print("2. Camera drivers are properly installed")
        print("3. Camera permissions are granted")
        print("4. Try running: python -c \"import cv2; print('OpenCV version:', cv2.__version__)\"")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize hand tracker, mudra recognizer, and visual effects
    hand_tracker = HandTracker()
    mudra_recognizer = MudraRecognizer()
    visual_effects = VisualEffects()
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    print("Indian Classical Mudra Detection System Started")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame from camera")
            # Try to reinitialize camera
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0)  # Try to reconnect
            if not cap.isOpened():
                print("Error: Lost camera connection")
                break
            continue
        
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Warning: Invalid frame received")
            continue
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Find hands
        results = hand_tracker.find_hands(frame)
        
        # Get hand landmarks and handedness
        hand_landmarks_list = hand_tracker.get_hand_landmarks(results)
        handedness_list = hand_tracker.get_handedness(results)
        
        # Draw landmarks on frame
        frame = hand_tracker.draw_landmarks(frame, results)
        
        # Recognize mudras for each detected hand with proper handedness
        for i, landmarks in enumerate(hand_landmarks_list):
            # Get hand label (Left/Right) from MediaPipe handedness
            hand_label = handedness_list[i] if i < len(handedness_list) else f"Hand {i+1}"
            
            # Recognize mudra with hand type for temporal smoothing
            mudra = mudra_recognizer.recognize_mudra(landmarks, hand_label)
            
            # Display mudra name on frame
            text = f"{hand_label}: {mudra}"
            
            # Calculate text position (above the wrist)
            wrist_x = int(landmarks[0][0] * frame.shape[1])
            wrist_y = int(landmarks[0][1] * frame.shape[0])
            
            # Draw background rectangle for better visibility
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, 
                         (wrist_x - 10, wrist_y - 40),
                         (wrist_x + text_size[0] + 10, wrist_y - 10),
                         (0, 0, 0), -1)
            
            # Draw text with color coding (green for known, red for unknown)
            text_color = (0, 255, 0) if mudra != "Unknown" else (0, 0, 255)
            cv2.putText(frame, text, (wrist_x, wrist_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Draw visual effects for detected mudras
            if mudra == "Pataka":
                # Draw special yellow circle with rays effect above the hand
                effect_y = wrist_y - 80
                frame = visual_effects.draw_pataka_effect(frame, wrist_x, effect_y, 40)
            elif mudra != "Unknown":
                # Draw simple badge for other mudras
                effect_y = wrist_y - 80
                frame = visual_effects.draw_mudra_badge(frame, mudra, wrist_x, effect_y, 25)
            
            # Display enhanced debug info for finger states (if enabled)
            if mudra_recognizer.debug_mode:
                debug_info = mudra_recognizer.get_debug_info()
                debug_y = wrist_y + 20
                
                # Display finger states
                finger_states = debug_info.get('finger_states', {})
                for finger, state in finger_states.items():
                    debug_text = f"{finger}: {state}"
                    cv2.putText(frame, debug_text, (wrist_x, debug_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    debug_y += 15
                
                # Display confidence score
                confidence = debug_info.get('confidence', 0)
                conf_text = f"Confidence: {confidence:.2f}"
                cv2.putText(frame, conf_text, (wrist_x, debug_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                debug_y += 15
                
                # Display finger angles for extended fingers
                finger_angles = debug_info.get('finger_angles', {})
                for finger, angle in finger_angles.items():
                    if finger_states.get(finger) == 'extended':
                        angle_text = f"{finger}: {angle:.0f}°"
                        cv2.putText(frame, angle_text, (wrist_x, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
                        debug_y += 12
        
        # Calculate and display FPS
        fps_counter += 1
        if fps_counter >= 10:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display FPS
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display supported mudras
        mudras_text = "30+ Mudras: Pataka, Tripataka, Ardhapataka, Kartarimukha, Mayura, Ardhachandra, Arala, Shukatunda, Mushti, Shikhara, Kapitta, Katakamukha, Suchi, Chandrakala, Padmakosha, Sarpashirsha, Mrigashirsha, Simhamukha, Kangula, Alapadma, Chatura, Bhramara, Hamsasya, Hamsapaksha, Sandamsha, Mukula, Tamrachuda, Trishula"
        cv2.putText(frame, mudras_text[:120] + "...", (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' or ESC to quit | Debug mode ON", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Indian Classical Mudra Detection', frame)
        
        # Check for quit keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # q, Q, or ESC
            print("Quit key pressed - closing application...")
            break
    
    # Cleanup
    print("Cleaning up resources...")
    
    # Release camera
    if cap is not None and cap.isOpened():
        cap.release()
        print("Camera released")
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    # Force close any remaining windows
    for i in range(4):
        cv2.waitKey(1)
    
    print("Mudra Detection System Stopped")


if __name__ == "__main__":
    main()
