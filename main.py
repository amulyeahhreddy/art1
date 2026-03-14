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
    
    # Debug mode and freeze frame state
    debug_mode = False
    freeze_frame = False
    
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
        hand_data = []
        if results.multi_hand_landmarks:
            for i, hand_lm in enumerate(results.multi_hand_landmarks):
                lm_list = [(lm.x, lm.y, lm.z) for lm in hand_lm.landmark]
                handedness = results.multi_handedness[i].classification[0].label
                hand_data.append((lm_list, handedness))

        debug_info = None
        if len(hand_data) == 0:
            mudra, score = "Unknown", 0.0
        elif len(hand_data) == 1:
            if debug_mode:
                mudra, score, debug_info = mudra_recognizer.recognize_single(
                    hand_data[0][0], hand_data[0][1], debug=True)
            else:
                mudra, score = mudra_recognizer.recognize_single(
                    hand_data[0][0], hand_data[0][1])
        else:  # 2 hands
            mudra, score = mudra_recognizer.recognize_two_hand(hand_data[0], hand_data[1])
            
            # Display mudra name and confidence
            if mudra != "Unknown":
                label = f"{mudra}  {int(score*100)}%"
                cv2.putText(frame, label, (20, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,180), 2)
            else:
                cv2.putText(frame, "Unknown", (20, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 2)
            
            # Draw visual effects for detected mudras
            if len(hand_data) == 1:  # Only draw effects for single hand
                lm, handedness = hand_data[0]
                wrist_x = int(lm[0][0] * frame.shape[1])
                wrist_y = int(lm[0][1] * frame.shape[0])
                
                if mudra == "Pataka":
                    # Draw special yellow circle with rays effect above the hand
                    effect_y = wrist_y - 80
                    frame = visual_effects.draw_pataka_effect(frame, wrist_x, effect_y, 40)
                elif mudra != "Unknown":
                    # Draw simple badge for other mudras
                    effect_y = wrist_y - 80
                    frame = visual_effects.draw_mudra_badge(frame, mudra, wrist_x, effect_y, 25)
            
            # Display debug overlay if enabled
            if debug_mode and debug_info:
                # Semi-transparent black background for debug panel
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 90), (350, 200), -1)
                
                # Debug information lines
                angles = debug_info.get('finger_angles', {})
                lines = [
                    f"Angles: T:{angles.get('thumb', 0):.0f} I:{angles.get('index', 0):.0f} M:{angles.get('middle', 0):.0f} R:{angles.get('ring', 0):.0f} P:{angles.get('pinky', 0):.0f}",
                    f"Thumb: {debug_info.get('thumb_state', 'unknown')} | Hand Size: {debug_info.get('hand_size', 0):.3f}",
                    f"Groups: {debug_info.get('stage1_groups', [])}",
                ]
                
                if 'top5' in debug_info:
                    lines.append("Top 5 Scores:")
                    for i, (name, scr) in enumerate(debug_info['top5']):
                        lines.append(f"{i+1}. {name}: {scr:.2f}")
                
                # Draw debug text
                for i, line in enumerate(lines):
                    cv2.putText(overlay, line, (10, 115 + i*22),
                                cv2.FONT_HERSHEY_PLAIN, 1.0, (200,255,200), 1)
                
                # Blend overlay with original frame
                cv2.addWeighted(overlay, 0.7, 0, 0, 0, frame)
            
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
        elif key == ord('d') or key == ord('D'):  # Toggle debug mode
            debug_mode = not debug_mode
            mudra_recognizer.debug_mode = debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('f') or key == ord('F'):  # Freeze frame
            freeze_frame = not freeze_frame
            print(f"Frame frozen: {'YES' if freeze_frame else 'NO'}")
        elif key == ord('s') or key == ord('S'):  # Save debug snapshot
            if debug_info:
                with open('debug_log.txt', 'a') as f:
                    f.write(f"Debug Snapshot - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Mudra: {mudra}, Score: {score:.2f}\n")
                    f.write(f"Hand Size: {debug_info.get('hand_size', 0):.3f}\n")
                    f.write(f"Groups: {debug_info.get('stage1_groups', [])}\n")
                    f.write(f"Thumb State: {debug_info.get('thumb_state', 'unknown')}\n")
                    angles = debug_info.get('finger_angles', {})
                    f.write(f"Angles: T:{angles.get('thumb', 0):.0f} I:{angles.get('index', 0):.0f} M:{angles.get('middle', 0):.0f} R:{angles.get('ring', 0):.0f} P:{angles.get('pinky', 0):.0f}\n")
                    if 'top5' in debug_info:
                        f.write("Top 5 Scores:\n")
                        for i, (name, scr) in enumerate(debug_info['top5']):
                            f.write(f"  {i+1}. {name}: {scr:.2f}\n")
                print("Debug snapshot saved to debug_log.txt")
            else:
                print("Debug mode not enabled - press 'd' to enable")
    
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
